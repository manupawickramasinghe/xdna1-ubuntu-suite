//! Minimal ONNX interpreter for face-embedding CNNs (batch 1).
//!
//! NPU (via [`Matmul`]): Conv with group == 1 (im2col), Gemm, MatMul with constant B.
//! CPU: depthwise Conv, BatchNormalization (folded into the producing Conv when
//! possible), PRelu/Relu/Add/Sub/Mul/Div, Flatten/Reshape/Dropout/Identity,
//! GlobalAveragePool.

use crate::npu::{Matmul, PackedB};
use anyhow::{anyhow, bail, Context, Result};
use prost::Message;
use rayon::prelude::*;
use std::collections::HashMap;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, data }
    }
}

fn tensor_from_proto(t: &onnx::TensorProto) -> Result<(Vec<usize>, Vec<f32>)> {
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    let raw = t.raw_data();
    let data: Vec<f32> = match t.data_type() {
        1 => {
            if !raw.is_empty() {
                raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            } else {
                t.float_data.clone()
            }
        }
        7 => {
            if !raw.is_empty() {
                raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
            } else {
                t.int64_data.iter().map(|&v| v as f32).collect()
            }
        }
        dt => bail!("tensor {}: unsupported data type {dt}", t.name()),
    };
    if data.len() != n.max(1) && !(n == 0 && data.len() == 1) {
        bail!("tensor {}: {} values for shape {:?}", t.name(), data.len(), shape);
    }
    Ok((shape, data))
}

#[derive(Default, Clone)]
struct Attrs {
    ints: HashMap<String, Vec<i64>>,
    floats: HashMap<String, f32>,
}

impl Attrs {
    fn from(n: &onnx::NodeProto) -> Self {
        let mut a = Attrs::default();
        for at in &n.attribute {
            let name = at.name().to_string();
            if !at.ints.is_empty() {
                a.ints.insert(name, at.ints.clone());
            } else if at.r#type() == onnx::attribute_proto::AttributeType::Int {
                a.ints.insert(name, vec![at.i()]);
            } else if at.r#type() == onnx::attribute_proto::AttributeType::Float {
                a.floats.insert(name, at.f());
            }
        }
        a
    }
    fn i(&self, k: &str, d: i64) -> i64 {
        self.ints.get(k).and_then(|v| v.first().copied()).unwrap_or(d)
    }
    fn is(&self, k: &str, d: &[i64]) -> Vec<i64> {
        self.ints.get(k).cloned().unwrap_or_else(|| d.to_vec())
    }
    fn f(&self, k: &str, d: f32) -> f32 {
        self.floats.get(k).copied().unwrap_or(d)
    }
}

struct Node {
    op: String,
    inputs: Vec<String>,
    output: String,
    attrs: Attrs,
    /// packed weight matrix for NPU ops
    packed: Option<PackedB>,
}

pub struct Model {
    consts: HashMap<String, Tensor>,
    nodes: Vec<Node>,
    pub input: String,
    pub output: String,
    pub npu_ops: usize,
}

fn conv_weight_matrix(w: &Tensor) -> Vec<f32> {
    // [OC, C*kh*kw] -> [C*kh*kw, OC]
    let oc = w.shape[0];
    let k = w.data.len() / oc;
    let mut b = vec![0f32; k * oc];
    for o in 0..oc {
        for i in 0..k {
            b[i * oc + o] = w.data[o * k + i];
        }
    }
    b
}

impl Model {
    pub fn load(path: &str, mm: &dyn Matmul) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let model = onnx::ModelProto::decode(bytes.as_slice()).context("decoding ONNX")?;
        let g = model.graph.ok_or_else(|| anyhow!("model has no graph"))?;

        let mut consts = HashMap::new();
        for t in &g.initializer {
            let (s, d) = tensor_from_proto(t)?;
            consts.insert(t.name().to_string(), Tensor::new(s, d));
        }
        let mut nodes = Vec::new();
        for n in &g.node {
            if n.op_type() == "Constant" {
                let t = n.attribute.iter().find_map(|a| a.t.as_ref()).ok_or_else(|| anyhow!("Constant without tensor"))?;
                let (s, d) = tensor_from_proto(t)?;
                let s = if s.is_empty() { vec![1] } else { s };
                consts.insert(n.output[0].clone(), Tensor::new(s, d));
                continue;
            }
            nodes.push(Node {
                op: n.op_type().to_string(),
                inputs: n.input.clone(),
                output: n.output[0].clone(),
                attrs: Attrs::from(n),
                packed: None,
            });
        }
        let input = g.input.iter().map(|i| i.name().to_string()).find(|n| !consts.contains_key(n)).ok_or_else(|| anyhow!("no graph input"))?;
        let output = g.output[0].name().to_string();

        let mut m = Model { consts, nodes, input, output, npu_ops: 0 };
        m.fold_batchnorm();
        m.pack(mm)?;
        Ok(m)
    }

    /// Fold BatchNormalization into the Conv producing its input (inference only).
    fn fold_batchnorm(&mut self) {
        let mut uses: HashMap<String, usize> = HashMap::new();
        for n in &self.nodes {
            for i in &n.inputs {
                *uses.entry(i.clone()).or_default() += 1;
            }
        }
        let mut drop = vec![false; self.nodes.len()];
        for bi in 0..self.nodes.len() {
            if self.nodes[bi].op != "BatchNormalization" {
                continue;
            }
            let src = self.nodes[bi].inputs[0].clone();
            let Some(ci) = self.nodes.iter().position(|n| n.output == src) else { continue };
            if self.nodes[ci].op != "Conv" || uses.get(&src).copied().unwrap_or(0) != 1 {
                continue;
            }
            let bn = &self.nodes[bi];
            let eps = bn.attrs.f("epsilon", 1e-5);
            let [s, b, mu, var] = [1, 2, 3, 4].map(|i| self.consts[&bn.inputs[i]].data.clone());
            let k: Vec<f32> = s.iter().zip(&var).map(|(s, v)| s / (v + eps).sqrt()).collect();
            let bn_out = bn.output.clone();
            let conv = &self.nodes[ci];
            let wname = conv.inputs[1].clone();
            let old_bias = conv.inputs.get(2).filter(|n| !n.is_empty()).map(|n| self.consts[n].data.clone());
            let w = self.consts.get_mut(&wname).unwrap();
            let per = w.data.len() / k.len();
            for (o, kk) in k.iter().enumerate() {
                for v in &mut w.data[o * per..(o + 1) * per] {
                    *v *= kk;
                }
            }
            let ob = old_bias.unwrap_or_else(|| vec![0.0; k.len()]);
            let nb: Vec<f32> = (0..k.len()).map(|o| (ob[o] - mu[o]) * k[o] + b[o]).collect();
            let bname = format!("{bn_out}__folded_bias");
            self.consts.insert(bname.clone(), Tensor::new(vec![nb.len()], nb));
            let conv = &mut self.nodes[ci];
            conv.inputs.truncate(2);
            conv.inputs.push(bname);
            conv.output = bn_out;
            drop[bi] = true;
        }
        let mut i = 0;
        self.nodes.retain(|_| {
            let keep = !drop[i];
            i += 1;
            keep
        });
    }

    fn pack(&mut self, mm: &dyn Matmul) -> Result<()> {
        for n in &mut self.nodes {
            match n.op.as_str() {
                "Conv" if n.attrs.i("group", 1) == 1 => {
                    let w = &self.consts[&n.inputs[1]];
                    let b = conv_weight_matrix(w);
                    n.packed = Some(mm.pack(&b, b.len() / w.shape[0], w.shape[0]));
                }
                "Gemm" => {
                    let w = &self.consts[&n.inputs[1]];
                    let alpha = n.attrs.f("alpha", 1.0);
                    let (r, c) = (w.shape[0], w.shape[1]);
                    // want B as [K, N]
                    let (k, nn, b): (usize, usize, Vec<f32>) = if n.attrs.i("transB", 0) == 1 {
                        let mut t = vec![0f32; r * c];
                        for i in 0..r {
                            for j in 0..c {
                                t[j * r + i] = w.data[i * c + j] * alpha;
                            }
                        }
                        (c, r, t)
                    } else {
                        (r, c, w.data.iter().map(|v| v * alpha).collect())
                    };
                    n.packed = Some(mm.pack(&b, k, nn));
                }
                "MatMul" if self.consts.contains_key(&n.inputs[1]) => {
                    let w = &self.consts[&n.inputs[1]];
                    n.packed = Some(mm.pack(&w.data, w.shape[0], w.shape[1]));
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn run(&mut self, mm: &mut dyn Matmul, x: Tensor) -> Result<Tensor> {
        let mut env: HashMap<&str, Tensor> = HashMap::new();
        env.insert(self.input.as_str(), x);
        let mut npu_ops = 0;
        for n in &self.nodes {
            let get = |i: usize| -> Result<&Tensor> {
                let name = &n.inputs[i];
                env.get(name.as_str()).or_else(|| self.consts.get(name)).ok_or_else(|| anyhow!("missing tensor {name}"))
            };
            let y = match n.op.as_str() {
                "Conv" => {
                    let x = get(0)?;
                    let w = &self.consts[&n.inputs[1]];
                    let bias = n.inputs.get(2).filter(|s| !s.is_empty()).map(|s| &self.consts[s]);
                    if n.packed.is_some() {
                        npu_ops += 1;
                    }
                    conv(x, w, bias, &n.attrs, n.packed.as_ref(), mm)?
                }
                "Gemm" | "MatMul" if n.packed.is_some() => {
                    let a = get(0)?;
                    let k = *a.shape.last().unwrap();
                    let m = a.data.len() / k;
                    let pb = n.packed.as_ref().unwrap();
                    let mut c = mm.matmul(&a.data, m, pb)?;
                    npu_ops += 1;
                    if n.op == "Gemm" && n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                        let bias = get(2)?;
                        let beta = n.attrs.f("beta", 1.0);
                        for row in c.chunks_mut(pb.n) {
                            for (j, v) in row.iter_mut().enumerate() {
                                *v += beta * bias.data[j % bias.data.len()];
                            }
                        }
                    }
                    Tensor::new(vec![m, pb.n], c)
                }
                "BatchNormalization" => {
                    let x = get(0)?;
                    let [s, b, mu, var] = [1, 2, 3, 4].map(|i| &self.consts[&n.inputs[i]].data);
                    let eps = n.attrs.f("epsilon", 1e-5);
                    let c = x.shape[1];
                    let per = x.data.len() / (x.shape[0] * c);
                    let mut d = x.data.clone();
                    d.par_chunks_mut(per).enumerate().for_each(|(i, ch)| {
                        let ci = i % c;
                        let k = s[ci] / (var[ci] + eps).sqrt();
                        let off = b[ci] - mu[ci] * k;
                        ch.iter_mut().for_each(|v| *v = *v * k + off);
                    });
                    Tensor::new(x.shape.clone(), d)
                }
                "PRelu" => {
                    let x = get(0)?;
                    let s = get(1)?;
                    if s.data.len() == x.shape[1] {
                        let per = x.data.len() / (x.shape[0] * x.shape[1]);
                        let c = x.shape[1];
                        let mut d = x.data.clone();
                        d.par_chunks_mut(per).enumerate().for_each(|(i, ch)| {
                            let a = s.data[i % c];
                            ch.iter_mut().for_each(|v| {
                                if *v < 0.0 {
                                    *v *= a
                                }
                            });
                        });
                        Tensor::new(x.shape.clone(), d)
                    } else {
                        binary(x, s, |v, a| if v < 0.0 { v * a } else { v })?
                    }
                }
                "Relu" => {
                    let x = get(0)?;
                    Tensor::new(x.shape.clone(), x.data.iter().map(|v| v.max(0.0)).collect())
                }
                "Add" => binary(get(0)?, get(1)?, |a, b| a + b)?,
                "Sub" => binary(get(0)?, get(1)?, |a, b| a - b)?,
                "Mul" => binary(get(0)?, get(1)?, |a, b| a * b)?,
                "Div" => binary(get(0)?, get(1)?, |a, b| a / b)?,
                "Dropout" | "Identity" => get(0)?.clone(),
                "Flatten" => {
                    let x = get(0)?;
                    let ax = n.attrs.i("axis", 1) as usize;
                    let outer: usize = x.shape[..ax].iter().product();
                    Tensor::new(vec![outer, x.data.len() / outer.max(1)], x.data.clone())
                }
                "Reshape" => {
                    let x = get(0)?;
                    let spec: Vec<i64> = get(1)?.data.iter().map(|&v| v as i64).collect();
                    let total = x.data.len();
                    let known: usize = spec.iter().enumerate().map(|(i, &d)| if d == 0 { x.shape[i] } else if d < 0 { 1 } else { d as usize }).product();
                    let shape = spec.iter().enumerate().map(|(i, &d)| if d == 0 { x.shape[i] } else if d < 0 { total / known } else { d as usize }).collect();
                    Tensor::new(shape, x.data.clone())
                }
                "GlobalAveragePool" => {
                    let x = get(0)?;
                    let (nb, c) = (x.shape[0], x.shape[1]);
                    let per = x.data.len() / (nb * c);
                    let d = x.data.chunks(per).map(|ch| ch.iter().sum::<f32>() / per as f32).collect();
                    Tensor::new(vec![nb, c, 1, 1], d)
                }
                op => bail!("ONNX op {op} is not supported"),
            };
            env.insert(n.output.as_str(), y);
        }
        self.npu_ops = npu_ops;
        env.remove(self.output.as_str()).ok_or_else(|| anyhow!("output {} not produced", self.output))
    }
}

/// numpy-style broadcasting for elementwise binary ops.
fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32 + Sync) -> Result<Tensor> {
    let r = a.shape.len().max(b.shape.len());
    let pad = |s: &[usize]| -> Vec<usize> { std::iter::repeat(1).take(r - s.len()).chain(s.iter().copied()).collect() };
    let (sa, sb) = (pad(&a.shape), pad(&b.shape));
    let mut out = vec![0usize; r];
    for i in 0..r {
        out[i] = match (sa[i], sb[i]) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => bail!("cannot broadcast {x} and {y}"),
        };
    }
    let strides = |s: &[usize]| -> Vec<usize> {
        let mut st = vec![0; r];
        let mut acc = 1;
        for i in (0..r).rev() {
            st[i] = if s[i] == 1 { 0 } else { acc };
            acc *= s[i];
        }
        st
    };
    let (ta, tb) = (strides(&sa), strides(&sb));
    let total: usize = out.iter().product();
    let data = (0..total)
        .into_par_iter()
        .map(|mut idx| {
            let (mut ia, mut ib) = (0, 0);
            for d in (0..r).rev() {
                let c = idx % out[d];
                idx /= out[d];
                ia += c * ta[d];
                ib += c * tb[d];
            }
            f(a.data[ia], b.data[ib])
        })
        .collect();
    Ok(Tensor::new(out, data))
}

fn conv(x: &Tensor, w: &Tensor, bias: Option<&Tensor>, at: &Attrs, packed: Option<&PackedB>, mm: &mut dyn Matmul) -> Result<Tensor> {
    let (c, h, wd) = (x.shape[1], x.shape[2], x.shape[3]);
    let (oc, kh, kw) = (w.shape[0], w.shape[2], w.shape[3]);
    let s = at.is("strides", &[1, 1]);
    let dl = at.is("dilations", &[1, 1]);
    let p = at.is("pads", &[0, 0, 0, 0]);
    let (sh, sw, dh, dw) = (s[0] as usize, s[1] as usize, dl[0] as usize, dl[1] as usize);
    let (pt, pl, pb, pr) = (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize);
    let oh = (h + pt + pb - dh * (kh - 1) - 1) / sh + 1;
    let ow = (wd + pl + pr - dw * (kw - 1) - 1) / sw + 1;
    let group = at.i("group", 1) as usize;
    let px = |ci: usize, yy: isize, xx: isize| -> f32 {
        if yy < 0 || xx < 0 || yy as usize >= h || xx as usize >= wd {
            0.0
        } else {
            x.data[(ci * h + yy as usize) * wd + xx as usize]
        }
    };

    let mut y = vec![0f32; oc * oh * ow];
    if let Some(pk) = packed {
        // im2col: A[OH*OW, C*kh*kw]
        let k = c * kh * kw;
        let mut a = vec![0f32; oh * ow * k];
        a.par_chunks_mut(k).enumerate().for_each(|(pos, row)| {
            let (oy, ox) = (pos / ow, pos % ow);
            for ci in 0..c {
                for i in 0..kh {
                    for j in 0..kw {
                        let yy = (oy * sh + i * dh) as isize - pt as isize;
                        let xx = (ox * sw + j * dw) as isize - pl as isize;
                        row[(ci * kh + i) * kw + j] = px(ci, yy, xx);
                    }
                }
            }
        });
        let cm = mm.matmul(&a, oh * ow, pk)?; // [OH*OW, OC]
        y.par_chunks_mut(oh * ow).enumerate().for_each(|(o, plane)| {
            for (pos, v) in plane.iter_mut().enumerate() {
                *v = cm[pos * oc + o];
            }
        });
    } else if group == c && group == oc {
        // depthwise
        y.par_chunks_mut(oh * ow).enumerate().for_each(|(ci, plane)| {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0f32;
                    for i in 0..kh {
                        for j in 0..kw {
                            let yy = (oy * sh + i * dh) as isize - pt as isize;
                            let xx = (ox * sw + j * dw) as isize - pl as isize;
                            acc += w.data[(ci * kh + i) * kw + j] * px(ci, yy, xx);
                        }
                    }
                    plane[oy * ow + ox] = acc;
                }
            }
        });
    } else {
        bail!("grouped conv with group={group} (C={c}, OC={oc}) is not supported");
    }
    if let Some(b) = bias {
        y.par_chunks_mut(oh * ow).enumerate().for_each(|(o, plane)| plane.iter_mut().for_each(|v| *v += b.data[o]));
    }
    Ok(Tensor::new(vec![1, oc, oh, ow], y))
}
