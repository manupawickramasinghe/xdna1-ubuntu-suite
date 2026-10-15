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
    /// runs as a matmul (regular Conv via im2col, Gemm, MatMul with constant B)
    npu: bool,
    /// weights packed for the matmul backend; built on first run, when M is known
    packed: Option<PackedB>,
}

pub struct Model {
    /// per-op-type wall time (seconds), filled when NPU_FACE_PROFILE is set
    pub profile: HashMap<String, f64>,
    /// when set, every NPU conv is recomputed on the CPU with bf16 numerics and
    /// the worst (lowest cosine, highest max-error) layer is recorded
    pub validate: bool,
    /// check only this NPU conv (index among NPU convs) on the next run; used to
    /// spot-check one rotating layer per request in production
    pub spot: Option<usize>,
    pub worst_layer: Option<(f64, f32, String)>,
    /// number of NPU convs (known after the first run)
    pub npu_convs: usize,
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
    pub fn load(path: &str) -> Result<Self> {
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
                npu: false,
                packed: None,
            });
        }
        let input = g.input.iter().map(|i| i.name().to_string()).find(|n| !consts.contains_key(n)).ok_or_else(|| anyhow!("no graph input"))?;
        let output = g.output[0].name().to_string();

        let mut m = Model { profile: HashMap::new(), validate: false, spot: None, worst_layer: None, npu_convs: 0, consts, nodes, input, output, npu_ops: 0 };
        m.fold_batchnorm();
        m.mark_matmuls();
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

    fn mark_matmuls(&mut self) {
        for n in &mut self.nodes {
            n.npu = match n.op.as_str() {
                "Conv" => n.attrs.i("group", 1) == 1,
                "Gemm" => true,
                "MatMul" => self.consts.contains_key(&n.inputs[1]),
                _ => false,
            };
        }
    }

    /// B matrix [K, N] (row-major) of a matmul node.
    fn weight_matrix(&self, n: &Node) -> (Vec<f32>, usize, usize) {
        let w = &self.consts[&n.inputs[1]];
        match n.op.as_str() {
            "Conv" => {
                let b = conv_weight_matrix(w);
                let oc = w.shape[0];
                (b.clone(), b.len() / oc, oc)
            }
            "Gemm" => {
                let alpha = n.attrs.f("alpha", 1.0);
                let (r, c) = (w.shape[0], w.shape[1]);
                if n.attrs.i("transB", 0) == 1 {
                    let mut t = vec![0f32; r * c];
                    for i in 0..r {
                        for j in 0..c {
                            t[j * r + i] = w.data[i * c + j] * alpha;
                        }
                    }
                    (t, c, r)
                } else {
                    (w.data.iter().map(|v| v * alpha).collect(), r, c)
                }
            }
            _ => (w.data.clone(), w.shape[0], w.shape[1]),
        }
    }

    pub fn run(&mut self, mm: &mut dyn Matmul, x: Tensor) -> Result<Tensor> {
        let mut env: HashMap<String, Tensor> = HashMap::new();
        env.insert(self.input.clone(), x);
        let mut npu_ops = 0;
        let profiling = std::env::var_os("NPU_FACE_PROFILE").is_some();
        let mut worst: Option<(f64, f32, String)> = None;
        let mut conv_idx = 0usize;
        for idx in 0..self.nodes.len() {
            let t_op = std::time::Instant::now();
            if self.nodes[idx].npu && self.nodes[idx].packed.is_none() {
                let n = &self.nodes[idx];
                let src = env.get(&n.inputs[0]).or_else(|| self.consts.get(&n.inputs[0])).ok_or_else(|| anyhow!("missing tensor {}", n.inputs[0]))?;
                let m = match n.op.as_str() {
                    "Conv" => conv_out_hw(src, &self.consts[&n.inputs[1]], &n.attrs).iter().product(),
                    _ => src.data.len() / src.shape.last().unwrap(),
                };
                let (b, k, nn) = self.weight_matrix(n);
                let packed = mm.pack(&b, m, k, nn);
                self.nodes[idx].packed = Some(packed);
            }
            let n = &self.nodes[idx];
            let get = |i: usize| -> Result<&Tensor> {
                let name = &n.inputs[i];
                env.get(name).or_else(|| self.consts.get(name)).ok_or_else(|| anyhow!("missing tensor {name}"))
            };
            let y = match n.op.as_str() {
                "Conv" => {
                    let x = get(0)?;
                    let w = &self.consts[&n.inputs[1]];
                    let bias = n.inputs.get(2).filter(|s| !s.is_empty()).map(|s| &self.consts[s]);
                    if n.packed.is_some() {
                        npu_ops += 1;
                    }
                    let check = n.npu && (self.validate || self.spot == Some(conv_idx));
                    if n.npu {
                        conv_idx += 1;
                    }
                    let v = if check { Some(&mut worst) } else { None };
                    conv(x, w, bias, &n.attrs, n.packed.as_ref(), mm, v)?
                }
                "Gemm" | "MatMul" if n.npu => {
                    let a = get(0)?;
                    let k = *a.shape.last().unwrap();
                    let m = a.data.len() / k;
                    let pb = n.packed.as_ref().unwrap();
                    let mut c = mm.matmul(&a.data, m, pb)?;
                    npu_ops += 1;
                    if n.op == "Gemm" && n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                        let bias = get(2)?;
                        let beta = n.attrs.f("beta", 1.0);
                        for row in c.chunks_mut(pb.n()) {
                            for (j, v) in row.iter_mut().enumerate() {
                                *v += beta * bias.data[j % bias.data.len()];
                            }
                        }
                    }
                    Tensor::new(vec![m, pb.n()], c)
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
            if profiling {
                *self.profile.entry(n.op.clone()).or_default() += t_op.elapsed().as_secs_f64();
            }
            env.insert(n.output.clone(), y);
        }
        self.npu_ops = npu_ops;
        self.npu_convs = conv_idx;
        if self.validate || self.spot.is_some() {
            self.worst_layer = worst;
        }
        env.remove(&self.output).ok_or_else(|| anyhow!("output {} not produced", self.output))
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

/// Fill one output row (all ox for a given oy) of the im2col matrix. Zero
/// padding is skipped: callers start from a zeroed buffer.
#[allow(clippy::too_many_arguments)]
fn im2col_row(
    x: &Tensor,
    oy: usize,
    ow: usize,
    (kh, kw): (usize, usize),
    (sh, sw): (usize, usize),
    (dh, dw): (usize, usize),
    (pt, pl): (usize, usize),
    k: usize,
    mut put: impl FnMut(usize, f32),
) {
    let (c, h, wd) = (x.shape[1], x.shape[2], x.shape[3]);
    for ci in 0..c {
        for i in 0..kh {
            let yy = (oy * sh + i * dh) as isize - pt as isize;
            if yy < 0 || yy as usize >= h {
                continue;
            }
            let src = &x.data[(ci * h + yy as usize) * wd..][..wd];
            for j in 0..kw {
                let col = (ci * kh + i) * kw + j;
                // valid ox range: 0 <= ox*sw + j*dw - pl < wd
                let off = (j * dw) as isize - pl as isize;
                let lo = if off >= 0 { 0 } else { ((-off) as usize).div_ceil(sw) };
                let hi = if (wd as isize - off) <= 0 { 0 } else { (((wd as isize - off) as usize) + sw - 1) / sw };
                for ox in lo..hi.min(ow) {
                    put(ox * k + col, src[(ox * sw) as usize + off as usize]);
                }
            }
        }
    }
}

fn conv_out_hw(x: &Tensor, w: &Tensor, at: &Attrs) -> [usize; 2] {
    let (h, wd) = (x.shape[2], x.shape[3]);
    let (kh, kw) = (w.shape[2], w.shape[3]);
    let s = at.is("strides", &[1, 1]);
    let dl = at.is("dilations", &[1, 1]);
    let p = at.is("pads", &[0, 0, 0, 0]);
    let oh = (h + (p[0] + p[2]) as usize - dl[0] as usize * (kh - 1) - 1) / s[0] as usize + 1;
    let ow = (wd + (p[1] + p[3]) as usize - dl[1] as usize * (kw - 1) - 1) / s[1] as usize + 1;
    [oh, ow]
}

#[allow(clippy::too_many_arguments)]
fn conv(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    at: &Attrs,
    packed: Option<&PackedB>,
    mm: &mut dyn Matmul,
    validate: Option<&mut Option<(f64, f32, String)>>,
) -> Result<Tensor> {
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
        // im2col: A[OH*OW, C*kh*kw]; one output row (oy) per task, edge-aware so
        // the inner loop over ox has no per-element bounds checks.
        let k = c * kh * kw;
        let cm = if mm.wants_bf16(pk) {
            let mut a = vec![0u16; oh * ow * k];
            a.par_chunks_mut(ow * k).enumerate().for_each(|(oy, blk)| {
                im2col_row(x, oy, ow, (kh, kw), (sh, sw), (dh, dw), (pt, pl), k, |idx, v| blk[idx] = crate::npu::bf16(v));
            });
            mm.matmul_bf16(&a, oh * ow, pk)?
        } else {
            let mut a = vec![0f32; oh * ow * k];
            a.par_chunks_mut(ow * k).enumerate().for_each(|(oy, blk)| {
                im2col_row(x, oy, ow, (kh, kw), (sh, sw), (dh, dw), (pt, pl), k, |idx, v| blk[idx] = v);
            });
            mm.matmul(&a, oh * ow, pk)?
        }; // [OH*OW, OC]
        if validate.is_some() || std::env::var_os("NPU_FACE_CHECK").is_some() {
            let mut a = vec![0f32; oh * ow * k];
            a.par_chunks_mut(ow * k).enumerate().for_each(|(oy, blk)| {
                im2col_row(x, oy, ow, (kh, kw), (sh, sw), (dh, dw), (pt, pl), k, |idx, v| blk[idx] = v);
            });
            let b = conv_weight_matrix(w);
            let r = crate::npu::cpu_matmul(&a, k, oc, &b.iter().map(|&v| crate::npu::from_bf16(crate::npu::bf16(v))).collect::<Vec<_>>(), true);
            let num: f64 = cm.iter().zip(&r).map(|(p, q)| (*p as f64) * (*q as f64)).sum();
            let den = (cm.iter().map(|p| (*p as f64).powi(2)).sum::<f64>() * r.iter().map(|q| (*q as f64).powi(2)).sum::<f64>()).sqrt();
            let maxerr = cm.iter().zip(&r).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            let maxref = r.iter().map(|q| q.abs()).fold(0f32, f32::max);
            let (cosv, rel) = (num / den, maxerr / maxref.max(1e-12));
            if std::env::var_os("NPU_FACE_CHECK").is_some() {
                eprintln!("check conv M={} K={} N={}: cosine {:.6} max_err/max_ref {:.2e}", oh * ow, k, oc, cosv, rel);
            }
            if let Some(v) = validate {
                let name = format!("conv M={} K={} N={}", oh * ow, k, oc);
                let worse = match v { None => true, Some((c, r, _)) => cosv < *c || rel > *r };
                if worse {
                    *v = Some((cosv.min(v.as_ref().map_or(1.0, |x| x.0)), rel.max(v.as_ref().map_or(0.0, |x| x.1)), name));
                }
            }
        }
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
