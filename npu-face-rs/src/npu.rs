//! Tiled bf16 matmul on the XDNA NPU through one fixed-shape XRT kernel.

use anyhow::{bail, Result};
use std::ffi::{c_char, CString};
use std::time::Instant;

#[repr(C)]
struct XmmRaw {
    _p: [u8; 0],
}

extern "C" {
    fn xmm_open(xclbin: *const c_char, insts: *const c_char, m: u32, k: u32, n: u32, err: *mut c_char, len: u32) -> *mut XmmRaw;
    fn xmm_a(h: *mut XmmRaw) -> *mut u16;
    fn xmm_b(h: *mut XmmRaw) -> *mut u16;
    fn xmm_run(h: *mut XmmRaw, timeout_ms: u32, err: *mut c_char, len: u32) -> *const f32;
    fn xmm_close(h: *mut XmmRaw);
}

/// Round-to-nearest-even f32 -> bf16.
#[inline]
pub fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

#[inline]
pub fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn err_str(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Weights of one matmul, pre-converted to bf16 and pre-split into kernel tiles.
pub struct PackedB {
    pub k: usize,
    pub n: usize,
    /// tiles[kb][nb] = TK x TN row-major bf16 (zero padded); NPU path
    tiles: Vec<Vec<Vec<u16>>>,
    /// full-precision K x N copy; CPU reference path
    f32_cache: Option<Vec<f32>>,
}

pub trait Matmul: Send {
    /// C[m,n] = A[m,k] (row-major f32, rounded to bf16) x B (packed), f32 accumulate.
    fn matmul(&mut self, a: &[f32], m: usize, b: &PackedB) -> Result<Vec<f32>>;
    fn pack(&self, b_kn: &[f32], k: usize, n: usize) -> PackedB;
    fn stats(&self) -> (u64, f64) {
        (0, 0.0)
    }
}

/// CPU reference, for self-tests. `bf16_inputs` reproduces the NPU's numerics
/// (bf16-rounded A and B, f32 accumulation); otherwise exact f32.
pub struct CpuMatmul {
    pub bf16_inputs: bool,
}

#[inline]
fn round_bf16(x: f32) -> f32 {
    from_bf16(bf16(x))
}

impl Matmul for CpuMatmul {
    fn pack(&self, b: &[f32], k: usize, n: usize) -> PackedB {
        let b = if self.bf16_inputs { b.iter().map(|&v| round_bf16(v)).collect() } else { b.to_vec() };
        PackedB { k, n, tiles: Vec::new(), f32_cache: Some(b) }
    }
    fn matmul(&mut self, a: &[f32], m: usize, b: &PackedB) -> Result<Vec<f32>> {
        let (k, n) = (b.k, b.n);
        let bf = b.f32_cache.as_ref().expect("cpu pack");
        let a: std::borrow::Cow<[f32]> = if self.bf16_inputs {
            a.iter().map(|&v| round_bf16(v)).collect::<Vec<_>>().into()
        } else {
            a.into()
        };
        let mut c = vec![0f32; m * n];
        use rayon::prelude::*;
        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for kk in 0..k {
                let av = a[i * k + kk];
                if av == 0.0 {
                    continue;
                }
                let br = &bf[kk * n..(kk + 1) * n];
                for j in 0..n {
                    row[j] += av * br[j];
                }
            }
        });
        Ok(c)
    }
}

pub struct NpuMatmul {
    h: *mut XmmRaw,
    pub tm: usize,
    pub tk: usize,
    pub tn: usize,
    calls: u64,
    npu_secs: f64,
}

// The XRT handle is only used from one thread at a time (behind &mut self).
unsafe impl Send for NpuMatmul {}

impl NpuMatmul {
    pub fn open(xclbin: &str, insts: &str, tm: usize, tk: usize, tn: usize) -> Result<Self> {
        let (x, i) = (CString::new(xclbin)?, CString::new(insts)?);
        let mut err = [0 as c_char; 512];
        let h = unsafe { xmm_open(x.as_ptr(), i.as_ptr(), tm as u32, tk as u32, tn as u32, err.as_mut_ptr(), 512) };
        if h.is_null() {
            bail!("xmm_open: {}", err_str(&err));
        }
        Ok(Self { h, tm, tk, tn, calls: 0, npu_secs: 0.0 })
    }

    fn run_tile(&mut self) -> Result<&[f32]> {
        let mut err = [0 as c_char; 512];
        let t0 = Instant::now();
        let p = unsafe { xmm_run(self.h, 1000, err.as_mut_ptr(), 512) };
        self.npu_secs += t0.elapsed().as_secs_f64();
        self.calls += 1;
        if p.is_null() {
            bail!("NPU run failed: {}", err_str(&err));
        }
        Ok(unsafe { std::slice::from_raw_parts(p, self.tm * self.tn) })
    }
}

impl Drop for NpuMatmul {
    fn drop(&mut self) {
        unsafe { xmm_close(self.h) }
    }
}

impl Matmul for NpuMatmul {
    fn pack(&self, b: &[f32], k: usize, n: usize) -> PackedB {
        let (tk, tn) = (self.tk, self.tn);
        let kb_n = k.div_ceil(tk);
        let nb_n = n.div_ceil(tn);
        let mut tiles = vec![vec![Vec::new(); nb_n]; kb_n];
        for (kb, row) in tiles.iter_mut().enumerate() {
            for (nb, t) in row.iter_mut().enumerate() {
                let mut tile = vec![0u16; tk * tn];
                for kk in 0..tk.min(k - kb * tk) {
                    let src = &b[(kb * tk + kk) * n + nb * tn..];
                    let w = tn.min(n - nb * tn);
                    for j in 0..w {
                        tile[kk * tn + j] = bf16(src[j]);
                    }
                }
                *t = tile;
            }
        }
        PackedB { k, n, tiles, f32_cache: None }
    }

    fn matmul(&mut self, a: &[f32], m: usize, b: &PackedB) -> Result<Vec<f32>> {
        let (tm, tk, tn) = (self.tm, self.tk, self.tn);
        let (k, n) = (b.k, b.n);
        let mut c = vec![0f32; m * n];
        for mb in 0..m.div_ceil(tm) {
            let rows = tm.min(m - mb * tm);
            for kb in 0..k.div_ceil(tk) {
                let cols = tk.min(k - kb * tk);
                // pack the A tile once per (mb, kb), reused for every nb
                {
                    let at = unsafe { std::slice::from_raw_parts_mut(xmm_a(self.h), tm * tk) };
                    at.fill(0);
                    for r in 0..rows {
                        let src = &a[(mb * tm + r) * k + kb * tk..][..cols];
                        let dst = &mut at[r * tk..r * tk + cols];
                        for (d, &s) in dst.iter_mut().zip(src) {
                            *d = bf16(s);
                        }
                    }
                }
                for nb in 0..n.div_ceil(tn) {
                    let w = tn.min(n - nb * tn);
                    {
                        let bt = unsafe { std::slice::from_raw_parts_mut(xmm_b(self.h), tk * tn) };
                        bt.copy_from_slice(&b.tiles[kb][nb]);
                    }
                    let ct = self.run_tile()?;
                    for r in 0..rows {
                        let dst = &mut c[(mb * tm + r) * n + nb * tn..][..w];
                        let src = &ct[r * tn..r * tn + w];
                        for (d, &s) in dst.iter_mut().zip(src) {
                            *d += s;
                        }
                    }
                }
            }
        }
        Ok(c)
    }

    fn stats(&self) -> (u64, f64) {
        (self.calls, self.npu_secs)
    }
}
