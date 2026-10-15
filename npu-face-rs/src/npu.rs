//! Tiled bf16 matmul on the XDNA NPU through a small set of fixed-shape XRT
//! kernels (mlir-aie whole_array, bf16 x bf16 -> f32).
//!
//! Every NPU call has ~120-130 us of fixed cost, so for each matmul the kernel
//! with the lowest estimated total time (calls x measured us/call) is chosen,
//! which mostly means "fewest, fullest tiles". Weights are packed for the
//! chosen kernel the first time a layer runs. Tiny matmuls (e.g. the final FC
//! with M = 1) run on the CPU, where they are faster.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::ffi::{c_char, CString};
use std::path::Path;
use std::time::Instant;

#[repr(C)]
struct XmmRaw {
    _p: [u8; 0],
}

extern "C" {
    fn xmm_open(xclbin: *const c_char, insts: *const c_char, m: u32, k: u32, n: u32, err: *mut c_char, len: u32) -> *mut XmmRaw;
    fn xmm_b_new(h: *mut XmmRaw, err: *mut c_char, len: u32) -> i32;
    fn xmm_b_ptr(h: *mut XmmRaw, id: i32) -> *mut u16;
    fn xmm_b_sync(h: *mut XmmRaw, id: i32, err: *mut c_char, len: u32) -> i32;
    fn xmm_a_ptr(h: *mut XmmRaw, slot: i32) -> *mut u16;
    fn xmm_start(h: *mut XmmRaw, slot: i32, b_id: i32, sync_a: i32, err: *mut c_char, len: u32) -> i32;
    fn xmm_wait(h: *mut XmmRaw, slot: i32, timeout_ms: u32, err: *mut c_char, len: u32) -> *const f32;
    fn xmm_close(h: *mut XmmRaw);
}

/// Round-to-nearest-even f32 -> bf16.
#[inline]
pub fn bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

#[inline]
pub fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline]
fn round_bf16(x: f32) -> f32 {
    from_bf16(bf16(x))
}

fn err_str(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Weights of one matmul, prepared for one execution path.
pub enum PackedB {
    /// b_ids[kb][nb] = device buffer holding that TK x TN bf16 tile of kernel `kernel`
    Npu { k: usize, n: usize, kernel: usize, b_ids: Vec<Vec<i32>> },
    /// full K x N matrix (optionally bf16-rounded) for the CPU path
    Cpu { k: usize, n: usize, b: Vec<f32> },
}

impl PackedB {
    pub fn n(&self) -> usize {
        match self {
            PackedB::Npu { n, .. } | PackedB::Cpu { n, .. } => *n,
        }
    }
}

pub trait Matmul: Send {
    /// Prepare B[k,n] (row-major f32) for matmuls with `m` rows of A.
    fn pack(&self, b: &[f32], m: usize, k: usize, n: usize) -> PackedB;
    /// C[m,n] = A[m,k] (row-major f32) x B.
    fn matmul(&mut self, a: &[f32], m: usize, b: &PackedB) -> Result<Vec<f32>>;
    /// Whether matmul_bf16 is supported (lets im2col emit bf16 directly).
    fn wants_bf16(&self, _b: &PackedB) -> bool {
        false
    }
    /// C[m,n] = A[m,k] (row-major bf16 bits) x B.
    fn matmul_bf16(&mut self, _a: &[u16], _m: usize, _b: &PackedB) -> Result<Vec<f32>> {
        bail!("bf16 input not supported")
    }
    fn stats(&self) -> (u64, f64) {
        (0, 0.0)
    }
}

pub fn cpu_matmul(a: &[f32], k: usize, n: usize, b: &[f32], round: bool) -> Vec<f32> {
    let m = a.len() / k;
    let a: Vec<f32> = if round { a.iter().map(|&v| round_bf16(v)).collect() } else { a.to_vec() };
    let mut c = vec![0f32; m * n];
    if m >= 16 {
        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for kk in 0..k {
                let av = a[i * k + kk];
                if av != 0.0 {
                    for (r, bv) in row.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                        *r += av * bv;
                    }
                }
            }
        });
    } else {
        // few rows (e.g. the final FC, m = 1): parallelise over output columns
        const NB: usize = 32;
        for i in 0..m {
            let row = &mut c[i * n..(i + 1) * n];
            row.par_chunks_mut(NB).enumerate().for_each(|(jb, out)| {
                let j0 = jb * NB;
                let w = out.len();
                for kk in 0..k {
                    let av = a[i * k + kk];
                    if av != 0.0 {
                        for (r, bv) in out.iter_mut().zip(&b[kk * n + j0..kk * n + j0 + w]) {
                            *r += av * bv;
                        }
                    }
                }
            });
        }
    }
    c
}

/// CPU reference. `bf16_inputs` reproduces the NPU's numerics (bf16-rounded A
/// and B, f32 accumulation); otherwise exact f32.
pub struct CpuMatmul {
    pub bf16_inputs: bool,
}

impl Matmul for CpuMatmul {
    fn pack(&self, b: &[f32], _m: usize, k: usize, n: usize) -> PackedB {
        let b = if self.bf16_inputs { b.iter().map(|&v| round_bf16(v)).collect() } else { b.to_vec() };
        PackedB::Cpu { k, n, b }
    }
    fn matmul(&mut self, a: &[f32], m: usize, b: &PackedB) -> Result<Vec<f32>> {
        let PackedB::Cpu { k, n, b } = b else { bail!("CPU matmul got NPU-packed weights") };
        Ok(cpu_matmul(&a[..m * k], *k, *n, b, self.bf16_inputs))
    }
}

struct Kernel {
    h: *mut XmmRaw,
    tm: usize,
    tk: usize,
    tn: usize,
    /// measured seconds per call
    t_call: f64,
}

pub struct NpuMatmul {
    kernels: Vec<Kernel>,
    /// seconds blocked in xmm_wait / spent filling A / accumulating C (profiling)
    pub t_wait: f64,
    pub t_fill: f64,
    pub t_acc: f64,
    /// matmuls with m below this run on the CPU (bf16 numerics, like the NPU)
    pub cpu_max_m: usize,
    calls: u64,
    npu_secs: f64,
}

// XRT handles are only used through &mut self, i.e. from one thread at a time.
unsafe impl Send for NpuMatmul {}

impl NpuMatmul {
    /// Load every `mm_bf16_f32_{M}x{K}x{N}.xclbin` (+ `.insts.bin`) in `dir`
    /// (at most 4: npu1 has only a handful of hardware contexts).
    pub fn open_dir(dir: &str) -> Result<Self> {
        let mut found = Vec::new();
        for e in std::fs::read_dir(dir).with_context(|| format!("reading kernel dir {dir}"))? {
            let p = e?.path();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let Some(dims) = name.strip_prefix("mm_bf16_f32_").and_then(|s| s.strip_suffix(".xclbin")) else { continue };
            let d: Vec<usize> = dims.split('x').filter_map(|v| v.parse().ok()).collect();
            if d.len() == 3 {
                found.push((d[0], d[1], d[2], p.clone()));
            }
        }
        if found.is_empty() {
            bail!("no mm_bf16_f32_*.xclbin kernels in {dir}");
        }
        found.sort();
        let mut kernels = Vec::new();
        for (tm, tk, tn, xclbin) in found.into_iter().take(4) {
            let insts = xclbin.with_extension("insts.bin");
            let mut k = Self::open_one(&xclbin, &insts, tm, tk, tn)?;
            // measure the real per-call cost with two runs in flight, as in use
            let mut err = [0 as c_char; 512];
            let b = unsafe { xmm_b_new(k.h, err.as_mut_ptr(), 512) };
            if b < 0 {
                bail!("allocating B buffer: {}", err_str(&err));
            }
            Self::start(&mut k, 0, b, true)?;
            Self::wait(&mut k, 0)?;
            let t0 = Instant::now();
            for i in 0..16 {
                if i >= 2 {
                    Self::wait(&mut k, i % 2)?;
                }
                Self::start(&mut k, i % 2, b, true)?;
            }
            Self::wait(&mut k, 0)?;
            Self::wait(&mut k, 1)?;
            k.t_call = t0.elapsed().as_secs_f64() / 16.0;
            kernels.push(k);
        }
        Ok(Self { kernels, t_wait: 0.0, t_fill: 0.0, t_acc: 0.0, cpu_max_m: 8, calls: 0, npu_secs: 0.0 })
    }

    fn open_one(xclbin: &Path, insts: &Path, tm: usize, tk: usize, tn: usize) -> Result<Kernel> {
        let x = CString::new(xclbin.to_string_lossy().as_bytes())?;
        let i = CString::new(insts.to_string_lossy().as_bytes())?;
        let mut err = [0 as c_char; 512];
        let h = unsafe { xmm_open(x.as_ptr(), i.as_ptr(), tm as u32, tk as u32, tn as u32, err.as_mut_ptr(), 512) };
        if h.is_null() {
            bail!("xmm_open {}: {}", xclbin.display(), err_str(&err));
        }
        Ok(Kernel { h, tm, tk, tn, t_call: 0.0 })
    }

    fn start(k: &mut Kernel, slot: usize, b_id: i32, sync_a: bool) -> Result<()> {
        let mut err = [0 as c_char; 512];
        if unsafe { xmm_start(k.h, slot as i32, b_id, sync_a as i32, err.as_mut_ptr(), 512) } != 0 {
            bail!("NPU start failed: {}", err_str(&err));
        }
        Ok(())
    }

    fn wait(k: &mut Kernel, slot: usize) -> Result<*const f32> {
        let mut err = [0 as c_char; 512];
        let p = unsafe { xmm_wait(k.h, slot as i32, 2000, err.as_mut_ptr(), 512) };
        if p.is_null() {
            bail!("NPU run failed: {}", err_str(&err));
        }
        Ok(p)
    }

    pub fn describe(&self) -> String {
        self.kernels
            .iter()
            .map(|k| format!("{}x{}x{} {:.0}us", k.tm, k.tk, k.tn, k.t_call * 1e6))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Run C = A x B tile by tile with two tiles in flight: while the NPU works
    /// on one slot, the host fills the other. `fill_a(dst_row, src_row, k0, cols, k)`
    /// writes one row segment of an A tile.
    fn tiled<F>(&mut self, m: usize, b: &PackedB, fill_a: F) -> Result<Vec<f32>>
    where
        F: Fn(&mut [u16], usize, usize, usize, usize) + Sync,
    {
        let PackedB::Npu { k, n, kernel: ki, b_ids } = b else { bail!("NPU matmul got CPU-packed weights") };
        let (k, n, ki) = (*k, *n, *ki);
        let (tm, tk, tn) = (self.kernels[ki].tm, self.kernels[ki].tk, self.kernels[ki].tn);
        let mut jobs = Vec::new();
        for mb in 0..m.div_ceil(tm) {
            for kb in 0..k.div_ceil(tk) {
                for nb in 0..n.div_ceil(tn) {
                    jobs.push((mb, kb, nb));
                }
            }
        }
        let mut c = vec![0f32; m * n];
        let mut inflight: [Option<(usize, usize, usize)>; 2] = [None, None];
        let mut loaded: [Option<(usize, usize)>; 2] = [None, None];
        let accumulate = |c: &mut [f32], (mb, _kb, nb): (usize, usize, usize), ct: &[f32]| {
            let rows = tm.min(m - mb * tm);
            let w = tn.min(n - nb * tn);
            // sequential: a tile is only 32-512 K floats, less than rayon's dispatch cost
            for (r, row) in c[mb * tm * n..(mb * tm + rows) * n].chunks_mut(n).enumerate() {
                for (d, &s) in row[nb * tn..nb * tn + w].iter_mut().zip(&ct[r * tn..r * tn + w]) {
                    *d += s;
                }
            }
        };
        let t_start = Instant::now();
        for (j, &(mb, kb, nb)) in jobs.iter().enumerate() {
            let slot = j % 2;
            if let Some(prev) = inflight[slot].take() {
                let t0 = Instant::now();
                let p = Self::wait(&mut self.kernels[ki], slot)?;
                let t1 = Instant::now();
                accumulate(&mut c, prev, unsafe { std::slice::from_raw_parts(p, tm * tn) });
                self.t_wait += (t1 - t0).as_secs_f64();
                self.t_acc += t1.elapsed().as_secs_f64();
            }
            let tf = Instant::now();
            let refill = loaded[slot] != Some((mb, kb));
            if refill {
                let rows = tm.min(m - mb * tm);
                let cols = tk.min(k - kb * tk);
                let at = unsafe { std::slice::from_raw_parts_mut(xmm_a_ptr(self.kernels[ki].h, slot as i32), tm * tk) };
                for (r, dst) in at.chunks_mut(tk).enumerate() {
                    if r < rows {
                        fill_a(dst, mb * tm + r, kb * tk, cols, k);
                        dst[cols..].fill(0);
                    } else {
                        dst.fill(0);
                    }
                }
                loaded[slot] = Some((mb, kb));
            }
            self.t_fill += tf.elapsed().as_secs_f64();
            Self::start(&mut self.kernels[ki], slot, b_ids[kb][nb], refill)?;
            self.calls += 1;
            inflight[slot] = Some((mb, kb, nb));
        }
        // drain in submission order
        let order = if jobs.len() % 2 == 0 { [0, 1] } else { [1, 0] };
        for slot in order {
            if let Some(prev) = inflight[slot].take() {
                let p = Self::wait(&mut self.kernels[ki], slot)?;
                accumulate(&mut c, prev, unsafe { std::slice::from_raw_parts(p, tm * tn) });
            }
        }
        self.npu_secs += t_start.elapsed().as_secs_f64();
        Ok(c)
    }

    /// Kernel tile shape used for a packed weight (diagnostics).
    pub fn kernel_of(&self, b: &PackedB) -> Option<(usize, usize, usize)> {
        match b {
            PackedB::Npu { kernel, .. } => self.kernels.get(*kernel).map(|k| (k.tm, k.tk, k.tn)),
            _ => None,
        }
    }

    fn best_kernel(&self, m: usize, k: usize, n: usize) -> usize {
        // host work per call (fill A, flush, accumulate) is ~0.15-0.2 ms; count it
        const HOST_PER_CALL: f64 = 0.18e-3;
        let cost = |kk: &Kernel| (m.div_ceil(kk.tm) * k.div_ceil(kk.tk) * n.div_ceil(kk.tn)) as f64 * (kk.t_call + HOST_PER_CALL);
        (0..self.kernels.len())
            .min_by(|&a, &b| cost(&self.kernels[a]).partial_cmp(&cost(&self.kernels[b])).unwrap())
            .unwrap()
    }
}

impl Drop for NpuMatmul {
    fn drop(&mut self) {
        for k in &self.kernels {
            unsafe { xmm_close(k.h) }
        }
    }
}

impl Matmul for NpuMatmul {
    fn pack(&self, b: &[f32], m: usize, k: usize, n: usize) -> PackedB {
        if m < self.cpu_max_m {
            return PackedB::Cpu { k, n, b: b.iter().map(|&v| round_bf16(v)).collect() };
        }
        let ki = self.best_kernel(m, k, n);
        let (h, tk, tn) = (self.kernels[ki].h, self.kernels[ki].tk, self.kernels[ki].tn);
        let mut b_ids = Vec::new();
        for kb in 0..k.div_ceil(tk) {
            let mut row = Vec::new();
            for nb in 0..n.div_ceil(tn) {
                let mut err = [0 as c_char; 512];
                let id = unsafe { xmm_b_new(h, err.as_mut_ptr(), 512) };
                assert!(id >= 0, "allocating NPU weight buffer: {}", err_str(&err));
                let tile = unsafe { std::slice::from_raw_parts_mut(xmm_b_ptr(h, id), tk * tn) };
                let w = tn.min(n - nb * tn);
                tile.par_chunks_mut(tn).enumerate().for_each(|(kk, dst)| {
                    dst.fill(0);
                    if kk < tk.min(k - kb * tk) {
                        let src = &b[(kb * tk + kk) * n + nb * tn..][..w];
                        for (d, &s) in dst[..w].iter_mut().zip(src) {
                            *d = bf16(s);
                        }
                    }
                });
                let rc = unsafe { xmm_b_sync(h, id, err.as_mut_ptr(), 512) };
                assert!(rc == 0, "syncing NPU weight buffer: {}", err_str(&err));
                row.push(id);
            }
            b_ids.push(row);
        }
        PackedB::Npu { k, n, kernel: ki, b_ids }
    }

    fn matmul(&mut self, a: &[f32], m: usize, b: &PackedB) -> Result<Vec<f32>> {
        if let PackedB::Cpu { k, n, b } = b {
            return Ok(cpu_matmul(&a[..m * k], *k, *n, b, true));
        }
        self.tiled(m, b, |dst: &mut [u16], src_row: usize, k0: usize, cols: usize, k: usize| {
            for (d, &s) in dst[..cols].iter_mut().zip(&a[src_row * k + k0..][..cols]) {
                *d = bf16(s);
            }
        })
    }

    fn wants_bf16(&self, b: &PackedB) -> bool {
        matches!(b, PackedB::Npu { .. })
    }

    fn matmul_bf16(&mut self, a: &[u16], m: usize, b: &PackedB) -> Result<Vec<f32>> {
        self.tiled(m, b, |dst: &mut [u16], src_row: usize, k0: usize, cols: usize, k: usize| {
            dst[..cols].copy_from_slice(&a[src_row * k + k0..][..cols]);
        })
    }

    fn stats(&self) -> (u64, f64) {
        (self.calls, self.npu_secs)
    }
}
