//! npu-face: face-embedding CNNs with conv/FC matmuls on the AMD XDNA NPU.
//!
//!   npu-face selftest --model arcface.onnx      NPU vs exact-f32 CPU on random inputs
//!   npu-face bench    --model arcface.onnx      latency per embedding
//!   npu-face serve    --model arcface.onnx [--listen 127.0.0.1:5800]
//!
//! Wire protocol (little-endian), one request at a time per connection:
//!   request : u32 magic "XFQ1" | u32 n | u32 c | u32 h | u32 w | f32[n*c*h*w]
//!   response: u32 magic "XFR1" | i32 status | u32 n | u32 dim | f32[n*dim]
//!             (status != 0: dim = byte length of a UTF-8 error message instead)

mod model;
mod npu;

use anyhow::{bail, Context, Result};
use model::{Model, Tensor};
use npu::{CpuMatmul, Matmul, NpuMatmul};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Instant;

const REQ_MAGIC: u32 = u32::from_le_bytes(*b"XFQ1");
const RSP_MAGIC: u32 = u32::from_le_bytes(*b"XFR1");

struct Opts {
    cmd: String,
    model: String,
    xclbin: String,
    insts: String,
    listen: String,
    iters: usize,
}

fn parse() -> Result<Opts> {
    let exe_dir = std::env::current_exe()?.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let kdir = std::env::var("NPU_FACE_KERNELS").unwrap_or_else(|_| {
        for cand in [exe_dir.join("kernels"), exe_dir.join("../../kernels"), "kernels".into()] {
            if cand.join("mm_bf16_f32_512x256x256.xclbin").exists() {
                return cand.to_string_lossy().into_owned();
            }
        }
        "kernels".into()
    });
    let mut o = Opts {
        cmd: String::new(),
        model: String::new(),
        xclbin: format!("{kdir}/mm_bf16_f32_512x256x256.xclbin"),
        insts: format!("{kdir}/mm_bf16_f32_512x256x256.insts.bin"),
        listen: "127.0.0.1:5800".into(),
        iters: 20,
    };
    let mut args = std::env::args().skip(1);
    o.cmd = args.next().unwrap_or_default();
    while let Some(a) = args.next() {
        let mut v = || args.next().context("missing value");
        match a.as_str() {
            "--model" => o.model = v()?,
            "--xclbin" => o.xclbin = v()?,
            "--insts" => o.insts = v()?,
            "--listen" => o.listen = v()?,
            "--iters" => o.iters = v()?.parse()?,
            _ => bail!("unknown option {a}"),
        }
    }
    if o.model.is_empty() || !["selftest", "bench", "serve"].contains(&o.cmd.as_str()) {
        bail!("usage: npu-face <selftest|bench|serve> --model FILE.onnx [--listen ADDR] [--xclbin F --insts F]");
    }
    Ok(o)
}

fn random_input(seed: u64) -> Tensor {
    // xorshift; values like a normalised face image in [-1, 1]
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let data = (0..3 * 112 * 112)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        })
        .collect();
    Tensor::new(vec![1, 3, 112, 112], data)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn open_npu(o: &Opts) -> Result<NpuMatmul> {
    NpuMatmul::open(&o.xclbin, &o.insts, 512, 256, 256).context("opening NPU matmul kernel")
}

fn selftest(o: &Opts) -> Result<()> {
    let mut npu = open_npu(o)?;
    let mut cpu = CpuMatmul { bf16_inputs: false };
    let mut m_npu = Model::load(&o.model, &npu)?;
    let mut m_cpu = Model::load(&o.model, &cpu)?;
    let mut worst = 1f32;
    for i in 0..o.iters as u64 {
        let x = random_input(i + 1);
        let a = m_npu.run(&mut npu, x.clone())?;
        let b = m_cpu.run(&mut cpu, x)?;
        let c = cosine(&a.data, &b.data);
        worst = worst.min(c);
    }
    let dim = m_npu.run(&mut npu, random_input(99))?.data.len();
    println!(
        "selftest: {} inputs, NPU vs f32 CPU embedding cosine min {:.5}, {} NPU ops/embedding, dim {}",
        o.iters, worst, m_npu.npu_ops, dim
    );
    // bf16-emulated CPU: isolates NPU faults from bf16 rounding of the model itself
    let mut emu = CpuMatmul { bf16_inputs: true };
    let mut m_emu = Model::load(&o.model, &emu)?;
    let (mut w_emu, mut w_ne) = (1f32, 1f32);
    for i in 0..o.iters as u64 {
        let x = random_input(i + 1);
        let a = m_npu.run(&mut npu, x.clone())?;
        let e = m_emu.run(&mut emu, x.clone())?;
        let b = m_cpu.run(&mut cpu, x)?;
        w_emu = w_emu.min(cosine(&e.data, &b.data));
        w_ne = w_ne.min(cosine(&a.data, &e.data));
    }
    println!("selftest: bf16-emulated CPU vs f32 CPU cosine min {w_emu:.5}; NPU vs bf16-emulated CPU min {w_ne:.5}");
    if w_ne < 0.999 {
        bail!("selftest FAILED: NPU disagrees with the bf16 CPU emulation ({w_ne:.5})");
    }
    if worst < 0.99 {
        bail!("selftest FAILED: cosine {worst:.5} < 0.99");
    }
    println!("selftest: PASS");
    Ok(())
}

fn bench(o: &Opts) -> Result<()> {
    let mut npu = open_npu(o)?;
    let mut m = Model::load(&o.model, &npu)?;
    m.run(&mut npu, random_input(1))?; // warm-up
    let (c0, s0) = npu.stats();
    let mut ts = Vec::new();
    for i in 0..o.iters as u64 {
        let t = Instant::now();
        m.run(&mut npu, random_input(i))?;
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (c1, s1) = npu.stats();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let calls = (c1 - c0) as f64 / o.iters as f64;
    println!(
        "bench: p50 {:.1} ms  p90 {:.1} ms per embedding; {:.0} NPU tile calls/embedding, {:.1} ms NPU time/embedding",
        ts[ts.len() / 2],
        ts[ts.len() * 9 / 10],
        calls,
        (s1 - s0) * 1e3 / o.iters as f64
    );
    Ok(())
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn respond(w: &mut impl Write, status: i32, n: u32, dim: u32, payload: &[u8]) -> std::io::Result<()> {
    let mut hdr = Vec::with_capacity(16);
    hdr.extend_from_slice(&RSP_MAGIC.to_le_bytes());
    hdr.extend_from_slice(&status.to_le_bytes());
    hdr.extend_from_slice(&n.to_le_bytes());
    hdr.extend_from_slice(&dim.to_le_bytes());
    w.write_all(&hdr)?;
    w.write_all(payload)
}

fn serve(o: &Opts) -> Result<()> {
    let mut npu = open_npu(o)?;
    let mut m = Model::load(&o.model, &npu)?;
    // Refuse to serve if the NPU disagrees with a CPU run using the same bf16
    // numerics (an exact-f32 comparison would also flag the model's own bf16 rounding).
    let mut cpu = CpuMatmul { bf16_inputs: true };
    let mut m_cpu = Model::load(&o.model, &cpu)?;
    let x = random_input(7);
    let c = cosine(&m.run(&mut npu, x.clone())?.data, &m_cpu.run(&mut cpu, x)?.data);
    drop(m_cpu);
    if c < 0.999 {
        bail!("startup check failed: NPU vs CPU cosine {c:.5}");
    }
    let listener = TcpListener::bind(&o.listen).with_context(|| format!("binding {}", o.listen))?;
    eprintln!("npu-face: serving {} on {} (startup check cosine {c:.5}, {} NPU ops/embedding)", o.model, o.listen, m.npu_ops);

    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        s.set_nodelay(true).ok();
        // One client at a time: the NPU context is a single serial resource.
        loop {
            let Ok(magic) = read_u32(&mut s) else { break };
            if magic != REQ_MAGIC {
                let _ = respond(&mut s, -71, 0, 13, b"bad magic\0\0\0\0");
                break;
            }
            let dims: Result<Vec<u32>, _> = (0..4).map(|_| read_u32(&mut s)).collect();
            let Ok(d) = dims else { break };
            let (n, c, h, w) = (d[0] as usize, d[1] as usize, d[2] as usize, d[3] as usize);
            if n == 0 || n > 64 || c * h * w > 3 * 512 * 512 {
                let msg = b"bad shape";
                let _ = respond(&mut s, -22, 0, msg.len() as u32, msg);
                break;
            }
            let mut buf = vec![0u8; n * c * h * w * 4];
            if s.read_exact(&mut buf).is_err() {
                break;
            }
            let data: Vec<f32> = buf.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let mut out = Vec::new();
            let mut dim = 0;
            let mut failure = None;
            for i in 0..n {
                let per = c * h * w;
                let x = Tensor::new(vec![1, c, h, w], data[i * per..(i + 1) * per].to_vec());
                match m.run(&mut npu, x) {
                    Ok(y) => {
                        dim = y.data.len();
                        out.extend(y.data.iter().flat_map(|v| v.to_le_bytes()));
                    }
                    Err(e) => {
                        failure = Some(e.to_string());
                        break;
                    }
                }
            }
            let ok = match failure {
                None => respond(&mut s, 0, n as u32, dim as u32, &out),
                Some(msg) => respond(&mut s, -5, 0, msg.len() as u32, msg.as_bytes()),
            };
            if ok.is_err() {
                break;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let o = parse()?;
    match o.cmd.as_str() {
        "selftest" => selftest(&o),
        "bench" => bench(&o),
        _ => serve(&o),
    }
}
