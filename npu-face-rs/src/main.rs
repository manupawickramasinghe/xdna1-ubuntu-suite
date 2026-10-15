//! npu-face: face-embedding CNNs with conv/FC matmuls on the AMD XDNA NPU.
//!
//!   npu-face selftest --model arcface.onnx      NPU vs exact-f32 CPU on random inputs
//!   npu-face bench    --model arcface.onnx      latency per embedding
//!   npu-face serve    --model arcface.onnx [--listen 127.0.0.1:5800] [--stats 127.0.0.1:5801]
//!
//! `serve` also answers `GET /stats` (JSON) on the --stats address: NPU busy %
//! since the previous request, embeddings/s, mean latency, plus driver
//! telemetry (H-clock, NPU buffer memory, column occupancy). Frigate's System
//! page polls it.
//!
//! Wire protocol (little-endian), one request at a time per connection:
//!   request : u32 magic "XFQ1" | u32 n | u32 c | u32 h | u32 w | f32[n*c*h*w]
//!   response: u32 magic "XFR1" | i32 status | u32 n | u32 dim | f32[n*dim]
//!             (status != 0: dim = byte length of a UTF-8 error message instead)

mod model;
mod npu;
#[allow(dead_code)]
mod telemetry;

use anyhow::{bail, Context, Result};
use model::{Model, Tensor};
use npu::{CpuMatmul, Matmul, NpuMatmul};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const REQ_MAGIC: u32 = u32::from_le_bytes(*b"XFQ1");
const RSP_MAGIC: u32 = u32::from_le_bytes(*b"XFR1");

struct Opts {
    cmd: String,
    model: String,
    kernels: String,
    listen: String,
    stats: String,
    iters: usize,
}

fn parse() -> Result<Opts> {
    let exe_dir = std::env::current_exe()?.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let kdir = std::env::var("NPU_FACE_KERNELS").unwrap_or_else(|_| {
        for cand in [exe_dir.join("kernels"), exe_dir.join("../../kernels"), "kernels".into()] {
            if cand.is_dir() && std::fs::read_dir(&cand).map(|mut d| d.any(|e| e.map(|e| e.file_name().to_string_lossy().ends_with(".xclbin")).unwrap_or(false))).unwrap_or(false) {
                return cand.to_string_lossy().into_owned();
            }
        }
        "kernels".into()
    });
    let mut o = Opts {
        cmd: String::new(),
        model: String::new(),
        kernels: kdir,
        listen: "127.0.0.1:5800".into(),
        stats: "127.0.0.1:5801".into(),
        iters: 20,
    };
    let mut args = std::env::args().skip(1);
    o.cmd = args.next().unwrap_or_default();
    while let Some(a) = args.next() {
        let mut v = || args.next().context("missing value");
        match a.as_str() {
            "--model" => o.model = v()?,
            "--kernels" => o.kernels = v()?,
            "--listen" => o.listen = v()?,
            "--stats" => o.stats = v()?,
            "--iters" => o.iters = v()?.parse()?,
            _ => bail!("unknown option {a}"),
        }
    }
    if o.model.is_empty() || !["selftest", "bench", "serve"].contains(&o.cmd.as_str()) {
        bail!("usage: npu-face <selftest|bench|serve> --model FILE.onnx [--listen ADDR] [--kernels DIR]");
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
    let npu = NpuMatmul::open_dir(&o.kernels).context("opening NPU matmul kernels")?;
    eprintln!("npu-face: kernels {}", npu.describe());
    Ok(npu)
}

/// Run one input with every NPU layer recomputed on the CPU (bf16 numerics).
/// Whole-network cosine alone can hide corrupted layers, so check each one.
fn validate_layers(m: &mut Model, npu: &mut NpuMatmul) -> Result<()> {
    m.validate = true;
    let r = m.run(npu, random_input(11));
    m.validate = false;
    r?;
    if let Some((cos, rel, name)) = m.worst_layer.take() {
        eprintln!("npu-face: per-layer check: worst cosine {cos:.6}, worst max_err/max_ref {rel:.2e} ({name})");
        check_layer(cos, rel, &name)?;
    }
    Ok(())
}

fn check_layer(cos: f64, rel: f32, name: &str) -> Result<()> {
    if cos < 0.9999 || rel > 1e-2 {
        bail!("NPU per-layer validation failed at {name}: cosine {cos:.6}, max_err/max_ref {rel:.2e}");
    }
    Ok(())
}

/// Full re-validation interval while serving (another process using the NPU
/// can corrupt results; see kernels-unsafe/README.md).
const REVALIDATE_SECS: u64 = 600;

fn selftest(o: &Opts) -> Result<()> {
    let mut npu = open_npu(o)?;
    let mut cpu = CpuMatmul { bf16_inputs: false };
    let mut m_npu = Model::load(&o.model)?;
    validate_layers(&mut m_npu, &mut npu)?;
    let mut m_cpu = Model::load(&o.model)?;
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
    let mut m_emu = Model::load(&o.model)?;
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
    let mut m = Model::load(&o.model)?;
    m.run(&mut npu, random_input(1))?; // warm-up
    let (c0, s0) = npu.stats();
    (npu.t_wait, npu.t_fill, npu.t_acc) = (0.0, 0.0, 0.0);
    let mut ts = Vec::new();
    for i in 0..o.iters as u64 {
        let t = Instant::now();
        m.run(&mut npu, random_input(i))?;
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (c1, s1) = npu.stats();
    let it = o.iters as f64;
    eprintln!(
        "  tile loop per embedding: wait {:.1} ms, fill A {:.1} ms, accumulate {:.1} ms",
        npu.t_wait * 1e3 / it,
        npu.t_fill * 1e3 / it,
        npu.t_acc * 1e3 / it
    );
    if !m.profile.is_empty() {
        let mut v: Vec<_> = m.profile.iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let total: f64 = v.iter().map(|x| x.1).sum();
        for (op, t) in v {
            eprintln!("  {op:20} {:7.1} ms/embedding ({:.0}%)", t * 1e3 / (o.iters + 1) as f64, t / total * 100.0);
        }
    }
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

#[derive(Default)]
struct ServiceStats {
    /// cumulative seconds the NPU pipeline was running tiles
    busy: f64,
    embeddings: u64,
    latency_sum: f64,
    started: Option<Instant>,
}

/// Minimal HTTP/1.0 `GET /stats` responder on its own thread, so stats never
/// wait behind a face request (and vice versa).
fn spawn_stats_server(addr: String, st: Arc<Mutex<ServiceStats>>, model: String) -> Result<()> {
    let listener = TcpListener::bind(&addr).with_context(|| format!("binding stats {addr}"))?;
    std::thread::spawn(move || {
        let mut dev = telemetry::discover().into_iter().next().and_then(|i| telemetry::Device::open(i).ok());
        let mut last: Option<(Instant, f64, u64)> = None;
        for conn in listener.incoming() {
            let Ok(mut c) = conn else { continue };
            let mut buf = [0u8; 1024];
            let n = c.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            if !req.starts_with("GET /stats") {
                let _ = c.write_all(b"HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                continue;
            }
            let now = Instant::now();
            let (busy, emb, lat_sum, uptime) = {
                let s = st.lock().unwrap();
                (s.busy, s.embeddings, s.latency_sum, s.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0))
            };
            let (busy_pct, eps) = match last {
                Some((t, b, e)) => {
                    let dt = now.duration_since(t).as_secs_f64().max(1e-3);
                    (((busy - b) / dt * 100.0).clamp(0.0, 100.0), (emb - e) as f64 / dt)
                }
                None => ((busy / uptime.max(1e-3) * 100.0).clamp(0.0, 100.0), emb as f64 / uptime.max(1e-3)),
            };
            last = Some((now, busy, emb));
            let t = dev.as_mut().map(|d| d.sample(false));
            let (clk, mem, cols, pm) = t
                .map(|t| (t.h_mhz, t.mem_used, t.util_pct, t.runtime_status))
                .unwrap_or((0, 0, 0.0, "unknown".into()));
            let body = format!(
                "{{\"npu\":{:.1},\"embeddings_per_s\":{:.3},\"embeddings\":{},\"mean_latency_ms\":{:.1},\"h_clock_mhz\":{},\"bo_mem_mib\":{:.1},\"column_occupancy\":{:.0},\"runtime_pm\":\"{}\",\"model\":\"{}\"}}",
                busy_pct,
                eps,
                emb,
                if emb > 0 { lat_sum / emb as f64 * 1e3 } else { 0.0 },
                clk,
                mem as f64 / 1048576.0,
                cols,
                pm,
                model.replace('"', "'")
            );
            let _ = c.write_all(
                format!("HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).as_bytes(),
            );
        }
    });
    Ok(())
}

fn serve(o: &Opts) -> Result<()> {
    let mut npu = open_npu(o)?;
    let mut m = Model::load(&o.model)?;
    // Refuse to serve if the NPU disagrees with a CPU run using the same bf16
    // numerics (an exact-f32 comparison would also flag the model's own bf16 rounding).
    let mut cpu = CpuMatmul { bf16_inputs: true };
    let mut m_cpu = Model::load(&o.model)?;
    let x = random_input(7);
    let c = cosine(&m.run(&mut npu, x.clone())?.data, &m_cpu.run(&mut cpu, x)?.data);
    drop(m_cpu);
    if c < 0.999 {
        bail!("startup check failed: NPU vs CPU cosine {c:.5}");
    }
    validate_layers(&mut m, &mut npu)?;
    let st = Arc::new(Mutex::new(ServiceStats { started: Some(Instant::now()), ..Default::default() }));
    spawn_stats_server(o.stats.clone(), st.clone(), o.model.clone())?;
    let listener = TcpListener::bind(&o.listen).with_context(|| format!("binding {}", o.listen))?;
    eprintln!("npu-face: serving {} on {} (stats on {}; startup check cosine {c:.5}, {} NPU ops/embedding)", o.model, o.listen, o.stats, m.npu_ops);

    let mut last_validated = Instant::now();
    let mut spot_idx = 0usize;
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
                if last_validated.elapsed().as_secs() >= REVALIDATE_SECS {
                    if let Err(e) = validate_layers(&mut m, &mut npu) {
                        let msg = format!("NPU self-check failed, restarting: {e}");
                        let _ = respond(&mut s, -5, 0, msg.len() as u32, msg.as_bytes());
                        bail!(msg);
                    }
                    last_validated = Instant::now();
                }
                // spot-check one rotating NPU layer per embedding
                if m.npu_convs > 0 {
                    m.spot = Some(spot_idx % m.npu_convs);
                    spot_idx += 1;
                }
                let (t0, busy0) = (Instant::now(), npu.stats().1);
                let r = m.run(&mut npu, x);
                m.spot = None;
                if let Some((cos, rel, name)) = m.worst_layer.take() {
                    if let Err(e) = check_layer(cos, rel, &name) {
                        let msg = format!("NPU spot-check failed, restarting: {e}");
                        let _ = respond(&mut s, -5, 0, msg.len() as u32, msg.as_bytes());
                        bail!(msg);
                    }
                }
                {
                    let mut s = st.lock().unwrap();
                    s.busy += npu.stats().1 - busy0;
                    s.embeddings += 1;
                    s.latency_sum += t0.elapsed().as_secs_f64();
                }
                match r {
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
