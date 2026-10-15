//! Minimal amdxdna telemetry over DRM_IOCTL_AMDXDNA_GET_INFO / GET_ARRAY.
//! Struct layouts mirror include/uapi/drm/amdxdna_accel.h (kernel 7.1).

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---- UAPI ------------------------------------------------------------------

const DRM_IOCTL_BASE: u8 = b'd';
const DRM_COMMAND_BASE: u8 = 0x40;
const DRM_AMDXDNA_GET_INFO: u8 = 7;
const DRM_AMDXDNA_GET_ARRAY: u8 = 10;

const QUERY_AIE_METADATA: u32 = 1;
const QUERY_CLOCK_METADATA: u32 = 3;
const QUERY_SENSORS: u32 = 4;
const QUERY_FIRMWARE_VERSION: u32 = 8;
const GET_POWER_MODE: u32 = 9;
const QUERY_RESOURCE_INFO: u32 = 12;

const ARRAY_HW_CONTEXT_ALL: u32 = 0;
const ARRAY_BO_USAGE: u32 = 6;

const SENSOR_TYPE_POWER: u8 = 0;
const SENSOR_TYPE_COLUMN_UTILIZATION: u8 = 1;
pub const HWCTX_STATE_ACTIVE: u32 = 1;

#[repr(C)]
#[derive(Default)]
struct GetInfo {
    param: u32,
    buffer_size: u32,
    buffer: u64,
}

#[repr(C)]
#[derive(Default)]
struct GetArray {
    param: u32,
    element_size: u32,
    num_element: u32,
    pad: u32,
    buffer: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct TileMetadata {
    row_count: u16,
    row_start: u16,
    dma_channel_count: u16,
    lock_count: u16,
    event_reg_count: u16,
    pad: [u16; 3],
}

#[repr(C)]
#[derive(Default)]
struct AieMetadata {
    col_size: u32,
    cols: u16,
    rows: u16,
    ver_major: u32,
    ver_minor: u32,
    core: TileMetadata,
    mem: TileMetadata,
    shim: TileMetadata,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Clock {
    name: [u8; 16],
    freq_mhz: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct ClockMetadata {
    mp_npu: Clock,
    h: Clock,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawSensor {
    label: [u8; 64],
    input: u32,
    max: u32,
    average: u32,
    highest: u32,
    status: [u8; 64],
    units: [u8; 16],
    unitm: i8,
    kind: u8,
    pad: [u8; 6],
}

#[repr(C)]
#[derive(Default)]
struct FwVersion {
    major: u32,
    minor: u32,
    patch: u32,
    build: u32,
}

#[repr(C)]
#[derive(Default)]
struct ResourceInfo {
    clk_max: u64,
    tops_max: u64,
    task_max: u64,
    tops_curr: u64,
    task_curr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct HwCtxEntry {
    context_id: u32,
    start_col: u32,
    num_col: u32,
    hwctx_id: u32,
    pid: i64,
    command_submissions: u64,
    command_completions: u64,
    migrations: u64,
    preemptions: u64,
    errors: u64,
    priority: u64,
    heap_usage: u64,
    suspensions: u64,
    state: u32,
    pasid: u32,
    gops: u32,
    fps: u32,
    dma_bandwidth: u32,
    latency: u32,
    frame_exec_time: u32,
    txn_op_idx: u32,
    ctx_pc: u32,
    fatal_error_type: u32,
    fatal_error_exception_type: u32,
    fatal_error_exception_pc: u32,
    fatal_error_app_module: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct BoUsage {
    pid: i64,
    total_usage: u64,
    internal_usage: u64,
    heap_usage: u64,
}

// Sizes measured against the running kernel (see docs/FINDINGS.md).
const _: () = assert!(std::mem::size_of::<AieMetadata>() == 64);
const _: () = assert!(std::mem::size_of::<ClockMetadata>() == 48);
const _: () = assert!(std::mem::size_of::<RawSensor>() == 168);
const _: () = assert!(std::mem::size_of::<ResourceInfo>() == 40);
const _: () = assert!(std::mem::size_of::<HwCtxEntry>() == 144);
const _: () = assert!(std::mem::size_of::<BoUsage>() == 32);

nix::ioctl_readwrite!(ioctl_get_info, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO, GetInfo);
nix::ioctl_readwrite!(ioctl_get_array, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_GET_ARRAY, GetArray);

// ---- Public model ----------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct DeviceInfo {
    pub index: u32,
    pub node: PathBuf,
    pub sysfs: PathBuf,
    pub name: String,
    pub pci: String,
    pub vbnv: String,
    pub fw: String,
    pub cols: u16,
    pub rows: u16,
    pub core_rows: u16,
    pub core_row_start: u16,
    pub mem_rows: u16,
    pub mem_row_start: u16,
    pub shim_rows: u16,
    pub aie_version: (u32, u32),
    pub clk_max_mhz: u64,
    pub task_max: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Context {
    pub id: u32,
    pub pid: i64,
    pub comm: String,
    pub start_col: u32,
    pub num_col: u32,
    pub submissions: u64,
    pub completions: u64,
    pub errors: u64,
    pub busy: bool,
    pub rate: f64, // completions/s since last sample
}

#[derive(Debug, Clone, Default)]
pub struct Process {
    pub pid: i64,
    pub comm: String,
    pub total: u64,
    pub internal: u64,
    pub heap: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    #[default]
    Suspended,
    Cached,
    Sensors,
    Occupancy,
}

#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub runtime_status: String,
    pub source: Source,
    pub mp_npu_mhz: u32,
    pub h_mhz: u32,
    pub power_mode: Option<u8>,
    pub power_mw: Option<f64>,
    pub column_util: Vec<f64>,
    pub util_pct: f64,
    pub busy_cols: Vec<bool>,
    pub tasks: u64,
    pub contexts: Vec<Context>,
    pub processes: Vec<Process>,
    pub mem_used: u64,
}

fn read_trim(p: impl AsRef<Path>) -> String {
    fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_default()
}

fn cstr(b: &[u8]) -> String {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..n]).into_owned()
}

pub fn comm(pid: i64) -> String {
    read_trim(format!("/proc/{pid}/comm"))
}

pub fn discover() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir("/sys/class/accel") else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(idx) = name.strip_prefix("accel").and_then(|n| n.parse().ok()) else { continue };
        let dev = e.path().join("device");
        let drv = fs::read_link(dev.join("driver")).unwrap_or_default();
        if drv.file_name().map(|f| f != "amdxdna").unwrap_or(true) {
            continue;
        }
        let device = read_trim(dev.join("device"));
        out.push(DeviceInfo {
            index: idx,
            node: PathBuf::from("/dev/accel").join(&name),
            sysfs: e.path(),
            name: match device.as_str() {
                "0x1502" => "AMD Ryzen AI NPU (XDNA1)".into(),
                "0x17f0" => "AMD Ryzen AI NPU (XDNA2)".into(),
                _ => "AMD NPU".into(),
            },
            pci: fs::canonicalize(&dev)
                .ok()
                .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
                .unwrap_or_default(),
            vbnv: read_trim(dev.join("vbnv")),
            fw: read_trim(dev.join("fw_version")),
            ..Default::default()
        });
    }
    out.sort_by_key(|d| d.index);
    out
}

pub struct Device {
    pub info: DeviceInfo,
    fd: OwnedFd,
    sensors_ok: bool,
    prev: HashMap<(i64, u32), u64>,
    prev_at: Option<Instant>,
    idle_streak: u32,
    was_suspended: bool,
    last_query: Option<Instant>,
    last: Sample,
}

impl Device {
    pub fn open(mut info: DeviceInfo) -> std::io::Result<Self> {
        let f = fs::OpenOptions::new().read(true).write(true).open(&info.node)?;
        let mut d = Device {
            fd: f.into(),
            info: DeviceInfo::default(),
            sensors_ok: true,
            prev: HashMap::new(),
            prev_at: None,
            idle_streak: 0,
            was_suspended: false,
            last_query: None,
            last: Sample::default(),
        };
        let mut md = AieMetadata::default();
        if d.get_info(QUERY_AIE_METADATA, &mut md) {
            info.cols = md.cols;
            info.rows = md.rows;
            info.core_rows = md.core.row_count;
            info.core_row_start = md.core.row_start;
            info.mem_rows = md.mem.row_count;
            info.mem_row_start = md.mem.row_start;
            info.shim_rows = md.shim.row_count;
            info.aie_version = (md.ver_major, md.ver_minor);
        }
        let mut fw = FwVersion::default();
        if d.get_info(QUERY_FIRMWARE_VERSION, &mut fw) {
            info.fw = format!("{}.{}.{}.{}", fw.major, fw.minor, fw.patch, fw.build);
        }
        let mut ri = ResourceInfo::default();
        if d.get_info(QUERY_RESOURCE_INFO, &mut ri) {
            info.clk_max_mhz = ri.clk_max;
            info.task_max = ri.task_max;
        }
        d.info = info;
        Ok(d)
    }

    fn get_info<T>(&self, param: u32, buf: &mut T) -> bool {
        self.get_info_raw(param, buf as *mut T as *mut u8, std::mem::size_of::<T>() as u32).is_some()
    }

    fn get_info_raw(&self, param: u32, buf: *mut u8, size: u32) -> Option<u32> {
        let mut gi = GetInfo { param, buffer_size: size, buffer: buf as u64 };
        // SAFETY: buf points to `size` writable bytes owned by the caller.
        unsafe { ioctl_get_info(self.fd.as_raw_fd(), &mut gi) }.ok().map(|_| gi.buffer_size)
    }

    fn get_array<T>(&self, param: u32, buf: &mut [T]) -> Option<usize> {
        let mut ga = GetArray {
            param,
            element_size: std::mem::size_of::<T>() as u32,
            num_element: buf.len() as u32,
            pad: 0,
            buffer: buf.as_mut_ptr() as u64,
        };
        // SAFETY: buffer holds num_element elements of element_size bytes.
        unsafe { ioctl_get_array(self.fd.as_raw_fd(), &mut ga) }.ok().map(|_| ga.num_element as usize)
    }

    pub fn runtime_status(&self) -> String {
        read_trim(self.info.sysfs.join("device/power/runtime_status"))
    }

    /// Same policy as xdna_telemetry.hpp: never wake a suspended NPU; after 3
    /// idle samples, re-query at most every 10 s so it can autosuspend.
    pub fn sample(&mut self, wake: bool) -> Sample {
        let now = Instant::now();
        let rs = self.runtime_status();
        let cols = self.info.cols.max(1) as usize;

        if rs == "suspended" && !wake {
            self.idle_streak = 0;
            self.was_suspended = true;
            self.prev.clear();
            let mut s = self.carry_over(Source::Suspended, rs);
            s.mp_npu_mhz = 0;
            s.h_mhz = 0;
            return s;
        }
        if self.was_suspended {
            self.idle_streak = 0;
        }
        self.was_suspended = false;
        if !wake
            && self.idle_streak >= 3
            && self.last_query.is_some_and(|t| now.duration_since(t) < Duration::from_secs(10))
        {
            return self.carry_over(Source::Cached, rs);
        }
        self.last_query = Some(now);

        let mut s = Sample { runtime_status: rs, busy_cols: vec![false; cols], ..Default::default() };
        let mut ck = ClockMetadata::default();
        if self.get_info(QUERY_CLOCK_METADATA, &mut ck) {
            s.mp_npu_mhz = ck.mp_npu.freq_mhz;
            s.h_mhz = ck.h.freq_mhz;
        }
        let mut pm = [0u8; 8];
        if self.get_info(GET_POWER_MODE, &mut pm) {
            s.power_mode = Some(pm[0]);
        }
        let mut ri = ResourceInfo::default();
        if self.get_info(QUERY_RESOURCE_INFO, &mut ri) {
            s.tasks = ri.task_curr;
        }

        if self.sensors_ok {
            let mut buf = vec![std::mem::MaybeUninit::<RawSensor>::zeroed(); 32];
            let size = (buf.len() * std::mem::size_of::<RawSensor>()) as u32;
            match self.get_info_raw(QUERY_SENSORS, buf.as_mut_ptr() as *mut u8, size) {
                Some(got) => {
                    let n = (got as usize / std::mem::size_of::<RawSensor>()).min(buf.len());
                    for m in &buf[..n] {
                        // SAFETY: zero-initialised and filled by the kernel; all-bytes-valid POD.
                        let r = unsafe { m.assume_init() };
                        let v = r.input as f64 * 10f64.powi(r.unitm as i32);
                        match r.kind {
                            SENSOR_TYPE_POWER => {
                                let units = cstr(&r.units);
                                s.power_mw = Some(match units.as_str() {
                                    "W" => v * 1000.0,
                                    "uW" => v / 1000.0,
                                    _ => v,
                                });
                            }
                            SENSOR_TYPE_COLUMN_UTILIZATION => s.column_util.push(v.clamp(0.0, 100.0)),
                            _ => {}
                        }
                    }
                }
                None => self.sensors_ok = false,
            }
        }

        let mut ctx = vec![HwCtxEntry::default(); 64];
        let n = self.get_array(ARRAY_HW_CONTEXT_ALL, &mut ctx).unwrap_or(0);
        let dt = self.prev_at.map(|t| now.duration_since(t).as_secs_f64()).unwrap_or(0.0);
        let mut next = HashMap::new();
        for e in &ctx[..n] {
            let key = (e.pid, e.context_id);
            let prev = self.prev.get(&key).copied();
            let advanced = prev.is_some_and(|p| e.command_completions > p);
            let busy = e.state == HWCTX_STATE_ACTIVE || advanced || e.command_submissions > e.command_completions;
            let rate = match prev {
                Some(p) if dt > 0.0 => e.command_completions.saturating_sub(p) as f64 / dt,
                _ => 0.0,
            };
            next.insert(key, e.command_completions);
            if busy {
                for c in e.start_col..(e.start_col + e.num_col) {
                    if let Some(b) = s.busy_cols.get_mut(c as usize) {
                        *b = true;
                    }
                }
            }
            s.contexts.push(Context {
                id: e.context_id,
                pid: e.pid,
                comm: comm(e.pid),
                start_col: e.start_col,
                num_col: e.num_col,
                submissions: e.command_submissions,
                completions: e.command_completions,
                errors: e.errors,
                busy,
                rate,
            });
        }
        self.prev = next;
        self.prev_at = Some(now);

        if !s.column_util.is_empty() {
            s.util_pct = s.column_util.iter().sum::<f64>() / s.column_util.len() as f64;
            s.source = Source::Sensors;
        } else {
            s.util_pct = 100.0 * s.busy_cols.iter().filter(|b| **b).count() as f64 / cols as f64;
            s.source = Source::Occupancy;
        }

        let pids: BTreeSet<i64> = s.contexts.iter().map(|c| c.pid).collect();
        for pid in pids {
            let mut u = [BoUsage { pid, ..Default::default() }];
            if self.get_array(ARRAY_BO_USAGE, &mut u).is_some() {
                s.mem_used += u[0].total_usage;
                s.processes.push(Process {
                    pid,
                    comm: comm(pid),
                    total: u[0].total_usage,
                    internal: u[0].internal_usage,
                    heap: u[0].heap_usage,
                });
            }
        }

        self.idle_streak = if s.util_pct > 0.0 { 0 } else { self.idle_streak + 1 };
        self.last = s.clone();
        s
    }

    /// Reuse the last sample's clocks and the BO usage of processes that are still alive.
    fn carry_over(&mut self, source: Source, rs: String) -> Sample {
        self.last.processes.retain(|p| Path::new(&format!("/proc/{}", p.pid)).exists());
        let cols = self.info.cols.max(1) as usize;
        Sample {
            runtime_status: rs,
            source,
            mp_npu_mhz: self.last.mp_npu_mhz,
            h_mhz: self.last.h_mhz,
            power_mode: self.last.power_mode,
            busy_cols: vec![false; cols],
            mem_used: self.last.processes.iter().map(|p| p.total).sum(),
            processes: self.last.processes.clone(),
            ..Default::default()
        }
    }
}
