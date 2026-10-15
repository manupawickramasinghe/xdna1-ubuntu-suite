// SPDX-License-Identifier: MIT
//
// xdna_telemetry.hpp — header-only C++20 telemetry reader for AMD XDNA NPUs
// (Phoenix / Hawk Point "npu1", PCI 1022:1502; Strix "npu4+", PCI 1022:17f0).
//
// Talks to the in-tree `amdxdna` kernel driver through /dev/accel/accelN and
// /sys/class/accel/accelN. No XRT, ROCm or libdrm dependency.
//
// Data sources, in order of preference:
//   * DRM_AMDXDNA_QUERY_AIE_METADATA / CLOCK_METADATA / FIRMWARE_VERSION (GET_INFO)
//   * DRM_AMDXDNA_QUERY_SENSORS (power + per-column utilisation) — only when the
//     platform's amd-pmf exposes NPU metrics; returns ENODEV otherwise
//   * DRM_IOCTL_AMDXDNA_GET_ARRAY: HW_CONTEXT_ALL (per-context column partition,
//     state, command counters) and BO_USAGE (per-process buffer memory)
//   * /proc/<pid>/fdinfo drm-* keys, if the driver ever exposes them
//
// Runtime PM: the NPU autosuspends ~5 s after its last job. Any GET_INFO ioctl
// resumes it, so sample() first reads power/runtime_status and, while the
// device is suspended, reports an idle sample without touching the device.
// A 1 Hz poller therefore does not keep an idle NPU awake.

#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#if __has_include(<drm/amdxdna_accel.h>) && !defined(XDNA_USE_VENDORED_UAPI)
#include <drm/amdxdna_accel.h>
#else
#include "amdxdna_accel.h"
#endif

namespace xdna {

namespace fs = std::filesystem;

namespace detail {

inline std::string read_line(const fs::path& p) {
	std::ifstream f(p);
	std::string s;
	if (f) std::getline(f, s);
	while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\r')) s.pop_back();
	return s;
}

inline uint32_t read_hex(const fs::path& p) {
	auto s = read_line(p);
	if (s.empty()) return 0;
	try { return static_cast<uint32_t>(std::stoul(s, nullptr, 16)); } catch (...) { return 0; }
}

template <size_t N>
inline std::string cstr(const uint8_t (&a)[N]) {
	size_t n = 0;
	while (n < N && a[n] != 0) ++n;
	return std::string(reinterpret_cast<const char*>(a), n);
}

inline uint64_t to_u64(const void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); }

} // namespace detail

//* Static per-device information, read once.
struct DeviceInfo {
	int index = -1;               // N in accelN
	fs::path dev_node;            // /dev/accel/accelN
	fs::path sysfs;               // /sys/class/accel/accelN
	uint16_t vendor = 0, device = 0;
	std::string pci_slot;         // 0000:10:00.1
	std::string name;             // human readable
	std::string vbnv;             // e.g. RyzenAI-npu1
	std::string fw_version;       // e.g. 1.5.5.391

	// AIE array topology (from DRM_AMDXDNA_QUERY_AIE_METADATA)
	uint16_t cols = 0, rows = 0;
	uint16_t core_rows = 0, mem_rows = 0, shim_rows = 0;
	uint32_t aie_major = 0, aie_minor = 0;
	uint32_t col_size = 0;

	uint64_t clk_max_mhz = 0;     // from QUERY_RESOURCE_INFO (0 if unsupported)
	uint64_t tops_max = 0;
	uint64_t task_max = 0;
};

//* One hardware context (a column partition owned by a process).
struct HwContext {
	uint32_t context_id = 0;
	uint32_t hwctx_id = 0;
	uint32_t start_col = 0, num_col = 0;
	int64_t pid = 0;
	uint64_t submissions = 0, completions = 0, errors = 0;
	uint64_t heap_usage = 0;      // bytes
	uint32_t state = 0;           // AMDXDNA_HWCTX_STATE_*
	uint64_t priority = 0;
	bool busy = false;            // derived: state ACTIVE or completions advanced since last sample
};

//* Per-process NPU memory usage (DRM_AMDXDNA_BO_USAGE).
struct ProcUsage {
	int64_t pid = 0;
	std::string comm;
	uint64_t total_bytes = 0, internal_bytes = 0, heap_bytes = 0;
	uint32_t contexts = 0;
};

struct Sensor {
	std::string label, units, status;
	uint32_t input = 0, max = 0, average = 0, highest = 0;
	int8_t unitm = 0;
	uint8_t type = 0;
	double value() const {
		double v = input;
		for (int i = 0; i < unitm; ++i) v *= 10.0;
		for (int i = 0; i > unitm; --i) v /= 10.0;
		return v;
	}
};

enum class UtilSource { none, suspended, sensors, column_occupancy };

inline const char* to_string(UtilSource s) {
	switch (s) {
		case UtilSource::suspended:        return "suspended";
		case UtilSource::sensors:          return "column-sensors";
		case UtilSource::column_occupancy: return "column-occupancy";
		default:                           return "none";
	}
}

//* A point-in-time sample.
struct Sample {
	std::chrono::steady_clock::time_point when{};
	bool suspended = true;        // runtime PM state at sample time
	std::string runtime_status;   // "active" / "suspended" / ...

	uint32_t mp_npu_clock_mhz = 0;
	uint32_t h_clock_mhz = 0;
	std::optional<uint8_t> power_mode;

	std::optional<double> power_mw;        // only with sensor support
	std::vector<double> column_util_pct;   // only with sensor support, one per column

	double util_pct = 0;          // 0..100, see util_source
	UtilSource util_source = UtilSource::none;
	uint32_t busy_cols = 0;

	uint64_t tops_curr = 0, tasks_curr = 0;

	std::vector<HwContext> contexts;
	std::vector<ProcUsage> processes;
	uint64_t mem_used_bytes = 0;  // sum of per-process total BO usage

	std::vector<Sensor> sensors;

	bool fresh = false;           // true if this sample issued ioctls; false = cached/idle
};

inline std::string device_name(uint16_t vendor, uint16_t device) {
	if (vendor != 0x1022) return "Unknown accelerator";
	switch (device) {
		case 0x1502: return "AMD Ryzen AI NPU (XDNA1)";
		case 0x17f0: return "AMD Ryzen AI NPU (XDNA2)";
		default:     return "AMD NPU";
	}
}

//* Enumerate /sys/class/accel/accel* bound to amdxdna.
inline std::vector<DeviceInfo> discover(const fs::path& sys_root = "/sys/class/accel") {
	std::vector<DeviceInfo> out;
	std::error_code ec;
	if (!fs::exists(sys_root, ec)) return out;
	for (const auto& e : fs::directory_iterator(sys_root, ec)) {
		auto fname = e.path().filename().string();
		if (!fname.starts_with("accel")) continue;
		auto drv = fs::read_symlink(e.path() / "device" / "driver", ec);
		if (ec || drv.filename() != "amdxdna") { ec.clear(); continue; }
		DeviceInfo d;
		try { d.index = std::stoi(fname.substr(5)); } catch (...) { continue; }
		d.sysfs = e.path();
		d.dev_node = fs::path("/dev/accel") / fname;
		d.vendor = static_cast<uint16_t>(detail::read_hex(e.path() / "device" / "vendor"));
		d.device = static_cast<uint16_t>(detail::read_hex(e.path() / "device" / "device"));
		d.pci_slot = fs::canonical(e.path() / "device", ec).filename().string();
		d.vbnv = detail::read_line(e.path() / "device" / "vbnv");
		d.fw_version = detail::read_line(e.path() / "device" / "fw_version");
		d.name = device_name(d.vendor, d.device);
		out.push_back(std::move(d));
	}
	std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.index < b.index; });
	return out;
}

//* Read a process's comm, or "" if gone / not permitted.
inline std::string proc_comm(int64_t pid) {
	return detail::read_line(fs::path("/proc") / std::to_string(pid) / "comm");
}

//* PIDs holding an open fd to the given accel node (best effort; other users'
//* processes are only visible to root).
inline std::set<int64_t> pids_using(const fs::path& dev_node) {
	std::set<int64_t> pids;
	std::error_code ec;
	auto target = fs::canonical(dev_node, ec);
	if (ec) return pids;
	for (const auto& p : fs::directory_iterator("/proc", ec)) {
		const auto name = p.path().filename().string();
		if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) continue;
		std::error_code ec2;
		for (const auto& fd : fs::directory_iterator(p.path() / "fd", ec2)) {
			std::error_code ec3;
			if (fs::read_symlink(fd.path(), ec3) == target) { pids.insert(std::stoll(name)); break; }
		}
	}
	return pids;
}

class Device {
public:
	Device() = default;
	explicit Device(DeviceInfo info) : info_(std::move(info)) {}
	Device(const Device&) = delete;
	Device& operator=(const Device&) = delete;
	Device(Device&& o) noexcept { *this = std::move(o); }
	Device& operator=(Device&& o) noexcept {
		if (this != &o) {
			close();
			info_ = std::move(o.info_);
			fd_ = std::exchange(o.fd_, -1);
			prev_completions_ = std::move(o.prev_completions_);
			sensors_supported_ = o.sensors_supported_;
			idle_streak_ = o.idle_streak_;
			was_suspended_ = o.was_suspended_;
			last_query_ = o.last_query_;
			last_mp_clk_ = o.last_mp_clk_;
			last_h_clk_ = o.last_h_clk_;
			last_processes_ = std::move(o.last_processes_);
			idle_samples_before_backoff = o.idle_samples_before_backoff;
			idle_backoff = o.idle_backoff;
			last_error_ = std::move(o.last_error_);
		}
		return *this;
	}
	~Device() { close(); }

	//* Open the device node and read static metadata. Returns false and sets
	//* last_error() if the node cannot be opened (typically: not in `render` group).
	bool open() {
		if (fd_ >= 0) return true;
		fd_ = ::open(info_.dev_node.c_str(), O_RDWR | O_CLOEXEC);
		if (fd_ < 0) { last_error_ = info_.dev_node.string() + ": " + std::strerror(errno); return false; }

		amdxdna_drm_query_aie_metadata md{};
		if (get_info(DRM_AMDXDNA_QUERY_AIE_METADATA, &md, sizeof md)) {
			info_.cols = md.cols;
			info_.rows = md.rows;
			info_.col_size = md.col_size;
			info_.aie_major = md.version.major;
			info_.aie_minor = md.version.minor;
			info_.core_rows = md.core.row_count;
			info_.mem_rows = md.mem.row_count;
			info_.shim_rows = md.shim.row_count;
		}
		amdxdna_drm_query_firmware_version fw{};
		if (get_info(DRM_AMDXDNA_QUERY_FIRMWARE_VERSION, &fw, sizeof fw))
			info_.fw_version = std::to_string(fw.major) + "." + std::to_string(fw.minor) + "." +
			                   std::to_string(fw.patch) + "." + std::to_string(fw.build);
		amdxdna_drm_get_resource_info ri{};
		if (get_info(DRM_AMDXDNA_QUERY_RESOURCE_INFO, &ri, sizeof ri)) {
			info_.clk_max_mhz = ri.npu_clk_max;
			info_.tops_max = ri.npu_tops_max;
			info_.task_max = ri.npu_task_max;
		}
		return true;
	}

	void close() {
		if (fd_ >= 0) ::close(fd_);
		fd_ = -1;
	}

	bool is_open() const { return fd_ >= 0; }
	const DeviceInfo& info() const { return info_; }
	const std::string& last_error() const { return last_error_; }
	int fd() const { return fd_; }

	std::string runtime_status() const {
		return detail::read_line(info_.sysfs / "device" / "power" / "runtime_status");
	}

	//* Take a sample. If `wake` is false (default) and the device is runtime-
	//* suspended, no ioctl is issued and an idle sample is returned.
	//* `extra_pids` are added to the BO-usage query (e.g. from pids_using()).
	Sample sample(bool wake = false, const std::set<int64_t>& extra_pids = {}) {
		Sample s;
		s.when = std::chrono::steady_clock::now();
		s.runtime_status = runtime_status();
		s.suspended = (s.runtime_status == "suspended");
		if (fd_ < 0) return s;
		if (s.suspended && !wake) {
			carry_over_processes(s);
			s.util_source = UtilSource::suspended;
			prev_completions_.clear();
			idle_streak_ = 0;
			was_suspended_ = true;
			return s;
		}
		//* Every ioctl resets the autosuspend timer. After a few idle samples, only
		//* re-query every idle_backoff so an otherwise idle NPU can suspend. A
		//* suspended -> active transition is never caused by us (we don't query
		//* while suspended), so it means a client started: query immediately.
		if (was_suspended_) idle_streak_ = 0;
		was_suspended_ = false;
		if (!wake && idle_streak_ >= idle_samples_before_backoff &&
		    s.when - last_query_ < idle_backoff) {
			s.mp_npu_clock_mhz = last_mp_clk_;
			s.h_clock_mhz = last_h_clk_;
			s.util_source = sensors_supported_ ? UtilSource::sensors : UtilSource::column_occupancy;
			carry_over_processes(s);
			return s;
		}
		last_query_ = s.when;
		s.fresh = true;

		amdxdna_drm_query_clock_metadata ck{};
		if (get_info(DRM_AMDXDNA_QUERY_CLOCK_METADATA, &ck, sizeof ck)) {
			s.mp_npu_clock_mhz = ck.mp_npu_clock.freq_mhz;
			s.h_clock_mhz = ck.h_clock.freq_mhz;
		}
		amdxdna_drm_get_power_mode pm{};
		if (get_info(DRM_AMDXDNA_GET_POWER_MODE, &pm, sizeof pm)) s.power_mode = pm.power_mode;
		amdxdna_drm_get_resource_info ri{};
		if (get_info(DRM_AMDXDNA_QUERY_RESOURCE_INFO, &ri, sizeof ri)) {
			s.tops_curr = ri.npu_tops_curr;
			s.tasks_curr = ri.npu_task_curr;
		}

		read_sensors(s);
		read_contexts(s);

		//* Utilisation: real column sensors when present, else column occupancy
		//* (columns owned by contexts that are ACTIVE or completed work since the
		//* last sample, divided by total columns).
		const uint32_t cols = info_.cols ? info_.cols : 1;
		if (!s.column_util_pct.empty()) {
			double sum = 0;
			for (double v : s.column_util_pct) sum += v;
			s.util_pct = sum / static_cast<double>(s.column_util_pct.size());
			s.util_source = UtilSource::sensors;
		} else {
			std::vector<bool> busy(cols, false);
			for (const auto& c : s.contexts)
				if (c.busy)
					for (uint32_t i = c.start_col; i < c.start_col + c.num_col && i < cols; ++i) busy[i] = true;
			for (bool b : busy) s.busy_cols += b;
			s.util_pct = 100.0 * s.busy_cols / cols;
			s.util_source = UtilSource::column_occupancy;
		}

		std::set<int64_t> pids = extra_pids;
		for (const auto& c : s.contexts) pids.insert(c.pid);
		read_bo_usage(s, pids);

		last_processes_ = s.processes;
		last_mp_clk_ = s.mp_npu_clock_mhz;
		last_h_clk_ = s.h_clock_mhz;
		idle_streak_ = s.util_pct > 0 ? 0 : idle_streak_ + 1;
		return s;
	}

	//* Back-off tuning (see sample()). Defaults suit a 1 s refresh and the
	//* driver's 5 s autosuspend delay.
	int idle_samples_before_backoff = 3;
	std::chrono::steady_clock::duration idle_backoff = std::chrono::seconds(10);

	bool sensors_supported() const { return sensors_supported_; }

	//* Raw GET_INFO wrapper. Returns true on success.
	bool get_info(uint32_t param, void* buf, uint32_t size, uint32_t* out_size = nullptr) const {
		amdxdna_drm_get_info gi{};
		gi.param = param;
		gi.buffer_size = size;
		gi.buffer = detail::to_u64(buf);
		const int r = ::ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_INFO, &gi);
		if (out_size) *out_size = gi.buffer_size;
		return r == 0;
	}

	//* Raw GET_ARRAY wrapper. Returns number of elements, or -errno.
	long get_array(uint32_t param, void* buf, uint32_t elem_size, uint32_t max_elems) const {
		amdxdna_drm_get_array ga{};
		ga.param = param;
		ga.element_size = elem_size;
		ga.num_element = max_elems;
		ga.buffer = detail::to_u64(buf);
		if (::ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_ARRAY, &ga) != 0) return -errno;
		return ga.num_element;
	}

private:
	//* BOs survive runtime suspend, so keep reporting the last-known usage of
	//* processes that are still alive when we skip the ioctls.
	void carry_over_processes(Sample& s) {
		std::error_code ec;
		std::erase_if(last_processes_, [&](const ProcUsage& p) {
			return !fs::exists(fs::path("/proc") / std::to_string(p.pid), ec);
		});
		s.processes = last_processes_;
		for (auto& p : s.processes) { p.contexts = 0; s.mem_used_bytes += p.total_bytes; }
	}

	void read_sensors(Sample& s) {
		if (!sensors_supported_) return;
		std::array<amdxdna_drm_query_sensor, 32> buf{};
		uint32_t got = 0;
		if (!get_info(DRM_AMDXDNA_QUERY_SENSORS, buf.data(), sizeof buf, &got)) {
			//* ENODEV/EOPNOTSUPP/EINVAL: platform has no PMF NPU metrics. Stop asking.
			if (errno == ENODEV || errno == EOPNOTSUPP || errno == EINVAL) sensors_supported_ = false;
			return;
		}
		const size_t n = std::min<size_t>(got / sizeof(amdxdna_drm_query_sensor), buf.size());
		for (size_t i = 0; i < n; ++i) {
			const auto& r = buf[i];
			Sensor x;
			x.label = detail::cstr(r.label);
			x.units = detail::cstr(r.units);
			x.status = detail::cstr(r.status);
			x.input = r.input; x.max = r.max; x.average = r.average; x.highest = r.highest;
			x.unitm = r.unitm; x.type = r.type;
			if (r.type == AMDXDNA_SENSOR_TYPE_POWER) {
				double v = x.value();                    // in `units`
				if (x.units == "W") v *= 1000.0;
				else if (x.units == "uW") v /= 1000.0;
				s.power_mw = v;
			} else if (r.type == AMDXDNA_SENSOR_TYPE_COLUMN_UTILIZATION) {
				s.column_util_pct.push_back(std::clamp(x.value(), 0.0, 100.0));
			}
			s.sensors.push_back(std::move(x));
		}
	}

	void read_contexts(Sample& s) {
		std::vector<amdxdna_drm_hwctx_entry> buf(64);
		const long n = get_array(DRM_AMDXDNA_HW_CONTEXT_ALL, buf.data(), sizeof(amdxdna_drm_hwctx_entry),
		                         static_cast<uint32_t>(buf.size()));
		std::map<uint64_t, uint64_t> next;
		for (long i = 0; i < n; ++i) {
			const auto& e = buf[static_cast<size_t>(i)];
			HwContext c;
			c.context_id = e.context_id;
			c.hwctx_id = e.hwctx_id;
			c.start_col = e.start_col;
			c.num_col = e.num_col;
			c.pid = e.pid;
			c.submissions = e.command_submissions;
			c.completions = e.command_completions;
			c.errors = e.errors;
			c.heap_usage = e.heap_usage;
			c.state = e.state;
			c.priority = e.priority;
			const uint64_t key = (static_cast<uint64_t>(e.pid) << 32) | e.context_id;
			auto it = prev_completions_.find(key);
			const bool advanced = it != prev_completions_.end() && e.command_completions > it->second;
			const bool pending = e.command_submissions > e.command_completions;
			c.busy = e.state == AMDXDNA_HWCTX_STATE_ACTIVE || advanced || pending;
			next[key] = e.command_completions;
			s.contexts.push_back(c);
		}
		prev_completions_ = std::move(next);
	}

	void read_bo_usage(Sample& s, const std::set<int64_t>& pids) {
		for (int64_t pid : pids) {
			amdxdna_drm_bo_usage u{};
			u.pid = pid;
			if (get_array(DRM_AMDXDNA_BO_USAGE, &u, sizeof u, 1) < 0) continue;
			ProcUsage p;
			p.pid = pid;
			p.comm = proc_comm(pid);
			p.total_bytes = u.total_usage;
			p.internal_bytes = u.internal_usage;
			p.heap_bytes = u.heap_usage;
			for (const auto& c : s.contexts) p.contexts += (c.pid == pid);
			s.mem_used_bytes += p.total_bytes;
			s.processes.push_back(std::move(p));
		}
	}

	DeviceInfo info_;
	int fd_ = -1;
	std::map<uint64_t, uint64_t> prev_completions_;
	bool sensors_supported_ = true;
	std::string last_error_;
	int idle_streak_ = 0;
	bool was_suspended_ = false;
	std::chrono::steady_clock::time_point last_query_{};
	uint32_t last_mp_clk_ = 0, last_h_clk_ = 0;
	std::vector<ProcUsage> last_processes_;
};

} // namespace xdna
