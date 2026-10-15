// SPDX-License-Identifier: MIT
//
// xdna_stat — print AMD XDNA NPU telemetry from the amdxdna driver.
//
//   xdna_stat              one sample (does not wake a suspended NPU)
//   xdna_stat -w           one sample, waking the NPU if needed
//   xdna_stat -i 1000      sample every 1000 ms until Ctrl-C
//   xdna_stat --json       machine-readable output

#include "xdna_telemetry.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <thread>

namespace {

volatile std::sig_atomic_t g_stop = 0;

std::string human_bytes(uint64_t b) {
	const char* units[] = {"B", "KiB", "MiB", "GiB"};
	double v = static_cast<double>(b);
	int u = 0;
	while (v >= 1024.0 && u < 3) { v /= 1024.0; ++u; }
	char buf[32];
	std::snprintf(buf, sizeof buf, u ? "%.1f %s" : "%.0f %s", v, units[u]);
	return buf;
}

std::string json_escape(const std::string& s) {
	std::string o;
	for (char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += c; }
		else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += c;
	}
	return o;
}

void print_text(const xdna::DeviceInfo& d, const xdna::Sample& s) {
	std::printf("%s  [%04x:%04x @ %s]  %s  fw %s\n", d.name.c_str(), d.vendor, d.device, d.pci_slot.c_str(),
	            d.vbnv.c_str(), d.fw_version.c_str());
	std::printf("  array      : %u cols x %u rows (core %u, mem %u, shim %u), AIE v%u.%u\n", d.cols, d.rows,
	            d.core_rows, d.mem_rows, d.shim_rows, d.aie_major, d.aie_minor);
	std::printf("  runtime pm : %s\n", s.runtime_status.c_str());
	if (s.suspended && s.util_source == xdna::UtilSource::suspended) {
		std::printf("  state      : idle (suspended; pass -w to wake and query)\n");
		return;
	}
	std::printf("  clocks     : MP-NPU %u MHz, H-clock %u MHz", s.mp_npu_clock_mhz, s.h_clock_mhz);
	if (d.clk_max_mhz) std::printf(" (max %llu)", static_cast<unsigned long long>(d.clk_max_mhz));
	std::printf("\n");
	if (s.power_mode) {
		static const char* modes[] = {"default", "low", "medium", "high", "turbo"};
		std::printf("  power mode : %s\n", *s.power_mode < 5 ? modes[*s.power_mode] : "?");
	}
	if (s.power_mw) std::printf("  power      : %.0f mW\n", *s.power_mw);
	else std::printf("  power      : n/a (no PMF NPU sensor on this platform)\n");
	std::printf("  utilisation: %.0f%% (%s, %u/%u cols busy)\n", s.util_pct, xdna::to_string(s.util_source),
	            s.busy_cols, d.cols);
	if (d.tops_max)
		std::printf("  resources  : %llu/%llu tasks, %llu TOPS reserved (max %llu)\n",
		            static_cast<unsigned long long>(s.tasks_curr), static_cast<unsigned long long>(d.task_max),
		            static_cast<unsigned long long>(s.tops_curr), static_cast<unsigned long long>(d.tops_max));
	std::printf("  bo memory  : %s\n", human_bytes(s.mem_used_bytes).c_str());
	for (const auto& c : s.contexts)
		std::printf("    ctx %-3u pid %-7lld %-15s cols %u-%u  %s  sub %llu  done %llu  err %llu\n", c.context_id,
		            static_cast<long long>(c.pid), xdna::proc_comm(c.pid).c_str(), c.start_col,
		            c.start_col + c.num_col - 1, c.busy ? "BUSY" : "idle",
		            static_cast<unsigned long long>(c.submissions), static_cast<unsigned long long>(c.completions),
		            static_cast<unsigned long long>(c.errors));
	for (const auto& p : s.processes)
		std::printf("    pid %-7lld %-15s %s (internal %s, heap %s)\n", static_cast<long long>(p.pid), p.comm.c_str(),
		            human_bytes(p.total_bytes).c_str(), human_bytes(p.internal_bytes).c_str(),
		            human_bytes(p.heap_bytes).c_str());
}

void print_json(const xdna::DeviceInfo& d, const xdna::Sample& s) {
	std::ostringstream o;
	o << "{\"device\":{\"index\":" << d.index << ",\"name\":\"" << json_escape(d.name) << "\",\"pci\":\""
	  << d.pci_slot << "\",\"vendor\":" << d.vendor << ",\"device\":" << d.device << ",\"vbnv\":\""
	  << json_escape(d.vbnv) << "\",\"fw\":\"" << json_escape(d.fw_version) << "\",\"cols\":" << d.cols
	  << ",\"rows\":" << d.rows << ",\"clk_max_mhz\":" << d.clk_max_mhz << "},";
	o << "\"runtime_status\":\"" << json_escape(s.runtime_status) << "\",\"suspended\":" << (s.suspended ? "true" : "false")
	  << ",\"mp_npu_clock_mhz\":" << s.mp_npu_clock_mhz << ",\"h_clock_mhz\":" << s.h_clock_mhz
	  << ",\"power_mw\":" << (s.power_mw ? std::to_string(*s.power_mw) : "null") << ",\"util_pct\":" << s.util_pct
	  << ",\"util_source\":\"" << xdna::to_string(s.util_source) << "\",\"busy_cols\":" << s.busy_cols
	  << ",\"mem_used_bytes\":" << s.mem_used_bytes << ",\"contexts\":[";
	for (size_t i = 0; i < s.contexts.size(); ++i) {
		const auto& c = s.contexts[i];
		o << (i ? "," : "") << "{\"id\":" << c.context_id << ",\"pid\":" << c.pid << ",\"start_col\":" << c.start_col
		  << ",\"num_col\":" << c.num_col << ",\"busy\":" << (c.busy ? "true" : "false") << ",\"submissions\":"
		  << c.submissions << ",\"completions\":" << c.completions << ",\"errors\":" << c.errors << "}";
	}
	o << "],\"processes\":[";
	for (size_t i = 0; i < s.processes.size(); ++i) {
		const auto& p = s.processes[i];
		o << (i ? "," : "") << "{\"pid\":" << p.pid << ",\"comm\":\"" << json_escape(p.comm) << "\",\"total_bytes\":"
		  << p.total_bytes << ",\"internal_bytes\":" << p.internal_bytes << ",\"heap_bytes\":" << p.heap_bytes << "}";
	}
	o << "]}";
	std::cout << o.str() << std::endl;
}

void usage(const char* argv0) {
	std::fprintf(stderr, "usage: %s [-w|--wake] [-i|--interval MS] [--json] [--scan-pids]\n", argv0);
}

} // namespace

int main(int argc, char** argv) {
	bool wake = false, json = false, scan = false;
	int interval_ms = 0;
	for (int i = 1; i < argc; ++i) {
		std::string_view a = argv[i];
		if (a == "-w" || a == "--wake") wake = true;
		else if (a == "--json") json = true;
		else if (a == "--scan-pids") scan = true;
		else if ((a == "-i" || a == "--interval") && i + 1 < argc) interval_ms = std::atoi(argv[++i]);
		else { usage(argv[0]); return a == "-h" || a == "--help" ? 0 : 2; }
	}

	auto infos = xdna::discover();
	if (infos.empty()) {
		std::fprintf(stderr, "No amdxdna device found under /sys/class/accel. Is the NPU enabled in BIOS and the "
		                     "amdxdna module loaded?\n");
		return 1;
	}
	std::vector<xdna::Device> devs;
	for (auto& i : infos) {
		xdna::Device d(i);
		if (!d.open()) {
			std::fprintf(stderr, "%s (add yourself to the 'render' group)\n", d.last_error().c_str());
			return 1;
		}
		devs.push_back(std::move(d));
	}

	std::signal(SIGINT, [](int) { g_stop = 1; });
	std::signal(SIGTERM, [](int) { g_stop = 1; });
	do {
		for (auto& d : devs) {
			auto pids = scan ? xdna::pids_using(d.info().dev_node) : std::set<int64_t>{};
			pids.erase(getpid());
			auto s = d.sample(wake, pids);
			if (json) print_json(d.info(), s);
			else print_text(d.info(), s);
		}
		if (interval_ms > 0 && !json) std::printf("\n");
		std::fflush(stdout);
		if (interval_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
	} while (interval_ms > 0 && !g_stop);
	return 0;
}
