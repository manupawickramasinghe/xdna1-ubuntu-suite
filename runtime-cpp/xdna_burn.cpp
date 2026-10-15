// SPDX-License-Identifier: MIT
//
// xdna_burn — NPU benchmark + reliability burn test using mlir-aie's
// vector_scalar_mul kernel (int16 out[i] = in[i] * scale).
//
//   xdna_burn --xclbin final_8192.xclbin --insts insts_8192.bin
//             [--elements 4096] [--contexts 1..5] [--bench-runs 2000]
//             [--minutes 10] [--json report.json]
//
// Phase 1 (bench): single context, latency distribution + throughput.
// Phase 2 (burn):  N threads, each with its own hardware context (so N
//                  columns are busy), fresh random inputs every run, every
//                  output verified. Reports every 10 s; exits non-zero on any
//                  mismatch, timeout or exception.

#include "xdna_runner.hpp"
#include "xdna_telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

namespace {

using clk = std::chrono::steady_clock;
std::atomic<bool> g_stop{false};

struct Opts {
	std::string xclbin, insts, kernel, json;
	size_t elements = 4096;
	int contexts = 1;
	int bench_runs = 2000;
	double minutes = 10;
	int timeout_ms = 1000;
	int out_pos = 1;   // vector_scalar_mul runtime sequence is (A, C, factor)
};

struct Stats {
	std::vector<double> lat_us;
	void add(double v) { lat_us.push_back(v); }
	double pct(double p) {
		if (lat_us.empty()) return 0;
		std::sort(lat_us.begin(), lat_us.end());
		return lat_us[std::min(lat_us.size() - 1, static_cast<size_t>(p / 100.0 * lat_us.size()))];
	}
	double avg() const {
		return lat_us.empty() ? 0 : std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / lat_us.size();
	}
};

struct Shared {
	std::atomic<uint64_t> runs{0}, mismatches{0}, bad_elems{0}, timeouts{0}, exceptions{0};
	std::mutex mu;
	std::vector<double> window_lat;   // latencies since last report
	std::string first_error;
};

xdna::RunnerConfig runner_cfg(const Opts& o) {
	xdna::RunnerConfig rc;
	rc.xclbin_path = o.xclbin;
	rc.insts_path = o.insts;
	rc.kernel_name = o.kernel;
	rc.input_bytes = {o.elements * sizeof(int16_t), sizeof(int32_t)};
	rc.output_bytes = o.elements * sizeof(int16_t);
	rc.out_pos = o.out_pos;
	return rc;
}

//* Fill fresh random inputs, run, verify. Returns number of wrong elements (or -1 on timeout).
long one_run(xdna::XdnaRunner& r, std::mt19937& rng, size_t n, int timeout_ms, double& lat_us) {
	std::uniform_int_distribution<int> val(-1000, 1000), sc(-30, 30);
	auto* in = reinterpret_cast<int16_t*>(r.input(0).data());
	for (size_t i = 0; i < n; ++i) in[i] = static_cast<int16_t>(val(rng));
	const int32_t scale = sc(rng);
	std::memcpy(r.input(1).data(), &scale, sizeof scale);
	// Poison the output so a run that silently does nothing can't pass.
	auto* out_poison = const_cast<uint8_t*>(r.output().data());
	std::memset(out_poison, 0xA5, r.output().size());

	const auto t0 = clk::now();
	try {
		r.run(std::chrono::milliseconds(timeout_ms));
	} catch (const std::runtime_error& e) {
		if (std::string(e.what()).find("timed out") != std::string::npos) return -1;
		throw;
	}
	lat_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();

	const auto* out = reinterpret_cast<const int16_t*>(r.output().data());
	long bad = 0;
	for (size_t i = 0; i < n; ++i)
		if (out[i] != static_cast<int16_t>(in[i] * scale)) ++bad;
	return bad;
}

int read_tctl_c() {
	// k10temp Tctl (CPU/APU package) — the NPU has no temperature sensor of its own.
	for (const auto& e : std::filesystem::directory_iterator("/sys/class/hwmon")) {
		if (xdna::detail::read_line(e.path() / "name") == "k10temp") {
			auto v = xdna::detail::read_line(e.path() / "temp1_input");
			return v.empty() ? -1 : std::stoi(v) / 1000;
		}
	}
	return -1;
}

std::string kernel_log_since(const std::string& since) {
	std::string cmd = "journalctl -k --no-pager -o short-monotonic --since '" + since + "' 2>/dev/null | grep -iE 'amdxdna|iommu|accel' | grep -viE 'audit'";
	std::string out;
	if (FILE* p = popen(cmd.c_str(), "r")) {
		char buf[512];
		while (fgets(buf, sizeof buf, p)) out += buf;
		pclose(p);
	}
	return out;
}

std::string now_str() {
	std::time_t t = std::time(nullptr);
	char b[32];
	std::strftime(b, sizeof b, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
	return b;
}

[[noreturn]] void usage(const char* a) {
	std::fprintf(stderr,
	             "usage: %s --xclbin F --insts F [--kernel MLIR_AIE] [--elements 4096] [--contexts N]\n"
	             "          [--bench-runs N] [--minutes M] [--timeout-ms N] [--json FILE]\n",
	             a);
	std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
	Opts o;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		auto next = [&]() -> std::string { if (i + 1 >= argc) usage(argv[0]); return argv[++i]; };
		if (a == "--xclbin") o.xclbin = next();
		else if (a == "--insts") o.insts = next();
		else if (a == "--kernel") o.kernel = next();
		else if (a == "--elements") o.elements = std::stoull(next());
		else if (a == "--contexts") o.contexts = std::stoi(next());
		else if (a == "--bench-runs") o.bench_runs = std::stoi(next());
		else if (a == "--minutes") o.minutes = std::stod(next());
		else if (a == "--timeout-ms") o.timeout_ms = std::stoi(next());
		else if (a == "--json") o.json = next();
		else if (a == "--out-pos") o.out_pos = std::stoi(next());
		else usage(argv[0]);
	}
	if (o.xclbin.empty() || o.insts.empty() || o.contexts < 1) usage(argv[0]);
	std::signal(SIGINT, [](int) { g_stop = true; });
	std::signal(SIGTERM, [](int) { g_stop = true; });

	const std::string started = now_str();
	auto infos = xdna::discover();
	std::optional<xdna::Device> tel;
	if (!infos.empty()) { tel.emplace(infos[0]); if (!tel->open()) tel.reset(); }

	std::printf("xdna_burn: %s, %zu int16 elements, started %s, Tctl %d C\n", o.xclbin.c_str(), o.elements,
	            started.c_str(), read_tctl_c());

	// ---------------- Phase 1: benchmark ----------------
	double ctx_ms = 0;
	Stats bench;
	uint64_t bench_bad = 0;
	double bench_wall_s = 0;
	{
		const auto t0 = clk::now();
		xdna::XdnaRunner r(runner_cfg(o));
		ctx_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
		std::mt19937 rng(1234);
		double lat;
		for (int i = 0; i < 50; ++i) one_run(r, rng, o.elements, o.timeout_ms, lat); // warm-up
		const auto b0 = clk::now();
		for (int i = 0; i < o.bench_runs && !g_stop; ++i) {
			long bad = one_run(r, rng, o.elements, o.timeout_ms, lat);
			if (bad != 0) ++bench_bad; else bench.add(lat);
		}
		bench_wall_s = std::chrono::duration<double>(clk::now() - b0).count();
	}
	const double bytes_per_run = 2.0 * o.elements * sizeof(int16_t);
	std::printf("\n[bench] context setup %.1f ms\n", ctx_ms);
	std::printf("[bench] %d runs: min %.1f  p50 %.1f  avg %.1f  p99 %.1f  max %.1f us, %.0f runs/s, %.1f MB/s host<->NPU, %" PRIu64 " bad\n",
	            o.bench_runs, bench.pct(0), bench.pct(50), bench.avg(), bench.pct(99), bench.pct(100),
	            o.bench_runs / bench_wall_s, o.bench_runs * bytes_per_run / bench_wall_s / 1e6, bench_bad);
	std::fflush(stdout);

	// ---------------- Phase 2: burn ----------------
	Shared sh;
	std::atomic<int> ready{0};
	std::vector<std::thread> threads;
	std::vector<std::string> ctx_errors(o.contexts);
	for (int t = 0; t < o.contexts; ++t) {
		threads.emplace_back([&, t] {
			try {
				xdna::XdnaRunner r(runner_cfg(o));
				std::mt19937 rng(42 + t);
				++ready;
				std::vector<double> local;
				while (!g_stop) {
					double lat = 0;
					long bad = one_run(r, rng, o.elements, o.timeout_ms, lat);
					++sh.runs;
					if (bad < 0) { ++sh.timeouts; continue; }
					if (bad > 0) {
						++sh.mismatches; sh.bad_elems += bad;
						std::lock_guard l(sh.mu);
						if (sh.first_error.empty())
							sh.first_error = "ctx " + std::to_string(t) + ": " + std::to_string(bad) + " wrong elements at run " + std::to_string(sh.runs.load());
					}
					local.push_back(lat);
					if (local.size() >= 256) {
						std::lock_guard l(sh.mu);
						sh.window_lat.insert(sh.window_lat.end(), local.begin(), local.end());
						local.clear();
					}
				}
			} catch (const std::exception& e) {
				++sh.exceptions;
				ctx_errors[t] = e.what();
				++ready;
			}
		});
	}
	while (ready < o.contexts) std::this_thread::sleep_for(std::chrono::milliseconds(10));

	std::printf("\n[burn] %d context(s) for %.1f min (Ctrl-C to stop early)\n", o.contexts, o.minutes);
	std::printf("[burn]  time   runs/s   p99 us  max us  H-clk  util  Tctl  mism  tmo  exc\n");
	const auto b0 = clk::now();
	const auto b_end = b0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(o.minutes * 60));
	uint64_t last_runs = 0;
	auto last_t = b0;
	int tctl_max = -1;
	uint32_t hclk_min = UINT32_MAX, hclk_max = 0;
	std::vector<std::string> report_lines;
	while (!g_stop && clk::now() < b_end && sh.exceptions == 0) {
		std::this_thread::sleep_for(std::chrono::seconds(10));
		const auto now = clk::now();
		Stats w;
		{ std::lock_guard l(sh.mu); w.lat_us.swap(sh.window_lat); }
		const uint64_t runs = sh.runs.load();
		const double rate = (runs - last_runs) / std::chrono::duration<double>(now - last_t).count();
		last_runs = runs; last_t = now;
		xdna::Sample s;
		if (tel) s = tel->sample(true);
		const int tctl = read_tctl_c();
		tctl_max = std::max(tctl_max, tctl);
		if (s.h_clock_mhz) { hclk_min = std::min(hclk_min, s.h_clock_mhz); hclk_max = std::max(hclk_max, s.h_clock_mhz); }
		char line[256];
		std::snprintf(line, sizeof line, "[burn] %5.0fs %8.0f %8.1f %7.1f %6u %4.0f%% %4dC %5" PRIu64 " %4" PRIu64 " %4" PRIu64,
		              std::chrono::duration<double>(now - b0).count(), rate, w.pct(99), w.pct(100), s.h_clock_mhz,
		              s.util_pct, tctl, sh.mismatches.load(), sh.timeouts.load(), sh.exceptions.load());
		std::puts(line);
		std::fflush(stdout);
		report_lines.push_back(line);
	}
	g_stop = true;
	for (auto& t : threads) t.join();
	const double burn_s = std::chrono::duration<double>(clk::now() - b0).count();

	const std::string klog = kernel_log_since(started);
	const bool pass = sh.mismatches == 0 && sh.timeouts == 0 && sh.exceptions == 0 && bench_bad == 0;

	std::printf("\n[result] burn %.0f s, %" PRIu64 " runs (%.0f runs/s), %.2f GB verified\n", burn_s, sh.runs.load(),
	            sh.runs / burn_s, sh.runs * o.elements * sizeof(int16_t) / 1e9);
	std::printf("[result] mismatched runs %" PRIu64 " (%" PRIu64 " elements), timeouts %" PRIu64 ", exceptions %" PRIu64 "\n",
	            sh.mismatches.load(), sh.bad_elems.load(), sh.timeouts.load(), sh.exceptions.load());
	if (!sh.first_error.empty()) std::printf("[result] first error: %s\n", sh.first_error.c_str());
	for (int t = 0; t < o.contexts; ++t)
		if (!ctx_errors[t].empty()) std::printf("[result] ctx %d exception: %s\n", t, ctx_errors[t].c_str());
	std::printf("[result] H-clock range %u-%u MHz, max Tctl %d C\n", hclk_min == UINT32_MAX ? 0 : hclk_min, hclk_max, tctl_max);
	std::printf("[result] kernel log (amdxdna/iommu) since start: %s\n", klog.empty() ? "clean" : "");
	if (!klog.empty()) std::fputs(klog.c_str(), stdout);
	std::printf("[result] %s\n", pass ? "PASS" : "FAIL");

	if (!o.json.empty()) {
		std::ofstream j(o.json);
		j << "{\"started\":\"" << started << "\",\"xclbin\":\"" << o.xclbin << "\",\"elements\":" << o.elements
		  << ",\"contexts\":" << o.contexts << ",\"bench\":{\"ctx_setup_ms\":" << ctx_ms << ",\"runs\":" << o.bench_runs
		  << ",\"min_us\":" << bench.pct(0) << ",\"p50_us\":" << bench.pct(50) << ",\"avg_us\":" << bench.avg()
		  << ",\"p99_us\":" << bench.pct(99) << ",\"max_us\":" << bench.pct(100)
		  << ",\"runs_per_s\":" << o.bench_runs / bench_wall_s << ",\"bad\":" << bench_bad << "},"
		  << "\"burn\":{\"seconds\":" << burn_s << ",\"runs\":" << sh.runs << ",\"mismatches\":" << sh.mismatches
		  << ",\"bad_elements\":" << sh.bad_elems << ",\"timeouts\":" << sh.timeouts << ",\"exceptions\":" << sh.exceptions
		  << ",\"hclk_min\":" << (hclk_min == UINT32_MAX ? 0 : hclk_min) << ",\"hclk_max\":" << hclk_max
		  << ",\"tctl_max_c\":" << tctl_max << ",\"kernel_log_clean\":" << (klog.empty() ? "true" : "false")
		  << "},\"pass\":" << (pass ? "true" : "false") << "}\n";
	}
	return pass ? 0 : 1;
}
