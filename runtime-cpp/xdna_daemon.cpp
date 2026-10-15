// SPDX-License-Identifier: MIT
//
// xdna_daemon — keep one NPU kernel loaded and serve inferences over a Unix
// socket, or generate a controllable NPU load with --benchmark-loop.
//
// Serve:
//   xdna_daemon --xclbin final.xclbin --insts insts.bin --in 4096,4 --out 4096
//               [--kernel MLIR_AIE] [--socket PATH]
//
// Load generator (watch it in btop / xdna_stat):
//   xdna_daemon --xclbin ... --insts ... --in ... --out ... --benchmark-loop
//               [--duty 50] [--period-ms 200] [--seconds 0]
//
// Wire protocol (little-endian, one request/response at a time per connection):
//   request : u32 magic 'XDNQ' | u32 n_inputs | u64 size[n_inputs] | payloads...
//   response: u32 magic 'XDNR' | i32 status (0 ok, else -errno) | u64 out_size | payload
//   On error, out_size is the length of a UTF-8 error message instead.
//
// Hardware context creation (xclbin load + partition setup) costs tens of ms.
// Holding it in a daemon makes each request cost only BO sync + dispatch.

#include "xdna_runner.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

constexpr uint32_t kReqMagic = 0x514e4458; // "XDNQ"
constexpr uint32_t kRspMagic = 0x524e4458; // "XDNR"
constexpr uint32_t kMaxInputs = 16;

std::atomic<bool> g_stop{false};

using clk = std::chrono::steady_clock;

struct Options {
	xdna::RunnerConfig rc;
	std::string socket_path;
	bool benchmark = false;
	int duty = 100;          // percent of each period spent dispatching
	int period_ms = 200;
	int seconds = 0;         // 0 = until Ctrl-C
	int timeout_ms = 1000;
};

[[noreturn]] void usage(const char* argv0, int code) {
	std::fprintf(code ? stderr : stdout,
	             "usage: %s --xclbin FILE --insts FILE --in BYTES[,BYTES...] --out BYTES\n"
	             "          [--kernel NAME] [--out-pos N] [--device N] [--timeout-ms N]\n"
	             "          [--socket PATH | --benchmark-loop [--duty 1-100] [--period-ms N] [--seconds N]]\n",
	             argv0);
	std::exit(code);
}

std::vector<size_t> parse_sizes(const std::string& s) {
	std::vector<size_t> out;
	std::stringstream ss(s);
	std::string tok;
	while (std::getline(ss, tok, ',')) out.push_back(std::stoull(tok));
	return out;
}

std::string default_socket() {
	if (geteuid() == 0) return "/run/xdna_infer.sock";
	if (const char* x = std::getenv("XDG_RUNTIME_DIR")) return std::string(x) + "/xdna_infer.sock";
	return "/tmp/xdna_infer-" + std::to_string(getuid()) + ".sock";
}

Options parse(int argc, char** argv) {
	Options o;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) usage(argv[0], 2);
			return argv[++i];
		};
		if (a == "--xclbin") o.rc.xclbin_path = next();
		else if (a == "--insts") o.rc.insts_path = next();
		else if (a == "--kernel") o.rc.kernel_name = next();
		else if (a == "--in") o.rc.input_bytes = parse_sizes(next());
		else if (a == "--out") o.rc.output_bytes = std::stoull(next());
		else if (a == "--out-pos") o.rc.out_pos = std::stoi(next());
		else if (a == "--device") o.rc.device_index = static_cast<unsigned>(std::stoul(next()));
		else if (a == "--socket") o.socket_path = next();
		else if (a == "--benchmark-loop") o.benchmark = true;
		else if (a == "--duty") o.duty = std::stoi(next());
		else if (a == "--period-ms") o.period_ms = std::stoi(next());
		else if (a == "--seconds") o.seconds = std::stoi(next());
		else if (a == "--timeout-ms") o.timeout_ms = std::stoi(next());
		else if (a == "-h" || a == "--help") usage(argv[0], 0);
		else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0], 2); }
	}
	if (o.rc.xclbin_path.empty() || o.rc.insts_path.empty() || o.rc.output_bytes == 0) usage(argv[0], 2);
	if (o.duty < 1 || o.duty > 100 || o.period_ms < 1) usage(argv[0], 2);
	if (o.socket_path.empty()) o.socket_path = default_socket();
	return o;
}

// ---------------------------------------------------------------------------
int benchmark(xdna::XdnaRunner& r, const Options& o) {
	// Fill inputs with a ramp so outputs are non-trivial.
	for (size_t i = 0; i < r.num_inputs(); ++i) {
		auto in = r.input(i);
		for (size_t b = 0; b < in.size(); ++b) in[b] = static_cast<uint8_t>(b);
	}
	std::printf("benchmark: kernel %s, duty %d%% of %d ms periods%s\n", r.config().kernel_name.c_str(), o.duty,
	            o.period_ms, o.seconds ? (", " + std::to_string(o.seconds) + " s").c_str() : ", Ctrl-C to stop");

	const auto period = std::chrono::milliseconds(o.period_ms);
	const auto busy = period * o.duty / 100;
	const auto t_end = o.seconds ? clk::now() + std::chrono::seconds(o.seconds) : clk::time_point::max();
	auto t_report = clk::now() + std::chrono::seconds(1);
	uint64_t runs = 0, total_runs = 0;
	std::vector<double> lat_us;

	while (!g_stop && clk::now() < t_end) {
		const auto p0 = clk::now();
		while (!g_stop && clk::now() - p0 < busy) {
			const auto t0 = clk::now();
			r.run(std::chrono::milliseconds(o.timeout_ms));
			lat_us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
			++runs;
		}
		const auto rest = period - (clk::now() - p0);
		if (rest > clk::duration::zero()) std::this_thread::sleep_for(rest);

		if (clk::now() >= t_report) {
			std::sort(lat_us.begin(), lat_us.end());
			const double avg = lat_us.empty() ? 0 : std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / lat_us.size();
			const double p99 = lat_us.empty() ? 0 : lat_us[std::min(lat_us.size() - 1, lat_us.size() * 99 / 100)];
			std::printf("  %6llu runs/s  avg %8.1f us  p99 %8.1f us\n", static_cast<unsigned long long>(runs), avg, p99);
			std::fflush(stdout);
			total_runs += runs;
			runs = 0;
			lat_us.clear();
			t_report += std::chrono::seconds(1);
		}
	}
	std::printf("benchmark: %llu runs total\n", static_cast<unsigned long long>(total_runs + runs));
	return 0;
}

// ---------------------------------------------------------------------------
bool read_full(int fd, void* buf, size_t n) {
	auto* p = static_cast<uint8_t*>(buf);
	while (n) {
		const ssize_t r = ::read(fd, p, n);
		if (r == 0) return false;
		if (r < 0) { if (errno == EINTR) continue; return false; }
		p += r; n -= static_cast<size_t>(r);
	}
	return true;
}

bool write_full(int fd, const void* buf, size_t n) {
	auto* p = static_cast<const uint8_t*>(buf);
	while (n) {
		const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
		if (w < 0) { if (errno == EINTR) continue; return false; }
		p += w; n -= static_cast<size_t>(w);
	}
	return true;
}

bool respond(int fd, int32_t status, std::span<const uint8_t> payload) {
	uint8_t hdr[16];
	const uint64_t n = payload.size();
	std::memcpy(hdr, &kRspMagic, 4);
	std::memcpy(hdr + 4, &status, 4);
	std::memcpy(hdr + 8, &n, 8);
	return write_full(fd, hdr, sizeof hdr) && (payload.empty() || write_full(fd, payload.data(), payload.size()));
}

bool respond_error(int fd, int32_t status, const std::string& msg) {
	return respond(fd, status, {reinterpret_cast<const uint8_t*>(msg.data()), msg.size()});
}

//* Serve one connection until the client closes it. Returns false on protocol error.
bool serve_client(int fd, xdna::XdnaRunner& r, const Options& o) {
	for (;;) {
		uint32_t magic = 0, n_in = 0;
		if (!read_full(fd, &magic, 4)) return true; // clean close
		if (magic != kReqMagic || !read_full(fd, &n_in, 4) || n_in > kMaxInputs) {
			respond_error(fd, -EPROTO, "bad request header");
			return false;
		}
		std::vector<uint64_t> sizes(n_in);
		if (n_in && !read_full(fd, sizes.data(), n_in * 8)) return false;
		if (n_in != r.num_inputs()) {
			respond_error(fd, -EINVAL, "expected " + std::to_string(r.num_inputs()) + " inputs");
			return false;
		}
		for (uint32_t i = 0; i < n_in; ++i) {
			if (sizes[i] != r.config().input_bytes[i]) {
				respond_error(fd, -EINVAL, "input " + std::to_string(i) + " must be " +
				                               std::to_string(r.config().input_bytes[i]) + " bytes");
				return false;
			}
			// Read straight into the pinned input BO: no intermediate copy.
			if (!read_full(fd, r.input(i).data(), sizes[i])) return false;
		}
		try {
			auto out = r.run(std::chrono::milliseconds(o.timeout_ms));
			if (!respond(fd, 0, out)) return false;
		} catch (const std::exception& e) {
			if (!respond_error(fd, -EIO, e.what())) return false;
		}
	}
}

int serve(xdna::XdnaRunner& r, const Options& o) {
	const int ls = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (ls < 0) { perror("socket"); return 1; }
	sockaddr_un addr{};
	addr.sun_family = AF_UNIX;
	if (o.socket_path.size() >= sizeof addr.sun_path) { std::fprintf(stderr, "socket path too long\n"); return 1; }
	std::strncpy(addr.sun_path, o.socket_path.c_str(), sizeof addr.sun_path - 1);
	::unlink(o.socket_path.c_str());
	if (::bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) { perror("bind"); return 1; }
	::chmod(o.socket_path.c_str(), 0660);
	if (::listen(ls, 8) < 0) { perror("listen"); return 1; }
	std::printf("xdna_daemon: kernel %s, %zu input(s), %zu-byte output, listening on %s\n",
	            r.config().kernel_name.c_str(), r.num_inputs(), r.config().output_bytes, o.socket_path.c_str());
	std::fflush(stdout);

	while (!g_stop) {
		pollfd p{ls, POLLIN, 0};
		const int pr = ::poll(&p, 1, 500);
		if (pr <= 0) continue;
		const int c = ::accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
		if (c < 0) continue;
		// The NPU context is a single serial resource; handle clients one at a time.
		serve_client(c, r, o);
		::close(c);
	}
	::close(ls);
	::unlink(o.socket_path.c_str());
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	Options o = parse(argc, argv);
	struct sigaction sa{};
	sa.sa_handler = [](int) { g_stop = true; };
	sigaction(SIGINT, &sa, nullptr);
	sigaction(SIGTERM, &sa, nullptr);

	try {
		const auto t0 = clk::now();
		xdna::XdnaRunner runner(o.rc);
		std::printf("xdna_daemon: context ready in %.1f ms\n",
		            std::chrono::duration<double, std::milli>(clk::now() - t0).count());
		return o.benchmark ? benchmark(runner, o) : serve(runner, o);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "xdna_daemon: %s\n", e.what());
		return 1;
	}
}
