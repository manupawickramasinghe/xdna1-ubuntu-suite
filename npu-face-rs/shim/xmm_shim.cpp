// SPDX-License-Identifier: MIT
//
// C ABI for one fixed-shape bf16 x bf16 -> f32 matmul tile kernel (mlir-aie
// whole_array: args opcode, instr, n_instr, A, B, C) with:
//   * a pool of B buffer objects, filled and synced once (static weights),
//   * two A/C slots and two xrt::run objects, so the host can prepare the next
//     tile while the NPU computes the current one (double buffering).

#include "xdna_runner.hpp"  // load_instructions()

#include <cstdio>
#include <exception>
#include <memory>
#include <vector>

namespace {

struct Xmm {
	unsigned m, k, n;
	xrt::device device;
	xrt::xclbin xclbin;
	xrt::hw_context ctx;
	xrt::kernel kernel;
	xrt::bo instr;
	uint32_t instr_words = 0;
	xrt::bo a[2], c[2];
	std::vector<xrt::bo> b_pool;
	xrt::run run[2];
	bool busy[2] = {false, false};
};

template <typename F>
bool guarded(char* err, unsigned len, F&& f) {
	try {
		f();
		return true;
	} catch (const std::exception& e) {
		std::snprintf(err, len, "%s", e.what());
		return false;
	}
}

}  // namespace

extern "C" {

void* xmm_open(const char* xclbin, const char* insts, unsigned m, unsigned k, unsigned n, char* err, unsigned len) {
	Xmm* h = nullptr;
	bool ok = guarded(err, len, [&] {
		auto x = std::make_unique<Xmm>();
		x->m = m;
		x->k = k;
		x->n = n;
		x->device = xrt::device(0u);
		x->xclbin = xrt::xclbin(std::string(xclbin));
		x->device.register_xclbin(x->xclbin);
		x->ctx = xrt::hw_context(x->device, x->xclbin.get_uuid());
		x->kernel = xrt::kernel(x->ctx, x->xclbin.get_kernels().front().get_name());
		auto words = xdna::load_instructions(insts);
		x->instr_words = static_cast<uint32_t>(words.size());
		x->instr = xrt::bo(x->device, words.size() * 4, xrt::bo::flags::cacheable, x->kernel.group_id(1));
		x->instr.write(words.data());
		x->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		for (int s = 0; s < 2; ++s) {
			x->a[s] = xrt::bo(x->device, size_t(m) * k * 2, xrt::bo::flags::host_only, x->kernel.group_id(3));
			x->c[s] = xrt::bo(x->device, size_t(m) * n * 4, xrt::bo::flags::host_only, x->kernel.group_id(5));
			x->run[s] = xrt::run(x->kernel);
			x->run[s].set_arg(0, 3u);
			x->run[s].set_arg(1, x->instr);
			x->run[s].set_arg(2, x->instr_words);
			x->run[s].set_arg(3, x->a[s]);
			x->run[s].set_arg(5, x->c[s]);
		}
		h = x.release();
	});
	return ok ? h : nullptr;
}

// New B buffer (k x n bf16); returns its id, or -1. Fill via xmm_b_ptr, then xmm_b_sync.
int xmm_b_new(void* hv, char* err, unsigned len) {
	auto* h = static_cast<Xmm*>(hv);
	int id = -1;
	guarded(err, len, [&] {
		h->b_pool.emplace_back(h->device, size_t(h->k) * h->n * 2, xrt::bo::flags::host_only, h->kernel.group_id(4));
		id = static_cast<int>(h->b_pool.size()) - 1;
	});
	return id;
}
uint16_t* xmm_b_ptr(void* hv, int id) { return static_cast<Xmm*>(hv)->b_pool.at(id).map<uint16_t*>(); }
int xmm_b_sync(void* hv, int id, char* err, unsigned len) {
	auto* h = static_cast<Xmm*>(hv);
	return guarded(err, len, [&] { h->b_pool.at(id).sync(XCL_BO_SYNC_BO_TO_DEVICE); }) ? 0 : -1;
}

uint16_t* xmm_a_ptr(void* hv, int slot) { return static_cast<Xmm*>(hv)->a[slot].map<uint16_t*>(); }

// Start C[slot] = A[slot] x B[b_id] asynchronously. sync_a: flush host writes
// to A[slot] first (skip when A[slot] is unchanged since its last run).
int xmm_start(void* hv, int slot, int b_id, int sync_a, char* err, unsigned len) {
	auto* h = static_cast<Xmm*>(hv);
	return guarded(err, len, [&] {
		if (sync_a) h->a[slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
		h->run[slot].set_arg(4, h->b_pool.at(b_id));
		h->run[slot].start();
		h->busy[slot] = true;
	}) ? 0 : -1;
}

// Wait for slot's run; returns C (valid until the slot is started again) or null.
const float* xmm_wait(void* hv, int slot, unsigned timeout_ms, char* err, unsigned len) {
	auto* h = static_cast<Xmm*>(hv);
	const float* out = nullptr;
	guarded(err, len, [&] {
		h->busy[slot] = false;
		if (h->run[slot].wait2(std::chrono::milliseconds(timeout_ms)) == std::cv_status::timeout) {
			h->run[slot].abort();
			throw std::runtime_error("NPU tile timed out");
		}
		h->c[slot].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		out = h->c[slot].map<const float*>();
	});
	return out;
}

void xmm_close(void* hv) {
	auto* h = static_cast<Xmm*>(hv);
	for (int s = 0; s < 2; ++s)
		if (h->busy[s]) {
			try { h->run[s].wait2(std::chrono::milliseconds(1000)); } catch (...) {}
		}
	delete h;
}

}  // extern "C"
