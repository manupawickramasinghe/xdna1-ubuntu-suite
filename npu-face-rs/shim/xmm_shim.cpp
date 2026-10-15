// SPDX-License-Identifier: MIT
//
// C ABI over xdna::XdnaRunner for one fixed-shape bf16 x bf16 -> f32 matmul
// tile kernel (mlir-aie whole_array, data args A, B, C). Rust writes A and B
// straight into the pinned input BOs and reads C from the output BO.

#include "xdna_runner.hpp"

#include <cstdio>
#include <exception>
#include <new>

extern "C" {

struct xmm;

struct xmm* xmm_open(const char* xclbin, const char* insts, unsigned m, unsigned k, unsigned n,
                     char* err, unsigned err_len) {
	try {
		xdna::RunnerConfig rc;
		rc.xclbin_path = xclbin;
		rc.insts_path = insts;
		rc.input_bytes = {size_t(m) * k * 2, size_t(k) * n * 2};
		rc.output_bytes = size_t(m) * n * 4;
		rc.out_pos = 2;  // mlir-aie whole_array: (A, B, C)
		return reinterpret_cast<xmm*>(new xdna::XdnaRunner(rc));
	} catch (const std::exception& e) {
		std::snprintf(err, err_len, "%s", e.what());
		return nullptr;
	}
}

uint16_t* xmm_a(struct xmm* h) { return reinterpret_cast<uint16_t*>(reinterpret_cast<xdna::XdnaRunner*>(h)->input(0).data()); }
uint16_t* xmm_b(struct xmm* h) { return reinterpret_cast<uint16_t*>(reinterpret_cast<xdna::XdnaRunner*>(h)->input(1).data()); }

// Runs the kernel; returns a pointer to C (valid until the next run), or null.
const float* xmm_run(struct xmm* h, unsigned timeout_ms, char* err, unsigned err_len) {
	try {
		auto out = reinterpret_cast<xdna::XdnaRunner*>(h)->run(std::chrono::milliseconds(timeout_ms));
		return reinterpret_cast<const float*>(out.data());
	} catch (const std::exception& e) {
		std::snprintf(err, err_len, "%s", e.what());
		return nullptr;
	}
}

void xmm_close(struct xmm* h) { delete reinterpret_cast<xdna::XdnaRunner*>(h); }

}  // extern "C"
