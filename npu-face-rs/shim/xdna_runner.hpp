// SPDX-License-Identifier: MIT
//
// xdna_runner.hpp — keep an XRT hardware context + pre-allocated buffers alive
// and dispatch mlir-aie style NPU kernels with minimal per-call overhead.
//
// Kernel ABI (mlir-aie / IRON, see runtime_lib/test_lib/xrt_test_wrapper.h):
//   arg 0  opcode (3 = run DPU instruction stream)
//   arg 1  instruction BO           group_id(1), cacheable
//   arg 2  instruction word count
//   arg 3+ data BOs in the design's runtime-sequence order   group_id(3 + i), host_only
//
// The output's position among the data BOs depends on the design: newer IRON
// designs built with `transform` take (in, out, *params), e.g. mlir-aie
// vector_scalar_mul is (A, C, factor) → out_pos = 1. Default: after all inputs.
//
// Buffers are allocated by XRT (page-aligned, pinned through the amdxdna
// driver), so callers never need aligned_alloc themselves.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

namespace xdna {

struct RunnerConfig {
	std::string xclbin_path;
	std::string insts_path;            // insts.bin (raw uint32 words) or insts.txt (hex per line)
	std::string kernel_name;           // empty: first kernel in the xclbin (mlir-aie uses "MLIR_AIE")
	std::vector<size_t> input_bytes;   // one entry per input BO
	size_t output_bytes = 0;
	unsigned device_index = 0;
	unsigned opcode = 3;
	int out_pos = -1;                  // index of the output among data BOs; -1 = last
};

//* Load an mlir-aie instruction stream. Binary files are raw little-endian
//* uint32 words; .txt files have one hex word per line.
inline std::vector<uint32_t> load_instructions(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("cannot open instruction file " + path);
	std::vector<uint32_t> words;
	if (path.ends_with(".txt")) {
		std::string line;
		while (std::getline(f, line)) {
			if (line.empty() || line[0] == '#') continue;
			words.push_back(static_cast<uint32_t>(std::stoul(line, nullptr, 16)));
		}
	} else {
		f.seekg(0, std::ios::end);
		const auto n = static_cast<size_t>(f.tellg());
		if (n % 4) throw std::runtime_error(path + ": size is not a multiple of 4");
		f.seekg(0);
		words.resize(n / 4);
		f.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(n));
	}
	if (words.empty()) throw std::runtime_error(path + ": empty instruction stream");
	return words;
}

class XdnaRunner {
public:
	explicit XdnaRunner(RunnerConfig cfg) : cfg_(std::move(cfg)) {
		if (cfg_.output_bytes == 0) throw std::invalid_argument("output_bytes must be > 0");

		device_ = xrt::device(cfg_.device_index);
		xclbin_ = xrt::xclbin(cfg_.xclbin_path);
		if (cfg_.kernel_name.empty()) {
			auto ks = xclbin_.get_kernels();
			if (ks.empty()) throw std::runtime_error(cfg_.xclbin_path + ": no kernels in xclbin");
			cfg_.kernel_name = ks.front().get_name();
		}
		device_.register_xclbin(xclbin_);
		ctx_ = xrt::hw_context(device_, xclbin_.get_uuid());
		kernel_ = xrt::kernel(ctx_, cfg_.kernel_name);

		const auto instr = load_instructions(cfg_.insts_path);
		instr_words_ = instr.size();
		bo_instr_ = xrt::bo(device_, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, kernel_.group_id(1));
		bo_instr_.write(instr.data());
		bo_instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

		const size_t n_data = cfg_.input_bytes.size() + 1;
		if (cfg_.out_pos < 0) cfg_.out_pos = static_cast<int>(n_data - 1);
		if (static_cast<size_t>(cfg_.out_pos) >= n_data) throw std::invalid_argument("out_pos out of range");
		out_arg_ = 3 + cfg_.out_pos;
		bo_out_ = xrt::bo(device_, cfg_.output_bytes, xrt::bo::flags::host_only, kernel_.group_id(out_arg_));
		for (size_t i = 0, slot = 0; i < cfg_.input_bytes.size(); ++i, ++slot) {
			if (static_cast<int>(slot) == cfg_.out_pos) ++slot;
			in_args_.push_back(static_cast<int>(3 + slot));
			bo_in_.emplace_back(device_, cfg_.input_bytes[i], xrt::bo::flags::host_only, kernel_.group_id(in_args_.back()));
			std::memset(bo_in_.back().map<uint8_t*>(), 0, cfg_.input_bytes[i]);
		}

		run_ = xrt::run(kernel_);
		run_.set_arg(0, cfg_.opcode);
		run_.set_arg(1, bo_instr_);
		run_.set_arg(2, static_cast<uint32_t>(instr_words_));
		for (size_t i = 0; i < bo_in_.size(); ++i) run_.set_arg(in_args_[i], bo_in_[i]);
		run_.set_arg(out_arg_, bo_out_);
	}

	const RunnerConfig& config() const { return cfg_; }
	size_t num_inputs() const { return bo_in_.size(); }

	//* Writable host view of input i (write into it, then call run()).
	std::span<uint8_t> input(size_t i) { return {bo_in_.at(i).map<uint8_t*>(), cfg_.input_bytes.at(i)}; }
	std::span<const uint8_t> output() const {
		return {const_cast<xrt::bo&>(bo_out_).map<const uint8_t*>(), cfg_.output_bytes};
	}

	//* Sync inputs, execute, sync output. Throws on error or timeout.
	std::span<const uint8_t> run(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
		for (auto& b : bo_in_) b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// Flush any CPU writes to the output BO too: otherwise dirty cache lines
		// can be written back over the NPU's results after it finishes.
		bo_out_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		run_.start();
		if (run_.wait2(timeout) == std::cv_status::timeout) {
			run_.abort();
			throw std::runtime_error("NPU run timed out after " + std::to_string(timeout.count()) + " ms");
		}
		bo_out_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		return output();
	}

	//* Copy inputs in, run, return the output view (valid until the next call).
	std::span<const uint8_t> infer(std::span<const std::span<const uint8_t>> inputs,
	                               std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
		if (inputs.size() != bo_in_.size())
			throw std::invalid_argument("expected " + std::to_string(bo_in_.size()) + " inputs, got " +
			                            std::to_string(inputs.size()));
		for (size_t i = 0; i < inputs.size(); ++i) {
			if (inputs[i].size() != cfg_.input_bytes[i])
				throw std::invalid_argument("input " + std::to_string(i) + ": expected " +
				                            std::to_string(cfg_.input_bytes[i]) + " bytes, got " +
				                            std::to_string(inputs[i].size()));
			std::memcpy(bo_in_[i].map<uint8_t*>(), inputs[i].data(), inputs[i].size());
		}
		return run(timeout);
	}

private:
	RunnerConfig cfg_;
	xrt::device device_;
	xrt::xclbin xclbin_;
	xrt::hw_context ctx_;
	xrt::kernel kernel_;
	xrt::bo bo_instr_;
	std::vector<xrt::bo> bo_in_;
	xrt::bo bo_out_;
	xrt::run run_;
	size_t instr_words_ = 0;
	int out_arg_ = 3;
	std::vector<int> in_args_;
};

} // namespace xdna
