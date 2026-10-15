#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run an ONNX CNN with its heavy layers on the AMD XDNA NPU.

IREE-AMD-AIE can't lower convolutions for npu1 yet, but it can run bf16 matmuls.
This engine interprets the ONNX graph on the host and lowers every regular
Conv (via im2col) and Gemm/MatMul to a padded bf16 matmul on the NPU:

  NPU : Conv (group == 1), Gemm, MatMul       → linalg.matmul bf16×bf16→f32
  CPU : depthwise Conv, BatchNorm (folded into the preceding Conv when possible),
        PRelu/Relu/Add/Mul/Sub/Flatten/Reshape/... (cheap elementwise ops)

Matmul shapes are padded to the AIE instruction grid (M, N multiples of 4,
K a multiple of 8). Each padded shape is compiled once with iree-compile and
cached as a .vmfb.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import numpy_helper

IREE_BUILD = Path(os.environ.get("IREE_AIE_BUILD", Path.home() / ".cache/iree-aie/build"))
PEANO = Path(
    os.environ.get(
        "PEANO_INSTALL_DIR",
        Path.home() / ".cache/mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie",
    )
)
CACHE = Path(os.environ.get("NPU_CNN_CACHE", Path.home() / ".cache/npu-cnn"))

BF16 = ml_dtypes.bfloat16


def _ceil(x: int, m: int) -> int:
    return (x + m - 1) // m * m


MATMUL_MLIR = """func.func @mm(%a: tensor<{M}x{K}xbf16>, %b: tensor<{K}x{N}xbf16>) -> tensor<{M}x{N}xf32> {{
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<{M}x{N}xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %r = linalg.matmul ins(%a, %b : tensor<{M}x{K}xbf16>, tensor<{K}x{N}xbf16>) outs(%f : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  return %r : tensor<{M}x{N}xf32>
}}
"""


class NpuMatmul:
    """Compiles and runs padded bf16 matmuls on the NPU through IREE."""

    # Larger M (e.g. 12544 for a 112x112 feature map) compiles but returns wrong
    # results on npu1 with this IREE-AMD-AIE build; 3136 and below verify exactly.
    MAX_M = 3136
    VERIFY_RUNS = 10

    def __init__(self, device: str = "amdxdna", stack_size: int = 4096) -> None:
        import iree.runtime as rt

        self.rt = rt
        self.device_name = device
        self.stack_size = stack_size
        self.config = rt.Config(device)
        self.fns: dict[tuple[int, int, int], object] = {}
        # The function handle doesn't keep its SystemContext alive; hold them here
        # or the module is freed and the next call segfaults.
        self._contexts: list[object] = []
        # shape -> True if the NPU result matched numpy on random data
        self.verified: dict[tuple[int, int, int], bool] = {}
        self.stats = defaultdict(lambda: [0, 0.0])  # shape -> [calls, seconds]
        CACHE.mkdir(parents=True, exist_ok=True)

    def _vmfb(self, M: int, K: int, N: int) -> Path:
        key = hashlib.sha1(f"{M}x{K}x{N}-{self.device_name}-{self.stack_size}".encode()).hexdigest()[:12]
        out = CACHE / f"mm_{M}x{K}x{N}_{key}.vmfb"
        if out.exists():
            return out
        src = CACHE / f"mm_{M}x{K}x{N}.mlir"
        src.write_text(MATMUL_MLIR.format(M=M, K=K, N=N))
        if self.device_name == "amdxdna":
            target = [
                "--iree-hal-target-backends=amd-aie",
                "--iree-amdaie-target-device=npu1_4col",
                f"--iree-amd-aie-peano-install-dir={PEANO}",
                f"--iree-amd-aie-install-dir={IREE_BUILD}",
                "--iree-amdaie-device-hal=amdxdna",
                f"--iree-amdaie-stack-size={self.stack_size}",
                "--iree-scheduling-optimize-bindings=false",
                "--iree-hal-memoization=false",
                "--iree-hal-indirect-command-buffers=false",
            ]
        else:  # CPU reference build of the exact same program
            target = ["--iree-hal-target-backends=llvm-cpu", "--iree-llvmcpu-target-cpu=host"]
        cmd = [str(IREE_BUILD / "tools/iree-compile"), str(src), *target, "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"iree-compile failed for {M}x{K}x{N}:\n{r.stderr[-2000:]}")
        return out

    def _fn(self, M: int, K: int, N: int):
        key = (M, K, N)
        if key not in self.fns:
            ctx = self.rt.SystemContext(config=self.config)
            ctx.add_vm_module(self.rt.VmModule.mmap(self.config.vm_instance, str(self._vmfb(M, K, N))))
            self._contexts.append(ctx)
            self.fns[key] = ctx.modules.module["mm"]
        return self.fns[key]

    def _run(self, M: int, K: int, N: int, ap: np.ndarray, bp: np.ndarray) -> np.ndarray:
        res = self._fn(M, K, N)(ap, bp)
        # np.asarray() is a view into the mapped device buffer, which is unmapped
        # when `res` is freed: copy the result out before dropping it.
        out = np.array(res.to_host(), copy=True)
        del res
        return out

    def _verify(self, M: int, K: int, N: int) -> bool:
        """Check a newly compiled shape against numpy before trusting it."""
        rng = np.random.default_rng(M * 131 + K * 17 + N)
        ok = True
        # Some shapes fail intermittently, so one passing run proves little.
        for _ in range(self.VERIFY_RUNS):
            a = rng.standard_normal((M, K)).astype(BF16)
            b = rng.standard_normal((K, N)).astype(BF16)
            ref = a.astype(np.float32) @ b.astype(np.float32)
            out = self._run(M, K, N, a, b)
            tol = 1e-2 * max(1.0, float(np.abs(ref).max()))
            ok = ok and bool(np.abs(out - ref).max() <= tol)
        if not ok:
            print(f"npu_cnn: matmul {M}x{K}x{N} failed NPU verification, using CPU")
        return ok

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a[m,k] @ b[k,n] in bf16 with f32 accumulation."""
        m, k = a.shape
        if m > self.MAX_M:
            return np.concatenate(
                [self(a[i : i + self.MAX_M], b) for i in range(0, m, self.MAX_M)], 0
            )
        n = b.shape[1]
        # K < 64 (e.g. 32) is intermittently wrong on npu1 (35-65% of calls in a
        # 20-run repeat test), so pad K up to at least 64.
        M, K, N = _ceil(max(m, 4), 4), max(64, _ceil(k, 8)), _ceil(max(n, 4), 4)
        key = (M, K, N)
        if key not in self.verified:
            self.verified[key] = self._verify(M, K, N)
        if not self.verified[key]:
            return a.astype(np.float32) @ b.astype(np.float32)
        ap = np.zeros((M, K), BF16)
        ap[:m, :k] = a
        bp = np.zeros((K, N), BF16)
        bp[:k, :n] = b
        t0 = time.perf_counter()
        out = self._run(M, K, N, ap, bp)
        s = self.stats[key]
        s[0] += 1
        s[1] += time.perf_counter() - t0
        return out[:m, :n]


def im2col(x: np.ndarray, kh: int, kw: int, sh: int, sw: int, pads, dh: int, dw: int):
    """x: [C,H,W] → cols [OH*OW, C*kh*kw], OH, OW."""
    c, h, w = x.shape
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)))
    oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
    cols = np.empty((c, kh, kw, oh, ow), x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, i, j] = xp[:, i * dh : i * dh + sh * oh : sh, j * dw : j * dw + sw * ow : sw]
    return cols.reshape(c * kh * kw, oh * ow).T, oh, ow


class OnnxNpuModel:
    """Minimal ONNX interpreter for face-embedding CNNs (batch size 1)."""

    def __init__(self, path: str, matmul: NpuMatmul | None = None) -> None:
        self.model = onnx.load(path)
        g = self.model.graph
        self.w = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in g.initializer}
        for n in g.node:
            if n.op_type == "Constant":
                self.w[n.output[0]] = numpy_helper.to_array(n.attribute[0].t).astype(np.float32)
        self.input = next(i.name for i in g.input if i.name not in self.w)
        self.output = g.output[0].name
        self.nodes = [n for n in g.node if n.op_type != "Constant"]
        self.mm = matmul or NpuMatmul()
        self.npu_ops = 0
        self._fold_batchnorm()
        self._prepare_weights()

    @staticmethod
    def _attrs(n):
        return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

    def _fold_batchnorm(self) -> None:
        """Fold BatchNormalization into the Conv that feeds it (inference only)."""
        consumers = defaultdict(list)
        for n in self.nodes:
            for i in n.input:
                consumers[i].append(n)
        folded = set()
        for n in self.nodes:
            if n.op_type != "BatchNormalization":
                continue
            prod = next((p for p in self.nodes if n.input[0] in p.output), None)
            if prod is None or prod.op_type != "Conv" or len(consumers[prod.output[0]]) != 1:
                continue
            scale, bias, mean, var = (self.w[x] for x in n.input[1:5])
            eps = self._attrs(n).get("epsilon", 1e-5)
            k = scale / np.sqrt(var + eps)
            wname = prod.input[1]
            self.w[wname] = self.w[wname] * k[:, None, None, None]
            b = self.w[prod.input[2]] if len(prod.input) > 2 else np.zeros_like(k)
            bname = prod.output[0] + "__folded_bias"
            self.w[bname] = (b - mean) * k + bias
            del prod.input[2:]
            prod.input.append(bname)
            prod.output[0] = n.output[0]
            folded.add(id(n))
        self.nodes = [n for n in self.nodes if id(n) not in folded]

    def _prepare_weights(self) -> None:
        """Pre-transpose + bf16-cast the matmul weight matrices once."""
        self.wmat = {}
        for n in self.nodes:
            a = self._attrs(n)
            if n.op_type == "Conv" and a.get("group", 1) == 1:
                w = self.w[n.input[1]]
                self.wmat[n.name or n.output[0]] = w.reshape(w.shape[0], -1).T.astype(BF16)
            elif n.op_type == "Gemm":
                w = self.w[n.input[1]]
                w = w.T if a.get("transB", 0) else w
                self.wmat[n.name or n.output[0]] = (w * a.get("alpha", 1.0)).astype(BF16)

    def _conv(self, n, x: np.ndarray) -> np.ndarray:
        a = self._attrs(n)
        w = self.w[n.input[1]]
        b = self.w[n.input[2]] if len(n.input) > 2 else None
        group = a.get("group", 1)
        kh, kw = w.shape[2:]
        sh, sw = a.get("strides", [1, 1])
        dh, dw = a.get("dilations", [1, 1])
        p = a.get("pads", [0, 0, 0, 0])
        pads = (p[0], p[1], p[2], p[3])
        x = x[0]
        if group == 1:
            cols, oh, ow = im2col(x, kh, kw, sh, sw, pads, dh, dw)
            y = self.mm(cols.astype(BF16), self.wmat[n.name or n.output[0]])  # [OH*OW, OC]
            self.npu_ops += 1
            y = y.T.reshape(w.shape[0], oh, ow)
        elif group == x.shape[0] == w.shape[0]:  # depthwise: cheap, CPU in f32
            xp = np.pad(x, ((0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
            oh = (x.shape[1] + pads[0] + pads[2] - dh * (kh - 1) - 1) // sh + 1
            ow = (x.shape[2] + pads[1] + pads[3] - dw * (kw - 1) - 1) // sw + 1
            y = np.zeros((x.shape[0], oh, ow), np.float32)
            for i in range(kh):
                for j in range(kw):
                    y += w[:, 0, i, j][:, None, None] * xp[
                        :, i * dh : i * dh + sh * oh : sh, j * dw : j * dw + sw * ow : sw
                    ]
        else:  # other grouped convs: CPU in f32
            cg = x.shape[0] // group
            outs = []
            for g in range(group):
                cols, oh, ow = im2col(x[g * cg : (g + 1) * cg], kh, kw, sh, sw, pads, dh, dw)
                wg = w[g * (w.shape[0] // group) : (g + 1) * (w.shape[0] // group)]
                outs.append((cols @ wg.reshape(wg.shape[0], -1).T).T.reshape(-1, oh, ow))
            y = np.concatenate(outs, 0)
        if b is not None:
            y = y + b[:, None, None]
        return y[None].astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        env = dict(self.w)
        env[self.input] = x.astype(np.float32)
        for n in self.nodes:
            i = [env[k] if k else None for k in n.input]
            a = self._attrs(n)
            op = n.op_type
            if op == "Conv":
                y = self._conv(n, i[0])
            elif op == "Gemm":
                y = self.mm(i[0].astype(BF16), self.wmat[n.name or n.output[0]])
                self.npu_ops += 1
                if len(i) > 2 and i[2] is not None:
                    y = y + a.get("beta", 1.0) * i[2]
            elif op == "MatMul":
                y = self.mm(i[0].reshape(-1, i[0].shape[-1]).astype(BF16), i[1].astype(BF16))
                self.npu_ops += 1
            elif op == "BatchNormalization":
                s, b, m, v = i[1:5]
                eps = a.get("epsilon", 1e-5)
                shape = (1, -1) + (1,) * (i[0].ndim - 2)
                y = (i[0] - m.reshape(shape)) / np.sqrt(v.reshape(shape) + eps) * s.reshape(shape) + b.reshape(shape)
            elif op == "PRelu":
                slope = i[1].reshape((1, -1) + (1,) * (i[0].ndim - 2)) if i[1].size == i[0].shape[1] else i[1]
                y = np.where(i[0] >= 0, i[0], i[0] * slope)
            elif op == "Relu":
                y = np.maximum(i[0], 0)
            elif op == "Add":
                y = i[0] + i[1]
            elif op == "Sub":
                y = i[0] - i[1]
            elif op == "Mul":
                y = i[0] * i[1]
            elif op == "Div":
                y = i[0] / i[1]
            elif op in ("Dropout", "Identity"):
                y = i[0]
            elif op == "Flatten":
                ax = a.get("axis", 1)
                y = i[0].reshape(int(np.prod(i[0].shape[:ax])), -1)
            elif op == "Reshape":
                y = i[0].reshape([int(d) for d in i[1]])
            elif op == "Transpose":
                y = np.transpose(i[0], a.get("perm"))
            elif op == "GlobalAveragePool":
                y = i[0].mean(axis=(2, 3), keepdims=True)
            else:
                raise NotImplementedError(f"ONNX op {op} not supported by npu_cnn")
            env[n.output[0]] = y.astype(np.float32)
        return env[self.output]
