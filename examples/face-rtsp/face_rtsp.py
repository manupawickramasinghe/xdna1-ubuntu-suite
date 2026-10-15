#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Face recognition on an RTSP stream: YuNet detection + SFace embeddings.

  face_rtsp.py enroll --name me --seconds 20      # average embeddings of the largest face
  face_rtsp.py run --seconds 60 [--embed iree]    # detect + recognise, log timings

Embedding backends:
  cpu   onnxruntime CPUExecutionProvider
  iree  IREE runtime module compiled for the NPU (see compile_sface_iree.sh)

Enrolment data (a 128-d embedding) is stored locally in ./enrolled/<name>.npy.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
ENROLLED = HERE / "enrolled"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"
COSINE_MATCH = 0.363  # SFace's recommended cosine threshold


class Embedder:
    """SFace on a 112x112 aligned face → L2-normalised 128-d vector."""

    def __init__(self, backend: str, vmfb: str | None, device: str, fn: str = "mxnet_converted_model"):
        # cv2's recognizer only does alignment here; inference runs on the chosen backend.
        self.align = cv2.FaceRecognizerSF.create(str(SFACE), "")
        self.backend = backend
        if backend == "cpu":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(SFACE), providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        elif backend == "iree":
            import iree.runtime as rt
            cfg = rt.Config(device)
            self.ctx = rt.SystemContext(config=cfg)
            self.ctx.add_vm_module(rt.VmModule.mmap(cfg.vm_instance, vmfb))
            self.fn = self.ctx.modules.module[fn]  # entry point named after the ONNX graph
        else:
            raise ValueError(backend)

    def __call__(self, frame, face) -> np.ndarray:
        crop = self.align.alignCrop(frame, face)                       # 112x112 BGR
        blob = cv2.dnn.blobFromImage(crop, 1.0, (112, 112), swapRB=True)  # 1x3x112x112 float32 RGB
        if self.backend == "cpu":
            v = self.sess.run(None, {self.inp: blob})[0]
        else:
            v = np.asarray(self.fn(blob))
        v = v.reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def open_stream(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"cannot open {url}")
    return cap


def detector_for(frame, score: float):
    return cv2.FaceDetectorYN.create(str(YUNET), "", (frame.shape[1], frame.shape[0]), score)


def cmd_enroll(a):
    cap = open_stream(a.url)
    emb = Embedder(a.embed, a.vmfb, a.device)
    ok, frame = cap.read()
    det = detector_for(frame, a.score)
    vecs, t_end = [], time.time() + a.seconds
    ENROLLED.mkdir(exist_ok=True)
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        _, faces = det.detect(frame)
        if faces is None:
            continue
        face = max(faces, key=lambda f: f[2] * f[3])  # largest face = the person being enrolled
        if face[2] < a.min_size:
            continue
        vecs.append(emb(frame, face))
        if len(vecs) == 1:
            x, y, w, h = map(int, face[:4])
            cv2.imwrite(str(ENROLLED / f"{a.name}_sample.jpg"), frame[max(0, y - h // 2):y + h + h // 2, max(0, x - w // 2):x + w + w // 2])
    if len(vecs) < 5:
        sys.exit(f"only {len(vecs)} usable face samples; face the camera and retry")
    m = np.mean(vecs, axis=0)
    m /= np.linalg.norm(m)
    np.save(ENROLLED / f"{a.name}.npy", m)
    spread = [float(v @ m) for v in vecs]
    print(f"enrolled '{a.name}' from {len(vecs)} samples; self-similarity min {min(spread):.3f} mean {np.mean(spread):.3f}")


def cmd_run(a):
    people = {p.stem: np.load(p) for p in ENROLLED.glob("*.npy")}
    if not people:
        sys.exit("nobody enrolled; run `enroll` first")
    cap = open_stream(a.url)
    emb = Embedder(a.embed, a.vmfb, a.device)
    ok, frame = cap.read()
    det = detector_for(frame, a.score)
    stats = {"frames": 0, "faces": 0, "matches": {}, "unknown": 0, "t_det_ms": [], "t_emb_ms": []}
    t_end, snap_t = time.time() + a.seconds, 0.0
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        stats["frames"] += 1
        t0 = time.perf_counter()
        _, faces = det.detect(frame)
        stats["t_det_ms"].append((time.perf_counter() - t0) * 1e3)
        for f in [] if faces is None else faces:
            if f[2] < a.min_size:
                continue
            t1 = time.perf_counter()
            v = emb(frame, f)
            stats["t_emb_ms"].append((time.perf_counter() - t1) * 1e3)
            name, best = max(((n, float(v @ e)) for n, e in people.items()), key=lambda x: x[1])
            label = name if best >= COSINE_MATCH else "unknown"
            stats["faces"] += 1
            if label == "unknown":
                stats["unknown"] += 1
            else:
                stats["matches"][label] = stats["matches"].get(label, 0) + 1
            x, y, w, h = map(int, f[:4])
            col = (0, 255, 0) if label != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
            cv2.putText(frame, f"{label} {best:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        if a.snapshot and time.time() - snap_t > 5:
            cv2.imwrite(a.snapshot, cv2.resize(frame, (960, 540)))
            snap_t = time.time()
    p = lambda xs, q: float(np.percentile(xs, q)) if xs else 0.0
    out = {
        "backend": a.embed,
        "frames": stats["frames"],
        "fps": stats["frames"] / a.seconds,
        "faces_embedded": stats["faces"],
        "matches": stats["matches"],
        "unknown": stats["unknown"],
        "detect_ms_p50": p(stats["t_det_ms"], 50),
        "embed_ms_p50": p(stats["t_emb_ms"], 50),
        "embed_ms_p99": p(stats["t_emb_ms"], 99),
    }
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("FACE_RTSP_URL", "rtsp://192.168.1.69:8554/stream"))
    ap.add_argument("--embed", choices=["cpu", "iree"], default="cpu")
    ap.add_argument("--vmfb", default=str(HERE / "build" / "sface_npu.vmfb"))
    ap.add_argument("--device", default="amdxdna", help="IREE HAL device (amdxdna = NPU, local-task = CPU)")
    ap.add_argument("--score", type=float, default=0.7)
    ap.add_argument("--min-size", type=int, default=40, help="ignore faces narrower than this (px)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("--name", required=True); e.add_argument("--seconds", type=float, default=20)
    r = sub.add_parser("run"); r.add_argument("--seconds", type=float, default=60); r.add_argument("--snapshot", default=None)
    a = ap.parse_args()
    {"enroll": cmd_enroll, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
