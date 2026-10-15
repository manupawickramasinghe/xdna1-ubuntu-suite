"""Model runner that offloads inference to an external NPU service.

Used for AMD XDNA (Ryzen AI) NPUs, which the Frigate container cannot drive
directly: the `npu-face` service (Rust, xdna1-ubuntu-suite) runs on the host
with access to /dev/accel/accel0, loads the same ONNX model file, and serves
embeddings over a small length-prefixed TCP protocol.

Select it with `device: npu` (default 127.0.0.1:5800) or `device: npu://host:port`.
"""

import json
import logging
import socket
import struct
import threading
from typing import Any

import numpy as np

from frigate.detectors.detection_runners import BaseModelRunner

logger = logging.getLogger(__name__)

REQ_MAGIC = b"XFQ1"
RSP_MAGIC = b"XFR1"
DEFAULT_ENDPOINT = ("127.0.0.1", 5800)


def parse_npu_device(device: str | None) -> tuple[str, int] | None:
    """Return the service endpoint if `device` selects the NPU service."""
    if not device:
        return None
    d = device.strip().lower()
    if d == "npu":
        return DEFAULT_ENDPOINT
    if d.startswith("npu://"):
        host, _, port = d[len("npu://") :].rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(
                f"invalid NPU endpoint {device!r}, expected npu://host:port"
            )
        return host, int(port)
    return None


def get_npu_service_stats(
    endpoint: tuple[str, int], timeout: float = 0.5
) -> dict[str, Any] | None:
    """Fetch NPU usage from the service's stats port (inference port + 1).

    Returns the shape the System page expects for `npu_usages` entries:
    {"npu": busy_percent, "mem": "<NPU buffer MiB> MiB", ...extra fields}.
    """
    host, port = endpoint
    try:
        with socket.create_connection((host, port + 1), timeout=timeout) as sock:
            sock.sendall(b"GET /stats HTTP/1.0\r\n\r\n")
            data = b""
            while chunk := sock.recv(4096):
                data += chunk
    except OSError:
        return None

    _, _, body = data.partition(b"\r\n\r\n")
    try:
        raw = json.loads(body)
    except ValueError:
        return None

    return {
        "npu": round(float(raw.get("npu", 0.0)), 1),
        "mem": f"{raw.get('bo_mem_mib', 0):.0f} MiB",
        "embeddings_per_s": raw.get("embeddings_per_s"),
        "mean_latency_ms": raw.get("mean_latency_ms"),
        "clock_mhz": raw.get("h_clock_mhz"),
    }


class NpuServiceRunner(BaseModelRunner):
    """Runs a single-input model on the npu-face service."""

    def __init__(
        self,
        model_path: str,
        device: str,
        endpoint: tuple[str, int],
        input_name: str = "data",
        input_width: int = 112,
        timeout: float = 5.0,
        **kwargs,
    ):
        super().__init__(model_path, device, **kwargs)
        self.endpoint = endpoint
        self.input_name = input_name
        self.input_width = input_width
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        logger.info(
            "Using NPU service at %s:%d for %s", endpoint[0], endpoint[1], model_path
        )

    def get_input_names(self) -> list[str]:
        return [self.input_name]

    def get_input_width(self) -> int:
        return self.input_width

    def _connect(self) -> socket.socket:
        if self._sock is None:
            sock = socket.create_connection(self.endpoint, timeout=self.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
        return self._sock

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("NPU service closed the connection")
            buf += chunk
        return bytes(buf)

    def _request(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"NPU service expects NCHW input, got shape {x.shape}")
        sock = self._connect()
        sock.sendall(REQ_MAGIC + struct.pack("<4I", *x.shape) + x.tobytes())
        magic, status, n, dim = struct.unpack("<4siII", self._recv_exact(sock, 16))
        if magic != RSP_MAGIC:
            raise ConnectionError("bad response from NPU service")
        payload = self._recv_exact(sock, dim if status else n * dim * 4)
        if status:
            raise RuntimeError(
                f"NPU service error {status}: {payload.decode(errors='replace')}"
            )
        return np.frombuffer(payload, dtype=np.float32).reshape(n, dim)

    def run(self, input: dict[str, Any]) -> Any | None:
        x = input[self.input_name]
        with self._lock:
            try:
                return [self._request(x)]
            except (OSError, ConnectionError):
                # one reconnect attempt, e.g. after the service restarted
                self._close()
                try:
                    return [self._request(x)]
                except (OSError, ConnectionError):
                    self._close()
                    logger.exception("NPU service unavailable at %s:%d", *self.endpoint)
                    return None
