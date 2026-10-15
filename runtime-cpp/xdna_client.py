#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal client for xdna_daemon's Unix-socket protocol.

    from xdna_client import XdnaClient
    with XdnaClient() as npu:
        out = npu.infer([frame_bytes, scale_bytes])

CLI smoke test (sends zero-filled inputs):
    python3 xdna_client.py --in 4096,4
"""
import argparse
import os
import socket
import struct
import time

REQ_MAGIC = 0x514E4458  # "XDNQ"
RSP_MAGIC = 0x524E4458  # "XDNR"


def default_socket() -> str:
    if os.geteuid() == 0:
        return "/run/xdna_infer.sock"
    rt = os.environ.get("XDG_RUNTIME_DIR")
    return f"{rt}/xdna_infer.sock" if rt else f"/tmp/xdna_infer-{os.getuid()}.sock"


class XdnaError(RuntimeError):
    pass


class XdnaClient:
    def __init__(self, path: str | None = None):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path or default_socket())

    def _recv(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("daemon closed the connection")
            buf += chunk
        return bytes(buf)

    def infer(self, inputs: list[bytes]) -> bytes:
        hdr = struct.pack("<II", REQ_MAGIC, len(inputs)) + b"".join(struct.pack("<Q", len(b)) for b in inputs)
        self.sock.sendall(hdr + b"".join(inputs))
        magic, status, n = struct.unpack("<IiQ", self._recv(16))
        if magic != RSP_MAGIC:
            raise XdnaError(f"bad response magic {magic:#x}")
        payload = self._recv(n)
        if status != 0:
            raise XdnaError(f"daemon error {status}: {payload.decode(errors='replace')}")
        return payload

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=None)
    ap.add_argument("--in", dest="sizes", required=True, help="comma-separated input sizes in bytes")
    ap.add_argument("-n", type=int, default=100, help="number of requests")
    a = ap.parse_args()
    ins = [bytes(int(s)) for s in a.sizes.split(",")]
    with XdnaClient(a.socket) as c:
        t0 = time.perf_counter()
        for _ in range(a.n):
            out = c.infer(ins)
        dt = (time.perf_counter() - t0) / a.n
    print(f"{a.n} requests, {dt * 1e6:.0f} us/request round trip, output {len(out)} bytes")
