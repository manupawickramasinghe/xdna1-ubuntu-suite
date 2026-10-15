# Frigate face recognition on the XDNA NPU

```text
Frigate container (host network)                     host
  face_recognition: {model_size: large, device: npu}
  ArcfaceEmbedding → get_optimized_runner()
     └─ NpuServiceRunner ── TCP 127.0.0.1:5800 ──▶ npu-face (Rust, systemd)
                                                     ArcFace IResNet-100, 104 conv/FC as
                                                     bf16 matmuls on /dev/accel/accel0 (XRT)
```

## NPU usage on the System page

`npu-face` also serves `GET http://127.0.0.1:5801/stats` (inference port + 1): NPU busy % since the
last poll, embeddings/s, mean latency, H-clock, NPU buffer memory. Frigate's `set_npu_usages()`
(patched by `patch_stats.py`) reports it as `npu_usages["amd-xdna"]`, which the existing **System →
General → NPU usage** chart renders. No web UI change is needed.

## Deploy (as done on the 8600G, 2026-10-15)

1. `npu-face-rs`: `CC=gcc-14 CXX=g++-14 cargo build --release`
2. Copy `arcface.onnx` into `/mnt/docker/frigate/config/model_cache/facedet/`, or let Frigate download it.
3. `sudo install -m644 npu-face.service /etc/systemd/system/ && sudo systemctl enable --now npu-face`.
   The service validates every NPU layer at startup and won't serve if any layer is wrong.
4. `docker build -t frigate-ncnn-vulkan-npu:latest .` (FROM the current image; adds the runner and the stats hook).
5. In `docker-compose.yml`, set `image: frigate-ncnn-vulkan-npu:latest`. In `config.yml`:
   ```yaml
   face_recognition:
     enabled: true
     model_size: large
     device: npu        # or npu://host:port
   ```
6. `docker compose up -d frigate`. The log should show `Using NPU service at 127.0.0.1:5800`.

## Rollback

```bash
cd /mnt/docker/frigate
cp config/config.yml.bak-npu-<ts> config/config.yml
cp docker-compose.yml.bak-npu-<ts> docker-compose.yml
docker compose up -d frigate
sudo systemctl disable --now npu-face
```

If the service is down, Frigate keeps running; the runner logs `NPU service unavailable` and face
recognition returns no result.
