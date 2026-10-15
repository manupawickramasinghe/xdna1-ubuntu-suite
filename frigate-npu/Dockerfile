# Adds the AMD XDNA NPU face-embedding runner to an existing Frigate image
# without changing anything else. Same code as Zektopic/frigate feat/npu-face-runner.
ARG BASE=frigate-ncnn-vulkan:latest
FROM ${BASE}

COPY npu_service_runner.py /opt/frigate/frigate/detectors/npu_service_runner.py
COPY patch_runners.py /tmp/patch_runners.py
RUN python3 /tmp/patch_runners.py /opt/frigate/frigate/detectors/detection_runners.py \
 && rm /tmp/patch_runners.py \
 && find /opt/frigate -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null; \
    python3 -c "import ast,sys; ast.parse(open('/opt/frigate/frigate/detectors/detection_runners.py').read())"
