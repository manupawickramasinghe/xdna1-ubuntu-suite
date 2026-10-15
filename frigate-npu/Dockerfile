# Adds the AMD XDNA NPU face-embedding runner to an existing Frigate image
# without changing anything else. Same code as Zektopic/frigate feat/npu-face-runner.
ARG BASE=frigate-ncnn-vulkan:latest
FROM ${BASE}

COPY npu_service_runner.py /opt/frigate/frigate/detectors/npu_service_runner.py
COPY patch_runners.py patch_stats.py /tmp/
RUN python3 /tmp/patch_runners.py /opt/frigate/frigate/detectors/detection_runners.py \
 && python3 /tmp/patch_stats.py /opt/frigate/frigate/stats/util.py \
 && rm /tmp/patch_runners.py /tmp/patch_stats.py \
 && find /opt/frigate -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null; \
    python3 -c "import ast; [ast.parse(open(f).read()) for f in ('/opt/frigate/frigate/detectors/detection_runners.py', '/opt/frigate/frigate/stats/util.py')]"
