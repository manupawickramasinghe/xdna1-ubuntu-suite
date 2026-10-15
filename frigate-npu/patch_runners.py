"""Insert the NPU service hook into get_optimized_runner() (idempotent)."""
import sys

path = sys.argv[1]
src = open(path).read()
if "parse_npu_device" in src:
    sys.exit(0)
anchor = '    """Get an optimized runner for the hardware."""\n    device = device or "AUTO"\n'
if src.count(anchor) != 1:
    sys.exit("get_optimized_runner anchor not found; refusing to patch")
hook = (
    "\n    # External NPU service (AMD XDNA); imported lazily to avoid a cycle\n"
    "    from frigate.detectors.npu_service_runner import NpuServiceRunner, parse_npu_device\n\n"
    "    if (npu_endpoint := parse_npu_device(device)) is not None:\n"
    "        return NpuServiceRunner(model_path, device, npu_endpoint, **kwargs)\n"
)
open(path, "w").write(src.replace(anchor, anchor + hook))
print("patched", path)
