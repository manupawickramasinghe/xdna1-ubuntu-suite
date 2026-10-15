"""Report the npu-face service's NPU usage in set_npu_usages() (idempotent)."""
import sys

path = sys.argv[1]
src = open(path).read()
if "get_npu_service_stats" in src:
    sys.exit(0)
anchor = '    if stats:\n        all_stats["npu_usages"] = stats'
if src.count(anchor) != 1 or "import asyncio" not in src:
    sys.exit("set_npu_usages anchor not found; refusing to patch")
hook = '''    # AMD XDNA NPU used through the npu-face service (face recognition)
    from frigate.detectors.npu_service_runner import (
        get_npu_service_stats,
        parse_npu_device,
    )

    endpoint = (
        parse_npu_device(config.face_recognition.device)
        if config.face_recognition.enabled
        else None
    )
    if endpoint is not None:
        xdna_usage = await asyncio.to_thread(get_npu_service_stats, endpoint)
        if xdna_usage:
            stats["amd-xdna"] = xdna_usage

'''
open(path, "w").write(src.replace(anchor, hook + anchor))
print("patched", path)
