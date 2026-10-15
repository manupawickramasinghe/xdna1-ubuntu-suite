#!/usr/bin/env bash
# Download YuNet + SFace (OpenCV zoo) and strip SFace's initializer-as-input entries,
# which otherwise turn the weights into runtime arguments for IREE.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
base=https://github.com/opencv/opencv_zoo/raw/main/models
curl -fL -o models/face_detection_yunet_2023mar.onnx "$base/face_detection_yunet/face_detection_yunet_2023mar.onnx"
curl -fL -o models/face_recognition_sface_2021dec.onnx "$base/face_recognition_sface/face_recognition_sface_2021dec.onnx"
python3 - <<'PY'
import onnx
p = "models/face_recognition_sface_2021dec.onnx"
m = onnx.load(p)
inits = {i.name for i in m.graph.initializer}
keep = [i for i in m.graph.input if i.name not in inits]
del m.graph.input[:]
m.graph.input.extend(keep)
onnx.checker.check_model(m)
onnx.save(m, p)
print("SFace inputs:", [i.name for i in m.graph.input])
PY
