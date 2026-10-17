"""
Export untrained B1-shaped models with a dynamic instrument count for latency benchmarking only
(no id embedding, random weights). Also checks ONNX Runtime CPU vs PyTorch parity for the trained model.

Usage: python -m trademaster_ml.export_bench --trained models/b1/v1 --out models/bench
"""
import argparse, json, os, struct

import numpy as np
import onnxruntime as ort
import torch

from .b1 import B1


def read_f32(path):
    with open(path, "rb") as f:
        nd = struct.unpack("<i", f.read(4))[0]
        shape = struct.unpack(f"<{nd}i", f.read(4 * nd))
        return np.frombuffer(f.read(), dtype="<f4").reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", default="private/models/b1/v1")
    ap.add_argument("--out", default="private/models/bench")
    a = ap.parse_args()

    x, m, ref = (read_f32(os.path.join(a.trained, f"parity_{n}.bin")) for n in ("x", "mask", "out"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(os.path.join(a.trained, "model.onnx"), so, providers=["CPUExecutionProvider"])
    got = s.run(None, {"x": x, "mask": m})[0]
    print(f"parity torch vs ORT-CPU (trained B1): max |diff| = {np.abs(got - ref).max():.2e}")

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = B1(1, 8, id_embedding=False).eval()
    x0, m0 = torch.randn(1, 30, 8), torch.ones(1, 30)
    path = os.path.join(a.out, "b1_dynamic_n.onnx")
    torch.onnx.export(model, (x0, m0), path, input_names=["x", "mask"], output_names=["out"],
                      dynamic_axes={"x": {0: "batch", 1: "n"}, "mask": {0: "batch", 1: "n"}, "out": {0: "batch", 1: "n"}},
                      opset_version=17, dynamo=False)
    with open(os.path.join(a.out, "README.json"), "w") as f:
        json.dump({"purpose": "latency benchmark only - random weights, not a trading model", "features": 8}, f, indent=1)
    print(f"exported {path}")


if __name__ == "__main__":
    main()
