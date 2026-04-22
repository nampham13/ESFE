import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ultralytics"))

from backbone import register

register()

import torch
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("ucmnet.yaml")
    model.info(verbose=True)

    dummy = torch.zeros(1, 3, 640, 640)
    out = model.model(dummy)

    if isinstance(out, dict):
        shapes = {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")}
    elif isinstance(out, (list, tuple)):
        shapes = [tuple(o.shape) if hasattr(o, "shape") else type(o).__name__ for o in out]
    elif hasattr(out, "shape"):
        shapes = tuple(out.shape)
    else:
        shapes = type(out).__name__

    print("Output shapes:", shapes)
