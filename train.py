import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ultralytics"))

from backbone import register

register()

import torch
from ultralytics import YOLO  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UCM-Net (ESFENet backbone) with Ultralytics YOLO.")
    parser.add_argument("--model", default="ucmnet.yaml", help="Path to model YAML.")
    parser.add_argument("--data", default="local_data/esfe_local.yaml", help="Dataset YAML path.")
    parser.add_argument("--weights", default="", help="Optional pretrained checkpoint to warm-start from.")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", default="ultralytics/runs", help="Output project directory.")
    parser.add_argument("--name", default="train_esfe_e3", help="Run name.")
    return parser.parse_args()


def load_backbone_neck_weights(model: YOLO, ckpt_path: str) -> int:
    """Load matching weights except detect head from checkpoint into model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    model_state = model.model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and "detect" not in k.lower()}
    model_state.update(filtered)
    model.model.load_state_dict(model_state, strict=False)
    return len(filtered)


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.model)

    if args.weights:
        loaded = load_backbone_neck_weights(model, args.weights)
        print(f"Loaded {loaded} layers from {args.weights}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,
        patience=50,
        project=args.project,
        name=args.name,
    )
