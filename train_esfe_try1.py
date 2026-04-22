import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ultralytics"))

from backbone import register

register()

from ultralytics import YOLO  # type: ignore[attr-defined]


if __name__ == "__main__":
    model = YOLO("ucmnet.yaml")
    model.train(
        data="local_data/esfe_local.yaml",
        epochs=200,
        imgsz=640,
        batch=4,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,
        patience=50,
        project="ultralytics/runs",
        name="esfe_try1",
    )
