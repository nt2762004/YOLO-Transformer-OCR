from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .utils import ensure_dir


def train_detector(
    data_yaml: Path,
    weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: int = 0,
    project_dir: Path | None = None,
    name: str = "yolo_textdet",
):
    model = YOLO(weights)
    kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "name": name,
    }
    if project_dir is not None:
        kwargs["project"] = str(project_dir)
        ensure_dir(project_dir)
    return model.train(**kwargs)


def sort_boxes(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    order = np.lexsort((boxes[:, 0], boxes[:, 1]))
    return boxes[order]


def detect_text_regions(yolo: YOLO, image: Image.Image | np.ndarray | str, device: str | torch.device = "cpu", conf: float = 0.25) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        image_array = np.array(image.convert("RGB"))
    else:
        image_array = image
    prediction = yolo.predict(source=image_array, device=device, conf=conf)[0]
    boxes = prediction.boxes.xyxy.cpu().numpy() if prediction.boxes is not None else np.zeros((0, 4), dtype=np.float32)
    return sort_boxes(boxes)


def crop_boxes(image: Image.Image, boxes: Iterable[Iterable[float]]) -> list[Image.Image]:
    crops = []
    width, height = image.size
    for box in boxes:
        x1, y1, x2, y2 = box
        left = max(0, min(int(round(x1)), width - 1))
        top = max(0, min(int(round(y1)), height - 1))
        right = max(left + 1, min(int(round(x2)), width))
        bottom = max(top + 1, min(int(round(y2)), height))
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def draw_boxes(image: np.ndarray, boxes: np.ndarray, texts: list[str] | None = None, confs: list[float] | None = None) -> np.ndarray:
    import cv2  # Lazy import
    
    canvas = image.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 2)
        label = "text"
        if texts and index < len(texts):
            label = texts[index]
        if confs and index < len(confs):
            label = f"{label} ({confs[index]:.2f})"
        cv2.putText(canvas, label[:60], (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1, cv2.LINE_AA)
    return canvas
