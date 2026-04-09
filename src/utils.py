from __future__ import annotations

import json
import math
import os
import random
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def generate_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def geometric_mean(values: Iterable[float]) -> float:
    filtered = [max(float(value), 1e-12) for value in values]
    if not filtered:
        return 0.0
    return math.exp(sum(math.log(value) for value in filtered) / len(filtered))
