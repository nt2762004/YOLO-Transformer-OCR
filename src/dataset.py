from __future__ import annotations

import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .tokenizer import CharTokenizer


class OCRRecognitionDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        metadata_csv: Path,
        tokenizer: CharTokenizer,
        max_len: int = 100,
        height: int = 32,
        max_width: int = 256,
        alb_transforms: A.Compose | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.height = height
        self.max_width = max_width
        self.npy_files: list[str] = []
        texts: list[str] = []

        with Path(metadata_csv).open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.npy_files.append(row["npy_name"])
                texts.append(row["text"])

        tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        self.input_ids = tokenized.input_ids
        self.attn_masks = tokenized.attention_mask

        if alb_transforms is None:
            self.alb_transforms = A.Compose(
                [
                    A.ToGray(always_apply=True),
                    A.Resize(self.height, self.max_width),
                    A.PadIfNeeded(self.height, self.max_width, border_mode=cv2.BORDER_CONSTANT, value=255),
                    ToTensorV2(),
                ]
            )
        else:
            self.alb_transforms = alb_transforms

    def __len__(self) -> int:
        return len(self.npy_files)

    def __getitem__(self, index: int):
        array = np.load(self.cache_dir / self.npy_files[index])
        if array.size == 0:
            array = np.ones((self.height, self.max_width, 3), dtype=np.uint8) * 255
        elif array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)

        augmented = self.alb_transforms(image=array)
        image = augmented["image"]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image.float(), self.input_ids[index], self.attn_masks[index]


def build_recognition_dataloader(
    dataset: OCRRecognitionDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
