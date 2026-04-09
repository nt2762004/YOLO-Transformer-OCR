from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .utils import clip_box, ensure_dir, load_json, normalize_text


def link_or_copy_file(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except Exception:
        try:
            destination.symlink_to(source)
        except Exception:
            shutil.copy2(source, destination)


def build_numpy_cache(transcript_json: Path, img_dir: Path, cache_dir: Path, prefix: str | None = None) -> Path:
    ensure_dir(cache_dir)
    metadata_path = cache_dir / "metadata.csv"

    data = load_json(transcript_json)
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["npy_name", "text", "image_name", "box_x1", "box_y1", "box_x2", "box_y2"])

        for image_name, items in data.items():
            image_path = img_dir / image_name
            if not image_path.exists():
                continue

            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
                width, height = image.size
                for index, item in enumerate(items):
                    box = item.get("box") or item.get("bbox")
                    text = normalize_text(item.get("text", ""))
                    if not box or not text:
                        continue

                    left, top, right, bottom = clip_box(box, width, height)
                    if right <= left or bottom <= top:
                        continue

                    crop = image.crop((left, top, right, bottom))
                    array = np.array(crop, dtype=np.uint8)
                    stem = Path(image_name).stem
                    npy_name = f"{prefix + '__' if prefix else ''}{stem}_{index}.npy"
                    np.save(cache_dir / npy_name, array)
                    writer.writerow([npy_name, text, image_name, left, top, right, bottom])

    return metadata_path


def collect_texts(transcript_jsons: Iterable[Path] | Path) -> list[str]:
    if isinstance(transcript_jsons, Path):
        transcript_jsons = [transcript_jsons]

    texts: list[str] = []
    for transcript_json in transcript_jsons:
        data = load_json(transcript_json)
        for items in data.values():
            for item in items:
                text = normalize_text(item.get("text", ""))
                if text:
                    texts.append(text)
    return texts


def build_detection_yaml(output_path: Path, train_images: Path, val_images: Path, class_name: str = "text") -> Path:
    ensure_dir(output_path.parent)
    yaml_text = "\n".join(
        [
            f'path: "{train_images.parent.parent.parent.as_posix()}"',
            f'train: "{train_images.as_posix()}"',
            f'val: "{val_images.as_posix()}"',
            "names:",
            f"  0: {class_name}",
        ]
    )
    output_path.write_text(yaml_text, encoding="utf-8")
    return output_path


def build_project_caches(transcript_jsons: Iterable[tuple[Path, Path, Path]]) -> list[Path]:
    metadata_paths = []
    for transcript_json, img_dir, cache_dir in transcript_jsons:
        metadata_paths.append(build_numpy_cache(transcript_json, img_dir, cache_dir))
    return metadata_paths


def prepare_combined_detection_dataset(
    dataset_specs: Iterable[tuple[str, Path, Path, Path, Path]],
    output_root: Path,
) -> Path:
    image_train_dir = output_root / "images" / "train"
    image_val_dir = output_root / "images" / "val"
    label_train_dir = output_root / "labels" / "train"
    label_val_dir = output_root / "labels" / "val"

    for directory in (image_train_dir, image_val_dir, label_train_dir, label_val_dir):
        ensure_dir(directory)

    for dataset_name, train_images, val_images, train_labels, val_labels in dataset_specs:
        for source_dir, destination_dir, extension in (
            (train_images, image_train_dir, None),
            (val_images, image_val_dir, None),
            (train_labels, label_train_dir, ".txt"),
            (val_labels, label_val_dir, ".txt"),
        ):
            if not source_dir.exists():
                continue
            for source_file in source_dir.iterdir():
                if not source_file.is_file():
                    continue
                if extension is not None and source_file.suffix.lower() != extension:
                    continue
                destination_file = destination_dir / f"{dataset_name}__{source_file.name}"
                link_or_copy_file(source_file, destination_file)

    return output_root


def build_combined_detection_yaml(output_path: Path, combined_root: Path, class_name: str = "text") -> Path:
    ensure_dir(output_path.parent)
    yaml_text = "\n".join(
        [
            f'path: "{combined_root.as_posix()}"',
            'train: "images/train"',
            'val: "images/val"',
            "names:",
            f"  0: {class_name}",
        ]
    )
    output_path.write_text(yaml_text, encoding="utf-8")
    return output_path
