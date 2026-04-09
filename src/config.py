from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    notebooks_dir: Path
    src_dir: Path
    artifacts_dir: Path
    combined_detection_dir: Path
    cache_dir: Path
    checkpoints_dir: Path
    tokenizer_dir: Path
    tokenizer_path: Path
    combined_detection_yaml: Path
    combined_detection_image_dir: Path
    combined_detection_label_dir: Path
    en_receipt_dir: Path
    en_receipt_train_images: Path
    en_receipt_val_images: Path
    en_receipt_test_images: Path
    en_receipt_train_labels: Path
    en_receipt_val_labels: Path
    en_receipt_train_transcripts: Path
    en_receipt_val_transcripts: Path
    vn_receipt_dir: Path
    vn_receipt_train_images: Path
    vn_receipt_val_images: Path
    vn_receipt_test_images: Path
    vn_receipt_train_labels: Path
    vn_receipt_val_labels: Path
    vn_receipt_train_transcripts: Path
    vn_receipt_val_transcripts: Path
    detection_yaml: Path


@dataclass(frozen=True)
class TrainingConfig:
    image_height: int = 32
    image_width: int = 256
    max_seq_len: int = 100
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    epochs: int = 30
    patience: int = 5
    d_model: int = 256
    nhead: int = 8
    decoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    seed: int = 42


def get_project_paths() -> ProjectPaths:
    root = get_project_root()
    data_dir = root / "data"
    notebooks_dir = root / "notebooks"
    src_dir = root / "src"
    artifacts_dir = root / "artifacts"
    combined_detection_dir = artifacts_dir / "combined_receipt"
    cache_dir = artifacts_dir / "cache"
    checkpoints_dir = artifacts_dir / "checkpoints"
    tokenizer_dir = artifacts_dir / "tokenizer"

    en_receipt_dir = data_dir / "en_receipt"
    vn_receipt_dir = data_dir / "vn_receipt"
    return ProjectPaths(
        root=root,
        data_dir=data_dir,
        notebooks_dir=notebooks_dir,
        src_dir=src_dir,
        artifacts_dir=artifacts_dir,
        combined_detection_dir=combined_detection_dir,
        cache_dir=cache_dir,
        checkpoints_dir=checkpoints_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer_path=tokenizer_dir / "char_tokenizer.json",
        combined_detection_yaml=data_dir / "combined_receipt.yaml",
        combined_detection_image_dir=combined_detection_dir / "images",
        combined_detection_label_dir=combined_detection_dir / "labels",
        en_receipt_dir=en_receipt_dir,
        en_receipt_train_images=en_receipt_dir / "images" / "train",
        en_receipt_val_images=en_receipt_dir / "images" / "valid",
        en_receipt_test_images=en_receipt_dir / "images" / "test",
        en_receipt_train_labels=en_receipt_dir / "labels" / "train",
        en_receipt_val_labels=en_receipt_dir / "labels" / "valid",
        en_receipt_train_transcripts=en_receipt_dir / "train_transcripts.json",
        en_receipt_val_transcripts=en_receipt_dir / "valid_transcripts.json",
        vn_receipt_dir=vn_receipt_dir,
        vn_receipt_train_images=vn_receipt_dir / "images" / "train",
        vn_receipt_val_images=vn_receipt_dir / "images" / "val",
        vn_receipt_test_images=vn_receipt_dir / "images" / "test",
        vn_receipt_train_labels=vn_receipt_dir / "labels" / "train",
        vn_receipt_val_labels=vn_receipt_dir / "labels" / "val",
        vn_receipt_train_transcripts=vn_receipt_dir / "train_transcripts.json",
        vn_receipt_val_transcripts=vn_receipt_dir / "val_transcripts.json",
        detection_yaml=data_dir / "vn_receipt.yaml",
    )


def get_training_config() -> TrainingConfig:
    return TrainingConfig()
