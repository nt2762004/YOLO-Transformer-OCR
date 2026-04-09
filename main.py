from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_project_paths, get_training_config
from src.detector import train_detector
from src.infer import load_recognition_checkpoint, recognize_text
from src.preprocess import (
    build_combined_detection_yaml,
    build_numpy_cache,
    collect_texts,
    prepare_combined_detection_dataset,
)
from src.tokenizer import load_or_train_tokenizer
from src.train import train_recognition
from src.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final_version OCR project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Build crop caches and detector config")
    preprocess.add_argument("--split", choices=["train", "val", "all"], default="all")

    subparsers.add_parser("train-tokenizer", help="Train the character tokenizer")

    train_rec = subparsers.add_parser("train-recognition", help="Train the recognition model from scratch")
    train_rec.add_argument("--epochs", type=int, default=None)

    train_det = subparsers.add_parser("train-detector", help="Train YOLO text detector")
    train_det.add_argument("--weights", default="yolov8n.pt")
    train_det.add_argument("--epochs", type=int, default=50)
    train_det.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    train_det.add_argument("--batch", type=int, default=16, help="Batch size (default: 16, reduce if OOM)")
    train_det.add_argument("--device", type=int, default=0, help="Device (0=GPU, -1=CPU, default: 0)")

    infer = subparsers.add_parser("infer", help="Run end-to-end inference")
    infer.add_argument("--image", type=str, required=True)
    infer.add_argument("--detector", type=str, required=True)
    infer.add_argument("--recognizer", type=str, required=True)

    return parser


def main() -> None:
    paths = get_project_paths()
    config = get_training_config()
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(paths.artifacts_dir)
    ensure_dir(paths.cache_dir)
    ensure_dir(paths.checkpoints_dir)
    ensure_dir(paths.tokenizer_dir)

    if args.command == "preprocess":
        recognition_specs = []

        if args.split in ("train", "all"):
            recognition_specs.extend(
                [
                    ("vn_receipt", paths.vn_receipt_train_transcripts, paths.vn_receipt_train_images, paths.cache_dir / "vn_receipt_train"),
                    ("en_receipt", paths.en_receipt_train_transcripts, paths.en_receipt_train_images, paths.cache_dir / "en_receipt_train"),
                ]
            )
        if args.split in ("val", "all"):
            recognition_specs.extend(
                [
                    ("vn_receipt", paths.vn_receipt_val_transcripts, paths.vn_receipt_val_images, paths.cache_dir / "vn_receipt_val"),
                    ("en_receipt", paths.en_receipt_val_transcripts, paths.en_receipt_val_images, paths.cache_dir / "en_receipt_val"),
                ]
            )

        for prefix, transcript_json, image_dir, cache_dir in recognition_specs:
            build_numpy_cache(transcript_json, image_dir, cache_dir, prefix=prefix)

        prepare_combined_detection_dataset(
            [
                ("vn_receipt", paths.vn_receipt_train_images, paths.vn_receipt_val_images, paths.vn_receipt_train_labels, paths.vn_receipt_val_labels),
                ("en_receipt", paths.en_receipt_train_images, paths.en_receipt_val_images, paths.en_receipt_train_labels, paths.en_receipt_val_labels),
            ],
            paths.combined_detection_dir,
        )
        build_combined_detection_yaml(paths.combined_detection_yaml, paths.combined_detection_dir)
        print(f"Preprocessing complete. Detection config: {paths.combined_detection_yaml}")
        return

    if args.command == "train-tokenizer":
        texts = collect_texts(
            [
                paths.vn_receipt_train_transcripts,
                paths.vn_receipt_val_transcripts,
                paths.en_receipt_train_transcripts,
                paths.en_receipt_val_transcripts,
            ]
        )
        tokenizer = load_or_train_tokenizer(texts, paths.tokenizer_path)
        print(f"Tokenizer saved to {paths.tokenizer_path} with vocab size {tokenizer.vocab_size}")
        return

    if args.command == "train-recognition":
        train_specs = [
            (paths.cache_dir / "vn_receipt_train", paths.cache_dir / "vn_receipt_train" / "metadata.csv"),
            (paths.cache_dir / "en_receipt_train", paths.cache_dir / "en_receipt_train" / "metadata.csv"),
        ]
        val_specs = [
            (paths.cache_dir / "vn_receipt_val", paths.cache_dir / "vn_receipt_val" / "metadata.csv"),
            (paths.cache_dir / "en_receipt_val", paths.cache_dir / "en_receipt_val" / "metadata.csv"),
        ]
        if any(not metadata_csv.exists() for _, metadata_csv in train_specs + val_specs):
            raise FileNotFoundError("Missing cache metadata. Run preprocess first.")

        train_texts = collect_texts([paths.vn_receipt_train_transcripts, paths.en_receipt_train_transcripts])
        val_texts = collect_texts([paths.vn_receipt_val_transcripts, paths.en_receipt_val_transcripts])
        epochs = args.epochs or config.epochs
        history, checkpoint_path, tokenizer_path = train_recognition(
            train_dataset_specs=train_specs,
            val_dataset_specs=val_specs,
            tokenizer_path=paths.tokenizer_path,
            checkpoint_path=paths.checkpoints_dir / "recognition_best.pt",
            train_texts=train_texts,
            val_texts=val_texts,
            max_len=config.max_seq_len,
            image_height=config.image_height,
            image_width=config.image_width,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            lr=config.lr,
            epochs=epochs,
            patience=config.patience,
            d_model=config.d_model,
            nhead=config.nhead,
            decoder_layers=config.decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            seed=config.seed,
        )
        print(f"Training complete. Checkpoint: {checkpoint_path}")
        print(f"Tokenizer: {tokenizer_path}")
        print(f"Last metrics: {history[-1] if history else {}}")
        return

    if args.command == "train-detector":
        if not paths.combined_detection_yaml.exists():
            prepare_combined_detection_dataset(
                [
                    ("vn_receipt", paths.vn_receipt_train_images, paths.vn_receipt_val_images, paths.vn_receipt_train_labels, paths.vn_receipt_val_labels),
                    ("en_receipt", paths.en_receipt_train_images, paths.en_receipt_val_images, paths.en_receipt_train_labels, paths.en_receipt_val_labels),
                ],
                paths.combined_detection_dir,
            )
            build_combined_detection_yaml(paths.combined_detection_yaml, paths.combined_detection_dir)
        train_detector(
            paths.combined_detection_yaml,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project_dir=paths.artifacts_dir / "detector_runs"
        )
        return

    if args.command == "infer":
        import torch
        from ultralytics import YOLO

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo = YOLO(args.detector)
        model, tokenizer = load_recognition_checkpoint(Path(args.recognizer), device)
        results = recognize_text(args.image, yolo, model.encoder, model.decoder, tokenizer, device, max_len=config.max_seq_len)
        for item in results:
            print(item)
        return


if __name__ == "__main__":
    main()
