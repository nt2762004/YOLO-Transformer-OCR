from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from .dataset import OCRRecognitionDataset, build_recognition_dataloader
from .model import OCRModel
from .tokenizer import load_or_train_tokenizer
from .utils import ensure_dir, seed_everything
from .visualize import plot_training_history


def greedy_decode(encoder: nn.Module, decoder: nn.Module, images: torch.Tensor, max_len: int, bos_token_id: int, eos_token_id: int) -> torch.Tensor:
    memory = encoder(images)
    batch_size = images.size(0)
    sequences = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=images.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

    for _ in range(max_len - 1):
        logits = decoder(sequences, memory)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sequences = torch.cat([sequences, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(eos_token_id)
        if finished.all():
            break

    return sequences


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, pad_token_id: int, epoch: int):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for images, labels, _ in tqdm(loader, desc=f"Epoch {epoch} [Train]"):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        decoder_input = labels[:, :-1]
        targets = labels[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        memory = encoder(images)
        logits = decoder(decoder_input, memory)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_epoch(encoder, decoder, loader, criterion, tokenizer, device, epoch: int, max_len: int):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    pred_texts: list[str] = []
    true_texts: list[str] = []

    for images, labels, _ in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        decoder_input = labels[:, :-1]
        targets = labels[:, 1:]

        memory = encoder(images)
        logits = decoder(decoder_input, memory)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item()

        seqs = greedy_decode(encoder, decoder, images, max_len=max_len, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
        pred_texts.extend(tokenizer.decode(seq.tolist()) for seq in seqs)
        true_texts.extend(tokenizer.decode(label.tolist()) for label in labels)

    from jiwer import cer, wer

    return {
        "val_loss": total_loss / max(len(loader), 1),
        "cer": cer(true_texts, pred_texts) if true_texts else 0.0,
        "wer": wer(true_texts, pred_texts) if true_texts else 0.0,
    }


def train_recognition(
    train_dataset_specs: list[tuple[Path, Path]],
    val_dataset_specs: list[tuple[Path, Path]],
    tokenizer_path: Path,
    checkpoint_path: Path,
    train_texts,
    val_texts,
    max_len: int = 100,
    image_height: int = 32,
    image_width: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    lr: float = 1e-4,
    epochs: int = 30,
    patience: int = 5,
    d_model: int = 256,
    nhead: int = 8,
    decoder_layers: int = 6,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    seed: int = 42,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_or_train_tokenizer(list(train_texts) + list(val_texts), tokenizer_path)

    train_datasets = [
        OCRRecognitionDataset(cache_dir, metadata_csv, tokenizer, max_len=max_len, height=image_height, max_width=image_width)
        for cache_dir, metadata_csv in train_dataset_specs
    ]
    val_datasets = [
        OCRRecognitionDataset(cache_dir, metadata_csv, tokenizer, max_len=max_len, height=image_height, max_width=image_width)
        for cache_dir, metadata_csv in val_dataset_specs
    ]

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    train_loader = build_recognition_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_recognition_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = OCRModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    ensure_dir(checkpoint_path.parent)
    best_val_loss = float("inf")
    stalled_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model.encoder, model.decoder, train_loader, criterion, optimizer, device, tokenizer.pad_token_id, epoch)
        metrics = evaluate_epoch(model.encoder, model.decoder, val_loader, criterion, tokenizer, device, epoch, max_len=max_len)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            stalled_epochs = 0
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "tokenizer_path": str(tokenizer_path),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "max_len": max_len,
                        "image_height": image_height,
                        "image_width": image_width,
                        "d_model": d_model,
                        "nhead": nhead,
                        "decoder_layers": decoder_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                    },
                },
                checkpoint_path,
            )
        else:
            stalled_epochs += 1
            if stalled_epochs >= patience:
                break

    # Save training history to JSON
    history_path = checkpoint_path.parent / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Generate and save plots
    images_dir = checkpoint_path.parent / "images"
    plot_training_history(history, images_dir)

    return history, checkpoint_path, tokenizer_path
