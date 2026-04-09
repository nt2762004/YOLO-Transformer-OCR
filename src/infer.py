from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from ultralytics import YOLO

from .detector import crop_boxes, detect_text_regions
from .model import OCRModel
from .tokenizer import CharTokenizer
from .utils import geometric_mean


def load_recognition_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharTokenizer.load(Path(checkpoint["tokenizer_path"]))
    config = checkpoint["config"]
    model = OCRModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(device)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model, tokenizer


def decode_logits_with_confidence(logits: torch.Tensor, tokenizer: CharTokenizer):
    probabilities = F.softmax(logits, dim=-1)
    token_ids = probabilities.argmax(-1).detach().cpu().tolist()
    tokens = []
    confidences = []
    for index, token_id in enumerate(token_ids):
        if token_id in tokenizer.all_special_ids:
            continue
        tokens.append(token_id)
        confidences.append(float(probabilities[index, token_id].item()))
    confidence = geometric_mean(confidences)
    return tokenizer.decode(tokens), confidence


def prepare_crop_transform(height: int = 32, width: int = 256) -> A.Compose:
    return A.Compose([
        A.ToGray(),
        A.Resize(height, width),
        A.PadIfNeeded(height, width, border_mode=0, value=255),
        ToTensorV2(),
    ])


@torch.no_grad()
def recognize_text(image, yolo: YOLO, encoder: torch.nn.Module, decoder: torch.nn.Module, tokenizer: CharTokenizer, device: torch.device, max_len: int = 100, bos_token_id: int | None = None, eos_token_id: int | None = None, conf: float = 0.25):
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")

    boxes = detect_text_regions(yolo, pil_image, device=device, conf=conf)
    crops = crop_boxes(pil_image, boxes)
    transform = prepare_crop_transform()

    results = []
    for box, crop in zip(boxes, crops):
        crop_array = np.array(crop)
        if crop_array.ndim == 2:
            crop_array = np.repeat(crop_array[:, :, None], 3, axis=2)
        augmented = transform(image=crop_array)
        tensor = augmented["image"].unsqueeze(0).float().to(device)

        memory = encoder(tensor)
        seq = torch.full((1, 1), bos_token_id or tokenizer.bos_token_id, dtype=torch.long, device=device)
        token_confidences = []
        for _ in range(max_len - 1):
            logits = decoder(seq, memory)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            token_confidences.append(float(probs[0, next_token.item()].item()))
            seq = torch.cat([seq, next_token], dim=1)
            if next_token.item() == (eos_token_id or tokenizer.eos_token_id):
                break

        text = tokenizer.decode(seq.squeeze(0).tolist())
        confidence = geometric_mean(token_confidences)
        results.append({"box": box.tolist() if hasattr(box, "tolist") else list(box), "text": text, "confidence": confidence})

    return results
