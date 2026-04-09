from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from .utils import ensure_dir, load_json, normalize_text, save_json


@dataclass
class CharTokenizer:
    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    unk_token: str = "[UNK]"
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def all_special_ids(self) -> list[int]:
        return [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id]

    def fit(self, texts: Iterable[str]) -> None:
        alphabet = set()
        for text in texts:
            alphabet.update(normalize_text(text))

        ordered_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        ordered_tokens.extend(sorted(token for token in alphabet if token not in ordered_tokens))

        self.token_to_id = {token: index for index, token in enumerate(ordered_tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def encode(self, text: str, max_length: int) -> list[int]:
        normalized = normalize_text(text)
        tokens = [self.bos_token_id]
        tokens.extend(self.token_to_id.get(char, self.unk_token_id) for char in normalized)
        tokens.append(self.eos_token_id)

        if len(tokens) > max_length:
            tokens = tokens[: max_length - 1] + [self.eos_token_id]

        if len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))

        return tokens

    def __call__(self, texts, truncation=True, padding="max_length", max_length=100, return_tensors=None):
        encoded = [self.encode(text, max_length=max_length) for text in texts]
        attention = [[1 if token != self.pad_token_id else 0 for token in sequence] for sequence in encoded]

        if return_tensors == "pt":
            import torch

            return SimpleNamespace(
                input_ids=torch.tensor(encoded, dtype=torch.long),
                attention_mask=torch.tensor(attention, dtype=torch.long),
            )

        return SimpleNamespace(input_ids=encoded, attention_mask=attention)

    def decode(self, ids, clean_up_tokenization_spaces: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in (self.pad_token_id, self.bos_token_id):
                continue
            if token_id == self.eos_token_id:
                break
            if token_id == self.unk_token_id:
                tokens.append("?")
            else:
                tokens.append(self.id_to_token.get(token_id, "?"))
        text = "".join(tokens)
        return text.strip() if clean_up_tokenization_spaces else text

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        save_json(
            path,
            {
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "token_to_id": self.token_to_id,
            },
        )

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = load_json(path)
        tokenizer = cls(
            pad_token=payload["pad_token"],
            bos_token=payload["bos_token"],
            eos_token=payload["eos_token"],
            unk_token=payload["unk_token"],
        )
        tokenizer.token_to_id = {str(token): int(index) for token, index in payload["token_to_id"].items()}
        tokenizer.id_to_token = {index: token for token, index in tokenizer.token_to_id.items()}
        return tokenizer


def train_char_tokenizer(texts: Iterable[str], output_path: Path) -> CharTokenizer:
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    tokenizer.save(output_path)
    return tokenizer


def load_or_train_tokenizer(texts: Iterable[str], output_path: Path) -> CharTokenizer:
    if output_path.exists():
        return CharTokenizer.load(output_path)
    return train_char_tokenizer(texts, output_path)
