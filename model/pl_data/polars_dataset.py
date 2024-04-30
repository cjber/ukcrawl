from pathlib import Path

import polars as pl
import torch
from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.common.utils import Consts, Labels


class PolarsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer=AutoTokenizer,
    ) -> None:
        self.tokenizer = tokenizer.from_pretrained(
            Consts.CLASS_MODEL, add_prefix_space=True
        )

        self.data = Dataset.from_polars(
            pl.read_parquet(path, columns=["Primary Category", "content", "URL"])
            .rename({"Primary Category": "label_name", "content": "text"})
            .unique("URL")
            .drop("URL")
            .with_columns(
                pl.col("label_name").replace(Labels.LABEL2ID).cast(int).alias("label")
            )
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx: dict[str, list] = self.data[index]
        encoding = self.tokenizer(
            idx["text"],
            return_attention_mask=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(idx["label"]),
        }
