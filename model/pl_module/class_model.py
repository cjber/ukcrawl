from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.utils import Consts, Labels


class WebsiteClassificationModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.optim = AdamW
        self.scheduler = ReduceLROnPlateau

        self.train_f1 = F1Score(task="multiclass", num_classes=Labels.COUNT)
        self.val_f1 = F1Score(task="multiclass", num_classes=Labels.COUNT)
        self.test_f1 = F1Score(task="multiclass", num_classes=Labels.COUNT)

        self.softmax = nn.Softmax(dim=1)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            Consts.CLASS_MODEL,
            num_labels=Labels.COUNT,
            return_dict=True,
            id2label=Labels.ID2LABEL,
            label2id=Labels.LABEL2ID,
            finetuning_task="classification",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Consts.CLASS_MODEL, add_prefix_space=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def step(self, batch, _: int):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        return {
            "preds": self.softmax(outputs["logits"]),
            "loss": outputs["loss"],
        }

    def training_step(self, batch, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        f1 = self.train_f1(step_out["preds"], batch["label"])
        self.log_dict(
            {"train_loss": loss, "train_f1": f1},
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        f1 = self.val_f1(step_out["preds"], batch["label"])
        self.log_dict({"val_loss": loss, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        f1 = self.test_f1(step_out["preds"], batch["label"])
        self.log_dict({"test_loss": loss, "test_f1": f1})
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if all(nd not in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        opt = self.optim(lr=2e-5, params=optimizer_grouped_parameters)
        scheduler = self.scheduler(optimizer=opt, patience=1, verbose=True, mode="min")

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
