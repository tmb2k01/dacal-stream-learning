import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.base import BasePredictor

class LightningOnlineModel(pl.LightningModule, BasePredictor):
    def __init__(self, backbone: nn.Module, task_type: str, lr: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.lr = lr

        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self.forward(x)
        loss = self._compute_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def _compute_loss(self, out, y):
        if self.task_type == "classification":
            return self.criterion(out, y)
        return self.criterion(out, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step_online(self, x):
        self.eval()
        with torch.no_grad():
            logits_or_output = self.forward(x)

        result = {"raw_output": logits_or_output}

        if self.task_type == "classification":
            probs = torch.softmax(logits_or_output, dim=-1)
            result["probs"] = probs
            result["point_prediction"] = torch.argmax(probs, dim=-1)
        else:
            result["point_prediction"] = logits_or_output

        return result

    def update_online(self, x, y):
        self.train()
        optimizer = self.optimizers() if self.trainer is not None else torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer.zero_grad()
        out = self.forward(x)
        loss = self._compute_loss(out, y)
        self.manual_backward(loss)
        optimizer.step()

    def reset_adaptation_state(self):
        # opcionális: optimizer reset, BN stat reset, memory reset
        pass