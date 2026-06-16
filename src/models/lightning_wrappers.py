from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from models.base import BasePredictor

try:
    from lightning import pytorch as pl
except ImportError:
    try:
        import pytorch_lightning as pl
    except ImportError:
        pl = None

LightningModuleBase = pl.LightningModule if pl is not None else nn.Module


class LightningOnlineModel(LightningModuleBase, BasePredictor):
    def __init__(
        self,
        backbone: nn.Module,
        task_type: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler: str | None = None,
        scheduler_t_max: int = 10,
    ):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_t_max = scheduler_t_max

        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)
        out = self.forward(x)
        loss = self._compute_loss(out, y)
        if hasattr(self, "log"):
            self.log("train_loss", loss)
            if self.task_type == "classification":
                acc = (out.argmax(dim=-1) == y).float().mean()
                self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)
        out = self.forward(x)
        loss = self._compute_loss(out, y)
        if hasattr(self, "log"):
            self.log("val_loss", loss)
            if self.task_type == "classification":
                acc = (out.argmax(dim=-1) == y).float().mean()
                self.log("val_acc", acc)
        return loss

    def _compute_loss(self, out, y):
        if self.task_type == "classification":
            return self.criterion(out, y)
        return self.criterion(out, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.scheduler == "cosine_annealing":
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.scheduler_t_max,
                ),
            }
        return optimizer

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
        x = self._ensure_tensor_batch(x)
        device = next(self.parameters()).device
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device).reshape(-1)
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        optimizer.zero_grad()
        out = self.forward(x)
        loss = self._compute_loss(out, y)
        loss.backward()
        optimizer.step()

    def reset_adaptation_state(self):
        pass

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        return batch

    @staticmethod
    def _ensure_tensor_batch(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            tensor = x.detach().clone().float()
        else:
            tensor = torch.from_numpy(np.asarray(x)).float()
        if tensor.ndim == 1 or tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor


class LightningImageClassifier(LightningOnlineModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str | None = "cosine_annealing",
        scheduler_t_max: int = 10,
        mean: Iterable[float] = (0.4914, 0.4822, 0.4465),
        std: Iterable[float] = (0.2470, 0.2435, 0.2616),
    ):
        super().__init__(
            backbone=backbone,
            task_type="classification",
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            scheduler_t_max=scheduler_t_max,
        )
        self.num_classes = num_classes
        self.mean = tuple(mean)
        self.std = tuple(std)

    def predict_step_online(self, x: Any) -> dict[str, Any]:
        x_tensor = self.prepare_image_batch(x, device=self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_tensor)
            probs = torch.softmax(logits, dim=-1)

        return {
            "raw_output": logits,
            "logits": logits,
            "probs": probs,
            "point_prediction": torch.argmax(probs, dim=-1),
            "uncertainty": 1.0 - probs.max(dim=-1).values,
        }

    def update_online(self, x: Any, y: Any) -> None:
        x_tensor = self.prepare_image_batch(x, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device).reshape(-1)
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        optimizer.zero_grad()
        logits = self.forward(x_tensor)
        loss = self.criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    def prepare_image_batch(self, x: Any, device: torch.device | str | None = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            tensor = x.detach().clone().float()
        else:
            tensor = torch.from_numpy(np.asarray(x)).float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got {tensor.ndim}")

        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)

        if tensor.max() > 2.0:
            tensor = tensor / 255.0

        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.to(device or self.device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class LightningSimpleCNNPredictor(LightningImageClassifier):
    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]):
        from models.simple_cnn import CIFAR10_MEAN, CIFAR10_STD, SimpleCNN

        training_cfg = model_cfg.get("training", {})
        model = cls(
            backbone=SimpleCNN(
                num_classes=model_cfg.get("num_classes", 10),
                dropout=model_cfg.get("dropout", 0.4),
            ),
            num_classes=model_cfg.get("num_classes", 10),
            lr=training_cfg.get("lr", model_cfg.get("lr", 1e-3)),
            weight_decay=training_cfg.get(
                "weight_decay",
                model_cfg.get("weight_decay", 1e-4),
            ),
            scheduler=training_cfg.get("scheduler", "cosine_annealing"),
            scheduler_t_max=training_cfg.get("epochs", 10),
            mean=model_cfg.get("mean", CIFAR10_MEAN),
            std=model_cfg.get("std", CIFAR10_STD),
        )
        device = model_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device)

    def load(self, checkpoint_path: str | Path) -> None:
        state = torch.load(checkpoint_path, map_location=self.device)
        try:
            self.backbone.load_state_dict(state)
        except RuntimeError:
            self.load_state_dict(state)

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), checkpoint)

    def fit_clean_cifar10(
        self,
        root: str | Path = "data",
        epochs: int = 10,
        batch_size: int = 128,
        num_workers: int = 0,
        seed: int = 42,
        calibration_size: int = 5000,
        accelerator: str = "auto",
        devices: int | str = "auto",
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_dataset = self._clean_cifar10_subset(
            root=root,
            seed=seed,
            calibration_size=calibration_size,
            split="train",
            transform=self._train_transform(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        if pl is None:
            self._fit_loader_without_lightning(train_loader, epochs=epochs)
            return

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(self, train_loader)

    def clean_cifar10_calibration_loader(
        self,
        root: str | Path = "data",
        calibration_size: int = 5000,
        batch_size: int = 128,
        num_workers: int = 0,
        seed: int = 42,
    ) -> DataLoader:
        calibration_dataset = self._clean_cifar10_subset(
            root=root,
            seed=seed,
            calibration_size=calibration_size,
            split="calibration",
            transform=self._eval_transform(),
        )
        return DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    @staticmethod
    def _clean_cifar10_subset(
        root: str | Path,
        seed: int,
        calibration_size: int,
        split: str,
        transform,
    ) -> Subset:
        dataset = CIFAR10(
            root=str(root),
            train=True,
            download=True,
            transform=transform,
        )
        if not 0 < calibration_size < len(dataset):
            raise ValueError(
                f"calibration_size must be between 1 and {len(dataset) - 1}, got {calibration_size}"
            )
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        calibration_indices = permutation[:calibration_size]
        train_indices = permutation[calibration_size:]
        if split == "train":
            return Subset(dataset, train_indices)
        if split == "calibration":
            return Subset(dataset, calibration_indices)
        raise ValueError(f"Unsupported CIFAR-10 split: {split}")

    def _fit_loader_without_lightning(self, train_loader: DataLoader, epochs: int) -> None:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            if self.scheduler == "cosine_annealing"
            else None
        )
        for _ in range(epochs):
            self.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

    def _train_transform(self):
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    def _eval_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
            ]
        )
