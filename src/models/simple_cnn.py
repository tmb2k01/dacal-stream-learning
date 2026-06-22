from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from models.base import BasePredictor

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SimpleCNN(L.LightningModule, BasePredictor):
    """Lightweight CIFAR-10 LightningModule used by the playground notebooks."""

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str | None = "cosine_annealing",
        scheduler_t_max: int = 10,
        mean: Iterable[float] = CIFAR10_MEAN,
        std: Iterable[float] = CIFAR10_STD,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_t_max = scheduler_t_max
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.criterion = nn.CrossEntropyLoss()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.get_embeddings(x)
        h = self.drop(h)
        return self.fc2(h)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.flatten(h)
        h = self.fc1(h)
        return self.relu(h)

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> SimpleCNN:
        training_cfg = model_cfg.get("training", {})
        model = cls(
            num_classes=model_cfg.get("num_classes", 10),
            dropout=model_cfg.get("dropout", 0.4),
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

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_acc", acc)
        return loss

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

    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, Any]:
        x, _ = self._unpack_batch(batch) if self._has_targets(batch) else (batch, None)
        return self.predict_step_online(x)

    def predict_step_online(self, x: Any) -> dict[str, Any]:
        x_tensor = self.prepare_image_batch(x, device=self._module_device)
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
        x_tensor = self.prepare_image_batch(x, device=self._module_device)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self._module_device).reshape(-1)
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

    def reset_adaptation_state(self) -> None:
        pass

    def load(self, checkpoint_path: str | Path) -> None:
        state = torch.load(checkpoint_path, map_location=self._module_device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(key.startswith("backbone.") for key in state):
            state = {
                key.removeprefix("backbone."): value
                for key, value in state.items()
                if key.startswith("backbone.")
            }
        self.load_state_dict(state)

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint)

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
        trainer = L.Trainer(
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

    def prepare_image_batch(
        self,
        x: Any,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
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
        return tensor.to(device or self._module_device)

    @property
    def _module_device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        return batch

    @staticmethod
    def _has_targets(batch: Any) -> bool:
        return isinstance(batch, dict) and "y" in batch or isinstance(batch, (tuple, list))

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

    def _train_transform(self):
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    @staticmethod
    def _eval_transform():
        return T.Compose(
            [
                T.ToTensor(),
            ]
        )
