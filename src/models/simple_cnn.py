from __future__ import annotations

from collections import deque
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

# Training-time augmentation of the clean CIFAR-10 train images. Deliberately free of
# noise/blur ops so the model never sees the corruption families used in the stream.
DEFAULT_AUGMENTATION: dict[str, Any] = {
    "random_crop_padding": 4,
    "horizontal_flip": True,
    "color_jitter": 0.2,
    "random_erasing": 0.0,
}


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
        online_lr: float | None = None,
        online_optimizer: str = "adam",
        freeze_bn_online: bool = True,
        reset_mode: str = "none",
        online_dropout: bool = True,
        online_steps: int = 1,
        online_replay_size: int = 0,
        online_replay_batch: int = 8,
        online_seed: int = 0,
        online_trainable: str = "all",
        online_anchor_batch: int = 0,
        augmentation: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if online_optimizer not in {"adam", "sgd"}:
            raise ValueError("online_optimizer must be 'adam' or 'sgd'")
        if reset_mode not in {"none", "restore_initial"}:
            raise ValueError("reset_mode must be 'none' or 'restore_initial'")
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_t_max = scheduler_t_max
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.online_lr = online_lr
        self.online_optimizer = online_optimizer
        self.freeze_bn_online = freeze_bn_online
        self.reset_mode = reset_mode
        if online_trainable not in {"all", "fc", "head"}:
            raise ValueError("online_trainable must be 'all', 'fc' or 'head'")
        if online_anchor_batch < 0:
            raise ValueError("online_anchor_batch must be >= 0")
        self.online_trainable = online_trainable
        self.online_anchor_batch = int(online_anchor_batch)
        self._anchor: tuple[torch.Tensor, torch.Tensor] | None = None
        if online_steps < 1:
            raise ValueError("online_steps must be >= 1")
        if online_replay_size < 0 or online_replay_batch < 0:
            raise ValueError("online_replay_size and online_replay_batch must be >= 0")
        self.online_dropout = online_dropout
        self.online_steps = int(online_steps)
        self.online_replay_size = int(online_replay_size)
        self.online_replay_batch = int(online_replay_batch)
        self.augmentation = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
        self._replay: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=self.online_replay_size or None
        )
        self._online_rng = np.random.default_rng(online_seed)
        self.online_updates = 0
        self._online_opt: torch.optim.Optimizer | None = None
        self._initial_state: dict[str, torch.Tensor] | None = None
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
            online_lr=model_cfg.get("online_lr"),
            online_optimizer=model_cfg.get("online_optimizer", "adam"),
            freeze_bn_online=model_cfg.get("freeze_bn_online", True),
            reset_mode=model_cfg.get("reset_mode", "none"),
            online_dropout=model_cfg.get("online_dropout", True),
            online_steps=model_cfg.get("online_steps", 1),
            online_replay_size=model_cfg.get("online_replay_size", 0),
            online_replay_batch=model_cfg.get("online_replay_batch", 8),
            online_seed=model_cfg.get("online_seed", 0),
            online_trainable=model_cfg.get("online_trainable", "all"),
            online_anchor_batch=model_cfg.get("online_anchor_batch", 0),
            augmentation=training_cfg.get("augmentation"),
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
            embedding = self.get_embeddings(x_tensor)
            logits = self.fc2(embedding)
            probs = torch.softmax(logits, dim=-1)

        return {
            "raw_output": logits,
            "logits": logits,
            "probs": probs,
            "point_prediction": torch.argmax(probs, dim=-1),
            "uncertainty": 1.0 - probs.max(dim=-1).values,
            # Penultimate-layer features, consumed by embedding drift detectors.
            "embedding": embedding,
        }

    def embed(self, x: Any) -> np.ndarray:
        """Return penultimate-layer embeddings as a numpy array."""
        x_tensor = self.prepare_image_batch(x, device=self._module_device)
        self.eval()
        with torch.no_grad():
            return self.get_embeddings(x_tensor).cpu().numpy()

    def predict_batched(self, X: Any, batch_size: int = 256) -> dict[str, np.ndarray]:
        """Batched inference returning numpy ``probs`` and ``embedding`` arrays."""
        probs, embeddings = [], []
        for start in range(0, len(X), batch_size):
            out = self.predict_step_online(X[start : start + batch_size])
            probs.append(out["probs"].cpu().numpy())
            embeddings.append(out["embedding"].cpu().numpy())
        return {"probs": np.concatenate(probs), "embedding": np.concatenate(embeddings)}

    def update_online(self, x: Any, y: Any) -> None:
        """Gradient update(s) on one newly labeled stream sample.

        Uses a persistent optimizer (``online_optimizer`` / ``online_lr``). Options:

        - ``freeze_bn_online``: BatchNorm stays in eval mode, so single-sample updates
          do not overwrite the clean running statistics.
        - ``online_dropout``: when false, dropout is disabled during the update.
        - ``online_replay_size`` / ``online_replay_batch``: the new sample is trained
          together with up to ``online_replay_batch`` earlier labeled stream samples.
        - ``online_steps``: gradient steps taken per labeled sample.
        - ``online_trainable``: ``all`` weights, ``fc`` (both dense layers) or ``head`` (last layer).
        - ``online_anchor_batch``: also train on this many clean *training* images (see
          :meth:`set_anchor_data`), which keeps single-sample updates from drifting the
          classifier toward the most recent label.
        """
        x_tensor = self.prepare_image_batch(x, device=self._module_device)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self._module_device).reshape(-1)
        x_batch, y_batch = self._with_replay(x_tensor, y_tensor)
        x_batch, y_batch = self._with_anchor(x_batch, y_batch)

        self.train()
        if self.freeze_bn_online:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if not self.online_dropout:
            self.drop.eval()
        optimizer = self._get_online_optimizer()
        for _ in range(self.online_steps):
            optimizer.zero_grad()
            loss = self.criterion(self.forward(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        self.eval()
        self.online_updates += 1

        if self.online_replay_size > 0:
            self._replay.append((x_tensor.detach(), y_tensor.detach()))

    def set_anchor_data(self, x: Any, y: Any) -> None:
        """Register labeled clean training images mixed into every online update."""
        x_tensor = self.prepare_image_batch(x, device=self._module_device)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self._module_device)
        self._anchor = (x_tensor, y_tensor.reshape(-1))

    def _with_anchor(
        self, x_tensor: torch.Tensor, y_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.online_anchor_batch == 0 or self._anchor is None:
            return x_tensor, y_tensor
        ax, ay = self._anchor
        picks = torch.as_tensor(
            self._online_rng.choice(len(ay), size=min(self.online_anchor_batch, len(ay)), replace=False),
            device=ax.device,
        )
        return torch.cat([x_tensor, ax[picks]], dim=0), torch.cat([y_tensor, ay[picks]], dim=0)

    def _online_parameters(self) -> list[nn.Parameter]:
        if self.online_trainable == "head":
            return list(self.fc2.parameters())
        if self.online_trainable == "fc":
            return [*self.fc1.parameters(), *self.fc2.parameters()]
        return list(self.parameters())

    def _with_replay(
        self, x_tensor: torch.Tensor, y_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.online_replay_size == 0 or self.online_replay_batch == 0 or not self._replay:
            return x_tensor, y_tensor
        k = min(self.online_replay_batch, len(self._replay))
        picks = self._online_rng.choice(len(self._replay), size=k, replace=False)
        xs = [x_tensor] + [self._replay[i][0] for i in picks]
        ys = [y_tensor] + [self._replay[i][1] for i in picks]
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def _get_online_optimizer(self) -> torch.optim.Optimizer:
        if self._online_opt is None:
            lr = self.online_lr if self.online_lr is not None else self.lr
            if self.online_optimizer == "sgd":
                self._online_opt = torch.optim.SGD(self._online_parameters(), lr=lr, momentum=0.9)
            else:
                self._online_opt = torch.optim.Adam(
                    self._online_parameters(), lr=lr, weight_decay=self.weight_decay
                )
        return self._online_opt

    def snapshot_initial_state(self) -> None:
        """Store the current (pretrained) weights as the reset target."""
        self._initial_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

    def reset_adaptation_state(self) -> None:
        """``reset_mode='none'`` keeps adapted weights; ``'restore_initial'`` reverts them."""
        self._online_opt = None
        if self.reset_mode == "restore_initial" and self._initial_state is not None:
            self.load_state_dict(self._initial_state)
            self._replay.clear()

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
        """Augmentation pipeline for clean CIFAR-10 training images (see ``DEFAULT_AUGMENTATION``)."""
        aug = self.augmentation
        ops: list[Any] = []
        if aug["random_crop_padding"]:
            ops.append(T.RandomCrop(32, padding=int(aug["random_crop_padding"])))
        if aug["horizontal_flip"]:
            ops.append(T.RandomHorizontalFlip())
        if aug["color_jitter"]:
            j = float(aug["color_jitter"])
            ops.append(T.ColorJitter(brightness=j, contrast=j, saturation=j))
        ops += [T.ToTensor(), T.Normalize(self.mean, self.std)]
        if aug["random_erasing"]:
            ops.append(T.RandomErasing(p=float(aug["random_erasing"])))
        return T.Compose(ops)

    @staticmethod
    def _eval_transform():
        return T.Compose(
            [
                T.ToTensor(),
            ]
        )
