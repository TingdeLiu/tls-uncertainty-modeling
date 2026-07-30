"""Training and inference utilities for RePN."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for one training fold."""

    epochs: int = 15
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15
    metric_scale: float = 1000.0


@dataclass(frozen=True)
class TrainingHistory:
    """Per-epoch RMSE values and the best validation score."""

    training_rmse: list[float]
    validation_rmse: list[float]
    best_validation_rmse: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_parts(
    labels: torch.Tensor,
    physical_feature_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 2 or labels.shape[1] != physical_feature_count + 1:
        raise ValueError(
            "labels must contain physical features followed by one target column"
        )
    return labels[:, :physical_feature_count], labels[:, -1:].contiguous()


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    checkpoint_path: str | Path,
    physical_feature_count: int = 4,
    config: TrainingConfig | None = None,
    show_progress: bool = True,
) -> TrainingHistory:
    """Train one fold and save the best validation checkpoint."""
    if config is None:
        config = TrainingConfig()
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(device)

    training_rmse: list[float] = []
    validation_rmse: list[float] = []
    best_validation_rmse = float("inf")
    unimproved_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        training_losses = []
        batches = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            leave=False,
            disable=not show_progress,
        )
        for neighbourhoods, labels in batches:
            features, target = _batch_parts(labels, physical_feature_count)
            neighbourhoods = neighbourhoods.to(device)
            features = features.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            prediction = model(neighbourhoods, features)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.detach().item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for neighbourhoods, labels in validation_loader:
                features, target = _batch_parts(labels, physical_feature_count)
                prediction = model(
                    neighbourhoods.to(device),
                    features.to(device),
                )
                validation_losses.append(
                    criterion(prediction, target.to(device)).item()
                )

        if not training_losses or not validation_losses:
            raise ValueError("training and validation loaders must not be empty")
        epoch_training_rmse = (
            float(np.sqrt(np.mean(training_losses))) * config.metric_scale
        )
        epoch_validation_rmse = (
            float(np.sqrt(np.mean(validation_losses))) * config.metric_scale
        )
        training_rmse.append(epoch_training_rmse)
        validation_rmse.append(epoch_validation_rmse)

        if epoch_validation_rmse < best_validation_rmse:
            best_validation_rmse = epoch_validation_rmse
            torch.save(model.state_dict(), checkpoint)
            unimproved_epochs = 0
        else:
            unimproved_epochs += 1
            if unimproved_epochs >= config.early_stopping_patience:
                break

    return TrainingHistory(
        training_rmse=training_rmse,
        validation_rmse=validation_rmse,
        best_validation_rmse=best_validation_rmse,
    )


def predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    physical_feature_count: int = 4,
) -> np.ndarray:
    """Return predictions for a labelled loader."""
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for neighbourhoods, labels in loader:
            features, _ = _batch_parts(labels, physical_feature_count)
            prediction = model(
                neighbourhoods.to(device),
                features.to(device),
            )
            predictions.append(prediction.cpu())
    if not predictions:
        raise ValueError("loader must not be empty")
    return torch.cat(predictions, dim=0).numpy()
