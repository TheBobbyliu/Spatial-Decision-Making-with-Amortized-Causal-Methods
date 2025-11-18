#!/usr/bin/env python3
"""Run linear regression experiments on the CDC SVI dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "cdcsvi_nohsdp_poverty_disc"
DEFAULT_CSV = DATA_DIR / "full_dataset.csv"
DEFAULT_FEATURE_DIR = DATA_DIR / "features_dinov3_vitl16"


@dataclass
class ExperimentResult:
    label: str
    mae: float
    r2: float
    remaining_covariates: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear regression ablations.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to full_dataset.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help="Directory containing .pt feature tensors (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quicker experiments.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val split and embedding training.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Dimension of the learned linear embeddings.",
    )
    parser.add_argument(
        "--embedding-epochs",
        type=int,
        default=50,
        help="Training epochs for the embedding model.",
    )
    parser.add_argument(
        "--embedding-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the embedding model optimizer.",
    )
    return parser.parse_args()


def load_tabular_data(csv_path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "row_id"})
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df)))
    if limit is not None:
        df = df.head(limit)
    return df


def load_feature_matrix(feature_dir: Path, row_ids: Sequence[int]) -> np.ndarray:
    tensors = []
    for row_id in row_ids:
        feature_path = feature_dir / f"{row_id}.pt"
        tensor = torch.load(feature_path, map_location="cpu")
        tensors.append(tensor.detach().view(-1).numpy().astype(np.float32, copy=False))
    return np.stack(tensors)


def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[LinearRegression, float, float]:
    model = LinearRegression()
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[val_idx])
    mae = mean_absolute_error(y[val_idx], preds)
    r2 = r2_score(y[val_idx], preds)
    return model, mae, r2


def covariate_masking_experiments(
    df: pd.DataFrame,
    covariates: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    extra_features: np.ndarray | None = None,
) -> List[ExperimentResult]:
    remaining = covariates.copy()
    y = df["outcome"].to_numpy(dtype=float)
    dropped_label = "baseline"
    results: List[ExperimentResult] = []

    while True:
        cols = remaining + ["treatment"]
        X_tabular = df[cols].to_numpy(dtype=float)
        if extra_features is not None:
            X = np.concatenate([X_tabular, extra_features], axis=1)
        else:
            X = X_tabular
        model, mae, r2 = fit_linear_regression(X, y, train_idx, val_idx)
        results.append(
            ExperimentResult(
                label=dropped_label,
                mae=mae,
                r2=r2,
                remaining_covariates=remaining.copy(),
            )
        )
        if not remaining:
            break
        cov_coefs = model.coef_[: len(remaining)]
        drop_idx = int(np.argmax(np.abs(cov_coefs)))
        dropped_label = f"drop {remaining[drop_idx]}"
        remaining.pop(drop_idx)

    return results


def features_to_targets(
    features: np.ndarray,
    targets: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    label: str,
) -> ExperimentResult:
    _, mae, r2 = fit_linear_regression(
        features, targets.to_numpy(dtype=float), train_idx, val_idx
    )
    return ExperimentResult(label=label, mae=mae, r2=r2, remaining_covariates=[])


import torch.nn as nn
class LinearEmbeddingModel(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        # self.embedding = torch.nn.Linear(input_dim, embedding_dim, bias=False)
        # self.head = torch.nn.Linear(embedding_dim, 1)
        self.embedding = nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, embedding_dim)
        )
        
        self.head = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embedding(x)
        out = self.head(z)
        return out, z


def train_linear_embedding(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    embedding_dim: int,
    epochs: int,
    lr: float,
    random_state: int,
) -> tuple[np.ndarray, float, float]:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearEmbeddingModel(features.shape[1], embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    X_train = torch.from_numpy(features[train_idx]).float().to(device)
    y_train = torch.from_numpy(targets[train_idx]).float().unsqueeze(1).to(device)

    X_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_val = torch.from_numpy(targets[val_idx]).float().unsqueeze(1).to(device)

    # Mini-batch training
    batch_size = 128

    for j in range(epochs):
        model.train()
        losses = []
        indices = torch.randperm(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            optimizer.zero_grad()
            preds, _ = model(X_train[batch_idx])
            loss = loss_fn(preds, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if j % 10 == 0:
            print("Epoch:", j+1, "Training loss:", sum(losses) / len(losses))

            with torch.no_grad():
                val_preds, _ = model(X_val)
                val_mae = mean_absolute_error(
                    y_val.cpu().numpy().ravel(), val_preds.cpu().numpy().ravel()
                )
                print("Epoch:", j+1, "Eval loss:", val_mae)

    with torch.no_grad():
        val_preds, _ = model(X_val)
        val_mae = mean_absolute_error(
            y_val.cpu().numpy().ravel(), val_preds.cpu().numpy().ravel()
        )
        val_r2 = r2_score(
            y_val.cpu().numpy().ravel(), val_preds.cpu().numpy().ravel()
        )
        all_embeddings = (
            model.embedding(torch.from_numpy(features).float().to(device))
            .cpu()
            .numpy()
        )
    return all_embeddings, val_mae, val_r2


def main() -> None:
    args = parse_args()
    df = load_tabular_data(args.csv_path, args.limit)
    covariates = [c for c in df.columns if c not in {"row_id", "treatment", "outcome"}]
    print(f"Loaded {len(df)} rows with {len(covariates)} covariates.")

    row_ids = df["row_id"].astype(int).tolist()
    features = load_feature_matrix(args.feature_dir, row_ids)
    print(f"Feature matrix shape: {features.shape}")

    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.random_state,
        shuffle=True,
    )
    y = df["outcome"].to_numpy(dtype=float)
    print(
        f"Train/val split: {len(train_idx)} train / {len(val_idx)} val "
        f"(val frac={args.val_fraction:.2f})"
    )

    print("\nExperiments (validation MAE | R^2):")
    cov_results = covariate_masking_experiments(df, covariates, train_idx, val_idx)
    for res in cov_results:
        print(
            f"[Covariate masking] {res.label:<18} -> "
            f"MAE={res.mae:.4f} | R^2={res.r2:.4f} | "
            f"remaining={len(res.remaining_covariates)}"
        )

    outcome_result = features_to_targets(
        features, df["outcome"], train_idx, val_idx, "features -> outcome"
    )
    print(
        f"[Features] {outcome_result.label:<18} -> "
        f"MAE={outcome_result.mae:.4f} | R^2={outcome_result.r2:.4f}"
    )

    print("\n[Features -> Covariates]")
    for cov in covariates:
        res = features_to_targets(features, df[cov], train_idx, val_idx, f"features -> {cov}")
        print(f"{cov:<12} MAE={res.mae:.4f} | R^2={res.r2:.4f}")

    print("\n[Covariate masking + Dino features]")
    cov_plus_feat = covariate_masking_experiments(
        df, covariates, train_idx, val_idx, extra_features=features
    )
    for res in cov_plus_feat:
        print(
            f"[Mask+Features] {res.label:<18} -> "
            f"MAE={res.mae:.4f} | R^2={res.r2:.4f} | "
            f"remaining={len(res.remaining_covariates)}"
        )

    print("\n[Learned linear embeddings]")
    embeddings, embed_val_mae, embed_val_r2 = train_linear_embedding(
        features,
        y,
        train_idx,
        val_idx,
        embedding_dim=args.embedding_dim,
        epochs=args.embedding_epochs,
        lr=args.embedding_lr,
        random_state=args.random_state,
    )
    print(
        f"Embedding model validation -> MAE={embed_val_mae:.4f} | R^2={embed_val_r2:.4f} "
        f"(dim={args.embedding_dim})"
    )
    cov_plus_embed = covariate_masking_experiments(
        df, covariates, train_idx, val_idx, extra_features=embeddings
    )
    for res in cov_plus_embed:
        print(
            f"[Mask+Embeds] {res.label:<18} -> "
            f"MAE={res.mae:.4f} | R^2={res.r2:.4f} | "
            f"remaining={len(res.remaining_covariates)}"
        )


if __name__ == "__main__":
    main()
