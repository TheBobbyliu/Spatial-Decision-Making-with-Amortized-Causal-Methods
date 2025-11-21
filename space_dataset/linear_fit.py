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
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import sys
sys.path.append("../../CausalPFN")
from src.causalpfn import CATEEstimator
# from causalpfn import CATEEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "cdcsvi_nohsdp_poverty_disc"
DEFAULT_CSV = DATA_DIR / "full_dataset.csv"
DEFAULT_FEATURE_DIR = DATA_DIR / "features_dinov3_vitl16"
PLOT_DIR = REPO_ROOT / "downloads" / "training_curves"
DROP_ORDER_INFO = [
    ("EP_NOINT", 0.10835388560054054, "no internet"),
    ("EP_DISABL", 0.052309518971272125, "disabled residents"),
    ("EP_AGE65", 0.03462559284916882, "age 65+"),
    ("EP_NOVEH", 0.03215625814964905, "no vehicle"),
    ("EP_LIMENG", 0.02169817521202527, "limited English proficiency"),
    ("EP_AGE17", 0.019733735688725287, "age 17 or younger"),
    ("EP_UNEMP", 0.01749401722098651, "unemployment"),
    ("RPL_THEME3", 0.01628507092659359, "SVI Theme 3 percentile"),
    ("EP_SNGPNT", 0.016162292726655373, "single-parent households"),
    ("EP_MUNIT", 0.013712076635576134, "multi-unit housing"),
    ("EP_MINRTY", 0.006506325206589632, "minority (non-white)"),
]
DROP_ORDER = [name for name, _, _ in DROP_ORDER_INFO]
DROP_DESCRIPTIONS = {name: desc for name, _, desc in DROP_ORDER_INFO}


def compute_pehe(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute PEHE between predicted and true CATE values."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def plot_training_curves(
    train_losses: List[float],
    val_metrics: List[float],
    label: str,
) -> Path:
    """Plot training/eval trends and store inside downloads/."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_metrics, label="Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE")
    ax.set_title(f"{label} training curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    sanitized_label = label.replace(" ", "_")
    output_path = PLOT_DIR / f"{sanitized_label}_curve.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@dataclass
class ExperimentResult:
    label: str
    mae: float
    r2: float
    remaining_covariates: List[str]
    preds: List[float]


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
        "--pca",
        action="store_true",
        help="use pca instead of training mlp"
    )
    parser.add_argument(
        "--joint-embedding",
        action="store_true",
        help="jointly train embeddings on covariates, outcome, and treatment"
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
        default=1e-3,
        help="Learning rate for the embedding model optimizer.",
    )
    parser.add_argument(
        "--cate-max-steps",
        type=int,
        default=None,
        help="Optional limit on the number of CATE masking steps (for quicker debugging).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="choose to standardize the feature or not"
    )
    return parser.parse_args()


def load_tabular_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "row_id"})
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df)))
    return df


def load_true_cate(csv_path: Path) -> np.ndarray | None:
    """Load ground-truth CATE from the cate.txt next to the csv."""
    cate_path = csv_path.parent / "cate.txt"
    if not cate_path.exists():
        return None
    values = np.loadtxt(cate_path, dtype=float)
    return values.reshape(-1)


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
    return model, preds, mae, r2


def covariate_masking_experiments(
    df: pd.DataFrame,
    covariates: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    extra_features: np.ndarray | None = None,
    drop_order: Sequence[str] | None = None,
) -> List[ExperimentResult]:
    remaining = covariates.copy()
    y = df["outcome"].to_numpy(dtype=float)
    dropped_label = "baseline"
    results: List[ExperimentResult] = []
    drop_sequence = [c for c in (drop_order or []) if c in remaining]
    drop_sequence.extend([c for c in remaining if c not in drop_sequence])
    drop_pointer = 0

    while True:
        cols = remaining + ["treatment"]
        X_tabular = df[cols].to_numpy(dtype=float)
        if extra_features is not None:
            X = np.concatenate([X_tabular, extra_features], axis=1)
        else:
            X = X_tabular
        model, preds, mae, r2 = fit_linear_regression(X, y, train_idx, val_idx)
        results.append(
            ExperimentResult(
                label=dropped_label,
                mae=mae,
                r2=r2,
                remaining_covariates=remaining.copy(),
                preds=preds
            )
        )
        if not remaining:
            break
        next_drop = None
        while drop_pointer < len(drop_sequence):
            candidate = drop_sequence[drop_pointer]
            drop_pointer += 1
            if candidate in remaining:
                next_drop = candidate
                break
        if next_drop is None:
            next_drop = remaining[0]
        desc = DROP_DESCRIPTIONS.get(next_drop)
        if desc:
            dropped_label = f"drop {next_drop} ({desc})"
        else:
            dropped_label = f"drop {next_drop}"
        remaining.remove(next_drop)

    return results


def features_to_targets(
    features: np.ndarray,
    targets: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    label: str,
) -> ExperimentResult:
    _, preds, mae, r2 = fit_linear_regression(
        features, targets.to_numpy(dtype=float), train_idx, val_idx
    )
    return ExperimentResult(label=label, mae=mae, r2=r2, remaining_covariates=[], preds=preds)


import torch.nn as nn
class LinearEmbeddingModel(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        # self.embedding = torch.nn.Linear(input_dim, embedding_dim, bias=False)
        # self.head = torch.nn.Linear(embedding_dim, 1)
        self.embedding = nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2, bias=False),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, embedding_dim)
        )
        
        self.head = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embedding(x)
        out = self.head(z)
        return out, z


class MultiTaskLinearEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        regression_targets: int,
        classification_targets: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            torch.nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, embedding_dim),
        )
        # self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.regression_head = torch.nn.Linear(embedding_dim, regression_targets)
        self.classification_head = torch.nn.Linear(embedding_dim, classification_targets)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        z = self.embedding(x)
        # neck = self.bnneck(z)
        reg_out = self.regression_head(z)
        cls_out = self.classification_head(z)
        return reg_out, cls_out, z

def run_pca(embedding_dim, features, train_idx):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=embedding_dim, whiten=True)
    pca.fit(features[train_idx])
    embeddings = pca.transform(features)
    return embeddings


def train_linear_embedding(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    embedding_dim: int,
    epochs: int,
    lr: float,
    random_state: int,
    cls: bool = False,
    curve_label: str | None = None,
) -> tuple[np.ndarray, float, float]:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearEmbeddingModel(features.shape[1], embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if cls:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    X_train = torch.from_numpy(features[train_idx]).float().to(device)
    y_train = torch.from_numpy(targets[train_idx]).float().unsqueeze(1).to(device)
    
    X_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_val = torch.from_numpy(targets[val_idx]).float().unsqueeze(1).to(device)

    # Mini-batch training
    batch_size = 128

    train_loss_history: List[float] = []
    val_metric_history: List[float] = []
    for _ in range(epochs):
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
        train_loss_history.append(float(np.mean(losses) if losses else 0.0))
        with torch.no_grad():
            val_preds, _ = model(X_val)
            if cls:
                val_preds = torch.sigmoid(val_preds)
            val_mae = mean_absolute_error(
                y_val.cpu().numpy().ravel(), val_preds.cpu().numpy().ravel()
            )
            val_metric_history.append(val_mae)

    label = curve_label or "embedding"
    plot_path = plot_training_curves(train_loss_history, val_metric_history, label)
    print(f"Saved training curve plot to {plot_path}")

    with torch.no_grad():
        val_preds, _ = model(X_val)
        if cls:
            val_preds = torch.sigmoid(val_preds)
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


def train_multitask_embedding(
    features: np.ndarray,
    covariate_targets: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    embedding_dim: int,
    epochs: int,
    lr: float,
    random_state: int,
    curve_label: str | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Jointly train embeddings to predict covariates/outcome and classify treatment."""
    # standardize regression as well
    y = outcome[train_idx]
    y_mean = y.mean()
    y_std = y.std() + 1e-6
    outcome = (outcome - y_mean) / y_std

    regression_targets = outcome.reshape(-1, 1)
    classification_targets = treatment.reshape(-1, 1)

    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLinearEmbeddingModel(
        input_dim=features.shape[1],
        embedding_dim=embedding_dim,
        regression_targets=regression_targets.shape[1],
        classification_targets=classification_targets.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    X_train = torch.from_numpy(features[train_idx]).float().to(device)
    y_reg_train = torch.from_numpy(regression_targets[train_idx]).float().to(device)
    y_cls_train = torch.from_numpy(classification_targets[train_idx]).float().to(device)

    X_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_reg_val = torch.from_numpy(regression_targets[val_idx]).float().to(device)
    y_cls_val = torch.from_numpy(classification_targets[val_idx]).float().to(device)

    batch_size = 128
    train_loss_history: List[float] = []
    val_metric_history: List[float] = []
    val_loss_history_reg = []
    val_loss_history_cls = []

    labels = y_cls_train.view(-1).long()  # shape [N], ints
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]   # weight per sample
    print(class_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_ds = TensorDataset(X_train, y_reg_train, y_cls_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )

    for ep in range(epochs):
        model.train()
        losses = []
        for X_b, y_reg_b, y_cls_b in train_loader:
            optimizer.zero_grad()
            reg_preds, cls_preds, _ = model(X_b)
            mse = mse_loss(reg_preds, y_reg_b)
            bce = bce_loss(cls_preds, y_cls_b)
            # loss = mse + bce
            loss = bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        train_loss_history.append(float(np.mean(losses) if losses else 0.0))

        with torch.no_grad():
            reg_val_pred, cls_val_pred, _ = model(X_val)
            print(cls_val_pred)
            val_reg_loss = mse_loss(reg_val_pred, y_reg_val)
            val_cls_loss = (
                bce_loss(cls_val_pred, y_cls_val) if cls_val_pred is not None else 0.0
            )
            val_metric_history.append(float((val_reg_loss + val_cls_loss).item()))
            val_loss_history_cls.append(val_cls_loss.item())
            val_loss_history_reg.append(val_reg_loss.item())

    label = curve_label or "joint_embedding"
    plot_path = plot_training_curves(train_loss_history, val_metric_history, label)
    _ = plot_training_curves(val_loss_history_reg, val_loss_history_cls, "reg_cls_losses_val")
    print(f"Saved training curve plot to {plot_path}")

    with torch.no_grad():
        reg_val_pred, cls_val_pred, _ = model(X_val)
        cov_preds = reg_val_pred[:, :-1]
        outcome_preds = reg_val_pred[:, -1]
        treatment_probs = (
            torch.sigmoid(cls_val_pred).cpu().numpy()
            if cls_val_pred is not None
            else None
        )

        # cov_mae = mean_absolute_error(
        #     y_reg_val[:, :-1].cpu().numpy(),
        #     cov_preds.cpu().numpy(),
        # )
        cov_mae = 0
        outcome_mae = mean_absolute_error(
            y_reg_val[:, -1].cpu().numpy(),
            outcome_preds.cpu().numpy(),
        )
        treatment_acc = (
            accuracy_score(
                y_cls_val.cpu().numpy().ravel().astype(int),
                (treatment_probs >= 0.5).astype(int).ravel(),
            )
            if treatment_probs is not None
            else float("nan")
        )

        all_embeddings = (
            model.embedding(torch.from_numpy(features).float().to(device)).cpu().numpy()
        )

    return all_embeddings, cov_mae, outcome_mae, treatment_acc

def _predict_potential_outcomes(estimator: CATEEstimator, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    estimator._check_fitted()
    X_context = estimator.X_train
    t_context = estimator.t_train
    y_context = estimator.y_train
    X_eval = X_query
    if estimator.max_feature_size is not None and X_eval.shape[1] > estimator.max_feature_size:
        X_eval = estimator.x_dim_transformer.transform(X_eval)

    zeros = np.zeros(X_eval.shape[0], dtype=t_context.dtype)
    ones = np.ones(X_eval.shape[0], dtype=t_context.dtype)
    mu0_mu1 = estimator._predict_cepo(
        X_context=X_context,
        t_context=t_context,
        y_context=y_context,
        X_query=np.concatenate([X_eval, X_eval], axis=0),
        t_query=np.concatenate([zeros, ones], axis=0),
        temperature=estimator.prediction_temperature,
    )
    mu0 = mu0_mu1[: X_eval.shape[0]]
    mu1 = mu0_mu1[X_eval.shape[0] :]
    return mu0, mu1


def fit_and_score_cate(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: str = "cpu",
) -> tuple[float, float, np.ndarray]:
    estimator = CATEEstimator(device=device)
    X_train = X[train_idx]
    t_train = treatment[train_idx]
    y_train = outcome[train_idx]
    estimator.fit(X_train, t_train, y_train)

    mu0, mu1 = _predict_potential_outcomes(estimator, X[val_idx])
    t_val = treatment[val_idx]
    y_val = outcome[val_idx]
    y_pred = np.where(t_val >= 0.5, mu1, mu0)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    cate = mu1 - mu0

    return y_pred, mae, r2, cate


def run_cate_masking_experiments(
    df: pd.DataFrame,
    schedule: List[ExperimentResult],
    treatment: np.ndarray,
    outcome: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    extra_features: np.ndarray | None,
    true_cate: np.ndarray | None,
    label_prefix: str,
    max_steps: int | None = None,
) -> None:
    for idx, res in enumerate(schedule):
        if max_steps is not None and idx >= max_steps:
            break
        if res.remaining_covariates:
            X_tabular = df[res.remaining_covariates].to_numpy(dtype=float)
        else:
            X_tabular = np.empty((len(df), 0), dtype=float)

        if extra_features is not None:
            X = np.concatenate([X_tabular, extra_features], axis=1)
        else:
            if X_tabular.shape[1] == 0:
                # Provide a constant feature so the estimator can still run.
                X = np.ones((len(df), 1), dtype=float)
            else:
                X = X_tabular
        X = X.astype(np.float32, copy=False)
        y_pred, mae, r2, cate = fit_and_score_cate(X, treatment, outcome, train_idx, val_idx)
        metrics = (
            f"{label_prefix} {res.label:<18} -> MAE={mae:.4f} | R^2={r2:.4f} | "
            f"CATE mean={cate.mean():.4f} std={cate.std():.4f}"
        )
        if true_cate is not None:
            pehe = compute_pehe(cate, true_cate[val_idx])
            metrics = f"{metrics} | PEHE={pehe:.4f}"
        print(
            metrics
        )


def main() -> None:
    args = parse_args()
    df = load_tabular_data(args.csv_path)
    true_cate = load_true_cate(args.csv_path)
    if true_cate is not None and len(true_cate) != len(df):
        print(
            f"Warning: cate.txt contains {len(true_cate)} rows but dataset has {len(df)}; "
            "skipping PEHE computation."
        )
        true_cate = None
    elif true_cate is None:
        print(
            f"Warning: cate.txt not found in {args.csv_path.parent}; "
            "skipping PEHE computation."
        )
    else:
        print("TRUE CATE:", true_cate.mean(), true_cate.std())

    ordered_covs = [c for c in DROP_ORDER if c in df.columns]
    other_covs = [
        c
        for c in df.columns
        if c not in {"row_id", "treatment", "outcome"} and c not in ordered_covs
    ]
    covariates = ordered_covs + other_covs
    print(f"Loaded {len(df)} rows with {len(covariates)} covariates.")

    row_ids = df["row_id"].astype(int).tolist()
    features = load_feature_matrix(args.feature_dir, row_ids)
    print(f"Feature matrix shape: {features.shape}")

    # try shuffle -> to see if the network is just learning random stuff
    # -> it is actually learning random stuff!
    # np.random.shuffle(features)

    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.random_state,
        shuffle=True,
        stratify=df["treatment"]
    )

    if args.standardize:
        print("standardize feature")
        train_features = features[train_idx]
        scaler = StandardScaler()
        scaler.fit(train_features)
        features = scaler.transform(features)

    scaler = StandardScaler()
    scaler.fit(df.loc[train_idx, covariates].to_numpy(dtype=float))
    df.loc[:, covariates] = scaler.transform(df[covariates].to_numpy(dtype=float))

    y = df["outcome"].to_numpy(dtype=float)

    treatment = df["treatment"].astype(float).to_numpy()

    print(
        f"Train/val split: {len(train_idx)} train / {len(val_idx)} val "
        f"(val frac={args.val_fraction:.2f})"
    )

    print("\nExperiments (validation MAE | R^2):")
    cov_results = covariate_masking_experiments(
        df, covariates, train_idx, val_idx, drop_order=DROP_ORDER
    )
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
        df,
        covariates,
        train_idx,
        val_idx,
        extra_features=features,
        drop_order=DROP_ORDER,
    )
    for res in cov_plus_feat:
        print(
            f"[Mask+Features] {res.label:<18} -> "
            f"MAE={res.mae:.4f} | R^2={res.r2:.4f} | "
            f"remaining={len(res.remaining_covariates)}"
        )

    print("\n[Learned linear embeddings]")
    if args.pca:
        print("Use PCA")
        embeddings = run_pca(args.embedding_dim, features, train_idx)
    elif args.joint_embedding:
        covariate_matrix = df[covariates].to_numpy(dtype=float)
        embeddings, cov_mae, outcome_mae, treatment_acc = train_multitask_embedding(
            features,
            covariate_matrix,
            y,
            treatment,
            train_idx,
            val_idx,
            embedding_dim=args.embedding_dim,
            epochs=args.embedding_epochs,
            lr=args.embedding_lr,
            random_state=args.random_state,
            curve_label="joint_multitask",
        )
        print(
            f"Joint embedding validation -> cov MAE={cov_mae:.4f} | "
            f"outcome MAE={outcome_mae:.4f} | treatment Acc={treatment_acc:.4f}"
        )
    else:
        embeddings, embed_val_mae, embed_val_r2 = train_linear_embedding(
            features,
            y,
            train_idx,
            val_idx,
            embedding_dim=args.embedding_dim,
            epochs=args.embedding_epochs,
            lr=args.embedding_lr,
            random_state=args.random_state,
            curve_label="outcome_regression",
        )
        print(
            f"Embedding model validation -> MAE={embed_val_mae:.4f} | R^2={embed_val_r2:.4f} "
            f"(dim={args.embedding_dim})"
        )

        embeddings, embed_val_mae, embed_val_r2 = train_linear_embedding(
            features,
            treatment,
            train_idx,
            val_idx,
            embedding_dim=args.embedding_dim,
            epochs=args.embedding_epochs,
            lr=args.embedding_lr,
            random_state=args.random_state,
            cls=True,
            curve_label="treatment_classification",
        )
        print(
            f"Embedding model validation -> MAE={embed_val_mae:.4f} | R^2={embed_val_r2:.4f} "
            f"(dim={args.embedding_dim})"
        )

    print("\n[Embeddings -> Covariates]")
    for cov in covariates:
        res = features_to_targets(embeddings, df[cov], train_idx, val_idx, f"embedding -> {cov}")
        print(f"{cov:<12} MAE={res.mae:.4f} | R^2={res.r2:.4f}")

    # for cov in covariates: 
    #     _, mae, r2 = train_linear_embedding(
    #         features,
    #         df[cov].to_numpy(),
    #         train_idx, 
    #         val_idx,
    #         embedding_dim=args.embedding_dim,
    #         epochs=args.embedding_epochs,
    #         lr=args.embedding_lr,
    #         random_state=args.random_state,
    #     )
    #     print(
    #         f"Embedding -> {cov} -> MAE={mae:.4f} | R^2={r2:.4f} "
    #     )

    cov_plus_embed = covariate_masking_experiments(
        df,
        covariates,
        train_idx,
        val_idx,
        extra_features=embeddings,
        drop_order=DROP_ORDER,
    )
    for res in cov_plus_embed:
        print(
            f"[Mask+Embeds] {res.label:<18} -> "
            f"MAE={res.mae:.4f} | R^2={res.r2:.4f} | "
            f"remaining={len(res.remaining_covariates)}"
        )

    print("\n[CATE: covariates only]: skip because already ran multiple times")
    # print("\n[CATE: covariates only]")
    # run_cate_masking_experiments(
    #     df,
    #     cov_results,
    #     treatment,
    #     y,
    #     train_idx,
    #     val_idx,
    #     extra_features=None,
    #     true_cate=true_cate,
    #     label_prefix="[CATE covs]",
    #     max_steps=args.cate_max_steps,
    # )

    print("\n[CATE: covariates + Dino features]: skip because already ran multiple times")
    # print("\n[CATE: covariates + Dino features]")
    # run_cate_masking_experiments(
    #     df,
    #     cov_results,
    #     treatment,
    #     y,
    #     train_idx,
    #     val_idx,
    #     extra_features=features,
    #     true_cate=true_cate,
    #     label_prefix="[CATE cov+feat]",
    #     max_steps=args.cate_max_steps,
    # )

    print("\n[CATE: covariates + embeddings]")
    run_cate_masking_experiments(
        df,
        cov_results,
        treatment,
        y,
        train_idx,
        val_idx,
        extra_features=embeddings,
        true_cate=true_cate,
        label_prefix="[CATE cov+feat]",
        max_steps=args.cate_max_steps,
    )


if __name__ == "__main__":
    main()
