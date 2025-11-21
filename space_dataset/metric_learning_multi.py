#!/usr/bin/env python3
"""Contrastive metric learning with multi-target BCE heads for treatment/outcome prediction."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "shrug"
DEFAULT_FEATURE_DIR = DEFAULT_DATA_DIR / "features"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "graph_adj.txt"
DEFAULT_CSV = DEFAULT_DATA_DIR / "full_dataset.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "metric_embeddings_multi.pt"


@dataclass
class GraphData:
    features: torch.Tensor
    labels: torch.Tensor
    node_ids: List[int]
    adjacency: List[Set[int]]


class EmbeddingMLP(nn.Module):
    """Small MLP similar to the one in metric_learning.py for learning embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


class MultiTargetMetricModel(nn.Module):
    """Shared encoder with a two-logit head for treatment/outcome."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = EmbeddingMLP(input_dim, hidden_dim, embedding_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return embeddings, logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive metric learning with multi-target BCE.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root directory of the dataset.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help="Directory that contains per-node .pt feature tensors.",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to graph_adj.txt containing adjacency lists.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV with treatment/outcome columns aligned to feature indices.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the learned embeddings (torch.save payload).",
    )
    parser.add_argument("--embedding-dim", type=int, default=32, help="Output embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden size of the MLP.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate inside the MLP.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of anchor nodes sampled per batch.")
    parser.add_argument("--positives-per-node", type=int, default=4, help="Positive samples per node.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional override for steps per epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for contrastive loss.")
    parser.add_argument("--classification-weight", type=float, default=1.0, help="Weight for BCE head loss.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Batch size for full-graph embedding export.")
    parser.add_argument("--device", type=str, default=None, help="Training device override (cpu or cuda).")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch interval for logging loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def load_feature_matrix(feature_dir: Path) -> tuple[torch.Tensor, List[int]]:
    feature_paths = sorted(feature_dir.glob("*.pt"), key=lambda p: int(p.stem))
    if not feature_paths:
        raise FileNotFoundError(f"No .pt files found inside {feature_dir}")

    features: List[torch.Tensor] = []
    node_ids: List[int] = []
    for path in feature_paths:
        node_id = int(path.stem)
        tensor = torch.load(path, map_location="cpu").float().view(-1)
        features.append(tensor)
        node_ids.append(node_id)
    stacked = torch.stack(features)
    return stacked, node_ids


def load_labels(csv_path: Path, node_ids: Sequence[int]) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "row_id"})
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df)))
    required_cols = {"treatment", "outcome"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    df = df.set_index("row_id")
    missing_rows = [node_id for node_id in node_ids if node_id not in df.index]
    if missing_rows:
        raise KeyError(f"Missing labels for node ids: {missing_rows[:5]} (and {len(missing_rows) - 5} more)")
    ordered = df.loc[node_ids, ["treatment", "outcome"]].to_numpy(dtype=np.float32)
    return torch.from_numpy(ordered)


def load_adjacency(graph_path: Path, node_to_idx: Dict[int, int], total_nodes: int) -> List[Set[int]]:
    adjacency: List[Set[int]] = [set() for _ in range(total_nodes)]
    with graph_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            node_id = int(tokens[0])
            anchor_idx = node_to_idx.get(node_id)
            if anchor_idx is None:
                continue
            for neighbor_token in tokens[1:]:
                neighbor_id = int(neighbor_token)
                neighbor_idx = node_to_idx.get(neighbor_id)
                if neighbor_idx is None or neighbor_idx == anchor_idx:
                    continue
                adjacency[anchor_idx].add(neighbor_idx)
                adjacency[neighbor_idx].add(anchor_idx)
    return adjacency


def build_graph_data(feature_dir: Path, graph_path: Path, csv_path: Path) -> GraphData:
    features, node_ids = load_feature_matrix(feature_dir)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = load_adjacency(graph_path, node_to_idx, len(node_ids))
    labels = load_labels(csv_path, node_ids)
    return GraphData(features=features, labels=labels, node_ids=node_ids, adjacency=adjacency)


def sample_batch_indices(num_nodes: int, batch_size: int, rng: random.Random) -> torch.Tensor:
    if num_nodes <= batch_size:
        indices = list(range(num_nodes))
        rng.shuffle(indices)
    else:
        indices = rng.sample(range(num_nodes), batch_size)
    return torch.tensor(indices, dtype=torch.long)


def choose_positives(
    anchor_indices: Sequence[int],
    adjacency: Sequence[Set[int]],
    positives_per_node: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    positives: Dict[int, List[int]] = {}
    for idx in anchor_indices:
        neighbors = list(adjacency[idx])
        if not neighbors:
            positives[idx] = [idx] * positives_per_node
            continue
        if len(neighbors) >= positives_per_node:
            positives[idx] = rng.sample(neighbors, positives_per_node)
        else:
            positives[idx] = [rng.choice(neighbors) for _ in range(positives_per_node)]
    return positives


def build_negative_sets(anchor_indices: Sequence[int], adjacency: Sequence[Set[int]]) -> Dict[int, List[int]]:
    batch_set = set(anchor_indices)
    negatives: Dict[int, List[int]] = {}
    for idx in anchor_indices:
        neighbor_set = adjacency[idx]
        neg_candidates = [other for other in batch_set if other != idx and other not in neighbor_set]
        if neg_candidates:
            negatives[idx] = neg_candidates
    return negatives


def compute_similarity(anchor: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    return -torch.sum((anchor.unsqueeze(0) - others) ** 2, dim=1)


def build_lookup(
    model: MultiTargetMetricModel,
    features: torch.Tensor,
    required_indices: Iterable[int],
    device: torch.device,
) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    required_list = sorted(set(required_indices))
    if not required_list:
        return {}
    required_tensor = torch.tensor(required_list, dtype=torch.long)
    embed_inputs = features.index_select(0, required_tensor).to(device)
    embeddings, logits = model(embed_inputs)
    return {
        idx: (embeddings[i], logits[i])
        for i, idx in enumerate(required_list)
    }


def contrastive_loss(
    lookup: Dict[int, tuple[torch.Tensor, torch.Tensor]],
    batch_indices: torch.Tensor,
    positives: Dict[int, List[int]],
    negatives: Dict[int, List[int]],
    temperature: float,
    device: torch.device,
) -> torch.Tensor | None:
    total_loss = torch.tensor(0.0, device=device)
    valid_nodes = 0
    for anchor_idx in batch_indices.tolist():
        neg_list = negatives.get(anchor_idx)
        pos_list = positives.get(anchor_idx)
        if not neg_list or not pos_list:
            continue
        anchor_entry = lookup.get(anchor_idx)
        if anchor_entry is None:
            continue
        anchor_emb = anchor_entry[0]
        pos_embs = [lookup[pos][0] for pos in pos_list if pos in lookup]
        neg_embs = [lookup[neg][0] for neg in neg_list if neg in lookup]
        if not pos_embs or not neg_embs:
            continue
        positive_embs = torch.stack(pos_embs)
        negative_embs = torch.stack(neg_embs)

        pos_scores = compute_similarity(anchor_emb, positive_embs)
        neg_scores = compute_similarity(anchor_emb, negative_embs)

        numerator = torch.sum(torch.exp(pos_scores / temperature))
        denominator = numerator + torch.sum(torch.exp(neg_scores / temperature))
        total_loss = total_loss - torch.log((numerator / denominator) + 1e-8)
        valid_nodes += 1

    if valid_nodes == 0:
        return None
    return total_loss / valid_nodes


def multitask_bce_loss(
    lookup: Dict[int, tuple[torch.Tensor, torch.Tensor]],
    batch_indices: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> torch.Tensor | None:
    logits: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for idx in batch_indices.tolist():
        entry = lookup.get(idx)
        if entry is None:
            continue
        logits.append(entry[1])
        targets.append(labels[idx].to(device))
    if not logits:
        return None
    logit_tensor = torch.stack(logits)
    target_tensor = torch.stack(targets)
    return loss_fn(logit_tensor, target_tensor)


def train_model(args: argparse.Namespace, graph_data: GraphData) -> tuple[MultiTargetMetricModel, torch.device]:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_nodes, input_dim = graph_data.features.shape
    model = MultiTargetMetricModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss()
    labels = graph_data.labels.to(device)

    steps_per_epoch = args.steps_per_epoch or max(1, num_nodes // args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_contrastive = 0.0
        epoch_bce = 0.0
        counted_steps = 0

        for _ in range(steps_per_epoch):
            batch_indices = sample_batch_indices(num_nodes, args.batch_size, rng)
            positives = choose_positives(batch_indices.tolist(), graph_data.adjacency, args.positives_per_node, rng)
            negatives = build_negative_sets(batch_indices.tolist(), graph_data.adjacency)

            required: Set[int] = set(batch_indices.tolist())
            for pos_list in positives.values():
                required.update(pos_list)
            for neg_list in negatives.values():
                required.update(neg_list)

            optimizer.zero_grad()
            lookup = build_lookup(model, graph_data.features, required, device)
            contrastive = contrastive_loss(
                lookup=lookup,
                batch_indices=batch_indices,
                positives=positives,
                negatives=negatives,
                temperature=args.temperature,
                device=device,
            )
            bce = multitask_bce_loss(
                lookup=lookup,
                batch_indices=batch_indices,
                labels=labels,
                loss_fn=bce_loss,
                device=device,
            )

            if contrastive is None and bce is None:
                continue

            loss_terms: List[torch.Tensor] = []
            if contrastive is not None:
                loss_terms.append(contrastive)
            if bce is not None:
                loss_terms.append(args.classification_weight * bce)
            loss = torch.stack(loss_terms).sum()
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if contrastive is not None:
                epoch_contrastive += float(contrastive.detach().cpu())
            if bce is not None:
                epoch_bce += float(bce.detach().cpu())
            counted_steps += 1

        if counted_steps == 0:
            raise RuntimeError("No valid batches were produced; check graph connectivity and batch sampling.")

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            avg_contrastive = epoch_contrastive / counted_steps if counted_steps else 0.0
            avg_bce = epoch_bce / counted_steps if counted_steps else 0.0
            print(
                f"Epoch {epoch:03d}/{args.epochs}: contrastive={avg_contrastive:.4f} "
                f"bce={avg_bce:.4f} (weight={args.classification_weight})"
            )

    return model, device


def export_embeddings(
    model: MultiTargetMetricModel,
    device: torch.device,
    features: torch.Tensor,
    node_ids: Sequence[int],
    output_path: Path,
    batch_size: int,
) -> None:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = features[start:end].to(device)
            batch_embeddings, batch_logits = model(batch)
            embeddings.append(batch_embeddings)
            logits_list.append(batch_logits)
    stacked_embeddings = torch.cat(embeddings).cpu()
    stacked_logits = torch.cat(logits_list).cpu()
    torch.save(
        {"node_ids": list(node_ids), "embeddings": stacked_embeddings, "logits": stacked_logits},
        output_path,
    )
    print(f"Saved embeddings and logits for {len(node_ids)} nodes to {output_path}")


def main() -> None:
    args = parse_args()
    graph_data = build_graph_data(args.feature_dir, args.graph_path, args.csv_path)
    model, device = train_model(args, graph_data)
    export_embeddings(
        model=model,
        device=device,
        features=graph_data.features,
        node_ids=graph_data.node_ids,
        output_path=args.output_path,
        batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()
