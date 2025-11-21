#!/usr/bin/env python3
"""Train a contrastive embedding on the CDC SVI graph using Sentinel features."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent
# DEFAULT_DATA_DIR = REPO_ROOT / "data" / "cdcsvi_nohsdp_poverty_disc"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "shrug"
# DEFAULT_FEATURE_DIR = DEFAULT_DATA_DIR / "features_dinov3_vitl16"
DEFAULT_FEATURE_DIR = DEFAULT_DATA_DIR / "features"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "graph_adj.txt"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "metric_embeddings.pt"

@dataclass
class GraphData:
    features: torch.Tensor
    node_ids: List[int]
    adjacency: List[Set[int]]


class EmbeddingMLP(nn.Module):
    """Small MLP similar to the one in linear_fit.py for learning embeddings."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive metric learning on CDC SVI graph.")
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


def build_graph_data(feature_dir: Path, graph_path: Path) -> GraphData:
    features, node_ids = load_feature_matrix(feature_dir)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = load_adjacency(graph_path, node_to_idx, len(node_ids))
    return GraphData(features=features, node_ids=node_ids, adjacency=adjacency)


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


def contrastive_loss(
    model: nn.Module,
    features: torch.Tensor,
    batch_indices: torch.Tensor,
    positives: Dict[int, List[int]],
    negatives: Dict[int, List[int]],
    temperature: float,
    device: torch.device,
) -> torch.Tensor | None:
    required: Set[int] = set(batch_indices.tolist())
    for pos_list in positives.values():
        required.update(pos_list)
    if not required:
        return None

    required_tensor = torch.tensor(sorted(required), dtype=torch.long)
    embed_inputs = features.index_select(0, required_tensor).to(device)
    embeddings = model(embed_inputs)
    idx_lookup = {idx: embeddings[i] for i, idx in enumerate(required_tensor.tolist())}

    total_loss = torch.zeros(1, device=device)
    valid_nodes = 0
    for anchor_idx in batch_indices.tolist():
        neg_list = negatives.get(anchor_idx)
        pos_list = positives.get(anchor_idx)
        if not neg_list or not pos_list:
            continue
        anchor_emb = idx_lookup[anchor_idx]
        positive_embs = torch.stack([idx_lookup[pos] for pos in pos_list])
        negative_embs = torch.stack([idx_lookup[neg] for neg in neg_list])

        pos_scores = compute_similarity(anchor_emb, positive_embs)
        neg_scores = compute_similarity(anchor_emb, negative_embs)

        numerator = torch.sum(torch.exp(pos_scores / temperature))
        denominator = numerator + torch.sum(torch.exp(neg_scores / temperature))
        loss_i = -torch.log((numerator / denominator) + 1e-8)
        total_loss = total_loss + loss_i
        valid_nodes += 1

    if valid_nodes == 0:
        return None
    return total_loss / valid_nodes


def train_model(args: argparse.Namespace, graph_data: GraphData) -> tuple[nn.Module, torch.device]:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_nodes, input_dim = graph_data.features.shape
    model = EmbeddingMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = args.steps_per_epoch or max(1, num_nodes // args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        counted_steps = 0
        for _ in range(steps_per_epoch):
            batch_indices = sample_batch_indices(num_nodes, args.batch_size, rng)
            positives = choose_positives(batch_indices.tolist(), graph_data.adjacency, args.positives_per_node, rng)
            negatives = build_negative_sets(batch_indices.tolist(), graph_data.adjacency)

            optimizer.zero_grad()
            loss = contrastive_loss(
                model=model,
                features=graph_data.features,
                batch_indices=batch_indices,
                positives=positives,
                negatives=negatives,
                temperature=args.temperature,
                device=device,
            )
            if loss is None:
                continue
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            counted_steps += 1

        if counted_steps == 0:
            raise RuntimeError("No valid batches were produced; check graph connectivity and batch sampling.")

        avg_loss = epoch_loss / counted_steps
        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs}: contrastive loss={avg_loss:.4f}")

    return model, device


def export_embeddings(
    model: nn.Module,
    device: torch.device,
    features: torch.Tensor,
    node_ids: Sequence[int],
    output_path: Path,
    batch_size: int,
) -> None:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = features[start:end].to(device)
            embeddings.append(model(batch))
    stacked = torch.cat(embeddings).cpu()
    torch.save({"node_ids": list(node_ids), "embeddings": stacked}, output_path)
    print(f"Saved embeddings for {len(node_ids)} nodes to {output_path}")


def main() -> None:
    args = parse_args()
    graph_data = build_graph_data(args.feature_dir, args.graph_path)
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
