#!/usr/bin/env python3
"""Plot CATE summaries using the cached linear_fit_*.txt logs."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATTERN = "linear_fit_result*.txt"
DEFAULT_CATE_PATH = (
    REPO_ROOT / "data" / "cdcsvi_nohsdp_poverty_disc" / "cate.txt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "downloads" / "plots"

SECTION_ALIASES = {
    "covariates only": "covariates only",
    "covariates + dino features": "covariates + Dino features",
    "covariates + embeddings": "covariates + Embeddings",
}

CATE_LINE = re.compile(
    r"\[(?P<prefix>CATE[^\]]+)\]\s*"
    r"(?P<label>[^-]+?)\s*->\s*"
    r"MAE=(?P<mae>[-+]?\d*\.\d+|\d+)\s*\|\s*"
    r"R\^2=(?P<r2>[-+]?\d*\.\d+|\d+)\s*\|\s*"
    r"CATE mean=(?P<mean>[-+]?\d*\.\d+|\d+)\s*std=(?P<std>[-+]?\d*\.\d+|\d+)"
    r"(?:\s*\|\s*PEHE=(?P<pehe>[-+]?\d*\.\d+|\d+))?"
)


@dataclass
class CateEntry:
    label: str
    mae: float
    r2: float
    mean: float
    std: float
    pehe: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw CATE distributions/PEHE using linear_fit logs."
    )
    parser.add_argument(
        "--log-pattern",
        default=DEFAULT_LOG_PATTERN,
        help="Glob for linear_fit_result*.txt files (default: %(default)s).",
    )
    parser.add_argument(
        "--cate-path",
        type=Path,
        default=DEFAULT_CATE_PATH,
        help="Ground-truth cate.txt path (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots (default: %(default)s).",
    )
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        return fh.readlines()


def parse_cate_logs(pattern: str) -> Dict[str, List[CateEntry]]:
    sections: Dict[str, List[CateEntry]] = {alias: [] for alias in SECTION_ALIASES.values()}
    log_files = sorted(REPO_ROOT.glob(pattern))
    if not log_files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}'")

    current_section: str | None = None
    seen_labels: Dict[str, set[str]] = {alias: set() for alias in sections}

    for log_file in log_files:
        for raw_line in read_lines(log_file):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[CATE:") and "]" in line:
                section_name = line.strip()[1:-1]  # remove brackets
                label = section_name.replace("CATE:", "").strip().lower()
                current_section = SECTION_ALIASES.get(label)
                continue
            match = CATE_LINE.match(line)
            if not match or current_section is None:
                continue
            entry = CateEntry(
                label=match.group("label").strip(),
                mae=float(match.group("mae")),
                r2=float(match.group("r2")),
                mean=float(match.group("mean")),
                std=float(match.group("std")),
                pehe=float(match.group("pehe")) if match.group("pehe") else None,
            )
            # Keep only the first occurrence for each label within a section.
            if entry.label not in seen_labels[current_section]:
                sections[current_section].append(entry)
                seen_labels[current_section].add(entry.label)
    return sections


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0:
        pdf = np.zeros_like(x)
        pdf[np.argmin(np.abs(x - mean))] = 1.0
        return pdf
    coef = 1.0 / (std * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coef * np.exp(exponent)


def plot_distributions(
    gt_cate: np.ndarray,
    baseline_stats: Dict[str, CateEntry],
    output_dir: Path,
) -> Path:
    ensure_output_dir(output_dir)
    labels = list(baseline_stats.keys())
    # Determine global range.
    mins = [gt_cate.min()]
    maxs = [gt_cate.max()]
    for entry in baseline_stats.values():
        mins.append(entry.mean - 4 * entry.std)
        maxs.append(entry.mean + 4 * entry.std)
    x = np.linspace(min(mins), max(maxs), 400)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gt_cate, bins=40, density=True, alpha=0.4, label="Ground-truth CATE")
    colors = plt.cm.tab10.colors
    for idx, label in enumerate(labels):
        entry = baseline_stats[label]
        ax.plot(
            x,
            gaussian_pdf(x, entry.mean, entry.std),
            label=f"{label} (mean={entry.mean:.2f}, std={entry.std:.2f})",
            color=colors[idx % len(colors)],
        )
    ax.set_xlabel("CATE value")
    ax.set_ylabel("Density")
    ax.set_title("CATE distributions")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "cate_distributions.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def align_labels(entries: Dict[str, List[CateEntry]]) -> Sequence[str]:
    """Use the first section with data to define the label ordering."""
    for section_entries in entries.values():
        if section_entries:
            return [entry.label for entry in section_entries]
    return []


def plot_pehe(
    entries: Dict[str, List[CateEntry]],
    output_dir: Path,
) -> Path:
    ensure_output_dir(output_dir)
    labels = align_labels(entries)
    if not labels:
        raise ValueError("No CATE entries with PEHE values found.")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    for idx, (section, section_entries) in enumerate(entries.items()):
        pehe_values = {entry.label: entry.pehe for entry in section_entries if entry.pehe is not None}
        if not pehe_values:
            continue
        y = [pehe_values.get(label, np.nan) for label in labels]
        ax.plot(
            x,
            y,
            marker="o",
            label=section,
            color=colors[idx % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("PEHE")
    ax.set_xlabel("Dropped features")
    ax.set_title("PEHE as features are masked")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "pehe_curve.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def load_ground_truth(cate_path: Path) -> np.ndarray:
    if not cate_path.exists():
        raise FileNotFoundError(f"cate.txt not found at {cate_path}")
    values = np.loadtxt(cate_path, dtype=float)
    return values.reshape(-1)


def main() -> None:
    args = parse_args()
    entries = parse_cate_logs(args.log_pattern)
    gt_cate = load_ground_truth(args.cate_path)

    # Collect baseline statistics for each requested section.
    baseline_stats = {}
    for section, section_entries in entries.items():
        if section_entries:
            baseline_stats[section] = section_entries[0]
    if len(baseline_stats) < 3:
        missing = [s for s in SECTION_ALIASES.values() if s not in baseline_stats]
        raise ValueError(f"Missing baseline entries for sections: {missing}")

    output_dir = ensure_output_dir(args.output_dir)
    dist_path = plot_distributions(gt_cate, baseline_stats, output_dir)
    pehe_path = plot_pehe(entries, output_dir)
    print(f"CATE distributions saved to {dist_path}")
    print(f"PEHE curve saved to {pehe_path}")


if __name__ == "__main__":
    main()
