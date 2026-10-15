"""Build submission figures from the two saved, one-time test evaluations.

This script reads saved metrics and copies existing confusion matrices. It does
not load data, checkpoints, or evaluate either model again.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
REPRO = OUT.parent / "reproduction"
SOURCES = (
    ("DenseNet161", REPRO / "test_metrics.csv", REPRO / "test_confusion_matrix.png", "confusion_matrix_densenet161.png"),
    ("ResNet18", REPRO / "resnet" / "test_metrics.csv", REPRO / "resnet" / "test_confusion_matrix.png", "confusion_matrix_resnet18.png"),
)
OUTPUTS = (
    "model_comparison.csv", "model_comparison.png", "confusion_matrix_densenet161.png",
    "confusion_matrix_resnet18.png", "class_f1_comparison.png", "evaluation_summary.txt",
)
METRICS = ("Accuracy", "Precision macro", "Recall macro", "F1 macro")
CLASSES = ("A", "B", "C", "D", "E")


def main() -> None:
    existing = [name for name in OUTPUTS if (OUT / name).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite final results: " + ", ".join(existing))
    rows = []
    for model, csv_path, image_path, target in SOURCES:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != 1 or source_rows[0]["Model"] != model:
            raise ValueError(f"Unexpected saved metrics: {csv_path}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(source_rows[0])
    if set(rows[0]) != set(rows[1]):
        raise ValueError("Saved metric schemas differ")
    if not (REPRO / "resnet" / "test_summary.txt").is_file():
        raise FileNotFoundError("Saved ResNet18 test summary is missing")

    with (OUT / "model_comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, row in enumerate(rows):
        values = [float(row[key]) for key in METRICS]
        bars = ax.bar(x + (index - 0.5) * 0.35, values, width=0.35, label=row["Model"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set(title="DenseNet161 vs ResNet18: Held-out Test Performance", xlabel="Evaluation metric",
           ylabel="Score", xticks=x, xticklabels=("Accuracy", "Precision", "Recall", "F1"), ylim=(0, 1.05))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=170)
    plt.close(fig)

    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, row in enumerate(rows):
        values = [float(row[f"F1 {name}"]) for name in CLASSES]
        bars = ax.bar(x + (index - 0.5) * 0.35, values, width=0.35, label=row["Model"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set(title="DenseNet161 vs ResNet18: Held-out Test F1 by User", xlabel="User class",
           ylabel="F1-score", xticks=x, xticklabels=CLASSES, ylim=(0, 1.08))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "class_f1_comparison.png", dpi=170)
    plt.close(fig)

    for _, _, source, target in SOURCES:
        shutil.copyfile(source, OUT / target)

    dense, resnet = rows
    delta = (float(dense["Accuracy"]) - float(resnet["Accuracy"])) * 100
    resnet_summary = (REPRO / "resnet" / "test_summary.txt").read_text(encoding="utf-8")
    error_line = next(line for line in resnet_summary.splitlines() if line.startswith("Most frequent error:"))
    summary = [
        "Final same-condition sEMG user classification comparison",
        "Saved 160/40/50 trial-file split; seed 42; five users A-E; 950 held-out test CWT windows.",
        "Released filtered CSVs; no duplicate notch/band-pass; 300-sample window, 150-sample hop, min-max, Morlet CWT; input (3, 32, 300).",
        "Validation selected the best checkpoint; the saved one-time held-out Test results were used without retraining or reevaluation.",
        "Precision, Recall, and F1 use macro averaging; scores are per CWT window.",
    ]
    for row in rows:
        summary.append(
            f"{row['Model']}: best epoch {row['Best epoch']}, validation accuracy {float(row['Validation Accuracy']):.4f}, "
            f"test accuracy {float(row['Accuracy']):.4f}, precision {float(row['Precision macro']):.4f}, "
            f"recall {float(row['Recall macro']):.4f}, F1 {float(row['F1 macro']):.4f}."
        )
        summary.append("Class F1: " + ", ".join(f"{name}={float(row[f'F1 {name}']):.4f}" for name in CLASSES))
    summary.extend((
        f"DenseNet161 test accuracy advantage: {delta:.2f} percentage points.",
        error_line,
        "DenseNet161 did not meet the 94% test accuracy reproduction target (actual 83.89%).",
        "The original 5-epoch DenseNet161 and 1-epoch ResNet18 results are initial experiments with different preprocessing, not this matched-condition comparison.",
    ))
    (OUT / "evaluation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Created {len(OUTPUTS)} final artifacts; DenseNet161 advantage {delta:.2f} percentage points")


if __name__ == "__main__":
    main()
