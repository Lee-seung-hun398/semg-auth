"""One-time held-out test evaluation of the selected reproduction checkpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_densenet_reproduction import (
    BATCH_SIZE, CHECKPOINTS, CLASSES, OUT, PROJECT, SEED,
    build_model, load_dataset, set_reproducible_state,
)

RESULTS = (
    "test_metrics.csv", "test_summary.txt", "test_confusion_matrix.png",
    "test_class_f1.png",
)


def test_files_from_manifest() -> list[Path]:
    manifest = json.loads((OUT / "split_manifest.json").read_text(encoding="utf-8"))
    groups = {name: manifest[name] for name in ("train", "validation", "test")}
    if tuple(map(len, groups.values())) != (160, 40, 50):
        raise ValueError("Unexpected split sizes in saved manifest")
    all_names = [name for group in groups.values() for name in group]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Overlapping CSVs in saved manifest")
    test = [(PROJECT / Path(name.replace("\\", "/"))).resolve() for name in groups["test"]]
    if any(not path.is_file() or not path.is_relative_to(PROJECT) for path in test):
        raise ValueError("Missing or out-of-project test CSV in saved manifest")
    if [sum(path.parent.name == label for path in test) for label in CLASSES] != [10] * 5:
        raise ValueError("Saved test split is not 10 trials per user")
    # Read the saved split only. No random split is regenerated here.
    return test


def main() -> None:
    existing = [str(OUT / name) for name in RESULTS if (OUT / name).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing test results: " + ", ".join(existing))

    set_reproducible_state()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        CHECKPOINTS / "densenet_best.pth", map_location="cpu", weights_only=True
    )
    if checkpoint.get("seed") != SEED or checkpoint.get("split_manifest") != "split_manifest.json":
        raise ValueError("Checkpoint seed or split does not match the reproduction experiment")
    best = checkpoint["best"]
    if int(checkpoint["epoch"]) != int(best["epoch"]):
        raise ValueError("Best checkpoint epoch metadata is inconsistent")
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    files = test_files_from_manifest()
    x_test, y_test = load_dataset(files)
    if x_test.shape[1:] != (3, 32, 300):
        raise ValueError(f"Unexpected CWT input shape: {x_test.shape}")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    predictions = []
    with torch.inference_mode():
        for signals, _ in loader:
            predictions.append(model(signals.to(device)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(predictions)
    labels = list(range(len(CLASSES)))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    class_f1 = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    with (OUT / RESULTS[0]).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Model", "Best epoch", "Validation Accuracy", "Accuracy", "Precision macro", "Recall macro", "F1 macro", *(f"F1 {name}" for name in CLASSES)))
        writer.writerow(("DenseNet161", best["epoch"], best["accuracy"], accuracy, precision, recall, f1, *class_f1))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of CWT windows")
    ax.set(title="DenseNet161 Held-out Test Confusion Matrix", xlabel="Predicted user", ylabel="Actual user",
           xticks=labels, yticks=labels, xticklabels=CLASSES, yticklabels=CLASSES)
    threshold = matrix.max() / 2
    for row in labels:
        for col in labels:
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(OUT / RESULTS[2], dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(CLASSES, class_f1, color="#3a78b4")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title="DenseNet161 Held-out Test F1 by User", xlabel="User class", ylabel="F1-score", ylim=(0, 1.08))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / RESULTS[3], dpi=170)
    plt.close(fig)

    summary = (
        f"Best epoch: {best['epoch']}\n"
        f"Validation Accuracy: {best['accuracy']:.4f}\n"
        f"Test Accuracy: {accuracy:.4f}\n"
        f"Test Precision: {precision:.4f} (macro)\n"
        f"Test Recall: {recall:.4f} (macro)\n"
        f"Test F1: {f1:.4f} (macro)\n"
        f"94% 기준 충족 여부: {'충족' if accuracy >= 0.94 else '미충족'}\n"
        f"Test CSV files: {len(files)}; test CWT windows: {len(y_test)}\n"
        "Preprocessing: released filtered CSV; no second notch/band-pass; "
        "300-sample window, 150-sample hop, min-max, Morlet CWT; input (3, 32, 300).\n"
        + "Class F1: " + ", ".join(f"{name}={value:.4f}" for name, value in zip(CLASSES, class_f1)) + "\n"
    )
    (OUT / RESULTS[1]).write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
