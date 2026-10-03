"""ResNet18 comparison on the saved reproduction split; select on validation only.

Run from the project root:
    .venv/Scripts/python.exe -B -u week3/reproduction/resnet/resnet_experiment.py --until 5
    .venv/Scripts/python.exe -B -u week3/reproduction/resnet/resnet_experiment.py --until 10 --resume
    .venv/Scripts/python.exe -B -u week3/reproduction/resnet/resnet_experiment.py --evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(REPRO))
from train_densenet_reproduction import (  # noqa: E402
    BATCH_SIZE, CLASSES, LEARNING_RATE, PROJECT, SEED, load_dataset,
    set_reproducible_state, validate,
)

HISTORY = HERE / "training_history.csv"
BEST = HERE / "resnet18_best.pth"
LATEST = HERE / "resnet18_latest.pth"
FIELDS = ("epoch", "train_loss", "train_accuracy", "validation_accuracy", "validation_macro_f1", "epoch_seconds", "elapsed_seconds")
RESULTS = ("test_metrics.csv", "test_summary.txt", "test_confusion_matrix.png", "test_class_f1.png")


def manifest_files() -> dict[str, list[Path]]:
    manifest = json.loads((REPRO / "split_manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split manifest keys")
    if [len(manifest[name]) for name in ("train", "validation", "test")] != [160, 40, 50]:
        raise ValueError("Unexpected split sizes")
    all_names = [name for group in manifest.values() for name in group]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Train, validation, and test share a CSV")
    groups = {
        name: [(PROJECT / Path(raw.replace("\\", "/"))).resolve() for raw in manifest[name]]
        for name in ("train", "validation", "test")
    }
    for name, files in groups.items():
        if any(not path.is_relative_to(PROJECT) or not path.is_file() for path in files):
            raise ValueError(f"Missing or out-of-project {name} CSV")
        expected = 32 if name == "train" else 8 if name == "validation" else 10
        if [sum(path.parent.name == label for path in files) for label in CLASSES] != [expected] * 5:
            raise ValueError(f"Unexpected class counts in {name}")
    return groups


def model_on(device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(device)


def loader(files: list[Path], shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    x, y = load_dataset(files)  # Same Week 2 windows/min-max/CWT; no duplicate filtering.
    if x.shape[1:] != (3, 32, 300):
        raise ValueError(f"Unexpected CWT shape: {x.shape}")
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator, num_workers=0,
    )


def save_history(rows: list[dict]) -> None:
    with HISTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train(until: int, resume: bool) -> None:
    if until not in (5, 10):
        raise ValueError("Only 5- and 10-epoch checkpoints are planned")
    if not resume and (HISTORY.exists() or LATEST.exists() or BEST.exists()):
        raise FileExistsError("Existing ResNet18 training results; use --resume")
    set_reproducible_state()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
    print("device:", device, flush=True)
    groups = manifest_files()
    print("CSV split: 160/40/50; no overlap. Loading train and validation only.", flush=True)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = loader(groups["train"], True, generator)
    validation_loader = loader(groups["validation"], False)
    model = model_on(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict] = []
    best = {"epoch": 0, "accuracy": -1.0, "macro_f1": -1.0}
    elapsed_before = 0.0
    start_epoch = 1
    if resume:
        checkpoint = torch.load(LATEST, map_location="cpu", weights_only=True)
        if checkpoint["seed"] != SEED or checkpoint["split_manifest"] != "../split_manifest.json":
            raise ValueError("Checkpoint does not match the saved split")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device.type == "cuda":
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        random.setstate(checkpoint["python_rng_state"])
        state = checkpoint["numpy_rng_state"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        generator.set_state(checkpoint["loader_rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        elapsed_before = checkpoint["elapsed_seconds"]
        with HISTORY.open(newline="", encoding="utf-8") as handle:
            rows = [{key: int(row[key]) if key == "epoch" else float(row[key]) for key in FIELDS}
                    for row in csv.DictReader(handle) if int(row["epoch"]) < start_epoch]
        if len(rows) != start_epoch - 1:
            raise ValueError("History does not match latest checkpoint")
        save_history(rows)
        print(f"Resuming after epoch {start_epoch - 1}", flush=True)
    if until < start_epoch:
        raise ValueError("Requested endpoint already passed")
    run_start = time.monotonic()
    for epoch in range(start_epoch, until + 1):
        epoch_start = time.monotonic()
        model.train()
        loss_sum = correct = total = 0
        for batch, (signals, labels) in enumerate(train_loader, 1):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signals)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((output.argmax(1) == labels).sum())
            total += len(labels)
            if batch % 50 == 0:
                print(f"epoch={epoch} batch={batch}/{len(train_loader)}", flush=True)
        accuracy, macro_f1 = validate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "validation_accuracy": accuracy, "validation_macro_f1": macro_f1,
               "epoch_seconds": time.monotonic() - epoch_start,
               "elapsed_seconds": elapsed_before + time.monotonic() - run_start}
        rows.append(row)
        if (accuracy, macro_f1) > (best["accuracy"], best["macro_f1"]):
            best = {"epoch": epoch, "accuracy": accuracy, "macro_f1": macro_f1}
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best": best,
                        "seed": SEED, "split_manifest": "../split_manifest.json"}, BEST)
        save_history(rows)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "best": best,
                      "elapsed_seconds": row["elapsed_seconds"], "seed": SEED,
                      "split_manifest": "../split_manifest.json", "python_rng_state": random.getstate(),
                      "numpy_rng_state": (lambda s: (s[0], s[1].tolist(), s[2], s[3], s[4]))(np.random.get_state()),
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                      "loader_rng_state": generator.get_state()}
        temporary = HERE / "resnet18_latest.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(LATEST)
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} train_accuracy={row['train_accuracy']:.4f} "
              f"validation_accuracy={accuracy:.4f} validation_macro_f1={macro_f1:.4f} "
              f"seconds={row['epoch_seconds']:.1f} best_epoch={best['epoch']}", flush=True)
    print(f"Completed through epoch {until}; held-out test was not evaluated.", flush=True)


def evaluate() -> None:
    existing = [name for name in RESULTS if (HERE / name).exists()]
    if existing:
        raise FileExistsError("Refusing repeat test evaluation or overwrite: " + ", ".join(existing))
    with (PROJECT / "week3" / "model_comparison.csv").open(newline="", encoding="utf-8") as handle:
        baseline = next(float(row["Accuracy"]) for row in csv.DictReader(handle) if row["Model"] == "ResNet18")
    set_reproducible_state()
    groups = manifest_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BEST, map_location="cpu", weights_only=True)
    if checkpoint["seed"] != SEED or checkpoint["split_manifest"] != "../split_manifest.json":
        raise ValueError("Best checkpoint does not match the saved split")
    best = checkpoint["best"]
    if checkpoint["epoch"] != best["epoch"]:
        raise ValueError("Best checkpoint metadata mismatch")
    model = model_on(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    test_loader = loader(groups["test"], False)
    actual, predicted = [], []
    with torch.inference_mode():
        for signals, labels in test_loader:
            actual.append(labels.numpy())
            predicted.append(model(signals.to(device)).argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(actual), np.concatenate(predicted)
    labels = list(range(len(CLASSES)))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    errors = int(off_diagonal.max())
    pairs = [(CLASSES[i], CLASSES[j]) for i, j in zip(*np.where(off_diagonal == errors)) if i != j] if errors else []

    with (HERE / RESULTS[0]).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Model", "Best epoch", "Validation Accuracy", "Accuracy", "Precision macro", "Recall macro", "F1 macro", *(f"F1 {name}" for name in CLASSES)))
        writer.writerow(("ResNet18", best["epoch"], best["accuracy"], accuracy, precision, recall, f1, *class_f1))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of CWT windows")
    ax.set(title="ResNet18 Held-out Test Confusion Matrix", xlabel="Predicted user", ylabel="Actual user",
           xticks=labels, yticks=labels, xticklabels=CLASSES, yticklabels=CLASSES)
    threshold = matrix.max() / 2
    for row in labels:
        for col in labels:
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(HERE / RESULTS[2], dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(CLASSES, class_f1, color="#c86b38")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title="ResNet18 Held-out Test F1 by User", xlabel="User class", ylabel="F1-score", ylim=(0, 1.08))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / RESULTS[3], dpi=170)
    plt.close(fig)
    with HISTORY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    training_seconds = float(rows[-1]["elapsed_seconds"])
    summary = (
        f"Best epoch: {best['epoch']}\nValidation Accuracy: {best['accuracy']:.4f}\n"
        f"Validation macro F1: {best['macro_f1']:.4f}\nTest Accuracy: {accuracy:.4f}\n"
        f"Test Precision macro: {precision:.4f}\nTest Recall macro: {recall:.4f}\nTest F1 macro: {f1:.4f}\n"
        + "Class F1: " + ", ".join(f"{name}={value:.4f}" for name, value in zip(CLASSES, class_f1)) + "\n"
        + ("Most frequent error: " + "; ".join(f"actual {a} -> predicted {b}: {errors}" for a, b in pairs) + "\n" if pairs else "Most frequent error: none\n")
        + f"Accuracy change vs. original 1-epoch ResNet18 ({baseline:.4f}): {(accuracy - baseline) * 100:+.2f} percentage points\n"
        + f"Test CSV files: {len(groups['test'])}; CWT windows: {len(y_true)}\n"
        + f"Training epoch time: {training_seconds:.1f} seconds\n"
        + "Macro averaging; counts and scores are per CWT window. Saved split_manifest.json; no duplicate filtering.\n"
        + "The original 1-epoch experiment used its previous preprocessing, so the change is descriptive rather than an isolated effect of training duration.\n"
    )
    (HERE / RESULTS[1]).write_text(summary, encoding="utf-8")
    print(summary, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--until", type=int, choices=(5, 10))
    group.add_argument("--evaluate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        if args.resume:
            parser.error("--resume is only for training")
        evaluate()
    else:
        train(args.until, args.resume)


if __name__ == "__main__":
    main()
