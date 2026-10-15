"""Train DenseNet161 on the saved 200-file train set and evaluate epochs 5/10.

No validation split or best-checkpoint selection is used. The held-out test
files are loaded only after epoch 10 has completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPRO = PROJECT / "week3" / "reproduction"
sys.path.insert(0, str(REPRO))
from train_densenet_reproduction import (  # noqa: E402
    BATCH_SIZE, CLASSES, LEARNING_RATE, SEED, build_model, load_dataset,
    set_reproducible_state,
)

CHECKPOINTS = HERE / "checkpoints"
HISTORY = HERE / "training_history.csv"
METRICS = HERE / "test_metrics.csv"
SUMMARY = HERE / "test_summary.txt"
LATEST = CHECKPOINTS / "densenet_latest.pth"
FIELDS = ("epoch", "train_loss", "train_accuracy", "epoch_seconds", "elapsed_seconds")


def split_files() -> tuple[list[Path], list[Path]]:
    manifest = json.loads((REPRO / "split_manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"train", "validation", "test"}:
        raise ValueError("Unexpected reproduction manifest keys")
    train_names = manifest["train"] + manifest["validation"]
    test_names = manifest["test"]
    if (len(train_names), len(test_names)) != (200, 50):
        raise ValueError("Expected the saved 200/50 file split")
    if set(train_names) & set(test_names) or len(set(train_names + test_names)) != 250:
        raise ValueError("Train/Test overlap or duplicate CSV in manifest")
    convert = lambda names: [(PROJECT / Path(name.replace("\\", "/"))).resolve() for name in names]
    train, test = convert(train_names), convert(test_names)
    for group_name, files, expected in (("train", train, 40), ("test", test, 10)):
        if any(not path.is_file() or not path.is_relative_to(PROJECT) for path in files):
            raise ValueError(f"Missing or out-of-project {group_name} CSV")
        if [sum(path.parent.name == label for path in files) for label in CLASSES] != [expected] * 5:
            raise ValueError(f"Unexpected class balance in {group_name}")
    return train, test


def save_history(rows: list[dict]) -> None:
    with HISTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                       generator: torch.Generator, elapsed: float) -> dict:
    state = np.random.get_state()
    return {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "seed": SEED,
        "source_manifest": "../reproduction/split_manifest.json", "train_csv_count": 200,
        "test_csv_count": 50, "elapsed_seconds": elapsed,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng_state": generator.get_state(),
    }


def train(resume: bool) -> None:
    CHECKPOINTS.mkdir(exist_ok=True)
    if not resume and (LATEST.exists() or HISTORY.exists() or METRICS.exists() or SUMMARY.exists()):
        raise FileExistsError("Full-train artifacts already exist; use --resume only for interrupted training")
    set_reproducible_state()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
    print("device:", device, flush=True)
    train_files, _ = split_files()
    print("Split verified: train=200, test=50, overlap=0. Loading train only.", flush=True)
    x_train, y_train = load_dataset(train_files)
    if x_train.shape[1:] != (3, 32, 300):
        raise ValueError(f"Unexpected input shape: {x_train.shape}")
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict] = []
    start_epoch, elapsed_before = 1, 0.0
    if resume:
        checkpoint = torch.load(LATEST, map_location="cpu", weights_only=True)
        if checkpoint["seed"] != SEED or checkpoint["train_csv_count"] != 200:
            raise ValueError("Latest checkpoint metadata mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device.type == "cuda":
            for state_values in optimizer.state.values():
                for key, value in state_values.items():
                    if isinstance(value, torch.Tensor):
                        state_values[key] = value.to(device)
        random.setstate(checkpoint["python_rng_state"])
        state = checkpoint["numpy_rng_state"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        generator.set_state(checkpoint["loader_rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_before = float(checkpoint["elapsed_seconds"])
        with HISTORY.open(newline="", encoding="utf-8") as handle:
            rows = [{key: int(row[key]) if key == "epoch" else float(row[key]) for key in FIELDS}
                    for row in csv.DictReader(handle) if int(row["epoch"]) < start_epoch]
        if len(rows) != start_epoch - 1:
            raise ValueError("History does not match latest checkpoint")
        save_history(rows)
        print(f"Resuming after epoch {start_epoch - 1}", flush=True)
    if start_epoch > 10:
        raise ValueError("Training through epoch 10 is already complete")
    run_start = time.monotonic()
    for epoch in range(start_epoch, 11):
        epoch_start = time.monotonic()
        model.train()
        loss_sum = correct = total = 0
        for batch, (signals, labels) in enumerate(loader, 1):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signals)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            correct += int((output.argmax(1) == labels).sum())
            total += len(labels)
            if batch % 50 == 0:
                print(f"epoch={epoch} batch={batch}/{len(loader)}", flush=True)
        elapsed = elapsed_before + time.monotonic() - run_start
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "epoch_seconds": time.monotonic() - epoch_start, "elapsed_seconds": elapsed}
        rows.append(row)
        save_history(rows)
        payload = checkpoint_payload(epoch, model, optimizer, generator, elapsed)
        temporary = CHECKPOINTS / "densenet_latest.tmp"
        torch.save(payload, temporary)
        temporary.replace(LATEST)
        if epoch in (5, 10):
            torch.save(payload, CHECKPOINTS / f"densenet_epoch_{epoch:02d}.pth")
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} train_accuracy={row['train_accuracy']:.4f} "
              f"seconds={row['epoch_seconds']:.1f}", flush=True)
    print("Training complete through epoch 10. Loading held-out test for the first evaluation.", flush=True)


def evaluate() -> None:
    if METRICS.exists() or SUMMARY.exists():
        raise FileExistsError("Refusing to overwrite or repeat saved Test evaluation")
    _, test_files = split_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test, y_test = load_dataset(test_files)
    if x_test.shape[1:] != (3, 32, 300):
        raise ValueError(f"Unexpected input shape: {x_test.shape}")
    loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    rows = []
    labels = list(range(len(CLASSES)))
    for epoch in (5, 10):
        checkpoint = torch.load(CHECKPOINTS / f"densenet_epoch_{epoch:02d}.pth",
                                map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != epoch or checkpoint["seed"] != SEED:
            raise ValueError(f"Epoch {epoch} checkpoint metadata mismatch")
        model = build_model().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for signals, _ in loader:
                predictions.append(model(signals.to(device)).argmax(1).cpu().numpy())
        y_pred = np.concatenate(predictions)
        rows.append({
            "Epoch": epoch, "Accuracy": accuracy_score(y_test, y_pred),
            "Precision macro": precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
            "Recall macro": recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
            "F1 macro": f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
        })
    with METRICS.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    baseline_week2, baseline_repro = 0.8716, 0.8389
    lines = [
        "DenseNet161 full-train fixed-epoch Test evaluation (200 train CSV / 50 test CSV)",
        "No validation split; no best checkpoint selection; same saved Test files and reproduction preprocessing.",
    ]
    for row in rows:
        lines.append(f"Epoch {row['Epoch']}: Accuracy={row['Accuracy']:.4f}, Precision macro={row['Precision macro']:.4f}, "
                     f"Recall macro={row['Recall macro']:.4f}, F1 macro={row['F1 macro']:.4f}")
    final_accuracy = float(rows[-1]["Accuracy"])
    lines.extend((
        f"10-epoch change vs week2 DenseNet 5 epoch 0.8716: {(final_accuracy - baseline_week2) * 100:+.2f} percentage points",
        f"10-epoch change vs reproduction DenseNet 0.8389: {(final_accuracy - baseline_repro) * 100:+.2f} percentage points",
        f"94% target: {'met' if final_accuracy >= 0.94 else 'not met'}",
        f"Test CSV files={len(test_files)}; Test CWT windows={len(y_test)}",
    ))
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.resume)
    evaluate()


if __name__ == "__main__":
    main()
