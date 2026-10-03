"""Validation-only DenseNet161 reproduction; the held-out test CSVs are never loaded.

Run from the project root:
    .venv/Scripts/python.exe -B week3/reproduction/train_densenet_reproduction.py --until 10
    .venv/Scripts/python.exe -B week3/reproduction/train_densenet_reproduction.py --until 15 --resume

The released CSVs are already filtered. We reuse Week 2 windowing,
normalization, and CWT, but skip its second filtering pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet161

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHECKPOINTS = OUT / "checkpoints"
sys.path.insert(0, str(PROJECT / "week3"))
import week3 as original  # noqa: E402: importing does not run week3.main()

SEED = 42
CLASSES = ("A", "B", "C", "D", "E")
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 45
HISTORY_FIELDS = (
    "epoch", "train_loss", "train_accuracy", "validation_accuracy",
    "validation_macro_f1", "epoch_seconds", "elapsed_seconds",
)


def set_reproducible_state() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def split_files() -> tuple[list[Path], list[Path]]:
    """Keep Week 2's held-out 20% intact; split only its 80% for validation."""
    files = [
        path
        for subject in CLASSES
        for path in sorted((PROJECT / "data" / "data" / subject).glob("*.csv"))
    ]
    labels = [original.LABEL_MAP[path.parent.name] for path in files]
    if len(files) != 250 or any(labels.count(i) != 50 for i in range(5)):
        raise ValueError("Expected 50 CSV trials for each of users A-E")
    old_train, test, old_labels, _ = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    train, validation = train_test_split(
        old_train, test_size=0.2, stratify=old_labels, random_state=SEED
    )
    if (len(train), len(validation), len(test)) != (160, 40, 50):
        raise AssertionError("Unexpected trial counts")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise AssertionError("A trial appears in more than one split")
    manifest = {
        name: [str(path.relative_to(PROJECT)) for path in group]
        for name, group in (("train", train), ("validation", validation), ("test", test))
    }
    previous = PROJECT / "week3" / "improved" / "split_manifest.json"
    if previous.is_file() and json.loads(previous.read_text(encoding="utf-8")) != manifest:
        raise AssertionError("Split differs from the existing Week 3 held-out test")
    path = OUT / "split_manifest.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise AssertionError("Saved reproduction split has changed")
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train, validation


def build_model() -> nn.Module:
    # The paper specifies 7x7/stride-2 and five output classes. Its text does
    # not disclose a precise dropout position/rate or a complete replacement
    # conv0 definition. Torchvision's stock conv0 already accepts 3-channel
    # 32x300 input and implements 7x7/stride-2, so no invented changes apply.
    model = densenet161(weights=None)
    conv0 = model.features.conv0
    if (conv0.in_channels, conv0.kernel_size, conv0.stride) != (3, (7, 7), (2, 2)):
        raise AssertionError("Torchvision DenseNet161 first convolution changed")
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    return model


def load_dataset(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    pipeline = original.week2_pipeline()
    pipeline["preprocess"] = lambda signal: signal  # released CSV is filtered
    return pipeline["build_dataset"](files)


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    with torch.inference_mode():
        for signals, labels in loader:
            guesses = model(signals.to(device)).argmax(dim=1).cpu().numpy()
            actual.append(labels.numpy())
            predicted.append(guesses)
    y_true = np.concatenate(actual)
    y_pred = np.concatenate(predicted)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", labels=range(5), zero_division=0)),
    )


def save_history(rows: list[dict[str, float | int]]) -> None:
    with (OUT / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for key, title, label, filename in (
        ("train_loss", "DenseNet161 Training Loss by Epoch", "Cross-entropy loss", "training_loss.png"),
        ("validation_accuracy", "DenseNet161 Validation Accuracy by Epoch", "Accuracy", "validation_accuracy.png"),
        ("validation_macro_f1", "DenseNet161 Validation Macro F1 by Epoch", "Macro F1", "validation_f1.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([row["epoch"] for row in rows], [row[key] for row in rows], marker="o")
        ax.set(title=title, xlabel="Epoch", ylabel=label)
        ax.set_xticks([row["epoch"] for row in rows])
        ax.grid(alpha=0.25)
        if key != "train_loss":
            ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)


def save_info(device: torch.device, rows: list[dict[str, float | int]], best: dict) -> None:
    first = float(rows[0]["epoch_seconds"]) if rows else None
    lines = [
        "DenseNet161 partial reproduction: validation only; no test CSV loaded.",
        "Paper: https://www.nature.com/articles/s41598-026-46294-3",
        "Filtered CSV source: https://github.com/sea3551/palm-sEMG-doorknob-filtered",
        f"Seed={SEED}; classes={','.join(CLASSES)}; learning_rate={LEARNING_RATE}; batch_size={BATCH_SIZE}; maximum_epochs={MAX_EPOCHS}.",
        "Split: stratified 80/20 trial files, random_state=42; then old train 80/20 train/validation; 160/40/50 trial files.",
        "The 50 held-out test CSVs are listed in split_manifest.json but never loaded or evaluated in this program.",
        "Released data already has 60 Hz notch and 20-500 Hz band-pass; no duplicate filter here.",
        "Week 2's 300-sample window, 150-sample hop, window min-max and Morlet CWT are reused without changes.",
        "Torchvision DenseNet161 random initialization; conv0=3->96, 7x7, stride 2; classifier=2208->5.",
        "The paper says conv0 was adjusted and dropout used, but omits the exact conv0 replacement and dropout position/rate. These are not invented here; exact architecture reproduction remains unverified.",
        "Adam, cross-entropy, no scheduler (none specified by paper); model selection by validation accuracy, macro F1 tie-break.",
        "torch.cuda.is_available()=" + str(torch.cuda.is_available()),
        "device=" + str(device),
        "device_name=" + (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "completed_epochs=" + str(len(rows)),
        "best_epoch=" + str(best["epoch"]),
        "best_validation_accuracy=" + f"{best['accuracy']:.6f}",
        "best_validation_macro_f1=" + f"{best['macro_f1']:.6f}",
    ]
    if first is not None:
        lines.extend((
            f"first_epoch_seconds={first:.2f}",
            f"estimated_45_epochs_hours={first * 45 / 3600:.2f}",
            f"mean_epoch_seconds={np.mean([row['epoch_seconds'] for row in rows]):.2f}",
        ))
    (OUT / "experiment_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", type=int, default=10, choices=range(5, MAX_EPOCHS + 1, 5))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print("GPU name or CPU:", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", flush=True)
    set_reproducible_state()
    CHECKPOINTS.mkdir(exist_ok=True)
    train_files, validation_files = split_files()
    print("Building train and validation CWT windows; test CSVs are not opened.", flush=True)
    x_train, y_train = load_dataset(train_files)
    x_validation, y_validation = load_dataset(validation_files)
    if x_train.shape[1:] != (3, 32, 300) or x_validation.shape[1:] != (3, 32, 300):
        raise AssertionError("Unexpected CWT input shape")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    latest = CHECKPOINTS / "densenet_latest.pth"
    best_path = CHECKPOINTS / "densenet_best.pth"
    history_path = OUT / "training_history.csv"
    rows: list[dict[str, float | int]] = []
    best = {"epoch": 0, "accuracy": -1.0, "macro_f1": -1.0}
    start_epoch = 1
    elapsed_before = 0.0
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError("Cannot resume: densenet_latest.pth is missing")
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device.type == "cuda":
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        random.setstate(checkpoint["python_rng_state"])
        numpy_state = checkpoint["numpy_rng_state"]
        np.random.set_state((
            numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
            numpy_state[2], numpy_state[3], numpy_state[4],
        ))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        generator.set_state(checkpoint["loader_rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = checkpoint["best"]
        elapsed_before = float(checkpoint["elapsed_seconds"])
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                {key: (int(row[key]) if key == "epoch" else float(row[key])) for key in HISTORY_FIELDS}
                for row in csv.DictReader(handle)
                if int(row["epoch"]) < start_epoch
            ]
        if len(rows) != start_epoch - 1:
            raise AssertionError("History is missing an epoch from the latest checkpoint")
        # A prior interruption may have written a history row before its
        # atomic latest checkpoint. Replay that epoch from the checkpoint.
        save_history(rows)
        print(f"Resuming after epoch {start_epoch - 1}", flush=True)
    elif latest.exists() or history_path.exists():
        raise FileExistsError("Reproduction artifacts exist; use --resume to preserve them")
    if args.until < start_epoch - 1:
        raise ValueError("--until precedes the latest completed epoch")
    criterion = nn.CrossEntropyLoss()
    run_start = time.monotonic()
    for epoch in range(start_epoch, args.until + 1):
        epoch_start = time.monotonic()
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        for batch_index, (signals, labels) in enumerate(train_loader, 1):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signals)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            correct += int((output.argmax(dim=1) == labels).sum().item())
            total += len(labels)
            if batch_index % 50 == 0:
                print(f"epoch={epoch} batch={batch_index}/{len(train_loader)}", flush=True)
        validation_accuracy, validation_macro_f1 = validate(model, validation_loader, device)
        epoch_seconds = time.monotonic() - epoch_start
        elapsed = elapsed_before + time.monotonic() - run_start
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_macro_f1,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        if (validation_accuracy, validation_macro_f1) > (best["accuracy"], best["macro_f1"]):
            best = {"epoch": epoch, "accuracy": validation_accuracy, "macro_f1": validation_macro_f1}
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "best": best,
                "seed": SEED, "split_manifest": "split_manifest.json",
            }, best_path)
        save_history(rows)
        save_info(device, rows, best)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_accuracy={row['train_accuracy']:.4f} "
            f"validation_accuracy={validation_accuracy:.4f} "
            f"validation_macro_f1={validation_macro_f1:.4f} "
            f"seconds={epoch_seconds:.1f} best_epoch={best['epoch']}", flush=True,
        )
        if epoch == 1:
            print(f"First epoch: {epoch_seconds / 60:.2f} min; estimated 45 epochs: {epoch_seconds * 45 / 3600:.2f} h", flush=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best": best,
            "elapsed_seconds": elapsed,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": (
                lambda state: (state[0], state[1].tolist(), state[2], state[3], state[4])
            )(np.random.get_state()),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader_rng_state": generator.get_state(),
        }
        temporary = CHECKPOINTS / "densenet_latest.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(latest)
        if epoch % 5 == 0:
            shutil.copyfile(latest, CHECKPOINTS / f"densenet_epoch_{epoch:02d}.pth")
            print(f"Milestone checkpoint saved: epoch {epoch}", flush=True)
    print(f"Completed through epoch {args.until}; test set was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
