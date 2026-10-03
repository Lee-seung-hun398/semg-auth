"""Week 3: evaluate the Week 2 DenseNet161 and train/evaluate ResNet18."""
from __future__ import annotations

import ast
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet161, resnet18

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
WEEK2 = ROOT / "week2" / "week2.py"
SUBJECTS = ["A", "B", "C", "D", "E"]
LABEL_MAP = {name: index for index, name in enumerate(SUBJECTS)}
BATCH = 16
RESNET_EPOCHS = 1
SEED = 42


def week2_pipeline():
    # Import only the original data-processing definitions. Importing week2.py
    # normally would execute its full DenseNet training at module import time.
    tree = ast.parse(WEEK2.read_bytes(), filename=str(WEEK2))
    wanted_functions = {"preprocess", "make_windows", "minmax", "to_cwt", "build_dataset"}
    wanted_assignments = {"FS", "WIN", "HOP", "SCALES"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in wanted_assignments for target in targets):
                selected.append(node)
    names = {node.name for node in selected if isinstance(node, ast.FunctionDef)}
    assert names == wanted_functions, f"Week 2 preprocessing functions changed: {names}"
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    scope = dict(np=np, pd=pd, pywt=pywt, butter=butter, iirnotch=iirnotch,
                 filtfilt=filtfilt, LABEL_MAP=LABEL_MAP)
    exec(compile(module, str(WEEK2), "exec"), scope)
    return scope


def predict(model, x, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=BATCH)
    result = []
    with torch.inference_mode():
        for (batch,) in loader:
            result.append(model(batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(result)


def score(y, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, labels=list(range(len(SUBJECTS))), average="macro", zero_division=0)
    _, _, class_f1, _ = precision_recall_fscore_support(
        y, prediction, labels=list(range(len(SUBJECTS))), average=None, zero_division=0)
    return dict(Accuracy=accuracy_score(y, prediction), Precision=precision,
                Recall=recall, F1=f1), class_f1


def confusion_plot(matrix, model_name, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(title=f"{model_name}: User Classification Confusion Matrix",
           xlabel="Predicted user", ylabel="Actual user")
    ax.set_xticks(range(len(SUBJECTS)), SUBJECTS)
    ax.set_yticks(range(len(SUBJECTS)), SUBJECTS)
    for row in range(len(SUBJECTS)):
        for col in range(len(SUBJECTS)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def grouped_plot(frame, x_labels, title, ylabel, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(x_labels))
    for index, model_name in enumerate(["DenseNet161", "ResNet18"]):
        ax.bar(x + (index - 0.5)*width, frame[model_name], width, label=model_name)
    ax.set_xticks(x, x_labels)
    ax.set(title=title, xlabel="Metric" if filename == "model_comparison.png" else "User class",
           ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def analysis_text(metrics, class_scores, matrices, split, elapsed):
    winner = max(metrics, key=lambda name: (metrics[name]["F1"], metrics[name]["Accuracy"]))
    lines = ["Week 3 sEMG user classification", "",
             "Dataset: A-E, 50 trials per user; file-level stratified 80/20 split.",
             "Split: test_size=0.2, random_state=42; no validation set in Week 2.",
             f"Training trials: {len(split[0])}; test trials: {len(split[1])}; test windows: {sum(matrices['DenseNet161'].ravel())}.",
             "Preprocessing: Week 2 notch 60 Hz, band-pass 20-499 Hz, 300 ms windows / 150 ms hop, min-max, Morlet CWT (3, 32, 300).",
             "Precision, Recall, and F1 averaging: macro, zero_division=0.",
             "Winner by macro F1 (accuracy tie-break): " + winner, ""]
    for name in ["DenseNet161", "ResNet18"]:
        values = metrics[name]
        matrix = matrices[name]
        best = np.flatnonzero(class_scores[name] == class_scores[name].max())
        misses = matrix.sum(axis=1) - np.diag(matrix)
        most_missed = np.flatnonzero(misses == misses.max())
        wrong = matrix.copy()
        np.fill_diagonal(wrong, 0)
        pairs = np.argwhere(wrong == wrong.max()) if wrong.max() else []
        pair_text = ", ".join(f"{SUBJECTS[i]} -> {SUBJECTS[j]} ({wrong[i,j]})" for i,j in pairs if i != j) or "none"
        lines += [f"{name}: Accuracy={values['Accuracy']:.4f}, Precision={values['Precision']:.4f}, Recall={values['Recall']:.4f}, F1={values['F1']:.4f}",
                  "  Best F1 class(es): " + ", ".join(f"{SUBJECTS[i]} ({class_scores[name][i]:.4f})" for i in best),
                  "  Most misclassified actual class(es): " + ", ".join(f"{SUBJECTS[i]} ({misses[i]} windows)" for i in most_missed),
                  "  Most frequent error pair(s): " + pair_text, ""]
    gaps = abs(class_scores["DenseNet161"] - class_scores["ResNet18"])
    large = np.flatnonzero(gaps >= 0.05)
    lines.append("Class F1 gaps >= 0.05: " + (
        ", ".join(f"{SUBJECTS[i]} ({gaps[i]:.4f}; higher: {'DenseNet161' if class_scores['DenseNet161'][i] > class_scores['ResNet18'][i] else 'ResNet18'})" for i in large)
        if len(large) else "none"))
    lines.append(f"Run time: {elapsed/60:.1f} minutes.")
    return "\n".join(lines) + "\n"


def main():
    started = time.monotonic()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    week2 = week2_pipeline()
    files = [path for subject in SUBJECTS for path in sorted((ROOT / "data" / "data" / subject).glob("*.csv"))]
    labels = [LABEL_MAP[path.parent.name] for path in files]
    assert len(files) == 250 and all(labels.count(i) == 50 for i in range(5))
    train_files, test_files, _, _ = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=42)
    assert not set(train_files) & set(test_files)
    print("Building training data with original Week 2 functions...", flush=True)
    x_train, y_train = week2["build_dataset"](train_files)
    print("Building shared test data...", flush=True)
    x_test, y_test = week2["build_dataset"](test_files)
    assert x_train.shape[1:] == x_test.shape[1:] == (3, 32, 300)
    print(f"Train={len(y_train)} windows, test={len(y_test)} windows", flush=True)
    checkpoint_path = ROOT / "week2" / "densenet161_epoch_5.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dense = densenet161(weights=None)
    dense.classifier = nn.Linear(dense.classifier.in_features, len(SUBJECTS))
    dense.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    dense = dense.to(device)
    dense_predictions = predict(dense, x_test, device)
    print("DenseNet161 evaluation complete", flush=True)
    del dense
    resnet = resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(SUBJECTS))
    resnet = resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    for epoch in range(RESNET_EPOCHS):
        resnet.train()
        epoch_start = time.monotonic()
        total_loss = 0.0
        for batch_number, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(resnet(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_number % 25 == 0 or batch_number == 1:
                elapsed = time.monotonic() - epoch_start
                estimate = elapsed / batch_number * len(train_loader) / 60
                print(f"Epoch {epoch+1}, batch {batch_number}/{len(train_loader)}, mean loss {total_loss/batch_number:.4f}, estimated epoch {estimate:.1f} min", flush=True)
        print(f"ResNet18 epoch {epoch+1} completed in {(time.monotonic()-epoch_start)/60:.1f} min", flush=True)
    resnet_predictions = predict(resnet, x_test, device)
    torch.save({"epoch": RESNET_EPOCHS, "model_state_dict": resnet.cpu().state_dict(),
                "classes": SUBJECTS, "seed": SEED}, OUT / "resnet18_epoch_1.pth")
    predictions = {"DenseNet161": dense_predictions, "ResNet18": resnet_predictions}
    metrics, class_scores, matrices = {}, {}, {}
    for name, prediction in predictions.items():
        metrics[name], class_scores[name] = score(y_test, prediction)
        matrices[name] = confusion_matrix(y_test, prediction, labels=list(range(len(SUBJECTS))))
        confusion_plot(matrices[name], name, f"confusion_matrix_{name.lower()}.png")
    comparison = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names="Model")
    comparison.to_csv(OUT / "model_comparison.csv", index=False, float_format="%.10f")
    grouped_plot({name: [metrics[name][metric] for metric in ["Accuracy","Precision","Recall","F1"]] for name in metrics},
                 ["Accuracy","Precision","Recall","F1"], "User Classification: Model Performance", "Score", "model_comparison.png", (0,1))
    grouped_plot(class_scores, SUBJECTS, "User Classification: F1 by User Class", "F1-score", "class_f1_comparison.png", (0,1))
    (OUT / "evaluation_summary.txt").write_text(
        analysis_text(metrics, class_scores, matrices, (train_files,test_files), time.monotonic()-started), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print("All outputs saved to", OUT, flush=True)


if __name__ == "__main__":
    main()
