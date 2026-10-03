"""Leakage-free Week 3 improvement; run dense, then resnet, then final."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet161, resnet18

import week3 as original

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "improved"
OUT.mkdir(exist_ok=True)
original.OUT = OUT
CLASSES = original.SUBJECTS
SEED = 42
BATCH = 16


def deterministic():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def split_files():
    files = [p for subject in CLASSES for p in sorted((ROOT / "data" / "data" / subject).glob("*.csv"))]
    assert len(files) == 250
    labels = [original.LABEL_MAP[p.parent.name] for p in files]
    old_train, test, old_labels, _ = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=SEED)
    train, val = train_test_split(old_train, test_size=0.2,
                                  stratify=old_labels, random_state=SEED)
    assert (len(train), len(val), len(test)) == (160, 40, 50)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    manifest = {part: [str(p.relative_to(ROOT)) for p in group]
                for part, group in [("train", train), ("validation", val), ("test", test)]}
    path = OUT / "split_manifest.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == manifest, "Split changed"
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train, val, test


def model_for(name):
    if name.startswith("densenet161"):
        model = densenet161(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    else:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def measure(model, data, labels, device):
    predictions = original.predict(model, data, device)
    accuracy = accuracy_score(labels, predictions)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    return accuracy, f1, predictions


def train(name):
    deterministic()
    train_files, val_files, _ = split_files()
    pipeline = original.week2_pipeline()
    print("Building 160 train / 40 validation trials with Week 2 preprocessing", flush=True)
    x_train, y_train = pipeline["build_dataset"](train_files)
    x_val, y_val = pipeline["build_dataset"](val_files)
    print(f"Windows: train={len(y_train)} validation={len(y_val)}", flush=True)
    assert x_train.shape[1:] == x_val.shape[1:] == (3, 32, 300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "densenet161_pretrained":
        weights_path = ROOT / "week3" / "pretrained" / "densenet161-8d451a50.pth"
        assert weights_path.is_file()
        model = densenet161(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
        for key in list(state):
            match = pattern.match(key)
            if match:
                state[match.group(1) + match.group(2)] = state.pop(key)
        model.load_state_dict(state, strict=True)
        del state
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
        initial_lr = 1e-4
    else:
        model = model_for(name)
        initial_lr = 1e-3
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train)),
                        batch_size=BATCH,shuffle=True,generator=generator,num_workers=0)
    max_epochs = 10 if name == "densenet161_pretrained" else (12 if name == "densenet161" else 10)
    best = (-1.0, -1.0)
    best_epoch = 0
    log = OUT / f"{name}_validation.csv"
    checkpoint_path = OUT / f"{name}_best.pth"
    with log.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Epoch","LearningRate","TrainLoss","ValidationAccuracy","ValidationF1Macro","Seconds"])
        for epoch in range(1,max_epochs+1):
            model.train()
            epoch_start=time.monotonic()
            total_loss=0.0
            for batch_index,(inputs,targets) in enumerate(loader,1):
                inputs,targets=inputs.to(device),targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(inputs),targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if batch_index in (1,25) or batch_index%50==0:
                    elapsed=time.monotonic()-epoch_start
                    print(f"{name} epoch {epoch}/{max_epochs} batch {batch_index}/{len(loader)} estimated {elapsed/batch_index*len(loader)/60:.1f} min/epoch",flush=True)
            val_accuracy,val_f1,_=measure(model,x_val,y_val,device)
            lr=optimizer.param_groups[0]["lr"]
            seconds=time.monotonic()-epoch_start
            writer.writerow([epoch,lr,total_loss/len(loader),val_accuracy,val_f1,seconds])
            stream.flush()
            print(f"{name} epoch={epoch} val_accuracy={val_accuracy:.4f} val_f1={val_f1:.4f} lr={lr:g} time={seconds/60:.1f}min",flush=True)
            if (val_accuracy,val_f1)>best:
                best=(val_accuracy,val_f1)
                best_epoch=epoch
                torch.save(dict(epoch=epoch,model_state_dict={key: value.detach().cpu() for key,value in model.state_dict().items()},
                                validation_accuracy=val_accuracy,validation_f1=val_f1,
                                learning_rate=lr,seed=SEED,classes=CLASSES,
                                split_manifest="split_manifest.json"),checkpoint_path)
                print(f"Best validation checkpoint saved: {checkpoint_path.name}",flush=True)
            scheduler.step(val_accuracy)
            if name.startswith("densenet161") and epoch>=(5 if name == "densenet161_pretrained" else 10) and epoch-best_epoch>=3:
                print("Stopped after validation plateau",flush=True)
                break
            if name == "densenet161_pretrained" and epoch>=5 and best[0]>=0.95:
                print("Stopped: validation accuracy >= 0.95",flush=True)
                break
            if name == "resnet18" and epoch>=5 and val_accuracy>=0.85:
                print("Stopped after 5+ epochs: validation target reached",flush=True)
                break
    print(f"{name} best epoch={best_epoch} validation_accuracy={best[0]:.4f}",flush=True)


def final():
    deterministic()
    train_files,val_files,test_files=split_files()
    del train_files,val_files
    pipeline=original.week2_pipeline()
    x_test,y_test=pipeline["build_dataset"](test_files)
    assert len(y_test)==950
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics,class_scores,matrices={}, {}, {}
    details={}
    candidates = [name for name in ("densenet161", "densenet161_pretrained") if (OUT/f"{name}_best.pth").is_file()]
    assert candidates, "No DenseNet161 validation checkpoint"
    dense_name = max(candidates, key=lambda name: (
        float(torch.load(OUT/f"{name}_best.pth",map_location="cpu",weights_only=True)["validation_accuracy"]),
        float(torch.load(OUT/f"{name}_best.pth",map_location="cpu",weights_only=True)["validation_f1"])))
    print("DenseNet161 selected by validation:",dense_name,flush=True)
    for name,display in [(dense_name,"DenseNet161"),("resnet18","ResNet18")]:
        checkpoint=torch.load(OUT/f"{name}_best.pth",map_location="cpu",weights_only=True)
        model=model_for(name)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        model=model.to(device)
        prediction=original.predict(model,x_test,device)
        metrics[display],class_scores[display]=original.score(y_test,prediction)
        matrices[display]=confusion_matrix(y_test,prediction,labels=list(range(len(CLASSES))))
        original.confusion_plot(matrices[display],display,f"confusion_matrix_{display.lower()}.png")
        details[display]=dict(source=name,epoch=checkpoint["epoch"],lr=checkpoint["learning_rate"],
                              validation_accuracy=checkpoint["validation_accuracy"],
                              validation_f1=checkpoint["validation_f1"])
        del model,checkpoint
    comparison=pd.DataFrame.from_dict(metrics,orient="index").reset_index(names="Model")
    comparison.to_csv(OUT/"model_comparison.csv",index=False,float_format="%.10f")
    original.grouped_plot({name:[metrics[name][key] for key in ["Accuracy","Precision","Recall","F1"]] for name in metrics},
                          ["Accuracy","Precision","Recall","F1"],"User Classification: Final Model Performance","Score","model_comparison.png",(0,1))
    original.grouped_plot(class_scores,CLASSES,"User Classification: Final F1 by User Class","F1-score","class_f1_comparison.png",(0,1))
    lines=["Final Week 3 evaluation: held-out test used only after validation checkpoint selection.",
           "Split: 250 trials -> 200 old train / 50 unchanged test (seed 42, stratified); 200 -> 160 train / 40 validation (seed 42, stratified).",
           "Test unit: CWT window; 950 test windows. Both models use the same test windows.",
           "Preprocessing: original Week 2 functions; no changes.",
           "Precision/Recall/F1: macro average, zero_division=0.",""]
    winner=max(metrics,key=lambda key:(metrics[key]["F1"],metrics[key]["Accuracy"]))
    lines.append(f"Higher macro F1: {winner}")
    for display in ["DenseNet161","ResNet18"]:
        m=metrics[display]; cm=matrices[display]; d=details[display]
        missed=cm.sum(axis=1)-cm.diagonal()
        wrong=cm.copy(); np.fill_diagonal(wrong,0)
        pairs=np.argwhere(wrong==wrong.max()) if wrong.max()>0 else []
        lines += [f"{display}: source={d['source']}, best epoch={d['epoch']}, lr={d['lr']:.6g}, val accuracy={d['validation_accuracy']:.4f}, test accuracy={m['Accuracy']:.4f}, precision={m['Precision']:.4f}, recall={m['Recall']:.4f}, F1={m['F1']:.4f}",
                  f"  Best F1 class: {CLASSES[int(np.argmax(class_scores[display]))]} ({max(class_scores[display]):.4f})",
                  f"  Most misclassified actual class: {CLASSES[int(np.argmax(missed))]} ({max(missed)} windows)",
                  "  Most frequent error pair(s): "+(", ".join(f"{CLASSES[i]} -> {CLASSES[j]} ({wrong[i,j]})" for i,j in pairs if i!=j) or "none")]
    gaps=abs(class_scores["DenseNet161"]-class_scores["ResNet18"])
    lines.append("Class F1 gaps >= 0.05: "+(", ".join(f"{CLASSES[i]} ({gaps[i]:.4f})" for i in np.flatnonzero(gaps>=.05)) or "none"))
    lines.append("Training logs: densenet161_validation.csv, resnet18_validation.csv; split: split_manifest.json.")
    (OUT/"evaluation_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(comparison.to_string(index=False),flush=True)
    print("Final outputs saved to",OUT,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["dense","dense_pretrained","resnet","final"])
    args=parser.parse_args()
    if args.stage=="dense": train("densenet161")
    elif args.stage=="dense_pretrained": train("densenet161_pretrained")
    elif args.stage=="resnet": train("resnet18")
    else: final()
