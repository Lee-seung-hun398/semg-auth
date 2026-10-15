from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
from sklearn.model_selection import train_test_split
from collections import Counter

import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, welch


DATA_ROOT = PROJECT_ROOT / "data" / "data"
FS = 1000


def preprocess(x, fs=FS):
    # 1) 60 Hz 전원선 잡음 제거
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    # 2) 20 ~ 499 Hz 대역통과 필터
    # 500 Hz는 Nyquist 주파수와 같기 때문에 499 사용
    b, a = butter(
        4,
        [20 / (fs / 2), 499 / (fs / 2)],
        btype="band"
    )

    x = filtfilt(b, a, x, axis=0)

    return x


# 테스트용: A 피험자 1번 시행
file_path = DATA_ROOT / "A" / "a (1).csv"

df = pd.read_csv(file_path)
x = df.to_numpy()

print("파일:", file_path)
print("원본 shape:", x.shape)

xf = preprocess(x)

print("필터 후 shape:", xf.shape)
print("필터 전 표준편차:", x.std())
print("필터 후 표준편차:", xf.std())

# --------------------------------------------------
# 2단계: 필터 전/후 주파수 스펙트럼 비교
# --------------------------------------------------

# 첫 번째 채널(Comp Ch 3) 사용
f_before, p_before = welch(
    x[:, 0],
    fs=FS,
    nperseg=512
)

f_after, p_after = welch(
    xf[:, 0],
    fs=FS,
    nperseg=512
)

plt.figure(figsize=(10, 4))

plt.semilogy(
    f_before,
    p_before,
    label="Before Filter",
    linewidth=0.8
)

plt.semilogy(
    f_after,
    p_after,
    label="After Filter",
    linewidth=0.8
)

# 60 Hz 위치 표시
plt.axvline(
    60,
    linestyle="--",
    label="60 Hz"
)

plt.xlim(0, 300)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Filter Before / After Spectrum")
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_check.png", dpi=150)
plt.close()

print("저장: filter_check.png")

# --------------------------------------------------
# 3단계: Sliding Window
# 300 ms window, 50% overlap
# --------------------------------------------------

WIN = 300
HOP = 150


def make_windows(x, win=WIN, hop=HOP):
    """
    x: (시간, 채널)
    return: (윈도우 수, win, 채널)
    """

    n = (len(x) - win) // hop + 1

    windows = [
        x[i * hop:i * hop + win]
        for i in range(n)
    ]

    return np.stack(windows)


w = make_windows(xf)

print()
print("[Sliding Window]")
print("입력 shape:", xf.shape)
print("윈도우 shape:", w.shape)
print("윈도우 개수:", len(w))

# --------------------------------------------------
# 4단계: Min-Max Normalization
# 각 윈도우를 0 ~ 1 범위로 정규화
# --------------------------------------------------

def minmax(w, eps=1e-8):
    """
    w: (윈도우 수, 시간, 채널)
    return: 각 윈도우별로 0~1 정규화된 데이터
    """

    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)

    return (w - mn) / (mx - mn + eps)


wn = minmax(w)

print()
print("[Min-Max Normalization]")
print("정규화 전 shape:", w.shape)
print("정규화 후 shape:", wn.shape)
print("최솟값:", wn.min())
print("최댓값:", wn.max())

# --------------------------------------------------
# 5단계: CWT (Continuous Wavelet Transform)
# (300, 2) -> (3, 32, 300)
# --------------------------------------------------

SCALES = np.arange(1, 33)


def to_cwt(one_window, wavelet="morl"):
    """
    one_window: (300, 2)

    return:
        (3, 32, 300)
    """

    maps = []

    # 기존 2개 sEMG 채널 각각 CWT 수행
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(
            one_window[:, ch],
            SCALES,
            wavelet
        )

        maps.append(np.abs(coef))

    # CNN 입력을 3채널로 맞추기 위해
    # 두 채널의 평균을 세 번째 채널로 사용
    maps.append(
        (maps[0] + maps[1]) / 2
    )

    return np.stack(maps).astype(np.float32)


# 첫 번째 윈도우 하나만 테스트
cwt_tensor = to_cwt(wn[0])

print()
print("[CWT]")
print("입력 window shape:", wn[0].shape)
print("CWT tensor shape:", cwt_tensor.shape)
print("dtype:", cwt_tensor.dtype)

# --------------------------------------------------
# CWT 결과 이미지 저장
# --------------------------------------------------

plt.figure(figsize=(10, 4))

plt.imshow(
    cwt_tensor[0],
    aspect="auto",
    origin="lower"
)

plt.xlabel("Time Sample")
plt.ylabel("Scale")
plt.title("CWT Example - Subject A / Trial 1 / Window 1 / Ch 1")

plt.colorbar(label="Magnitude")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cwt_example.png", dpi=150)
plt.close()

print("저장: cwt_example.png")

# --------------------------------------------------
# 6단계: Train / Test 데이터셋 만들기
# 시행(파일) 단위로 먼저 분할하여 데이터 누수 방지
# --------------------------------------------------

SUBJECTS = ["A", "B", "C", "D", "E"]

LABEL_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

# 전체 CSV 파일과 정답 라벨 수집
files = []
labels = []

for subject in SUBJECTS:
    subject_files = sorted(
        (DATA_ROOT / subject).glob("*.csv")
    )

    files.extend(subject_files)
    labels.extend(
        [LABEL_MAP[subject]] * len(subject_files)
    )


print()
print("[Dataset Split]")
print("전체 시행 수:", len(files))
print("전체 클래스 분포:", Counter(labels))


# --------------------------------------------------
# 중요:
# 윈도우를 만들기 전에 시행 파일 단위로 먼저 분할
# --------------------------------------------------

tr_files, te_files, tr_labels, te_labels = train_test_split(
    files,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)


print("학습 시행 수:", len(tr_files))
print("테스트 시행 수:", len(te_files))

print(
    "학습 시행 클래스 분포:",
    Counter(tr_labels)
)

print(
    "테스트 시행 클래스 분포:",
    Counter(te_labels)
)


def build_dataset(file_list):
    """
    시행 파일 목록을 받아
    전처리 -> Window -> Normalization -> CWT 수행

    return:
        X: (샘플 수, 3, 32, 300)
        y: (샘플 수,)
    """

    X = []
    y = []

    for file_path in file_list:
        # 파일의 상위 폴더 A~E가 클래스
        subject = file_path.parent.name
        label = LABEL_MAP[subject]

        # CSV 읽기
        df = pd.read_csv(file_path)
        signal = df.to_numpy()

        # 1. 필터
        signal = preprocess(signal)

        # 2. Sliding Window
        windows = make_windows(signal)

        # 3. Min-Max Normalization
        windows = minmax(windows)

        # 4. 각 Window를 CWT로 변환
        for window in windows:
            X.append(to_cwt(window))
            y.append(label)

    return (
        np.stack(X).astype(np.float32),
        np.array(y, dtype=np.int64)
    )


print()
print("학습 데이터 생성 중...")

Xtr, ytr = build_dataset(tr_files)

print("테스트 데이터 생성 중...")

Xte, yte = build_dataset(te_files)


print()
print("[Dataset Result]")

print("X_train:", Xtr.shape)
print("y_train:", ytr.shape)

print("X_test:", Xte.shape)
print("y_test:", yte.shape)

print(
    "학습 샘플 클래스 분포:",
    Counter(ytr.tolist())
)

print(
    "테스트 샘플 클래스 분포:",
    Counter(yte.tolist())
)

# --------------------------------------------------
# 7단계: DenseNet161 학습
# 우선 1 Epoch만 실행하여 학습 시간 확인
# --------------------------------------------------

import time
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("[Training]")
print("사용 장치:", device)


# --------------------------------------------------
# NumPy -> PyTorch Tensor
# torch.from_numpy()는 불필요한 복사를 줄일 수 있음
# --------------------------------------------------

X_train_tensor = torch.from_numpy(Xtr)
y_train_tensor = torch.from_numpy(ytr)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

print("학습 샘플 수:", len(train_dataset))
print("배치 수:", len(train_loader))


# --------------------------------------------------
# DenseNet161
# 강의자료와 동일하게 사전학습 가중치 사용 안 함
# --------------------------------------------------

model = densenet161(weights=None)

# DenseNet161 classifier: 2208 -> 5 classes
model.classifier = nn.Linear(
    2208,
    5
)

model = model.to(device)


# --------------------------------------------------
# Optimizer / Loss
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 5 Epoch 학습
# --------------------------------------------------

EPOCHS = 5
START_EPOCH = 0
epoch_losses = []


# --------------------------------------------------
# 기존 체크포인트에서 이어서 학습
# --------------------------------------------------

resume_path = Path(OUTPUT_DIR / "densenet161_epoch_2.pth")

if resume_path.exists():

    checkpoint = torch.load(
        resume_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    START_EPOCH = checkpoint["epoch"]

    # 기존 Epoch 1~2 Loss 복구
    for i in range(1, START_EPOCH + 1):

        old_checkpoint = torch.load(
            OUTPUT_DIR / f"densenet161_epoch_{i}.pth",
            map_location="cpu"
        )

        epoch_losses.append(
            old_checkpoint["loss"]
        )

    print()
    print(
        f"체크포인트 복구 완료: "
        f"Epoch {START_EPOCH}부터 이어서 진행"
    )


start_time = time.time()


for epoch in range(START_EPOCH, EPOCHS):
    epoch_start = time.time()

    model.train()
    total_loss = 0.0

    for batch_idx, (xb, yb) in enumerate(train_loader):

        xb = xb.to(device)
        yb = yb.to(device)

        output = model(xb)
        loss = criterion(output, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS}"
                f" | Batch {batch_idx + 1}/{len(train_loader)}"
                f" | Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)
    epoch_losses.append(average_loss)

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f" | Average Loss: {average_loss:.4f}"
        f" | Time: {epoch_time:.1f}초"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
        },
        OUTPUT_DIR / f"densenet161_epoch_{epoch + 1}.pth"
    )