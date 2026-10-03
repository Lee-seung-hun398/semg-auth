from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data" / "data"
FS = 1000

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 공통: DataFrame -> PNG
# --------------------------------------------------

def save_table_png(df, title, output_path, width=11, row_height=0.55):
    height = max(2.5, len(df) * row_height + 1.5)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=15)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("저장:", output_path)


# --------------------------------------------------
# 1. 실제 데이터 구조 확인
# --------------------------------------------------

files = sorted(DATA_ROOT.glob("*/*.csv"))

subject_counts = Counter(
    file.parent.name
    for file in files
)

example_df = pd.read_csv(files[0])

dataset_summary = pd.DataFrame([
    ["전체 CSV 파일 수", len(files)],
    ["피험자 수", len(subject_counts)],
    ["피험자", ", ".join(sorted(subject_counts.keys()))],
    ["피험자별 시행 수", ", ".join(
        f"{k}: {v}" for k, v in sorted(subject_counts.items())
    )],
    ["예시 파일", str(files[0])],
    ["원신호 Shape", str(example_df.shape)],
    ["채널 수", example_df.shape[1]],
    ["샘플링 주파수", f"{FS} Hz"],
    ["시행 길이", f"{len(example_df) / FS:.1f} sec"],
], columns=["항목", "실제 확인 결과"])


print("\n[Week 1 Dataset Summary]")
print(dataset_summary.to_string(index=False))

dataset_summary.to_csv(
    OUTPUT_DIR / "week1_dataset_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

save_table_png(
    dataset_summary,
    "1주차 데이터셋 요약",
    OUTPUT_DIR / "week1_dataset_summary.png"
)


# --------------------------------------------------
# 2. A / B / C 첫 시행의 구간별 RMS 측정
#
# 0~1초 : Grasp
# 1~2초 : Rotation
# 2~3초 : Stationary
# --------------------------------------------------

PHASES = {
    "Grasp (0~1s)": (0, 1000),
    "Rotation (1~2s)": (1000, 2000),
    "Stationary (2~3s)": (2000, 3000),
}

rows = []

for subject in ["A", "B", "C"]:
    file_path = DATA_ROOT / subject / f"{subject.lower()} (1).csv"

    df = pd.read_csv(file_path)
    x = df.to_numpy()

    for phase_name, (start, end) in PHASES.items():
        segment = x[start:end]

        for ch in range(segment.shape[1]):
            rms = np.sqrt(
                np.mean(segment[:, ch] ** 2)
            )

            mean_abs = np.mean(
                np.abs(segment[:, ch])
            )

            peak_abs = np.max(
                np.abs(segment[:, ch])
            )

            rows.append([
                subject,
                phase_name,
                f"Ch {ch + 1}",
                rms,
                mean_abs,
                peak_abs
            ])


phase_stats = pd.DataFrame(
    rows,
    columns=[
        "Subject",
        "Phase",
        "Channel",
        "RMS",
        "Mean Abs",
        "Peak Abs"
    ]
)

print("\n[Phase Statistics]")
print(phase_stats.to_string(index=False))

phase_stats.to_csv(
    OUTPUT_DIR / "week1_phase_stats.csv",
    index=False,
    encoding="utf-8-sig"
)


# PNG용으로 소수점 정리
phase_stats_png = phase_stats.copy()

for col in ["RMS", "Mean Abs", "Peak Abs"]:
    phase_stats_png[col] = phase_stats_png[col].map(
        lambda v: f"{v:.4f}"
    )

save_table_png(
    phase_stats_png,
    "A/B/C 피험자 구간별 sEMG 통계",
    OUTPUT_DIR / "week1_phase_stats.png",
    width=12,
    row_height=0.42
)


# --------------------------------------------------
# 3. 선행연구 비교표
# 강의자료 Table 1 + 이번 손바닥 연구
# --------------------------------------------------

prior_studies = pd.DataFrame([
    [
        "손목",
        50,
        8,
        "박수치기",
        "GAN + DNN",
        "97.94"
    ],
    [
        "전완",
        80,
        8,
        "스마트폰 잠금해제",
        "Siamese CNN",
        "92.06"
    ],
    [
        "전완",
        5,
        4,
        "손동작 6종",
        "DWT/EWT/EMD + CNN",
        "~95.62"
    ],
    [
        "이두·삼두",
        40,
        12,
        "손동작 3종",
        "CQT + CNN",
        "97.50"
    ],
    [
        "전완",
        21,
        4,
        "손 펴기",
        "DWT/CWT + CNN",
        "~99.21"
    ],
    [
        "손바닥",
        5,
        2,
        "문손잡이 회전",
        "CWT + DenseNet161",
        "94.00"
    ],
], columns=[
    "측정 부위",
    "인원",
    "채널",
    "동작",
    "특징 추출 + 모델",
    "정확도 (%)"
])


prior_studies.to_csv(
    OUTPUT_DIR / "week1_prior_studies.csv",
    index=False,
    encoding="utf-8-sig"
)

save_table_png(
    prior_studies,
    "sEMG 사용자 식별 선행연구 비교",
    OUTPUT_DIR / "week1_prior_studies.png",
    width=13
)


print()
print("=== Week 1 결과 생성 완료 ===")
print("- week1_dataset_summary.csv")
print("- week1_dataset_summary.png")
print("- week1_phase_stats.csv")
print("- week1_phase_stats.png")
print("- week1_prior_studies.csv")
print("- week1_prior_studies.png")