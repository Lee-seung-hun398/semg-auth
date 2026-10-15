from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "data"
FS = 1000

subjects = ["A", "B", "C"]

for subject in subjects:
    file_path = DATA_ROOT / subject / f"{subject.lower()} (1).csv"

    df = pd.read_csv(file_path)

    print(f"\n[{subject}]")
    print("파일:", file_path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("최솟값:", df.min().to_dict())
    print("최댓값:", df.max().to_dict())

    # 1000 Hz이므로 샘플 번호를 초 단위로 변환
    t = np.arange(len(df)) / FS

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True
    )

    for ch, column in enumerate(df.columns):
        axes[ch].plot(
            t,
            df[column],
            linewidth=0.7
        )

        # 1초: 파지 → 회전
        # 2초: 회전 → 정지
        axes[ch].axvline(1.0, linestyle="--")
        axes[ch].axvline(2.0, linestyle="--")

        axes[ch].set_ylabel(column)
        axes[ch].grid(alpha=0.2)

    axes[0].set_title(
        f"Subject {subject} - sEMG Trial 1"
    )

    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()

    output_name = PROJECT_ROOT / "week1" / f"signal_{subject}.png"
    plt.savefig(output_name, dpi=150)

    print("저장:", output_name)

    plt.close()

print("\n완료: A, B, C 그래프 생성")