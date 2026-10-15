from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

def save_table_png(df, title, out_path, fig_width=10, row_height=0.6, font_size=11):
    fig_height = max(2.5, len(df) * row_height + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"저장 완료: {out_path}")


def save_text_png(title, lines, out_path, fig_width=10, fig_height=4):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    text = "\n".join(lines)
    ax.text(
        0.02, 0.95, text,
        va="top", ha="left",
        fontsize=12,
        wrap=True
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"저장 완료: {out_path}")


# --------------------------------------------------
# 1) 데이터셋 요약 표
# --------------------------------------------------
dataset_summary_df = pd.DataFrame([
    ["피험자 수", "5명 (A~E)"],
    ["전체 원본 시행 수", "250"],
    ["피험자별 원본 시행 수", "50"],
    ["원신호 shape", "(3000, 2)"],
    ["샘플링 주파수", "1000 Hz"],
    ["윈도우 길이", "300 samples (300 ms)"],
    ["Hop 크기", "150 samples (150 ms)"],
    ["Overlap", "50%"],
    ["시행당 윈도우 수", "19"],
    ["학습 원본 시행 수", "200"],
    ["테스트 원본 시행 수", "50"],
    ["학습 샘플 수", "3800"],
    ["테스트 샘플 수", "950"],
    ["모델 입력 shape", "(3, 32, 300)"],
], columns=["항목", "값"])

save_table_png(
    dataset_summary_df,
    "Dataset Summary",
    OUTPUT_DIR / "dataset_summary.png"
)


# --------------------------------------------------
# 2) 클래스 분포 표
# --------------------------------------------------
class_distribution_df = pd.DataFrame([
    ["A", 40, 10, 760, 190],
    ["B", 40, 10, 760, 190],
    ["C", 40, 10, 760, 190],
    ["D", 40, 10, 760, 190],
    ["E", 40, 10, 760, 190],
    ["합계", 200, 50, 3800, 950],
], columns=[
    "피험자", "Train 시행 수", "Test 시행 수", "Train 샘플 수", "Test 샘플 수"
])

save_table_png(
    class_distribution_df,
    "Class Distribution",
    OUTPUT_DIR / "class_distribution.png"
)


# --------------------------------------------------
# 3) 설정 근거 메모
# --------------------------------------------------
setting_lines = [
    "[300ms Window 설정 근거]",
    "원신호는 1000Hz로 샘플링된 3초 길이의 (3000, 2) 데이터이며,",
    "300ms window와 150ms hop을 적용한 결과 한 시행당 19개의 구간을 생성할 수 있었다.",
    "이 설정은 시간적 근활성 패턴을 유지하면서도 학습 샘플 수를 충분히 확보하기 위한 것이다.",
    "",
    "[CWT Scale 32 설정 근거]",
    "각 300ms window에 대해 32개의 CWT scale을 적용하여",
    "(3, 32, 300) 형태의 시간-주파수 텐서를 생성하였다.",
    "이 설정은 시간-주파수 정보를 표현하면서도 연산량이 과도하게 커지지 않도록 하기 위한 것이다."
]

save_text_png(
    "Setting Rationale Memo",
    setting_lines,
    OUTPUT_DIR / "setting_memo.png",
    fig_height=5.5
)


# --------------------------------------------------
# 4) Loss 요약 표
# --------------------------------------------------
loss_df = pd.DataFrame([
    [1, 0.9925],
    [2, 0.6880],
    [3, 0.5766],
    [4, 0.4991],
    [5, 0.4199],
], columns=["Epoch", "Average Loss"])

save_table_png(
    loss_df,
    "Training Loss Summary",
    OUTPUT_DIR / "loss_summary.png"
)