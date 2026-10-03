from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
from collections import Counter
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 기본 설정
# ============================================================

FS = 1000  # Hz
SIGNAL_SECONDS = 3

PHASES = {
    "Grasp": (0.0, 1.0),
    "Rotation": (1.0, 2.0),
    "Stationary": (2.0, 3.0),
}

SUBJECTS = ["A", "B", "C", "D", "E"]


# ============================================================
# 한글 폰트 설정
# ============================================================

def setup_korean_font():
    """
    Windows에서는 맑은 고딕을 우선 사용.
    없을 경우 matplotlib 기본 폰트 사용.
    """
    candidates = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    ]

    for font_path in candidates:
        if font_path.exists():
            font_prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = font_prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[폰트] {font_prop.get_name()}")
            return

    print("[경고] 맑은 고딕을 찾지 못했습니다.")
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 데이터 폴더 찾기
# ============================================================

def find_data_root():
    candidates = [
        (PROJECT_ROOT / "data") / "data",
        (PROJECT_ROOT / "data"),
    ]

    for root in candidates:
        if not root.exists():
            continue

        subject_dirs = [root / subject for subject in SUBJECTS]

        if all(d.exists() for d in subject_dirs):
            return root

    raise FileNotFoundError(
        "A~E 데이터 폴더를 찾지 못했습니다.\n"
        "예상 위치: C:\\semg-auth\\data\\data\\A ~ E"
    )


# ============================================================
# CSV 읽기
# ============================================================

def load_signal(path: Path):
    df = pd.read_csv(path)

    # 숫자로 변환 가능한 열만 사용
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.dropna(axis=0, how="all")

    if numeric.shape[1] < 2:
        raise ValueError(
            f"{path} 에서 2개 채널을 찾지 못했습니다. "
            f"현재 shape={numeric.shape}"
        )

    # 이번 데이터셋은 2채널 사용
    x = numeric.iloc[:, :2].to_numpy(dtype=np.float64)

    return x


# ============================================================
# RMS / MAV / STD / Peak 계산
# ============================================================

def calculate_features(segment):
    """
    segment: (time, channel)

    반환:
        채널별 RMS
        채널별 MAV
        채널별 STD
        채널별 Peak
        두 채널 전체 Combined RMS
    """

    rms = np.sqrt(np.mean(segment ** 2, axis=0))
    mav = np.mean(np.abs(segment), axis=0)
    std = np.std(segment, axis=0)
    peak = np.max(np.abs(segment), axis=0)

    combined_rms = np.sqrt(np.mean(segment ** 2))

    return rms, mav, std, peak, combined_rms


# ============================================================
# DataFrame -> PNG 표
# ============================================================

def save_table_png(
    df,
    title,
    output_path,
    width=12,
    row_height=0.55,
    font_size=10,
):
    height = max(2.8, 1.5 + len(df) * row_height)

    fig, ax = plt.subplots(figsize=(width, height))

    ax.axis("off")
    ax.set_title(
        title,
        fontsize=15,
        pad=16,
        fontweight="bold",
    )

    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.45)

    try:
        table.auto_set_column_width(
            col=list(range(len(df.columns)))
        )
    except Exception:
        pass

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[저장] {output_path}")


# ============================================================
# 1. 데이터셋 전체 분석
# ============================================================

def analyze_dataset(data_root):
    files = sorted(data_root.glob("*/*.csv"))

    if not files:
        raise RuntimeError(
            f"CSV 파일이 없습니다: {data_root}"
        )

    subject_counts = Counter(
        f.parent.name
        for f in files
    )

    example_signal = load_signal(files[0])

    summary = pd.DataFrame(
        [
            ["전체 CSV 파일 수", len(files)],
            ["피험자 수", len(subject_counts)],
            ["피험자", ", ".join(sorted(subject_counts.keys()))],
            [
                "피험자별 시행 수",
                ", ".join(
                    f"{s}: {subject_counts[s]}"
                    for s in sorted(subject_counts)
                ),
            ],
            ["원신호 Shape", str(example_signal.shape)],
            ["채널 수", example_signal.shape[1]],
            ["샘플링 주파수", f"{FS} Hz"],
            [
                "시행 길이",
                f"{len(example_signal) / FS:.1f} sec",
            ],
        ],
        columns=[
            "항목",
            "실제 확인 결과",
        ],
    )

    print("\n==========================================")
    print("1주차 데이터셋 요약")
    print("==========================================")
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "week1_dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_table_png(
        summary,
        "1주차 데이터셋 요약",
        OUTPUT_DIR / "week1_dataset_summary.png",
        width=10,
    )

    return files, subject_counts


# ============================================================
# 2. 피험자 A/B/C 파형 PNG
# ============================================================

def save_signal_examples(data_root):
    for subject in ["A", "B", "C"]:
        files = sorted(
            (data_root / subject).glob("*.csv")
        )

        if not files:
            print(f"[경고] {subject} 파일 없음")
            continue

        path = files[0]
        x = load_signal(path)

        t = np.arange(len(x)) / FS

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 5),
            sharex=True,
        )

        for ch in range(2):
            axes[ch].plot(
                t,
                x[:, ch],
                linewidth=0.6,
            )

            axes[ch].axvline(
                1.0,
                linestyle="--",
                linewidth=1.0,
            )

            axes[ch].axvline(
                2.0,
                linestyle="--",
                linewidth=1.0,
            )

            axes[ch].set_ylabel(
                f"Channel {ch + 1}"
            )

            axes[ch].grid(
                alpha=0.2
            )

        axes[0].text(
            0.17,
            0.93,
            "Grasp",
            transform=axes[0].transAxes,
            ha="center",
        )

        axes[0].text(
            0.50,
            0.93,
            "Rotation",
            transform=axes[0].transAxes,
            ha="center",
        )

        axes[0].text(
            0.83,
            0.93,
            "Stationary",
            transform=axes[0].transAxes,
            ha="center",
        )

        axes[1].set_xlabel("Time (s)")

        fig.suptitle(
            f"Subject {subject} sEMG Signal\n{path.name}",
            fontsize=14,
        )

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"signal_{subject}.png"

        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"[저장] {output_path} "
            f"({path.name})"
        )


# ============================================================
# 3. 250개 CSV 전체 구간별 특징 계산
# ============================================================

def extract_all_features(files):
    rows = []

    for index, path in enumerate(files, start=1):
        subject = path.parent.name
        x = load_signal(path)

        required_samples = SIGNAL_SECONDS * FS

        if len(x) < required_samples:
            print(
                f"[경고] 길이가 짧아서 제외: "
                f"{path} ({len(x)} samples)"
            )
            continue

        # 첫 3초 분석
        x = x[:required_samples]

        for phase_name, (
            start_sec,
            end_sec,
        ) in PHASES.items():

            start = int(start_sec * FS)
            end = int(end_sec * FS)

            segment = x[start:end]

            (
                rms,
                mav,
                std,
                peak,
                combined_rms,
            ) = calculate_features(segment)

            for ch in range(2):
                rows.append(
                    {
                        "Subject": subject,
                        "File": path.name,
                        "Phase": phase_name,
                        "Channel": f"Ch{ch + 1}",
                        "RMS": rms[ch],
                        "MAV": mav[ch],
                        "STD": std[ch],
                        "Peak": peak[ch],
                        "Combined_RMS": combined_rms,
                    }
                )

        if index % 25 == 0:
            print(
                f"[분석] {index}/{len(files)}"
            )

    stats = pd.DataFrame(rows)

    stats.to_csv(
        OUTPUT_DIR / "week1_all_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[저장] week1_all_features.csv "
        f"({len(stats)} rows)"
    )

    return stats


# ============================================================
# 4. 파지 / 회전 / 정지 RMS 비교
# ============================================================

def save_phase_rms_comparison(stats):
    phase_order = [
        "Grasp",
        "Rotation",
        "Stationary",
    ]

    channel_order = [
        "Ch1",
        "Ch2",
    ]

    mean_values = {}
    std_values = {}

    for channel in channel_order:
        means = []
        stds = []

        for phase in phase_order:
            values = stats[
                (stats["Phase"] == phase)
                & (stats["Channel"] == channel)
            ]["RMS"]

            means.append(values.mean())
            stds.append(values.std())

        mean_values[channel] = means
        std_values[channel] = stds

    x = np.arange(len(phase_order))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        x - width / 2,
        mean_values["Ch1"],
        width,
        yerr=std_values["Ch1"],
        capsize=4,
        label="Channel 1",
    )

    ax.bar(
        x + width / 2,
        mean_values["Ch2"],
        width,
        yerr=std_values["Ch2"],
        capsize=4,
        label="Channel 2",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Grasp\n(0~1 s)",
            "Rotation\n(1~2 s)",
            "Stationary\n(2~3 s)",
        ]
    )

    ax.set_ylabel("RMS")
    ax.set_title(
        "파지 · 회전 · 정지 구간별 평균 RMS"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "phase_rms_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "[저장] phase_rms_comparison.png"
    )


# ============================================================
# 5. A~E 사용자별 회전 RMS 분포
# ============================================================

def save_subject_rms_distribution(stats):
    # 같은 Trial에서 Ch1/Ch2가 각각 한 행씩 있기 때문에
    # Combined_RMS는 파일 단위로 하나만 남긴다.
    rotation = (
        stats[
            stats["Phase"] == "Rotation"
        ][
            [
                "Subject",
                "File",
                "Combined_RMS",
            ]
        ]
        .drop_duplicates(
            subset=[
                "Subject",
                "File",
            ]
        )
        .copy()
    )

    data = []

    for subject in SUBJECTS:
        values = rotation[
            rotation["Subject"] == subject
        ]["Combined_RMS"].to_numpy()

        data.append(values)

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.boxplot(
    data,
    tick_labels=SUBJECTS,
    showmeans=True,
    )

    ax.set_xlabel("Subject")
    ax.set_ylabel(
        "Combined RMS"
    )

    ax.set_title(
        "사용자별 회전 구간 RMS 분포"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "subject_rms_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "[저장] subject_rms_distribution.png"
    )

    return rotation


# ============================================================
# 6. 사용자별 회전 RMS 요약표
# ============================================================

def save_rotation_summary(rotation):
    summary = (
        rotation
        .groupby("Subject")["Combined_RMS"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        "피험자",
        "시행 수",
        "평균 RMS",
        "표준편차",
        "최솟값",
        "최댓값",
    ]

    display = summary.copy()

    for col in [
        "평균 RMS",
        "표준편차",
        "최솟값",
        "최댓값",
    ]:
        display[col] = display[col].map(
            lambda x: f"{x:.4f}"
        )

    save_table_png(
        display,
        "사용자별 회전 구간 RMS 요약",
        OUTPUT_DIR / "subject_rotation_rms_summary.png",
        width=11,
    )

    summary.to_csv(
        OUTPUT_DIR / "subject_rotation_rms_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary


# ============================================================
# 7. 구간별 RMS 요약표
# ============================================================

def save_phase_summary(stats):
    summary = (
        stats
        .groupby(
            [
                "Phase",
                "Channel",
            ]
        )["RMS"]
        .agg(
            [
                "count",
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        "구간",
        "채널",
        "샘플 수",
        "평균 RMS",
        "표준편차",
    ]

    display = summary.copy()

    display["평균 RMS"] = (
        display["평균 RMS"]
        .map(lambda x: f"{x:.4f}")
    )

    display["표준편차"] = (
        display["표준편차"]
        .map(lambda x: f"{x:.4f}")
    )

    save_table_png(
        display,
        "구간별 RMS 통계",
        OUTPUT_DIR / "phase_rms_summary.png",
        width=10,
    )

    summary.to_csv(
        OUTPUT_DIR / "phase_rms_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary


# ============================================================
# 8. 선행연구 비교표
# ============================================================

def save_prior_studies_table():
    studies = pd.DataFrame(
        [
            [
                "손목",
                50,
                8,
                "박수치기",
                "GAN + DNN",
                "97.94",
            ],
            [
                "전완",
                80,
                8,
                "스마트폰 잠금해제",
                "Siamese CNN",
                "92.06",
            ],
            [
                "전완",
                5,
                4,
                "손동작 6종",
                "DWT/EWT/EMD + CNN",
                "~95.62",
            ],
            [
                "이두·삼두",
                40,
                12,
                "손동작 3종",
                "CQT + CNN",
                "97.50",
            ],
            [
                "전완",
                21,
                4,
                "손 펴기",
                "DWT/CWT + CNN",
                "~99.21",
            ],
            [
                "손바닥",
                5,
                2,
                "문손잡이 회전",
                "CWT + DenseNet161",
                "94.00",
            ],
        ],
        columns=[
            "측정 부위",
            "인원",
            "채널",
            "동작",
            "특징 추출 + 모델",
            "정확도 (%)",
        ],
    )

    studies.to_csv(
        OUTPUT_DIR / "week1_prior_studies.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_table_png(
        studies,
        "sEMG 사용자 식별 선행연구 비교",
        OUTPUT_DIR / "week1_prior_studies.png",
        width=14,
        row_height=0.65,
        font_size=9,
    )


# ============================================================
# 9. 데이터 기반 관찰 후보 TXT
# ============================================================

def save_observation_candidates(
    stats,
    rotation_summary,
):
    lines = []

    # -----------------------------
    # 전체 구간 RMS
    # -----------------------------
    phase_means = (
        stats
        .groupby("Phase")["RMS"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    highest_phase = phase_means.index[0]

    lines.append(
        "[데이터 기반 관찰 후보]"
    )

    lines.append("")

    lines.append(
        "1. 전체 RMS 평균이 가장 큰 구간"
    )

    for phase, value in phase_means.items():
        lines.append(
            f"   - {phase}: {value:.6f}"
        )

    lines.append(
        f"   -> 가장 큰 구간: {highest_phase}"
    )

    lines.append("")

    # -----------------------------
    # 사용자별 회전 RMS
    # -----------------------------
    lines.append(
        "2. 사용자별 Rotation 구간 Combined RMS"
    )

    for _, row in rotation_summary.iterrows():
        lines.append(
            f"   - {row['피험자']}: "
            f"평균={row['평균 RMS']:.6f}, "
            f"표준편차={row['표준편차']:.6f}"
        )

    lines.append("")

    # -----------------------------
    # 채널 비교
    # -----------------------------
    channel_means = (
        stats[
            stats["Phase"] == "Rotation"
        ]
        .groupby("Channel")["RMS"]
        .mean()
    )

    lines.append(
        "3. Rotation 구간 채널별 평균 RMS"
    )

    for channel, value in channel_means.items():
        lines.append(
            f"   - {channel}: {value:.6f}"
        )

    lines.append("")

    lines.append(
        "※ 위 값은 관찰 결과입니다."
    )

    lines.append(
        "※ 정확도나 인증 가능성을 의미하는 것은 아닙니다."
    )

    lines.append(
        "※ 사용자 식별 성능은 별도의 모델 평가가 필요합니다."
    )

    text = "\n".join(lines)

    Path(
        OUTPUT_DIR / "week1_observation_candidates.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )

    print()
    print(text)

    print(
        "\n[저장] "
        "week1_observation_candidates.txt"
    )


# ============================================================
# Main
# ============================================================

def main():
    setup_korean_font()

    print(
        "=========================================="
    )

    print(
        "Week 1 sEMG Analysis"
    )

    print(
        "=========================================="
    )

    print(
        f"Python: {platform.python_version()}"
    )

    data_root = find_data_root()

    print(
        f"Data root: {data_root.resolve()}"
    )

    # 1. 데이터 구조
    files, subject_counts = analyze_dataset(
        data_root
    )

    print()
    print(
        "피험자별 파일 수:"
    )

    for subject in SUBJECTS:
        print(
            f"  {subject}: "
            f"{subject_counts.get(subject, 0)}"
        )

    # 2. A/B/C 신호
    save_signal_examples(
        data_root
    )

    # 3. 전체 특징 계산
    stats = extract_all_features(
        files
    )

    # 4. 구간별 RMS 그래프
    save_phase_rms_comparison(
        stats
    )

    # 5. 사용자별 Rotation RMS
    rotation = save_subject_rms_distribution(
        stats
    )

    # 6. 사용자별 요약표
    rotation_summary = save_rotation_summary(
        rotation
    )

    # 7. 구간별 통계표
    save_phase_summary(
        stats
    )

    # 8. 선행연구 비교표
    save_prior_studies_table()

    # 9. 관찰 후보
    save_observation_candidates(
        stats,
        rotation_summary,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "Week 1 분석 완료"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()