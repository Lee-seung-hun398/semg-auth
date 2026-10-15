"""Read-only Week 1 inspection and descriptive plots; no filtering or ML.

Run from any directory with the shared .venv Python. Marker-relative 2--3 s
is a requested, provisional interval, not a verified motion annotation.
"""
from pathlib import Path
from collections import Counter
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "step2" / "data"
OUT = ROOT / "step2" / "outputs" / "week1"
FS, LENGTH = 1000, 5000
COLS = [f"bandpass {i}" for i in range(1, 6)]
MUSCLES = ["Extensor carpi radialis", "Extensor digitorum",
           "Flexor digitorum superficialis", "Biceps brachii", "Triceps brachii"]
GESTURES = {"fist": "fist", "open": "open", "pro": "pronation", "extor": "supination"}
CLASSES = [f"{angle}_{g}" for angle in (30, 60) for g in GESTURES.values()]
CAUTION = "주의: mark=0초 및 Motion=2~3초는 미확정 가정"
LOG = []
FIGURES = []


def snapshot(paths):
    result = {}
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        st = p.stat()
        result[str(p)] = (h.hexdigest(), st.st_size, st.st_mtime_ns)
    return result


def savefig(fig, name):
    # Verify every Korean glyph used in tables/titles before rendering.
    from matplotlib.ft2font import FT2Font
    glyphs = FT2Font(str(Path("C:/Windows/Fonts/malgun.ttf"))).get_charmap()
    from matplotlib.text import Text
    for obj in fig.findobj(match=Text):
        for ch in obj.get_text():
            if "\uac00" <= ch <= "\ud7a3" and ord(ch) not in glyphs:
                raise RuntimeError(f"Missing Korean glyph: {ch}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    missing = [str(w.message) for w in caught if "Glyph" in str(w.message)]
    if missing:
        raise RuntimeError("Font rendering failure: " + str(missing))
    plt.close(fig)
    with Image.open(OUT / name) as img:
        img.verify()
    FIGURES.append(name)


def table(frame, name, title, footer=""):
    fig, ax = plt.subplots(figsize=(14, max(4, .43 * (len(frame) + 2) + (2.8 if footer else 0))))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=15)
    tbl = ax.table(cellText=frame.astype(str).values, colLabels=frame.columns,
                   cellLoc="center", loc="upper center", bbox=[0, .36 if footer else .03, 1, .60 if footer else .90])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(range(len(frame.columns)))
    for (row, _), cell in tbl.get_celld().items():
        cell.set_facecolor("#dbe9f4" if row == 0 else "#f4f6f8" if row % 2 else "white")
    if footer:
        ax.text(0, .02, footer, transform=ax.transAxes, fontsize=10, va="bottom", linespacing=1.6)
    savefig(fig, name)


def phase_axes(ax):
    for lo, hi, color, text in [(0, 2, "#edf1f5", "Rest (가정)"),
                                (2, 3, "#fff0cf", "Motion (가정)"),
                                (3, 5, "#edf1f5", "Rest (가정)")]:
        ax.axvspan(lo, hi, color=color, alpha=.55)
        ax.text((lo + hi) / 2, .96, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.axvline(2, color="gray", linestyle="--")
    ax.axvline(3, color="gray", linestyle="--")
    ax.set_xlim(0, 5)
    ax.set_ylabel("원본 진폭\n(단위 미확인)")
    ax.grid(alpha=.2)


def main():
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if not font.is_file():
        raise FileNotFoundError("Malgun Gothic required: " + str(font))
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False})
    files = sorted(DATA.glob("*.csv"))
    if len(files) != 64:
        raise ValueError(f"Expected 64 CSV, found {len(files)}")
    step1_files = sorted(p for p in (ROOT / "step1").rglob("*") if p.is_file())
    original = snapshot(files)
    step1_before = snapshot(step1_files)
    OUT.mkdir(parents=True, exist_ok=True)
    records, rms_rows, phase_rows = [], [], []
    examples = {}
    for path in files:
        match = re.fullmatch(r"([a-h])_(30|60) (fist|open|pro|extor)", path.stem)
        if not match:
            raise ValueError(f"Unknown filename: {path.name}")
        subject, angle, gesture = match.groups()
        d = pd.read_csv(path)
        if list(d.columns) != COLS + ["time", "onset", "mark"] or d.isna().any().any():
            raise ValueError(f"Unexpected columns or missing values: {path.name}")
        if not np.isfinite(d.to_numpy()).all():
            raise ValueError(f"Nonfinite values: {path.name}")
        if not np.allclose(np.diff(d.time), .001, rtol=0, atol=1e-9):
            raise ValueError(f"Unexpected time spacing: {path.name}")
        if not set(d.mark.unique()) <= {0, 1} or not set(d.onset.unique()) <= {0, 1}:
            raise ValueError(f"Unexpected marker values: {path.name}")
        marks = np.flatnonzero(d.mark.to_numpy() == 1)
        onset = np.flatnonzero(d.onset.to_numpy() == 1)
        if len(marks) != 30 or not np.array_equal(onset, marks[:1]):
            raise ValueError(f"Unexpected onset/mark: {path.name}")
        gaps = np.diff(marks)
        if not np.all(np.abs(gaps - LENGTH) <= 1):
            raise ValueError(f"Marker interval not near 5000: {path.name}")
        if any(i + LENGTH > len(d) for i in marks):
            raise ValueError(f"Incomplete 5 s trial: {path.name}")
        x = d[COLS].to_numpy()
        # Fixed-size marker-relative windows. No filtering or interpolation.
        # 4999 gaps create a 1-sample overlap; 5001 gaps leave a 1-sample gap.
        trials = np.stack([x[i:i + LENGTH] for i in marks])
        bins = np.sqrt(np.mean(trials.reshape(30, 5, 1000, 5) ** 2, axis=2))
        for trial, b in enumerate(bins, 1):
            phase_rows.append({"subject": subject, "angle": int(angle), "gesture": gesture,
                               "trial": trial, **{f"{i}~{i+1}s_mean_channel_RMS": float(v) for i, v in enumerate(b.mean(axis=1))}})
        records.append({"file": path.name, "subject": subject, "angle": int(angle),
                        "class": f"{angle}_{GESTURES[gesture]}", "rows": len(d), "trials": len(marks),
                        "gap_counts": dict(Counter(gaps.tolist()))})
        LOG.append(f"{path.name}: rows={len(d)}, mark indices(0-based)={marks.tolist()}, gap counts={dict(Counter(gaps.tolist()))}, onset={onset.tolist()}")
        if gesture == "fist":
            values = np.sqrt(np.mean(trials[:, 2000:3000, :] ** 2, axis=1))
            for trial, row in enumerate(values, 1):
                rms_rows.append({"subject": subject, "angle": int(angle), "trial": trial,
                                 "mark_index": int(marks[trial - 1]), **{f"Ch{c+1}": float(v) for c, v in enumerate(row)}})
            if subject == "a":
                examples[int(angle)] = (trials, marks, d.time.to_numpy()[marks])
    assert len({(r['subject'], r['class']) for r in records}) == 64
    meta = pd.DataFrame(records)
    dist = meta.groupby("class").agg(subject_count=("subject", "nunique"), trial_count=("trials", "sum")).reindex(CLASSES).reset_index()
    dist.to_csv(OUT / "class_distribution.csv", index=False, encoding="utf-8-sig")
    table(pd.read_csv(OUT / "class_distribution.csv"), "class_distribution.png", "8클래스 분포 — mark 이벤트 기준 반복 수")
    summary = [("피험자 수", meta.subject.nunique()), ("클래스 수", len(dist)), ("조건당 반복 수 (피험자별)", 30),
               ("전체 trial 수 (mark 기준)", int(meta.trials.sum())), ("sampling rate (Hz)", FS),
               ("sEMG 채널 수", 5), ("CSV 파일 수", len(meta)), ("CSV 전체 행 수", int(meta.rows.sum())),
               ("분석 trial 길이", "5초 / 5000 samples (mark 기준 잠정 창)"),
               ("실제 mark 간격", "4999~5001 samples"), ("mark 의미", "trial 시작점인지 미확정"),
               ("프로토콜", "휴식 2초 + 동작 1초 + 휴식 2초")]
    summary += [(f"{r['class']} trial 수", int(r['trial_count'])) for r in dist.to_dict("records")]
    pd.DataFrame(summary, columns=["항목", "값"]).to_csv(OUT / "week1_dataset_summary.csv", index=False, encoding="utf-8-sig")
    table(pd.read_csv(OUT / "week1_dataset_summary.csv"), "week1_dataset_summary.png", "Step 2 Week 1 데이터 요약")
    raw = pd.DataFrame(rms_rows)
    raw.to_csv(OUT / "trial_rms_audit.csv", index=False, encoding="utf-8-sig")
    phases = pd.DataFrame(phase_rows)
    phases.to_csv(OUT / "marker_phase_diagnostics.csv", index=False, encoding="utf-8-sig")
    population = []
    for ch, muscle in enumerate(MUSCLES, 1):
        row = {"channel": f"Ch{ch}", "muscle": muscle, "interval": "mark+2~3s (motion unconfirmed)", "unit": "original CSV amplitude (unspecified)"}
        for angle in (30, 60):
            vals = raw.loc[raw.angle == angle, f"Ch{ch}"]
            row.update({f"mean_{angle}": vals.mean(), f"std_{angle}": vals.std(ddof=1), f"n_{angle}": len(vals)})
        row["difference_60_minus_30"] = row["mean_60"] - row["mean_30"]
        population.append(row)
    pd.DataFrame(population).to_csv(OUT / "angle_rms_comparison.csv", index=False, encoding="utf-8-sig")
    # Plot the CSV readback, so the saved chart and CSV share the same numbers.
    pop = pd.read_csv(OUT / "angle_rms_comparison.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(5)
    for angle, offset, color in [(30, -.18, "#3576ab"), (60, .18, "#dc8434")]:
        bars = ax.bar(positions + offset, pop[f"mean_{angle}"], width=.36,
                      yerr=pop[f"std_{angle}"], capsize=4, color=color, label=f"{angle}° (n=240)")
        ax.bar_label(bars, labels=[f"{v:.6f}" for v in pop[f"mean_{angle}"]], padding=4, fontsize=9)
    ax.set_xticks(positions, pop.channel)
    ax.set_ylabel("RMS (원본 진폭 단위 미확인)")
    ax.set_title("fist: mark 기준 2~3초의 RMS 평균 ± 표본 표준편차\n" + CAUTION)
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.text(.1, .01, "8명 × 30회/각도. trial 간 SD(ddof=1); 독립 표본의 신뢰구간이 아님. 음의 오차막대는 평균-SD 표시.", fontsize=9)
    savefig(fig, "angle_rms_comparison.png")
    trial30, mark30, times30 = examples[30]
    trial60, mark60, times60 = examples[60]
    full_rms = np.sqrt(np.mean(trial30 ** 2, axis=(1, 2)))
    selected = int(np.argsort(full_rms, kind="stable")[14])
    LOG.append(f"대표 선택: a/30/fist의 완전한 유한값 30개 창 중 전체 5초·5채널 RMS 중앙 순위(정렬 15번째, 동률 원래 순서). trial={selected+1}, RMS={full_rms[selected]:.12g}. 임상/동작 타이밍 정상 판정이 아님.")
    for angle, marks, times in [(30, mark30, times30), (60, mark60, times60)]:
        LOG.append(f"비교: a/{angle}/fist trial={selected+1}, mark index={int(marks[selected])}, source time={times[selected]}, rows=[{marks[selected]}, {marks[selected]+LENGTH}), motion-assumed rows=[{marks[selected]+2000}, {marks[selected]+3000})")
    t = np.arange(LENGTH) / FS
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    for c, ax in enumerate(axes):
        phase_axes(ax); ax.plot(t, trial30[selected, :, c], lw=.55, color="#3576ab")
        ax.set_title(f"Ch{c+1} {MUSCLES[c]}", loc="left", fontsize=11)
    axes[-1].set_xlabel("mark 기준 경과 시간 (s)")
    fig.suptitle(f"a / 30° fist / trial {selected+1} / mark index {mark30[selected]}\n{CAUTION}")
    fig.tight_layout(rect=(0, 0, 1, .95)); savefig(fig, "ch5_example.png")
    fig, axes = plt.subplots(5, 2, figsize=(17, 12), sharex=True, sharey="row")
    for c in range(5):
        for column, (angle, data, color) in enumerate([(30, trial30, "#3576ab"), (60, trial60, "#dc8434")]):
            ax = axes[c, column]; phase_axes(ax); ax.plot(t, data[selected, :, c], lw=.55, color=color)
            ax.set_title(f"Ch{c+1} {MUSCLES[c]} — {angle}°", fontsize=10)
    for ax in axes[-1]: ax.set_xlabel("mark 기준 경과 시간 (s)")
    fig.suptitle(f"a / fist / 같은 trial {selected+1} — 채널별 좌우 공통 y축\n{CAUTION}")
    fig.tight_layout(rect=(0, 0, 1, .95)); savefig(fig, "angle_comparison_fist.png")
    design = pd.DataFrame([{"Class": c, "Gesture": c.split("_", 1)[1], "Elbow angle": c.split("_")[0] + "°", "정의": c.split("_")[0] + "°에서 " + c.split("_", 1)[1] + " 수행"} for c in CLASSES])
    footer = ("1. Gesture × Angle: 동작과 자세의 조합을 하나의 라벨로 표현하여 조건을 함께 구분하는 설계. 성능은 아직 미평가.\n"
              "2. 별도 예측: 각각 4종/2종으로 해석 가능하나, 두 예측의 오차가 조합 결과에 함께 영향을 줌.\n"
              "   공동 8-class는 조합을 직접 표현하지만 조건별 데이터 수가 줄고 새 조합마다 라벨이 추가됨.\n"
              "3. 각도 3종 → 12클래스, 4종 → 16클래스. 동일 반복 수 유지 시 수집량 증가; 고정 수집량이면 클래스별 자료 감소.\n"
              "4. Ch1 Extensor carpi radialis / Ch2 Extensor digitorum / Ch3 Flexor digitorum superficialis\n"
              "   Ch4 Biceps brachii / Ch5 Triceps brachii (근육 대응은 저장소 README 기준).")
    table(design, "class_design.png", "Gesture × Elbow angle 클래스 설계안", footer)
    differences = pop.difference_60_minus_30.abs()
    obs = ["Week 1 관찰 메모 — 모델 학습 없음", CAUTION,
           "mark를 trial 시작점으로 확정할 근거가 부족하여 아래 RMS는 mark 기준 2~3초의 잠정 구간 결과임.",
           "신호 진폭 단위는 CSV/README에서 확인되지 않아 원본 수치 단위를 유지함.",
           "프로토콜의 2초 휴식/1초 동작/2초 휴식과 비교하기 위해 mark 뒤 1초 bin별 RMS를 점검함."]
    for subset_name, frame in [("전체 64조건", phases), ("fist 16조건", phases[phases.gesture == "fist"])]:
        cols = [f"{i}~{i+1}s_mean_channel_RMS" for i in range(5)]
        means = frame[cols].mean()
        obs.append(f"{subset_name} mark-relative 1초 bin RMS 평균(5채널 평균): {means.to_dict()}")
        obs.append(f"{subset_name} trial별 최대 1초 bin 빈도: {frame[cols].idxmax(axis=1).value_counts().to_dict()}")
    obs += ["fist의 mark+1~2초 RMS가 큰 패턴은 mark+2~3초를 유일한 Motion으로 확정하는 가정과 맞지 않음.",
            "mark가 신호 동작 시작/프로토콜 시작/다른 기준점 중 무엇인지는 직접적인 이벤트 의미 자료 없이 확정할 수 없음.",
            "단순 최대 진폭으로 타이밍을 자동 보정하지 않았으며, 1~2초 RMS는 진단에만 사용함.",
            f"절대 평균 RMS 차이 최대: {pop.loc[differences.idxmax(), 'channel']}",
            f"절대 평균 RMS 차이 최소: {pop.loc[differences.idxmin(), 'channel']}"]
    subject_means = raw.groupby(["subject", "angle"])[[f"Ch{i}" for i in range(1, 6)]].mean()
    for row in pop.to_dict("records"):
        ch = row['channel']
        obs.append(f"{ch}: 30°={row['mean_30']:.9g} ± {row['std_30']:.9g}; 60°={row['mean_60']:.9g} ± {row['std_60']:.9g}; 60−30={row['difference_60_minus_30']:.9g}; 더 큰 평균={60 if row['difference_60_minus_30'] > 0 else 30}°")
        for angle in (30, 60):
            vals = subject_means.xs(angle, level="angle")[ch]
            cv = vals.std(ddof=1) / vals.mean()
            obs.append(f"  {angle}° 피험자별 평균 범위={vals.min():.9g}~{vals.max():.9g}, 피험자 평균의 SD={vals.std(ddof=1):.9g}, CV={cv:.3f}; 개인차를 수치로 기술하며 크다는 이분법 판정은 하지 않음.")
        paired = subject_means.xs(60, level="angle")[ch] - subject_means.xs(30, level="angle")[ch]
        obs.append(f"  피험자별 60−30 평균 차이: {paired.to_dict()}; 60°가 큰 피험자={int((paired>0).sum())}/8")
    for c in range(5):
        observations = []
        for angle, data in [(30, trial30), (60, trial60)]:
            sig = data[selected, :, c]
            peak = int(np.argmax(np.abs(sig)))
            observations.append(f"{angle}°: 최대절대진폭={abs(sig[peak]):.9g}, 위치={peak/FS:.3f}s, 1초 bin RMS={np.sqrt(np.mean(sig.reshape(5,1000)**2,axis=1)).tolist()}")
        obs.append(f"대표 파형 Ch{c+1}: " + " / ".join(observations))
    obs += ["대표 파형 하나의 차이는 전체 피험자에게 일반화하지 않음. 원본 진폭 그대로 표시하고 채널별 비교 y축을 공유함.",
            "RMS는 240개 trial/각도의 기술 통계이며 동일 피험자의 반복은 독립 피험자가 아님. 통계 검정, 정확도 평가 또는 분류 가능성 판단은 하지 않음.",
            "mark 간격 4999/5001에 대해서도 5000샘플 창을 그대로 사용: 일부 경계 1샘플 중복/공백이 발생. 보간/재표본화 없음.",
            "trial 시작 의미가 확정되기 전 실제 Motion RMS라고 단정해서 제출하지 말 것.", "", "파일별 검증 및 대표 trial 선택 로그:"] + LOG
    assert snapshot(files) == original, "Original CSV hash/mtime changed"
    assert sorted(p for p in DATA.glob('*.csv')) == files
    assert sorted(p for p in (ROOT / 'step1').rglob('*') if p.is_file()) == step1_files
    assert snapshot(step1_files) == step1_before, "Step 1 changed"
    obs += ["", f"검증: PNG {len(FIGURES)}개 실제 저장 및 PIL 디코딩 검증 완료.",
            "한글: Windows malgun.ttf 명시, 사용 한글 glyph 존재 및 렌더링 누락 경고 없음.",
            "요약표/분포표/RMS PNG는 저장 CSV를 다시 읽어 생성하여 값 일치.",
            "원본 CSV 64개 SHA256/size/mtime_ns 동일; Step 1 전체 파일 SHA256/size/mtime_ns 및 파일 목록 동일."]
    (OUT / "week1_observations.txt").write_text("\n".join(obs) + "\n", encoding="utf-8")
    (OUT / "analysis_log.txt").write_text("\n".join(obs[-5:] + LOG) + "\n", encoding="utf-8")
    print("WARNING:", CAUTION)
    print(pop[["channel", "mean_30", "std_30", "mean_60", "std_60", "difference_60_minus_30"]].to_string(index=False))
    print("Selected trial:", selected + 1, "marks:", mark30[selected], mark60[selected])
    print("Outputs:", OUT, "PNG:", len(FIGURES))
    print("Integrity verified: all CSVs and Step 1 unchanged")


if __name__ == "__main__":
    main()
