#!/usr/bin/env python3
# 使用方法（请在 LOVON 项目根目录执行）：
#
# 1. 分析视频并输出逐帧 CSV（默认保存为"原视频名_adaptive_blur.csv"）：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4
#
# 2. 同时生成变化曲线（需要 matplotlib）：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 \
#        --plot path/to/threshold_curve.png
#
# 2b. 单独画 Adaptive Threshold 曲线：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 \
#        --plot-adaptive-only path/to/adaptive_threshold.png
#
# 2c. 保存阈值变化点对应帧图像：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 \
#        --save-frames path/to/frames_dir
#
# 3. 一键保存全部三种图（推荐）：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 --all
#    → 自动生成：
#      - <video>_full.png         （Laplacian variance + Adaptive threshold 双面板）
#      - <video>_adaptive.png     （Adaptive threshold 单独曲线）
#      - <video>_frames/          （阈值变化点帧图像）
#
# 4. 指定输出和门控参数：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 \
#        --csv path/to/results.csv --initial-threshold 150 \
#        --update-interval 2 --quantile 0.2 --alpha 0.2

"""Analyze the adaptive blur threshold over a prerecorded video."""

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.adaptive_blur_threshold import AdaptiveBlurThreshold


CSV_FIELDS = (
    "frame_index",
    "timestamp_s",
    "laplacian_variance",
    "threshold_used",
    "is_blurry",
    "threshold_after_update",
    "window_updated",
    "window_quantile",
    "window_sample_count",
)


# ── Global style ──────────────────────────────────────────────────
_STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "savefig.facecolor": "#FAFAFA",
}
_COLORS = {
    "variance": "#8EACC0",       # soft steel blue fill
    "variance_line": "#5A9DB8",  # darker blue line
    "threshold": "#E04848",      # vibrant red
    "threshold_fill": "#E0484822",
    "blurry": "#3B82F6",         # bright blue
    "update": "#1A1A1A",         # near-black
    "grid": "#E0E0E0",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the adaptive Laplacian blur threshold along a video and "
            "save frame-level results to CSV."
        )
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Output CSV path (default: <video>_adaptive_blur.csv)",
    )
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="Output path for the full two-panel PNG (Laplacian + threshold)",
    )
    parser.add_argument(
        "--plot-adaptive-only", type=Path, default=None,
        help="Output path for the single-panel adaptive threshold PNG",
    )
    parser.add_argument(
        "--save-frames", type=Path, default=None,
        help="Directory to save frame images at each threshold update point",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=(
            "Auto-generate all three outputs next to the video file: "
            "<stem>_full.png, <stem>_adaptive.png, <stem>_frames/"
        ),
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--initial-threshold", type=float, default=150.0)
    parser.add_argument("--update-interval", type=float, default=2.0)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.2)
    return parser.parse_args()


def resolve_csv_path(video_path, requested_path):
    if requested_path is not None:
        return requested_path
    return video_path.with_name(f"{video_path.stem}_adaptive_blur.csv")


def frame_timestamp(capture, frame_index, fps, previous_timestamp,
                    use_metadata_timestamp=True):
    metadata_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    indexed_timestamp = frame_index / fps
    if frame_index == 0:
        if use_metadata_timestamp and math.isfinite(metadata_timestamp):
            return max(0.0, metadata_timestamp)
        return 0.0
    if (use_metadata_timestamp and math.isfinite(metadata_timestamp)
            and metadata_timestamp > previous_timestamp):
        return metadata_timestamp
    if indexed_timestamp > previous_timestamp:
        return indexed_timestamp
    return previous_timestamp + 1.0 / fps


# ── Plot 1: Full two-panel (Laplacian variance + Adaptive threshold) ──
def save_plot(output_path, timestamps, variances, thresholds,
              blurry_timestamps, blurry_variances,
              update_timestamps, update_thresholds):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib; pip install matplotlib"
        ) from exc

    plt.rcParams.update(_STYLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_sig, ax_thr) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.10},
    )

    # ── top panel: Laplacian variance ──
    ax_sig.fill_between(timestamps, variances, alpha=0.30,
                        color=_COLORS["variance"], linewidth=0)
    ax_sig.plot(timestamps, variances, color=_COLORS["variance_line"],
                linewidth=0.9, label="Laplacian variance", zorder=2)
    ax_sig.step(timestamps, thresholds, where="post",
                color=_COLORS["threshold"], linewidth=1.6,
                label="Threshold used", zorder=3)
    if blurry_timestamps:
        ax_sig.scatter(blurry_timestamps, blurry_variances,
                       color=_COLORS["blurry"], s=6, alpha=0.45,
                       label="Rejected (blurry)", zorder=4, edgecolors="none")
    ax_sig.set_ylabel("Laplacian variance")
    ax_sig.legend(loc="upper right", framealpha=0.85, fontsize=10)
    ax_sig.spines["top"].set_visible(False)
    ax_sig.spines["right"].set_visible(False)

    # ── bottom panel: adaptive threshold ──
    ax_thr.step(timestamps, thresholds, where="post",
                color=_COLORS["threshold"], linewidth=2.0,
                label="Adaptive threshold")
    ax_thr.fill_between(timestamps, thresholds, step="post",
                        alpha=0.08, color=_COLORS["threshold"])
    if update_timestamps:
        ax_thr.scatter(update_timestamps, update_thresholds,
                       color=_COLORS["update"], s=28, zorder=5,
                       label="Window update", edgecolors="white", linewidths=0.8)
        for idx, (ut, uv) in enumerate(
                zip(update_timestamps, update_thresholds)):
            ax_thr.annotate(
                f"#{idx+1}", (ut, uv),
                textcoords="offset points", xytext=(0, 12),
                fontsize=7.5, ha="center", fontweight="bold",
                color=_COLORS["update"],
                arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                lw=0.6, shrinkA=0, shrinkB=4),
            )
        ax_thr.legend(loc="upper right", framealpha=0.85, fontsize=10)
    ax_thr.set_xlabel("Video time (s)")
    ax_thr.set_ylabel("Adaptive threshold")
    ax_thr.spines["top"].set_visible(False)
    ax_thr.spines["right"].set_visible(False)
    ax_thr.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: Adaptive threshold only ──
def save_adaptive_only_plot(output_path, timestamps, thresholds,
                            update_timestamps, update_thresholds):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib; pip install matplotlib"
        ) from exc

    plt.rcParams.update(_STYLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.step(timestamps, thresholds, where="post",
            color=_COLORS["threshold"], linewidth=2.2,
            label="Adaptive threshold")
    ax.fill_between(timestamps, thresholds, step="post",
                    alpha=0.10, color=_COLORS["threshold"])
    if update_timestamps:
        ax.scatter(update_timestamps, update_thresholds,
                   color=_COLORS["update"], s=30, zorder=5,
                   label="Window update", edgecolors="white", linewidths=0.8)
        for idx, (ut, uv) in enumerate(
                zip(update_timestamps, update_thresholds)):
            ax.annotate(
                f"#{idx+1}", (ut, uv),
                textcoords="offset points", xytext=(0, 12),
                fontsize=8, ha="center", fontweight="bold",
                color=_COLORS["update"],
                arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                lw=0.6, shrinkA=0, shrinkB=4),
            )
        ax.legend(loc="upper right", framealpha=0.85, fontsize=10)

    ax.set_xlabel("Video time (s)")
    ax.set_ylabel("Adaptive threshold")
    ax.set_title("Adaptive Blur Threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Core analysis ──
def analyze_video(args):
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    # --all: auto-set all output paths
    if args.all:
        stem = args.video.stem
        parent = args.video.parent
        args.plot = args.plot or parent / f"{stem}_full.png"
        args.plot_adaptive_only = (
            args.plot_adaptive_only or parent / f"{stem}_adaptive.png"
        )
        args.save_frames = args.save_frames or parent / f"{stem}_frames"

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    metadata_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = args.fps if args.fps is not None else metadata_fps
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(
            "Video FPS is unavailable; provide a positive value with --fps."
        )

    gate = AdaptiveBlurThreshold(
        initial_threshold=args.initial_threshold,
        update_interval=args.update_interval,
        quantile=args.quantile,
        alpha=args.alpha,
    )
    csv_path = resolve_csv_path(args.video, args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps, variances, thresholds = [], [], []
    blurry_timestamps, blurry_variances = [], []
    update_timestamps, update_thresholds = [], []
    saved_frame_paths = []
    blurry_count = update_count = frame_index = 0
    previous_timestamp = -1.0

    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                timestamp = frame_timestamp(
                    capture, frame_index, fps, previous_timestamp,
                    use_metadata_timestamp=args.fps is None,
                )
                previous_timestamp = timestamp
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                threshold_used = gate.threshold
                is_blurry = lap_var < threshold_used
                threshold_update = gate.observe(lap_var, timestamp=timestamp)

                if is_blurry:
                    blurry_count += 1
                    blurry_timestamps.append(timestamp)
                    blurry_variances.append(lap_var)

                row = {
                    "frame_index": frame_index,
                    "timestamp_s": f"{timestamp:.6f}",
                    "laplacian_variance": f"{lap_var:.6f}",
                    "threshold_used": f"{threshold_used:.6f}",
                    "is_blurry": int(is_blurry),
                    "threshold_after_update": f"{gate.threshold:.6f}",
                    "window_updated": int(threshold_update is not None),
                    "window_quantile": "",
                    "window_sample_count": "",
                }

                if threshold_update is not None:
                    update_count += 1
                    update_timestamps.append(timestamp)
                    update_thresholds.append(threshold_update.threshold)
                    row["window_quantile"] = (
                        f"{threshold_update.lower_quantile:.6f}")
                    row["window_sample_count"] = threshold_update.sample_count
                    print(
                        f"[Update {update_count:03d}] t={timestamp:.2f}s, "
                        f"frames={threshold_update.sample_count}, "
                        f"q{args.quantile:.2f}="
                        f"{threshold_update.lower_quantile:.2f}, "
                        f"threshold {threshold_update.previous_threshold:.2f}"
                        f" -> {threshold_update.threshold:.2f}"
                    )
                    if args.save_frames is not None:
                        fname = f"frame_{update_count:03d}_t{timestamp:.2f}s.png"
                        fpath = args.save_frames / fname
                        cv2.imwrite(str(fpath), frame)
                        saved_frame_paths.append(fpath)

                writer.writerow(row)
                timestamps.append(timestamp)
                variances.append(lap_var)
                thresholds.append(threshold_used)
                frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames could be decoded from: {args.video}")

    # ── generate plots ──
    if args.plot is not None:
        save_plot(args.plot, timestamps, variances, thresholds,
                  blurry_timestamps, blurry_variances,
                  update_timestamps, update_thresholds)

    if args.plot_adaptive_only is not None:
        save_adaptive_only_plot(args.plot_adaptive_only, timestamps,
                                thresholds, update_timestamps,
                                update_thresholds)

    duration = timestamps[-1]
    print(f"\n{'='*50}")
    print(f"  Analysis complete")
    print(f"  Video        : {args.video}")
    print(f"  Frames       : {frame_index}  FPS: {fps:.3f}  Duration: {duration:.2f}s")
    print(f"  Blurry       : {blurry_count} ({100*blurry_count/frame_index:.1f}%)")
    print(f"  Updates      : {update_count}")
    print(f"  Threshold    : {gate.initial_threshold:.2f} -> {gate.threshold:.2f}")
    print(f"  CSV          : {csv_path}")
    if args.plot:
        print(f"  Full plot    : {args.plot}")
    if args.plot_adaptive_only:
        print(f"  Adaptive plot: {args.plot_adaptive_only}")
    if args.save_frames:
        print(f"  Frames dir   : {args.save_frames} ({len(saved_frame_paths)} images)")
    print(f"{'='*50}")


def main():
    args = parse_args()
    try:
        analyze_video(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
