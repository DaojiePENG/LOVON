#!/usr/bin/env python3
# 使用方法（请在 LOVON 项目根目录执行）：
#
# 1. 分析视频并输出逐帧 CSV（默认保存为“原视频名_adaptive_blur.csv”）：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4
#
# 2. 同时生成变化曲线（需要 matplotlib）：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 \
#        --plot path/to/threshold_curve.png
#
# 3. 指定输出和门控参数：
#    python scripts/analyze_blur_threshold_video.py path/to/video.mp4 \
#        --csv path/to/results.csv --initial-threshold 150 \
#        --update-interval 2 --quantile 0.2 --alpha 0.2
#
# CSV 中 threshold_used 是当前帧实际使用的判定阈值，threshold_after_update
# 是吸收当前帧的原始 Laplacian 方差后得到的阈值，用于后续帧。窗口更新时，
# 终端会输出更新时间、分位数、窗口帧数以及更新前后的阈值。

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the adaptive Laplacian blur threshold along a video and "
            "save frame-level results to CSV."
        )
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <video>_adaptive_blur.csv)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional output path for a PNG threshold curve",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video FPS when its metadata is missing or inaccurate",
    )
    parser.add_argument(
        "--initial-threshold",
        type=float,
        default=150.0,
        help="Initial Laplacian-variance threshold (default: 150)",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=2.0,
        help="Threshold update interval in video seconds (default: 2)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.2,
        help="Raw-variance quantile used by each window (default: 0.2)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Exponential smoothing factor (default: 0.2)",
    )
    return parser.parse_args()


def resolve_csv_path(video_path, requested_path):
    if requested_path is not None:
        return requested_path
    return video_path.with_name(f"{video_path.stem}_adaptive_blur.csv")


def frame_timestamp(
    capture,
    frame_index,
    fps,
    previous_timestamp,
    use_metadata_timestamp=True,
):
    """Use the video timeline, falling back to frame index divided by FPS."""

    metadata_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    indexed_timestamp = frame_index / fps

    if frame_index == 0:
        if use_metadata_timestamp and math.isfinite(metadata_timestamp):
            return max(0.0, metadata_timestamp)
        return 0.0
    if (
        use_metadata_timestamp
        and math.isfinite(metadata_timestamp)
        and metadata_timestamp > previous_timestamp
    ):
        return metadata_timestamp
    if indexed_timestamp > previous_timestamp:
        return indexed_timestamp
    return previous_timestamp + 1.0 / fps


def save_plot(
    output_path,
    timestamps,
    variances,
    thresholds,
    blurry_timestamps,
    blurry_variances,
    update_timestamps,
    update_thresholds,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib; install it with `pip install matplotlib`."
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (signal_axis, threshold_axis) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )

    signal_axis.plot(
        timestamps,
        variances,
        color="0.55",
        linewidth=0.8,
        label="Laplacian variance",
    )
    signal_axis.step(
        timestamps,
        thresholds,
        where="post",
        color="tab:red",
        linewidth=1.5,
        label="Threshold used",
    )
    if blurry_timestamps:
        signal_axis.scatter(
            blurry_timestamps,
            blurry_variances,
            color="tab:blue",
            s=8,
            label="Rejected frame",
            zorder=3,
        )
    signal_axis.set_ylabel("Laplacian variance")
    signal_axis.grid(alpha=0.25)
    signal_axis.legend(loc="best")

    threshold_axis.step(
        timestamps,
        thresholds,
        where="post",
        color="tab:red",
        linewidth=1.8,
    )
    if update_timestamps:
        threshold_axis.scatter(
            update_timestamps,
            update_thresholds,
            color="black",
            s=18,
            label="Window update",
            zorder=3,
        )
        threshold_axis.legend(loc="best")
    threshold_axis.set_xlabel("Video time (s)")
    threshold_axis.set_ylabel("Adaptive threshold")
    threshold_axis.grid(alpha=0.25)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_video(args):
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    metadata_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = args.fps if args.fps is not None else metadata_fps
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video FPS is unavailable; provide a positive value with --fps.")

    gate = AdaptiveBlurThreshold(
        initial_threshold=args.initial_threshold,
        update_interval=args.update_interval,
        quantile=args.quantile,
        alpha=args.alpha,
    )
    csv_path = resolve_csv_path(args.video, args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = []
    variances = []
    thresholds = []
    blurry_timestamps = []
    blurry_variances = []
    update_timestamps = []
    update_thresholds = []
    blurry_count = 0
    update_count = 0
    frame_index = 0
    previous_timestamp = -1.0

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                timestamp = frame_timestamp(
                    capture,
                    frame_index,
                    fps,
                    previous_timestamp,
                    use_metadata_timestamp=args.fps is None,
                )
                previous_timestamp = timestamp
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_variance = float(
                    cv2.Laplacian(gray, cv2.CV_64F).var()
                )

                threshold_used = gate.threshold
                is_blurry = laplacian_variance < threshold_used
                threshold_update = gate.observe(
                    laplacian_variance, timestamp=timestamp
                )

                if is_blurry:
                    blurry_count += 1
                    blurry_timestamps.append(timestamp)
                    blurry_variances.append(laplacian_variance)

                row = {
                    "frame_index": frame_index,
                    "timestamp_s": f"{timestamp:.6f}",
                    "laplacian_variance": f"{laplacian_variance:.6f}",
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
                        f"{threshold_update.lower_quantile:.6f}"
                    )
                    row["window_sample_count"] = threshold_update.sample_count
                    print(
                        f"[Update {update_count:03d}] t={timestamp:.2f}s, "
                        f"frames={threshold_update.sample_count}, "
                        f"q{args.quantile:.2f}="
                        f"{threshold_update.lower_quantile:.2f}, "
                        f"threshold {threshold_update.previous_threshold:.2f} "
                        f"-> {threshold_update.threshold:.2f}"
                    )

                writer.writerow(row)
                timestamps.append(timestamp)
                variances.append(laplacian_variance)
                thresholds.append(threshold_used)
                frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames could be decoded from: {args.video}")

    if args.plot is not None:
        save_plot(
            args.plot,
            timestamps,
            variances,
            thresholds,
            blurry_timestamps,
            blurry_variances,
            update_timestamps,
            update_thresholds,
        )

    duration = timestamps[-1]
    print("\nAnalysis complete")
    print(f"  Video: {args.video}")
    print(f"  Frames / FPS / duration: {frame_index} / {fps:.3f} / {duration:.2f}s")
    print(
        f"  Blurry frames: {blurry_count} "
        f"({100.0 * blurry_count / frame_index:.2f}%)"
    )
    print(f"  Threshold updates: {update_count}")
    print(
        f"  Initial / final threshold: "
        f"{gate.initial_threshold:.2f} / {gate.threshold:.2f}"
    )
    print(f"  CSV: {csv_path}")
    if args.plot is not None:
        print(f"  Plot: {args.plot}")


def main():
    args = parse_args()
    try:
        analyze_video(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
