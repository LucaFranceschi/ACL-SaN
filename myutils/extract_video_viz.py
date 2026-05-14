"""
extract_video_viz.py
--------------------
For each video in a directory, saves:
  - <video_name>_frame.png   : the central frame
  - <video_name>_waveform.png: the audio waveform

Dependencies:
    pip install opencv-python moviepy matplotlib numpy

Usage:
    python extract_video_viz.py --input_dir ./videos --output_dir ./output
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ──────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────

def extract_central_frame(video_path: Path) -> np.ndarray:
    """Return the central frame of a video as an RGB numpy array."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Could not read frame count for: {video_path}")

    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError(f"Could not read frame {mid_frame} from: {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_frame_figure(frame: np.ndarray, out_path: Path, title: str) -> None:
    """Save the frame as a clean matplotlib figure."""
    h, w = frame.shape[:2]
    dpi = 150
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(frame)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  ✓ Frame   → {out_path}")


# ──────────────────────────────────────────────
# Waveform extraction
# ──────────────────────────────────────────────

def extract_audio_waveform(video_path: Path):
    """
    Return (samples, sample_rate) from the video's audio track.
    Falls back gracefully if there is no audio.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        raise ImportError("moviepy is required: pip install moviepy")

    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            return None, None

        fps = clip.audio.fps
        # to_soundarray returns shape (n_samples,) or (n_samples, n_channels)
        samples = clip.audio.to_soundarray(fps=fps)

    # Mono-mix if stereo
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    return samples.astype(np.float32), int(fps)


def save_waveform_figure(
    samples: np.ndarray,
    sample_rate: int,
    out_path: Path,
    title: str,
) -> None:
    """Save the audio waveform as a matplotlib figure."""
    duration = len(samples) / sample_rate
    time_axis = np.linspace(0, duration, num=len(samples))

    # Downsample for plotting speed (keep at most 200 k points)
    max_pts = 200_000
    if len(samples) > max_pts:
        step = len(samples) // max_pts
        time_axis = time_axis[::step]
        samples = samples[::step]

    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=150)
    ax.plot(time_axis, samples, linewidth=0.4, color="#2563eb", alpha=0.85)
    ax.fill_between(time_axis, samples, alpha=0.15, color="#2563eb")

    ax.set_xlim(0, duration)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Waveform → {out_path}")


def save_no_audio_figure(out_path: Path, title: str) -> None:
    """Placeholder figure when a video has no audio track."""
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=150)
    ax.text(
        0.5, 0.5, "No audio track",
        ha="center", va="center", fontsize=14, color="gray",
        transform=ax.transAxes,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Waveform → {out_path}  (no audio — placeholder saved)")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def process_video(video_path: Path, output_dir: Path) -> None:
    stem = video_path.stem
    print(f"\n▶ {video_path.name}")

    # --- Central frame ---
    try:
        frame = extract_central_frame(video_path)
        frame_out = output_dir / f"{stem}_frame.png"
        save_frame_figure(frame, frame_out, title=f"{stem} — central frame")
    except Exception as exc:
        print(f"  ✗ Frame extraction failed: {exc}", file=sys.stderr)

    # --- Audio waveform ---
    try:
        samples, sr = extract_audio_waveform(video_path)
        waveform_out = output_dir / f"{stem}_waveform.png"
        if samples is None:
            save_no_audio_figure(waveform_out, title=f"{stem} — audio waveform")
        else:
            save_waveform_figure(samples, sr, waveform_out, title=f"{stem} — audio waveform")
    except Exception as exc:
        print(f"  ✗ Waveform extraction failed: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Extract central frame + audio waveform from every video in a directory."
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        default=Path("."),
        help="Directory containing video files (default: current directory)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        default=None,
        help="Directory to write output images (default: <input_dir>/video_viz)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir / "video_viz"
    )

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No video files found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(videos)} video(s) in '{input_dir}'")
    print(f"Saving output to '{output_dir}'\n{'─' * 50}")

    for video_path in videos:
        process_video(video_path, output_dir)

    print(f"\n{'─' * 50}\nDone. {len(videos)} video(s) processed.")


if __name__ == "__main__":
    main()