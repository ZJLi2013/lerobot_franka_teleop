#!/usr/bin/env python3
"""
Video post-process: pick a ROI on one frame, then crop every frame to that ROI
and resize back to the source video resolution.

Workflow:
  1) Extract a reference frame (or use pick-roi directly on the video).
  2) pick-roi: interactive ROI selection (needs GUI), saves ROI JSON.
  3) process: apply ROI to the full video and write a new file.

You can also create/edit the ROI JSON by hand after saving a frame as image.

Note on AV1 inputs/outputs (LeRobot datasets):
  LeRobot stores videos as AV1 in MP4 (libsvtav1, yuv420p, g=2, crf=30).
  If your OpenCV/FFmpeg build only ships a hardware AV1 decoder, you may see
  "Your platform doesn't support hardware accelerated AV1 decoding".
  This script auto-uses PyAV (installed with `lerobot`) for decoding when it
  is available, which decodes AV1 in software via libdav1d. As a fallback, the
  OpenCV path explicitly disables hardware acceleration.

  `process` writes output with PyAV using the same encoder settings as LeRobot
  by default (`--codec libsvtav1`), so the result can be dropped back into a
  LeRobot dataset directly. Use `--codec opencv` if you want OpenCV's
  cv2.VideoWriter (with `--fourcc`) instead.

Usage examples
--------------

1) Extract one frame as an image (optional, for offline ROI picking).
   `-n` is the 0-based frame index.

       python3 scripts/tools/video_roi_reframe.py extract-frame \\
           /path/to/input.mp4 -o ref.png -n 0

2) Pick a ROI and save it as JSON. Choose ONE of the two forms.
   In the OpenCV window: drag a rectangle, press ENTER/SPACE to confirm, C to cancel.

   a) Pick directly on a video frame (needs GUI):

       python3 scripts/tools/video_roi_reframe.py pick-roi \\
           --video /path/to/input.mp4 -o roi.json -n 0

   b) Pick on a reference image saved by step 1:

       python3 scripts/tools/video_roi_reframe.py pick-roi \\
           --image ref.png -o roi.json

3) Apply the ROI to the whole video. Output keeps the source resolution:
   each frame is cropped to the ROI, then resized back to the source W x H.

       python3 scripts/tools/video_roi_reframe.py process \\
           /path/to/input.mp4 -r roi.json -o /path/to/output.mp4

   By default the output is encoded with PyAV using libsvtav1 (LeRobot's
   format). Override the codec or quality if needed:

       python3 scripts/tools/video_roi_reframe.py process \\
           input.mp4 -r roi.json -o out.mp4 --codec libsvtav1 --crf 30
       python3 scripts/tools/video_roi_reframe.py process \\
           input.mp4 -r roi.json -o out.mp4 --codec h264 --crf 23
       python3 scripts/tools/video_roi_reframe.py process \\
           input.mp4 -r roi.json -o out.mp4 --codec opencv --fourcc mp4v

You can force a specific read backend via `--backend {auto,pyav,opencv}` on
`extract-frame`, `pick-roi`, and `process`. Default is `auto`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

import cv2  # noqa: E402


def _read_roi(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for key in ("x", "y", "w", "h"):
        if key not in data:
            raise ValueError(f"ROI file missing key '{key}': {path}")
    return data


def _clip_roi(x: int, y: int, w: int, h: int, fw: int, fh: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    return x, y, w, h


def _try_import_av():
    try:
        import av  # type: ignore

        return av
    except Exception:
        return None


def _read_frame_pyav(path: Path, frame_index: int) -> tuple[Any, int, int, float] | None:
    """Return (bgr_image, width, height, fps) from the given frame index using PyAV.

    Returns None when PyAV is unavailable or the frame cannot be decoded.
    """
    av = _try_import_av()
    if av is None:
        return None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            try:
                stream.thread_type = "AUTO"
            except Exception:
                pass
            w = int(stream.codec_context.width or 0)
            h = int(stream.codec_context.height or 0)
            rate = stream.average_rate or stream.base_rate or 30
            fps = float(rate)
            count = 0
            for frame in container.decode(stream):
                if count == frame_index:
                    img = frame.to_ndarray(format="bgr24")
                    if w == 0 or h == 0:
                        h, w = img.shape[:2]
                    return img, w, h, fps
                count += 1
        print(f"PyAV: frame_index {frame_index} out of range (decoded {count}).", file=sys.stderr)
        return None
    except Exception as e:
        print(f"PyAV decode error: {e}", file=sys.stderr)
        return None


def _iter_frames_pyav(path: Path) -> tuple[int, int, float, Iterator[Any]] | None:
    """Return (width, height, fps, generator-of-bgr-frames) using PyAV, or None."""
    av = _try_import_av()
    if av is None:
        return None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        w = int(stream.codec_context.width or 0)
        h = int(stream.codec_context.height or 0)
        rate = stream.average_rate or stream.base_rate or 30
        fps = float(rate)

        def gen() -> Iterator[Any]:
            try:
                for frame in container.decode(stream):
                    yield frame.to_ndarray(format="bgr24")
            finally:
                container.close()

        return w, h, fps, gen()
    except Exception as e:
        print(f"PyAV open error: {e}", file=sys.stderr)
        return None


def _open_capture_opencv(path: Path) -> cv2.VideoCapture | None:
    """Open OpenCV VideoCapture with hardware acceleration disabled."""
    try:
        cap = cv2.VideoCapture(
            str(path),
            cv2.CAP_FFMPEG,
            [int(cv2.CAP_PROP_HW_ACCELERATION), int(cv2.VIDEO_ACCELERATION_NONE)],
        )
    except Exception:
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _read_frame_opencv(path: Path, frame_index: int) -> tuple[Any, int, int, float] | None:
    cap = _open_capture_opencv(path)
    if cap is None:
        print(f"OpenCV: cannot open video: {path}", file=sys.stderr)
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_index < 0 or (n > 0 and frame_index >= n):
        print(f"OpenCV: frame_index {frame_index} out of range (frame_count={n}).", file=sys.stderr)
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        return None
    if w == 0 or h == 0:
        h, w = img.shape[:2]
    return img, w, h, fps


def _iter_frames_opencv(path: Path) -> tuple[int, int, float, Iterator[Any]] | None:
    cap = _open_capture_opencv(path)
    if cap is None:
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def gen() -> Iterator[Any]:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return w, h, fps, gen()


def _read_frame(path: Path, frame_index: int, backend: str) -> tuple[Any, int, int, float] | None:
    if backend in ("auto", "pyav"):
        out = _read_frame_pyav(path, frame_index)
        if out is not None:
            return out
        if backend == "pyav":
            return None
        print("PyAV unavailable or failed; falling back to OpenCV.", file=sys.stderr)
    return _read_frame_opencv(path, frame_index)


def _iter_frames(path: Path, backend: str) -> tuple[int, int, float, Iterator[Any]] | None:
    if backend in ("auto", "pyav"):
        out = _iter_frames_pyav(path)
        if out is not None:
            return out
        if backend == "pyav":
            return None
        print("PyAV unavailable or failed; falling back to OpenCV.", file=sys.stderr)
    return _iter_frames_opencv(path)


def cmd_extract_frame(args: argparse.Namespace) -> int:
    out = _read_frame(args.video, args.frame_index, args.backend)
    if out is None:
        print(f"Error: failed to read frame {args.frame_index}", file=sys.stderr)
        return 1
    img, _w, _h, _fps = out
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), img):
        print(f"Error: failed to write image: {out_path}", file=sys.stderr)
        return 1
    print(f"Wrote frame {args.frame_index} to {out_path}")
    return 0


def cmd_pick_roi(args: argparse.Namespace) -> int:
    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"Error: cannot read image: {args.image}", file=sys.stderr)
            return 1
        src_wh = (img.shape[1], img.shape[0])
        frame_index = args.frame_index
    else:
        out = _read_frame(args.video, args.frame_index, args.backend)
        if out is None:
            print(f"Error: failed to read frame {args.frame_index}", file=sys.stderr)
            return 1
        img, w, h, _fps = out
        src_wh = (w, h)
        frame_index = args.frame_index

    win = "ROI: drag rectangle, ENTER/SPACE confirm, C cancel"
    r = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
    if w <= 0 or h <= 0:
        print("ROI selection canceled or empty.", file=sys.stderr)
        return 1

    x, y, w, h = _clip_roi(x, y, w, h, src_wh[0], src_wh[1])
    payload = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "source_width": src_wh[0],
        "source_height": src_wh[1],
        "frame_index": frame_index,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ROI to {out_path}")
    print(json.dumps(payload, indent=2))
    return 0


def _process_with_pyav(
    args: argparse.Namespace,
    frames: Iterator[Any],
    out_w: int,
    out_h: int,
    fps: float,
    roi: tuple[int, int, int, int],
) -> int:
    """Crop+resize each frame and re-encode with PyAV (LeRobot-compatible)."""
    av = _try_import_av()
    if av is None:
        print("Error: --codec libsvtav1/h264/hevc requires `av` (pip install av).", file=sys.stderr)
        return 1

    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass

    x, y, w, h = roi
    options: dict[str, str] = {}
    if args.gop is not None:
        options["g"] = str(args.gop)
    if args.crf is not None:
        options["crf"] = str(args.crf)

    rate = Fraction(fps).limit_denominator(60000) if fps > 0 else Fraction(30, 1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    try:
        with av.open(str(out_path), "w") as output:
            stream = output.add_stream(args.codec, rate, options=options)
            stream.pix_fmt = args.pix_fmt
            stream.width = out_w
            stream.height = out_h

            for frame in frames:
                crop = frame[y : y + h, x : x + w]
                if crop.size == 0:
                    print(f"Error: empty crop at frame {n}", file=sys.stderr)
                    break
                stretched = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                vf = av.VideoFrame.from_ndarray(stretched, format="bgr24")
                for packet in stream.encode(vf):
                    output.mux(packet)
                n += 1
                if args.max_frames is not None and n >= args.max_frames:
                    break

            for packet in stream.encode():
                output.mux(packet)
    except Exception as e:
        print(f"Error: PyAV encoding failed: {e}", file=sys.stderr)
        return 1

    print(
        f"Wrote {n} frames to {out_path} at {out_w}x{out_h}, "
        f"{float(rate):.3f} fps, codec={args.codec}, pix_fmt={args.pix_fmt}"
    )
    return 0


def _process_with_opencv(
    args: argparse.Namespace,
    frames: Iterator[Any],
    out_w: int,
    out_h: int,
    fps: float,
    roi: tuple[int, int, int, int],
) -> int:
    x, y, w, h = roi
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(
            f"Error: VideoWriter failed (path={out_path}, fourcc={args.fourcc}, "
            f"fps={fps}, size=({out_w},{out_h})). Try --fourcc avc1 or XVID.",
            file=sys.stderr,
        )
        return 1

    n = 0
    for frame in frames:
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            print(f"Error: empty crop at frame {n}", file=sys.stderr)
            break
        stretched = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(stretched)
        n += 1
        if args.max_frames is not None and n >= args.max_frames:
            break

    writer.release()
    print(f"Wrote {n} frames to {out_path} at {out_w}x{out_h}, {fps:.3f} fps, codec=opencv:{args.fourcc}")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    roi_data = _read_roi(Path(args.roi))
    x, y, w, h = int(roi_data["x"]), int(roi_data["y"]), int(roi_data["w"]), int(roi_data["h"])

    info = _iter_frames(args.video, args.backend)
    if info is None:
        print(f"Error: cannot open video: {args.video}", file=sys.stderr)
        return 1
    out_w, out_h, fps, frames = info

    roi = _clip_roi(x, y, w, h, out_w, out_h)

    if args.codec == "opencv":
        return _process_with_opencv(args, frames, out_w, out_h, fps, roi)
    return _process_with_pyav(args, frames, out_w, out_h, fps, roi)


def _add_backend_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=("auto", "pyav", "opencv"),
        default="auto",
        help="Video read backend. Default 'auto' = PyAV (good for AV1) then OpenCV.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crop video to a fixed ROI and resize each frame back to the source resolution.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract-frame", help="Save one frame as an image for external ROI picking.")
    p_ext.add_argument("video", type=Path, help="Input video path.")
    p_ext.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output image path (e.g. ref.png).",
    )
    p_ext.add_argument(
        "-n",
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to save (0-based). Default: 0.",
    )
    _add_backend_arg(p_ext)
    p_ext.set_defaults(func=cmd_extract_frame)

    p_pick = sub.add_parser(
        "pick-roi",
        help="Interactive ROI on a video frame or image; writes ROI JSON (needs GUI).",
    )
    g = p_pick.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=Path, help="Input video path.")
    g.add_argument("--image", type=Path, help="Reference image (from extract-frame).")
    p_pick.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output ROI JSON path.",
    )
    p_pick.add_argument(
        "-n",
        "--frame-index",
        type=int,
        default=0,
        help="Frame index when using --video (0-based). Default: 0.",
    )
    _add_backend_arg(p_pick)
    p_pick.set_defaults(func=cmd_pick_roi)

    p_proc = sub.add_parser("process", help="Apply ROI JSON to the whole video.")
    p_proc.add_argument("video", type=Path, help="Input video path.")
    p_proc.add_argument(
        "-r",
        "--roi",
        type=Path,
        required=True,
        help="ROI JSON from pick-roi or hand-edited.",
    )
    p_proc.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output video path.",
    )
    p_proc.add_argument(
        "--codec",
        choices=("libsvtav1", "h264", "hevc", "opencv"),
        default="libsvtav1",
        help="Output codec. Default 'libsvtav1' matches LeRobot. Use 'opencv' to "
        "fall back to cv2.VideoWriter (configure with --fourcc).",
    )
    p_proc.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        help="Pixel format for PyAV codecs (default: yuv420p).",
    )
    p_proc.add_argument(
        "--crf",
        type=int,
        default=30,
        help="Constant Rate Factor for PyAV codecs (default: 30; lower = better quality).",
    )
    p_proc.add_argument(
        "--gop",
        type=int,
        default=2,
        help="GOP size 'g' for PyAV codecs (default: 2, matches LeRobot).",
    )
    p_proc.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC when --codec=opencv (default: mp4v). Examples: avc1, XVID.",
    )
    p_proc.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit on frames written.",
    )
    _add_backend_arg(p_proc)
    p_proc.set_defaults(func=cmd_process)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
