#!/usr/bin/env python3
"""
Video post-process: pick a ROI on one frame, then crop every frame to that ROI
and resize back to the source video resolution.

Workflow:
  1) Extract a reference frame (or use pick-roi directly on the video).
  2) pick-roi: interactive ROI selection (needs GUI), saves ROI JSON.
  3) process: apply ROI to the full video and write a new file.

You can also create/edit the ROI JSON by hand after saving a frame as image.

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

   If the default encoder cannot write the file, switch FourCC, e.g.:

       python3 scripts/tools/video_roi_reframe.py process \\
           /path/to/input.mp4 -r roi.json -o /path/to/output.mp4 --fourcc avc1
       # or XVID with a .avi container
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2


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


def cmd_extract_frame(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: cannot open video: {args.video}", file=sys.stderr)
        return 1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = args.frame_index
    if idx < 0 or (n > 0 and idx >= n):
        print(f"Error: frame_index {idx} out of range (frame_count={n})", file=sys.stderr)
        cap.release()
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"Error: failed to read frame {args.frame_index}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        print(f"Error: failed to write image: {out}", file=sys.stderr)
        return 1
    print(f"Wrote frame {args.frame_index} to {out}")
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
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"Error: cannot open video: {args.video}", file=sys.stderr)
            return 1
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idx = args.frame_index
        if idx < 0 or (n > 0 and idx >= n):
            print(f"Error: frame_index {idx} out of range (frame_count={n})", file=sys.stderr)
            cap.release()
            return 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            print(f"Error: failed to read frame {idx}", file=sys.stderr)
            return 1
        src_wh = (fw, fh)
        frame_index = idx

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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ROI to {out}")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    roi = _read_roi(Path(args.roi))
    x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: cannot open video: {args.video}", file=sys.stderr)
        return 1

    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    x, y, w, h = _clip_roi(x, y, w, h, out_w, out_h)

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
        cap.release()
        return 1

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            print(f"Error: empty crop at frame {n}", file=sys.stderr)
            break
        stretched = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(stretched)
        n += 1
        if args.max_frames is not None and n >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"Wrote {n} frames to {out_path} at {out_w}x{out_h}, {fps:.3f} fps")
    return 0


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
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC for VideoWriter (default: mp4v). Examples: avc1, XVID.",
    )
    p_proc.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit on frames written.",
    )
    p_proc.set_defaults(func=cmd_process)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
