"""Convert a demo video (outputs/demos/<run>/demo.mp4) into a small GIF for the README.

Example:
    python tools/make_demo_gif.py outputs/demos/camera_xxx/demo.mp4 assets/demo/le2i_fall.gif --width 560 --fps 8
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image


def main():
    """Read the demo video, keep frames at the requested fps and size, and write an optimised GIF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=560, help="GIF width in pixels")
    parser.add_argument("--fps", type=float, default=8.0, help="GIF frames per second")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="end time in seconds (default: end of video)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(src_fps / args.fps))
    frames, index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        seconds = index / src_fps
        index += 1
        if seconds < args.start or (args.end is not None and seconds > args.end) or (index - 1) % step:
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).quantize(colors=128))
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {args.video}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=round(1000 * step / src_fps), loop=0, optimize=True)
    print(f"Saved {args.output} ({len(frames)} frames, {args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
