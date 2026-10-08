"""Extract start/middle/end GIF frames for human visual quality review."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=Path, default=Path("assets/demos"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.media.glob("*.gif")):
        with Image.open(path) as image:
            count = getattr(image, "n_frames", 1)
            for label, frame_index in (("start", 0), ("middle", count // 2), ("end", count - 1)):
                image.seek(frame_index)
                output = args.output / f"{path.stem}-{label}.png"
                image.convert("RGB").save(output)
                print(f"{path.name}\t{count} frames\t{image.size[0]}x{image.size[1]}\t{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
