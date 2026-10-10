"""Create a verified playable GIF from one durable escape-team episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.escape_recording import record_episode_gif


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("team_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args(argv)
    result = record_episode_gif(
        args.team_root, args.output, episode_id=args.episode_id,
        fps=args.fps, width=args.width,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
