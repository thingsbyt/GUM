from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import GUMHarness
from .interface import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="GUM — Growing Understanding Machine")
    parser.add_argument("--workspace", type=Path, required=True); sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-world"); create.add_argument("--seed", type=int, required=True); create.add_argument("--difficulty", type=int, default=0)
    practice = sub.add_parser("practice"); practice.add_argument("--world", type=Path); practice.add_argument("--episodes", type=int, default=600)
    status = sub.add_parser("status"); verify = sub.add_parser("verify")
    chat = sub.add_parser("say"); chat.add_argument("text")
    web = sub.add_parser("serve"); web.add_argument("--port", type=int, default=8765)
    web.add_argument("--release", type=Path, help="optional GUM research package shown in Studio")
    web.add_argument("--open-browser", action="store_true", help="open GUM Studio in the default browser")
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.workspace, port=args.port, release_root=args.release,
              open_browser=args.open_browser)
        return 0
    harness = GUMHarness(args.workspace)
    if args.command == "create-world": result = {"world": str(harness.create_world(seed=args.seed, difficulty=args.difficulty))}
    elif args.command == "practice":
        if args.world: harness.load_world(args.world)
        result = harness.practice(training_episodes=args.episodes)
    elif args.command == "status": result = harness.status()
    elif args.command == "verify": result = harness.ledger.verify()
    else: result = {"response": harness.communicate(args.text)}
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
