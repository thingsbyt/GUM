"""Launch the local cooperative escape chamber watch interface."""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import webbrowser

from gum.school.escape_team import EscapeTeam
from gum.school.escape_watch import build_escape_watch_server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("work/escape-chamber-live"))
    parser.add_argument("--source-policy", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8783)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    manifest = args.workspace / "ESCAPE_TEAM.json"
    team = (
        EscapeTeam.load(args.workspace, device=args.device)
        if manifest.exists()
        else EscapeTeam.create(
            args.workspace,
            source_policy=args.source_policy,
            device=args.device,
        )
    )
    server, url, _ = build_escape_watch_server(team, host=args.host, port=args.port)
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"GUM cooperative escape watch: {url}", flush=True)
    print("Learning starts only when Start learning is pressed.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
