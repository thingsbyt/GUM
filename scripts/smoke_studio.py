"""End-to-end local smoke test for GUM Studio and its teaching API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def request(url: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as response:
        return json.loads(response.read())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "gum", "--workspace", str(args.workspace),
               "serve", "--release", str(root), "--port", str(args.port)]
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(50):
            try:
                request(url, "/api/teaching/state")
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read())
                time.sleep(0.1)
        else:
            raise RuntimeError("Studio did not become ready")

        starter = request(url, "/api/teaching/starter", {})
        request(url, "/api/teaching/scene", {"mode": "ambiguity"})
        question = request(url, "/api/teaching/begin", {"text": "approach the red object"})
        answer = request(url, "/api/teaching/answer", {"text": "the square"})
        assert starter["learned_word_count"] == 6
        assert question["last_turn"]["status"] == "clarify"
        assert answer["last_turn"]["status"] == "execute"
        print(json.dumps({"passed": True, "learned_words": 6,
                          "question": question["last_turn"]["response"],
                          "resolved_target": answer["last_turn"]["target_id"]}, indent=2))
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
