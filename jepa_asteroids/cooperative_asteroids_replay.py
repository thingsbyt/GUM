"""Record the two pixel views and learning state as an inspectable GIF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .cooperative_asteroids import DuoAsteroids, DuoConfig, TwoRocketTeam


def _panel(frames, step, info, status):
    canvas = Image.new("RGB", (640, 226), (18, 21, 29))
    canvas.paste(Image.fromarray(frames[0]), (0, 0)); canvas.paste(Image.fromarray(frames[1]), (320, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 184), "ROCKET A - LOCAL CAMERA", fill=(74, 224, 211))
    draw.text((327, 184), "ROCKET B - LOCAL CAMERA", fill=(245, 158, 72))
    grounded = sum(bool(row) for row in status["controls_grounded"])
    line = (f"step {step:03d} | controls grounded {grounded}/2 | hits {info['hits']:03d} "
            f"({info['hits_by_agent'][0]:02d}+{info['hits_by_agent'][1]:02d}) | "
            f"wave {info['wave']} | shared target messages {status['target_messages']}")
    draw.text((7, 204), line, fill=(239, 242, 247))
    return canvas


def record(output, trace_output, *, seed=1_440_001, max_steps=240):
    game = DuoAsteroids(DuoConfig(max_steps=max_steps), seed); team = TwoRocketTeam(communicate=True)
    frames = game.reset(seed); team.begin(frames)
    info = {"hits": 0, "hits_by_agent": [0, 0], "wave": 1}
    movie = [_panel(frames, 0, info, team.status())]; trace = []
    while not (game.terminated or game.truncated):
        actions = team.act(frames); later, reward, terminated, truncated, info = game.step(actions)
        team.observe(frames, actions, later, reward, terminated or truncated); status = team.status()
        trace.append({"step": game.steps, "actions": actions, "reward": reward, "hits": info["hits"],
                      "hits_by_agent": info["hits_by_agent"], "wave": info["wave"],
                      "alive": info["alive"], "controls_grounded": status["both_controls_grounded"],
                      "target_messages": status["target_messages"]})
        frames = later
        if game.steps % 2 == 0 or terminated or truncated: movie.append(_panel(frames, game.steps, info, status))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    movie[0].save(output, save_all=True, append_images=movie[1:], duration=90, loop=0, optimize=False)
    payload = {"format": "wailah-two-rocket-asteroids-v20-replay-v1", "seed": seed,
               "result": {"hits": info["hits"], "hits_by_agent": info["hits_by_agent"],
                          "friendly_fire": info["friendly_fire"], "alive": info["alive"], "wave": info["wave"]},
               "learner": team.status(), "trace": trace}
    Path(trace_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"output": str(output), "frames": len(movie), **payload["result"]}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True); parser.add_argument("--seed", type=int, default=1_440_001)
    parser.add_argument("--max-steps", type=int, default=240)
    args = parser.parse_args(argv); print(json.dumps(record(args.output, args.trace, seed=args.seed,
                                                              max_steps=args.max_steps), indent=2))


if __name__ == "__main__": raise SystemExit(main())
