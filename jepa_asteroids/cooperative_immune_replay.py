"""Create an inspectable two-view GIF without changing the frozen learner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cooperative_immune_savior import CooperativeImmuneSaviorGame, CooperativeImmuneTeam


def _panel(frames, round_index, status, reward=0.0):
    images = [Image.fromarray(np.transpose(frame, (1, 2, 0))).resize((224, 252), Image.Resampling.NEAREST)
              for frame in frames]
    canvas = Image.new("RGB", (448, 300), (18, 21, 29)); canvas.paste(images[0], (0, 0)); canvas.paste(images[1], (224, 0))
    draw = ImageDraw.Draw(canvas); draw.text((6, 256), "Guardian A", fill=(105, 220, 255))
    draw.text((230, 256), "Guardian B", fill=(255, 177, 90))
    line = (f"round {round_index:03d} | mode: {status['cooperation_mode']} | "
            f"recipe: {str(status['joint_recipe_learned']).lower()} | "
            f"growth: {str(status['growing_concept_learned']).lower()} | reward: {reward:g}")
    draw.text((6, 276), line, fill=(240, 242, 247)); return canvas


def record_episode(output, trace_output, seed=1_310_003, budget=1500):
    game = CooperativeImmuneSaviorGame(seed, max_steps=budget); team = CooperativeImmuneTeam()
    frames = game.reset(); team.begin(frames); movie = [_panel(frames, 0, team.status())]; trace = []
    success = False
    for round_index in range(1, budget + 1):
        actions = team.act(frames); later, reward, done, info = game.step(actions)
        learned = team.observe(frames, actions, later, reward, done); status = team.status()
        trace.append({"round": round_index, "actions": actions, "reward": reward,
                      "mode": status["cooperation_mode"], "recipe": status["joint_recipe_learned"],
                      "growth": status["growing_concept_learned"], "done": done})
        if round_index % 2 == 0 or done or (trace[-2]["mode"] != status["cooperation_mode"] if len(trace) > 1 else False):
            movie.append(_panel(later, round_index, status, reward))
        frames = later
        if done: success = bool(info["success"]); break
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    movie[0].save(output, save_all=True, append_images=movie[1:], duration=125, loop=0, optimize=False)
    Path(trace_output).write_text(json.dumps({"seed": seed, "success": success,
        "rounds": len(trace), "healthy_damaged": int(info.get("healthy_damaged", -1)),
        "final_status": team.status(), "trace": trace}, indent=2), encoding="utf-8")
    return {"success": success, "rounds": len(trace), "frames": len(movie), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True); parser.add_argument("--seed", type=int, default=1_310_003)
    args = parser.parse_args(argv); print(json.dumps(record_episode(args.output, args.trace, args.seed), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
