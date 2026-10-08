"""Human-readable recording of the frozen cooperative disease-game learner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cooperative_immune_savior import CooperativeImmuneSaviorGame, CooperativeImmuneTeam


CANVAS = (1060, 700)
MAP_X, MAP_Y, CELL = 28, 102, 56
ELEMENT_COLORS = ((97, 183, 255), (184, 125, 255), (255, 200, 76), (70, 218, 188))


def _font(size, bold=False):
    name = r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"
    try: return ImageFont.truetype(name, size)
    except OSError: return ImageFont.load_default()


def _center(draw, box, text, font, fill):
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(((box[0] + box[2] - width) / 2, (box[1] + box[3] - height) / 2 - bounds[1]),
              text, font=font, fill=fill)


def _phase(game, status, done, success):
    if done: return "BODY SAVED" if success else "EPISODE ENDED"
    if not all(status["controls_grounded"]): return "DISCOVERING UNKNOWN CONTROLS"
    if not status["map_complete"]: return "MAPPING THE BODY"
    if status["cooperation_mode"] == "independent": return "TESTING WHETHER COOPERATION IS NECESSARY"
    if not status["joint_recipe_learned"]: return "COORDINATING INGREDIENT EXPERIMENTS"
    if not game.treatment: return "RECREATING THE DISCOVERED CURE"
    return "REMOVING DISEASE — PROTECTING HEALTHY CELLS"


def _star(draw, center, outer, inner, fill, outline):
    points = []
    for index in range(16):
        angle = -math.pi / 2 + index * math.pi / 8; radius = outer if index % 2 == 0 else inner
        points.append((center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius))
    draw.polygon(points, fill=fill, outline=outline)


def _cell_center(cell):
    row, col = cell; return MAP_X + col * CELL + CELL // 2, MAP_Y + row * CELL + CELL // 2


def _draw_world(draw, game):
    map_size = 9 * CELL
    draw.rounded_rectangle((MAP_X - 9, MAP_Y - 9, MAP_X + map_size + 9, MAP_Y + map_size + 9),
                           radius=18, fill=(28, 42, 55), outline=(93, 126, 145), width=2)
    for row in range(9):
        for col in range(9):
            x0, y0 = MAP_X + col * CELL, MAP_Y + row * CELL
            wall = (row, col) in game.walls
            fill = (57, 71, 83) if wall else ((29, 55, 64) if (row + col) % 2 else (32, 61, 69))
            draw.rounded_rectangle((x0 + 2, y0 + 2, x0 + CELL - 2, y0 + CELL - 2), radius=8,
                                   fill=fill, outline=(44, 80, 87))
            if wall:
                draw.line((x0 + 12, y0 + 13, x0 + CELL - 12, y0 + CELL - 13), fill=(99, 113, 124), width=5)
                draw.line((x0 + CELL - 12, y0 + 13, x0 + 12, y0 + CELL - 13), fill=(99, 113, 124), width=5)

    for index, name in enumerate(game.element_names):
        if name not in game.available_elements: continue
        cx, cy = _cell_center(game.fixed[name]); color = ELEMENT_COLORS[index]
        draw.polygon(((cx, cy - 17), (cx + 17, cy), (cx, cy + 17), (cx - 17, cy)),
                     fill=color, outline=(245, 249, 255))
        _center(draw, (cx - 18, cy - 18, cx + 18, cy + 18), chr(65 + index), _font(16, True), (9, 18, 29))

    cx, cy = _cell_center(game.fixed["synthesizer"])
    draw.rounded_rectangle((cx - 20, cy - 20, cx + 20, cy + 20), radius=9,
                           fill=(224, 232, 242), outline=(255, 255, 255), width=2)
    draw.line((cx - 10, cy, cx + 10, cy), fill=(41, 57, 75), width=4)
    draw.line((cx, cy - 10, cx, cy + 10), fill=(41, 57, 75), width=4)

    for cell in game.healthy:
        cx, cy = _cell_center(cell)
        draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=(87, 220, 139), outline=(184, 255, 211), width=2)
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(39, 132, 83))
    for cell in game.disease:
        _star(draw, _cell_center(cell), 21, 13, (234, 73, 91), (255, 178, 185))

    for index, cell in enumerate(game.positions):
        cx, cy = _cell_center(cell); color = ((74, 211, 255), (255, 166, 75))[index]
        draw.ellipse((cx - 21, cy - 21, cx + 21, cy + 21), fill=(14, 23, 35), outline=color, width=4)
        _center(draw, (cx - 18, cy - 18, cx + 18, cy + 18), "AB"[index], _font(19, True), color)
        if game.treatment:
            draw.ellipse((cx - 25, cy - 25, cx + 25, cy + 25), outline=(255, 224, 104), width=2)


def _inventory_card(draw, game, index, x, y):
    color = ((74, 211, 255), (255, 166, 75))[index]
    draw.rounded_rectangle((x, y, x + 218, y + 88), radius=12, fill=(23, 32, 46), outline=color, width=2)
    draw.text((x + 13, y + 10), f"GUARDIAN {'AB'[index]} INVENTORY", font=_font(13, True), fill=color)
    item = game.inventories[index]
    if item is None:
        draw.text((x + 13, y + 47), "empty", font=_font(17), fill=(145, 157, 176))
    else:
        number = int(item.replace("element", "")); ingredient = f"INGREDIENT {chr(65 + number)}"
        draw.ellipse((x + 14, y + 43, x + 37, y + 66), fill=ELEMENT_COLORS[number])
        draw.text((x + 47, y + 44), ingredient, font=_font(16, True), fill=(238, 242, 248))


def _panel(frames, game, round_index, status, reward, event, done, success):
    canvas = Image.new("RGB", CANVAS, (10, 17, 27)); draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), "IMMUNE SAVIOR — TWO GUARDIANS INSIDE A LIVING BODY", font=_font(25, True), fill=(240, 245, 251))
    draw.text((29, 55), "Human explanation view — labels and full map are never shown to the agents",
              font=_font(14), fill=(147, 165, 186))
    phase = _phase(game, status, done, success)
    draw.rounded_rectangle((28, 76, 1032, 96), radius=8, fill=(28, 43, 60))
    _center(draw, (28, 76, 1032, 96), phase, _font(12, True),
            (99, 236, 171) if success else (255, 214, 104))
    _draw_world(draw, game)
    draw.text((28, 625), "LEGEND", font=_font(13, True), fill=(163, 179, 200))
    draw.ellipse((104, 625, 123, 644), fill=(87, 220, 139)); draw.text((130, 626), "healthy cell", font=_font(13), fill=(225, 232, 241))
    _star(draw, (260, 635), 12, 7, (234, 73, 91), (255, 178, 185)); draw.text((280, 626), "disease", font=_font(13), fill=(225, 232, 241))
    draw.polygon(((379, 623), (391, 635), (379, 647), (367, 635)), fill=ELEMENT_COLORS[0]); draw.text((400, 626), "ingredient", font=_font(13), fill=(225, 232, 241))
    draw.rounded_rectangle((449, 624, 470, 646), radius=4, fill=(224, 232, 242)); draw.text((478, 626), "mix lab", font=_font(13), fill=(225, 232, 241))

    # Actual private observations, exactly as received by the learners.
    for index, frame in enumerate(frames):
        image = Image.fromarray(np.transpose(frame, (1, 2, 0))).resize((224, 252), Image.Resampling.NEAREST)
        x = 578 + index * 230; canvas.paste(image, (x, 112))
        draw.rectangle((x, 112, x + 223, 363), outline=((74, 211, 255), (255, 166, 75))[index], width=2)
        draw.text((x, 371), f"Guardian {'AB'[index]}'s actual pixels", font=_font(12, True), fill=((74, 211, 255), (255, 166, 75))[index])
    _inventory_card(draw, game, 0, 578, 405); _inventory_card(draw, game, 1, 808, 405)
    recipe = "DISCOVERED" if status["joint_recipe_learned"] else "unknown"
    cure = "READY" if game.treatment else "not synthesized"
    draw.rounded_rectangle((578, 506, 1026, 606), radius=14, fill=(22, 33, 48))
    draw.text((594, 519), f"Round {round_index:03d}   |   Cure recipe: {recipe}   |   Cure: {cure}",
              font=_font(14, True), fill=(238, 242, 248))
    draw.text((594, 548), f"Disease remaining: {len(game.disease)}   Healthy protected: {len(game.healthy)}   Damaged: {game.damaged_healthy}",
              font=_font(14), fill=(102, 231, 166) if game.damaged_healthy == 0 else (255, 101, 112))
    readable_event = event.replace("-", " ").replace("private element", "ingredient")
    draw.text((594, 578), f"Latest event: {readable_event}   |   reward: {reward:g}", font=_font(13), fill=(255, 209, 105))
    decision = status.get("cooperation_decision")
    decision_text = "Cooperation decision: testing"
    if decision: decision_text = f"Cooperation decision: {'NECESSARY' if decision.get('necessary') else 'not necessary'}"
    draw.text((578, 627), decision_text, font=_font(14, True), fill=(190, 205, 224))
    draw.text((578, 654), f"Mode: {status['cooperation_mode']}  |  shared messages: {status['shared_messages']}  |  joint experiments: {status['joint_experiments']}",
              font=_font(13), fill=(147, 165, 186))
    return canvas


def record(output, trace_output, *, seed=2_410_003, budget=1500):
    game = CooperativeImmuneSaviorGame(seed, max_steps=budget); team = CooperativeImmuneTeam()
    frames = game.reset(); team.begin(frames); status = team.status(); trace = []
    movie = [_panel(frames, game, 0, status, 0.0, "entering tissue", False, False)]
    success = False; previous_phase = _phase(game, status, False, False)
    for round_index in range(1, budget + 1):
        actions = team.act(frames); later, reward, done, info = game.step(actions)
        team.observe(frames, actions, later, reward, done); status = team.status()
        events = info.get("events_audit_only", ["no event", "no event"])
        event = " + ".join(sorted(set(value for value in events if value not in ("moved", "blocked")))) or "moving / observing"
        phase = _phase(game, status, done, bool(info.get("success")))
        trace.append({"round": round_index, "actions": actions, "event": event, "reward": reward,
                      "phase": phase, "disease_remaining": info.get("disease_remaining"),
                      "healthy_damaged": info.get("healthy_damaged"), "status": status})
        # Keep motion readable and preserve every meaningful transition.
        meaningful = event != "moving / observing" or phase != previous_phase or done
        if round_index % 2 == 0 or meaningful:
            movie.append(_panel(later, game, round_index, status, reward, event, done, bool(info.get("success"))))
        frames = later; previous_phase = phase
        if done: success = bool(info.get("success")); break
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    movie[0].save(output, save_all=True, append_images=movie[1:], duration=135, loop=0, optimize=False)
    payload = {"format": "wailah-cooperative-immune-human-replay-v24", "seed": seed,
        "success": success, "rounds": len(trace), "frames": len(movie),
        "healthy_damaged": game.damaged_healthy, "healthy_remaining": len(game.healthy),
        "disease_remaining": len(game.disease), "failed_mixtures": game.failed_mixtures,
        "treatment_created": game.treatment, "final_status": team.status(), "trace": trace,
        "disclosure": "global labels are a human-only explanatory overlay; learner input remains two private RGB views, shared scalar reward, and termination"}
    Path(trace_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {key: payload[key] for key in ("success", "rounds", "frames", "healthy_damaged",
                                            "disease_remaining", "failed_mixtures", "treatment_created")}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True); parser.add_argument("--seed", type=int, default=2_410_003)
    parser.add_argument("--budget", type=int, default=1500)
    args = parser.parse_args(argv); print(json.dumps(record(args.output, args.trace, seed=args.seed, budget=args.budget), indent=2))


if __name__ == "__main__": raise SystemExit(main())
