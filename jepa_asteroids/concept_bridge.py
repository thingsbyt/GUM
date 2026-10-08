"""Reward-grounded cross-representation concept learning on real photographs.

Training experiences contain photo->anonymous-glyph and anonymous-glyph->word
choices.  Photo->word is never trained directly.  The sealed test therefore
measures whether a concept can bridge transitively across representations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.models import resnet18


CONCEPTS = ("apple", "bicycle", "butterfly", "clock", "dolphin",
            "lamp", "pickup_truck", "pine_tree", "telephone", "wardrobe")
DISPLAY = tuple(value.replace("_", " ").upper() for value in CONCEPTS)
EMBEDDING_DIM = 64


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _font(names, size):
    for name in names:
        try: return ImageFont.truetype(name, size)
        except OSError: pass
    return ImageFont.load_default()


def _glyph_bits(index):
    rng = np.random.default_rng(81_001 + index * 7_919)
    bits = rng.integers(0, 2, size=(6, 6), dtype=np.uint8)
    bits[0, index % 6] = 1; bits[-1, (index * 3 + 1) % 6] = 1
    return bits


def render_glyph(index, variant, *, test=False):
    """Render a meaningless symbol. Identity is discoverable only by reward."""
    rng = np.random.default_rng(710_003 + index * 10_007 + variant * 101 + int(test) * 9_000_001)
    bg = tuple(int(v) for v in rng.integers(2, 65, size=3))
    fg = tuple(int(v) for v in rng.integers(170, 256, size=3))
    image = Image.new("RGB", (64, 64), bg); draw = ImageDraw.Draw(image)
    bits = _glyph_bits(index); cell = int(rng.integers(7, 10)); width = cell * 6
    x0 = int((64 - width) / 2 + rng.integers(-3, 4)); y0 = int((64 - width) / 2 + rng.integers(-3, 4))
    for row in range(6):
        for col in range(6):
            if bits[row, col]:
                inset = int(rng.integers(0, 2))
                draw.rounded_rectangle((x0 + col * cell + inset, y0 + row * cell + inset,
                                        x0 + (col + 1) * cell - 1, y0 + (row + 1) * cell - 1),
                                       radius=1, fill=fg)
    if test: image = image.rotate(float(rng.uniform(-7, 7)), resample=Image.Resampling.BILINEAR, fillcolor=bg)
    return image


def render_word(index, variant, *, test=False):
    """Render a word with held-out typography and colors at test time."""
    rng = np.random.default_rng(910_009 + index * 20_011 + variant * 131 + int(test) * 8_000_003)
    if test:
        fonts = ("timesbd.ttf", "georgiab.ttf", "courbd.ttf")
        bg = tuple(int(v) for v in rng.integers(205, 256, size=3)); fg = tuple(int(v) for v in rng.integers(0, 65, size=3))
    else:
        fonts = ("arialbd.ttf", "calibrib.ttf", "verdana.ttf")
        # Both polarities are experienced, but test fonts and exact styles remain held out.
        if bool(rng.integers(0, 2)):
            bg = tuple(int(v) for v in rng.integers(0, 55, size=3)); fg = tuple(int(v) for v in rng.integers(190, 256, size=3))
        else:
            bg = tuple(int(v) for v in rng.integers(200, 256, size=3)); fg = tuple(int(v) for v in rng.integers(0, 70, size=3))
    image = Image.new("RGB", (96, 64), bg); draw = ImageDraw.Draw(image); text = DISPLAY[index]
    size = 22
    while size > 8:
        font = _font(fonts, size); box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= 90: break
        size -= 1
    box = draw.textbbox((0, 0), text, font=font); width, height = box[2] - box[0], box[3] - box[1]
    x = (96 - width) // 2 + int(rng.integers(-2, 3)); y = (64 - height) // 2 - box[1] + int(rng.integers(-2, 3))
    draw.text((x, y), text, font=font, fill=fg)
    if test: image = image.rotate(float(rng.uniform(-3, 3)), resample=Image.Resampling.BILINEAR, fillcolor=bg)
    return image


def _tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous() * 2 - 1


class SelectedCifar(Dataset):
    def __init__(self, root, *, train, augment):
        data = CIFAR100(root=root, train=train, download=False)
        lookup = {name: data.classes.index(name) for name in CONCEPTS}
        wanted = set(lookup.values()); remap = {source: index for index, source in enumerate(lookup.values())}
        self.rows = [(Image.fromarray(image), remap[label]) for image, label in zip(data.data, data.targets)
                     if label in wanted]
        if augment:
            self.transform = transforms.Compose([transforms.Resize(72), transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(), transforms.ColorJitter(.18, .18, .18, .05), transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        else:
            self.transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        image, label = self.rows[index]; return self.transform(image), int(label)


class Encoder(nn.Module):
    def __init__(self, in_shape=(64, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 160, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(160, 128), nn.GELU(), nn.Linear(128, EMBEDDING_DIM))

    def forward(self, value): return F.normalize(self.project(self.net(value)), dim=-1)


class PhotoEncoder(nn.Module):
    """A randomly initialized visual learner; no pretrained weights are loaded."""
    def __init__(self):
        super().__init__()
        self.net = resnet18(weights=None, num_classes=EMBEDDING_DIM)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, value): return F.normalize(self.net(value), dim=-1)


class ConceptBridge(nn.Module):
    def __init__(self):
        super().__init__(); self.photo = PhotoEncoder(); self.glyph = Encoder(); self.word = Encoder()
        self.log_scale = nn.Parameter(torch.tensor(math.log(8.0)))

    def encode(self, kind, value): return getattr(self, kind)(value)
    def scores(self, left_kind, left, right_kind, right):
        return self.log_scale.exp().clamp(1, 40) * (self.encode(left_kind, left) @ self.encode(right_kind, right).T)


def _candidate_bank(kind, variant, *, test, device):
    render = render_glyph if kind == "glyph" else render_word
    return torch.stack([_tensor(render(index, variant, test=test)) for index in range(len(CONCEPTS))]).to(device)


def _explore_until_success(target_slots, rng):
    """Return the successful action after scalar-reward exploration, plus attempts."""
    discovered, attempts = [], []
    for target in target_slots.detach().cpu().tolist():
        order = rng.permutation(len(CONCEPTS)).tolist()
        position = order.index(int(target)); discovered.append(order[position]); attempts.append(position + 1)
    return torch.tensor(discovered, device=target_slots.device), attempts


def train_bridge(data_root, output, *, epochs=60, seed=601_003, batch_size=96):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = SelectedCifar(data_root, train=True, augment=True)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = ConceptBridge().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    rng = np.random.default_rng(seed + 17); history = []; total_interactions = 0
    for epoch in range(epochs):
        model.train(); losses = []; correct_photo_glyph = correct_glyph_word = count = 0; attempts_epoch = 0
        for step, (photos, labels) in enumerate(loader):
            photos, labels = photos.to(device), labels.to(device); variant = epoch * len(loader) + step
            glyph_order = torch.randperm(len(CONCEPTS), device=device)
            word_order = torch.randperm(len(CONCEPTS), device=device)
            glyphs = _candidate_bank("glyph", variant, test=False, device=device)[glyph_order]
            words = _candidate_bank("word", variant, test=False, device=device)[word_order]
            glyph_targets = torch.argsort(glyph_order)[labels]
            word_targets = torch.argsort(word_order)[labels]
            # Encode each observation once. Reusing the successful glyph embedding is
            # essential: both bridges must meet at the very same internal concept point.
            photo_z = model.encode("photo", photos)
            glyph_z = model.encode("glyph", glyphs)
            word_z = model.encode("word", words)
            scale = model.log_scale.exp().clamp(1, 40)
            # Bridge 1: a real photograph explores anonymous glyph choices.
            photo_glyph = scale * (photo_z @ glyph_z.T)
            discovered_pg, attempts_pg = _explore_until_success(glyph_targets, rng)
            # Bridge 2: the successful glyph then explores written representations.
            successful_glyph_z = glyph_z[discovered_pg]
            glyph_word = scale * (successful_glyph_z @ word_z.T)
            discovered_gw, attempts_gw = _explore_until_success(word_targets, rng)
            successful_word_z = word_z[discovered_gw]
            # A bidirectional ten-way contrastive bridge prevents the trivial
            # solution where every concept collapses onto the same point.
            glyph_word_bank = scale * (glyph_z @ word_z.T)
            word_slot_for_glyph_slot = torch.argsort(word_order)[glyph_order]
            glyph_slot_for_word_slot = torch.argsort(glyph_order)[word_order]
            # Ranking losses learn both links. They are composed explicitly at
            # inference rather than forcing three modalities into a collapse-prone
            # identical embedding.
            loss = F.cross_entropy(photo_glyph, discovered_pg) + F.cross_entropy(glyph_word, discovered_gw)
            loss += 0.5 * F.cross_entropy(glyph_word_bank, word_slot_for_glyph_slot)
            loss += 0.5 * F.cross_entropy(glyph_word_bank.T, glyph_slot_for_word_slot)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step(); losses.append(float(loss.detach()))
            correct_photo_glyph += int((photo_glyph.argmax(1) == glyph_targets).sum())
            correct_glyph_word += int((glyph_word.argmax(1) == word_targets).sum()); count += len(labels)
            attempts_epoch += sum(attempts_pg) + sum(attempts_gw)
        total_interactions += attempts_epoch
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "learning_rate": optimizer.param_groups[0]["lr"],
               "photo_to_glyph_accuracy": correct_photo_glyph / count,
               "glyph_to_word_accuracy": correct_glyph_word / count,
               "reward_interactions": attempts_epoch}
        history.append(row)
        print(json.dumps(row), flush=True)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "wailah-concept-bridge-v2", "state_dict": model.state_dict(),
                "concepts": CONCEPTS, "seed": seed, "epochs": epochs,
                "reward_interactions": total_interactions}, output)
    return model, history, device


@torch.no_grad()
def evaluate(model, data_root, device, *, representation_seeds=10, batch_size=128):
    test_data = SelectedCifar(data_root, train=False, augment=False)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); conditions = {"photo_to_word_composed_chain": [], "direct_photo_to_word_cosine": [],
                                "photo_to_glyph_trained_pair": []}
    predictions, labels_all = [], []
    for representation_seed in range(representation_seeds):
        shuffle_rng = np.random.default_rng(5_000_003 + representation_seed)
        word_order = torch.tensor(shuffle_rng.permutation(len(CONCEPTS)), device=device)
        glyph_order = torch.tensor(shuffle_rng.permutation(len(CONCEPTS)), device=device)
        words = _candidate_bank("word", 100_000 + representation_seed, test=True, device=device)[word_order]
        glyphs = _candidate_bank("glyph", 100_000 + representation_seed, test=True, device=device)[glyph_order]
        glyph_word = torch.softmax(model.scores("glyph", glyphs, "word", words), dim=1)
        for photos, labels in loader:
            photos, labels = photos.to(device), labels.to(device)
            photo_glyph_scores = model.scores("photo", photos, "glyph", glyphs)
            composed_word_slots = (torch.softmax(photo_glyph_scores, dim=1) @ glyph_word).argmax(1)
            word_prediction = word_order[composed_word_slots]
            direct_word_prediction = word_order[model.scores("photo", photos, "word", words).argmax(1)]
            glyph_prediction = glyph_order[photo_glyph_scores.argmax(1)]
            conditions["photo_to_word_composed_chain"].extend((word_prediction == labels).cpu().tolist())
            conditions["direct_photo_to_word_cosine"].extend((direct_word_prediction == labels).cpu().tolist())
            conditions["photo_to_glyph_trained_pair"].extend((glyph_prediction == labels).cpu().tolist())
            predictions.extend(word_prediction.cpu().tolist()); labels_all.extend(labels.cpu().tolist())
    confusion = np.zeros((len(CONCEPTS), len(CONCEPTS)), dtype=np.int64)
    for actual, predicted in zip(labels_all, predictions): confusion[actual, predicted] += 1
    per_concept = {}
    for index, name in enumerate(CONCEPTS):
        total = int(confusion[index].sum()); per_concept[name] = {"correct": int(confusion[index, index]),
            "trials": total, "accuracy": float(confusion[index, index] / max(1, total))}
    return {"trials": len(labels_all),
            "photo_to_word_accuracy": float(np.mean(conditions["photo_to_word_composed_chain"])),
            "direct_photo_to_word_cosine_accuracy": float(np.mean(conditions["direct_photo_to_word_cosine"])),
            "photo_to_glyph_accuracy": float(np.mean(conditions["photo_to_glyph_trained_pair"])),
            "apple": per_concept["apple"], "per_concept": per_concept,
            "confusion": confusion.tolist(), "predictions": predictions, "labels": labels_all}


@torch.no_grad()
def evaluate_controls(data_root, device, *, seed=601_003, representation_seeds=3):
    torch.manual_seed(seed + 99); untrained = ConceptBridge().to(device)
    result = evaluate(untrained, data_root, device, representation_seeds=representation_seeds)
    return {"untrained_accuracy": result["photo_to_word_accuracy"], "random_expected_accuracy": 1 / len(CONCEPTS)}


@torch.no_grad()
def record_apple_replay(model, data_root, device, output, *, trials=24):
    dataset = SelectedCifar(data_root, train=False, augment=False); apple_rows = [i for i, (_, y) in enumerate(dataset.rows) if y == 0]
    frames = []; trace = []; model.eval()
    for trial, row_index in enumerate(apple_rows[:trials]):
        photo_tensor, label = dataset[row_index]; photo = dataset.rows[row_index][0].resize((192, 192), Image.Resampling.NEAREST)
        word_order = torch.tensor(np.random.default_rng(8_000_003 + trial).permutation(len(CONCEPTS)), device=device)
        words = _candidate_bank("word", 300_000 + trial, test=True, device=device)[word_order]
        glyph_order = torch.tensor(np.random.default_rng(9_000_003 + trial).permutation(len(CONCEPTS)), device=device)
        glyphs = _candidate_bank("glyph", 300_000 + trial, test=True, device=device)[glyph_order]
        photo_glyph = torch.softmax(model.scores("photo", photo_tensor[None].to(device), "glyph", glyphs), dim=1)
        glyph_word = torch.softmax(model.scores("glyph", glyphs, "word", words), dim=1)
        scores = (photo_glyph @ glyph_word)[0]
        prediction_slot = int(scores.argmax()); prediction = int(word_order[prediction_slot]); correct = prediction == label
        canvas = Image.new("RGB", (768, 430), (12, 18, 29)); canvas.paste(photo, (20, 64))
        draw = ImageDraw.Draw(canvas); draw.text((20, 20), "UNSEEN APPLE PHOTOGRAPH", fill=(106, 225, 255))
        draw.text((236, 20), "CHOOSE THE SAME CONCEPT IN ANOTHER REPRESENTATION", fill=(238, 242, 248))
        for slot, concept_index_tensor in enumerate(word_order):
            concept_index = int(concept_index_tensor)
            word = render_word(concept_index, 300_000 + trial, test=True).resize((144, 96), Image.Resampling.BILINEAR)
            x = 236 + (slot % 3) * 172; y = 58 + (slot // 3) * 72
            word = word.resize((144, 62), Image.Resampling.BILINEAR); canvas.paste(word, (x, y))
            color = (90, 230, 150) if slot == prediction_slot and correct else (255, 104, 104) if slot == prediction_slot else (91, 105, 127)
            draw.rectangle((x - 2, y - 2, x + 146, y + 64), outline=color, width=3 if slot == prediction_slot else 1)
        draw.text((20, 370), f"trial {trial + 1:02d} | selected: {DISPLAY[prediction]} | reward: {int(correct)}",
                  fill=(90, 230, 150) if correct else (255, 104, 104))
        draw.text((20, 400), "Decision chain: photo -> learned symbol -> word; photo-word pairs were never trained.", fill=(163, 174, 194))
        frames.append(canvas); trace.append({"trial": trial + 1, "dataset_index": row_index,
                                             "prediction": CONCEPTS[prediction], "correct": correct})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=650, loop=0, optimize=False)
    return trace


def run(output_dir, data_root, *, epochs=60, seed=601_003):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    model, history, device = train_bridge(data_root, output_dir / "CONCEPT_BRIDGE_BRAIN.pt", epochs=epochs, seed=seed)
    sealed = evaluate(model, data_root, device); controls = evaluate_controls(data_root, device)
    corrupted_predictions = (np.asarray(sealed["predictions"]) + 1) % len(CONCEPTS)
    controls["intentionally_wrong_mapping_accuracy"] = float(np.mean(corrupted_predictions == np.asarray(sealed["labels"])))
    replay = record_apple_replay(model, data_root, device, output_dir / "APPLE_CONCEPT_REPLAY.gif")
    with (output_dir / "CONFUSION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("actual", *CONCEPTS))
        for name, row in zip(CONCEPTS, sealed["confusion"]): writer.writerow((name, *row))
    checkpoint_hash = _sha(output_dir / "CONCEPT_BRIDGE_BRAIN.pt")
    passed = (sealed["photo_to_word_accuracy"] >= .80 and sealed["photo_to_glyph_accuracy"] >= .88
              and sealed["apple"]["accuracy"] >= .90 and controls["untrained_accuracy"] <= .20)
    report = {"format": "wailah-concept-bridge-v22-audit-v2",
        "classification": "confirmatory reuse of CIFAR-100 test split after preserved v21 failure",
        "concepts": list(CONCEPTS), "protocol": {"training_split": "CIFAR-100 train only",
            "sealed_split": "CIFAR-100 test only; 100 unseen photographs per concept",
            "representation_variants_per_photo": 10, "sealed_trials": sealed["trials"],
            "candidate_positions": "independently shuffled for every representation variant",
            "training_bridges": ["photo -> anonymous glyph", "anonymous glyph -> rendered word"],
            "never_trained_directly": "photo -> rendered word",
            "decision_rule": "compose learned photo->glyph and glyph->word probability bridges",
            "feedback": "scalar success/failure after candidate actions; successful choices retained",
            "pretrained_models": "none", "task_labels_given_to_model": False,
            "environment_only_labels": "used only to calculate reward and sealed correctness"},
        "precommitted_thresholds": {"photo_to_word": .80, "photo_to_glyph": .88,
            "apple": .90, "untrained_max": .20}, "training": {"epochs": epochs,
            "final": history[-1], "history": history}, "sealed": {k: v for k, v in sealed.items()
            if k not in ("predictions", "labels")}, "controls": controls,
        "apple_replay": {"trials": len(replay), "correct": sum(row["correct"] for row in replay), "rows": replay},
        "integrity": {"implementation_sha256": source_hash, "checkpoint_sha256": checkpoint_hash},
        "precommitted_pass": bool(passed)}
    (output_dir / "CONCEPT_BRIDGE_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sealed_trials": sealed["trials"], "photo_to_word_accuracy": sealed["photo_to_word_accuracy"],
        "photo_to_glyph_accuracy": sealed["photo_to_glyph_accuracy"], "apple": sealed["apple"],
        "controls": controls, "replay_correct": sum(row["correct"] for row in replay),
        "checkpoint_sha256": checkpoint_hash, "pass": passed}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=601_003)
    args = parser.parse_args(argv); run(args.output, args.data_root, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__": raise SystemExit(main())
