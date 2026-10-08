"""Use a learned Concept Bridge checkpoint on an ordinary image file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from .concept_bridge import CONCEPTS, ConceptBridge, _candidate_bank


@torch.no_grad()
def identify_file(image_path, checkpoint_path, *, variants=20):
    """Return a ten-concept probability distribution for a real image file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConceptBridge().to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    photo = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))])(image)[None].to(device)
    total = torch.zeros(len(CONCEPTS), device=device)
    for variant in range(variants):
        rng = np.random.default_rng(11_000_003 + variant)
        glyph_order = torch.tensor(rng.permutation(len(CONCEPTS)), device=device)
        word_order = torch.tensor(rng.permutation(len(CONCEPTS)), device=device)
        glyphs = _candidate_bank("glyph", 500_000 + variant, test=True, device=device)[glyph_order]
        words = _candidate_bank("word", 500_000 + variant, test=True, device=device)[word_order]
        photo_glyph = torch.softmax(model.scores("photo", photo, "glyph", glyphs), dim=1)
        glyph_word = torch.softmax(model.scores("glyph", glyphs, "word", words), dim=1)
        word_probability = (photo_glyph @ glyph_word)[0]
        total[word_order] += word_probability
    probability = total / total.sum()
    ranking = torch.argsort(probability, descending=True).cpu().tolist()
    return {"image": str(Path(image_path).resolve()), "variants": variants,
        "prediction": CONCEPTS[ranking[0]], "confidence": float(probability[ranking[0]]),
        "ranking": [{"concept": CONCEPTS[index], "probability": float(probability[index])}
                    for index in ranking],
        "warning": "closed set: it must choose one of the ten known concepts"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Identify an image with the learned ten-concept bridge")
    parser.add_argument("image", type=Path); parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variants", type=int, default=20)
    args = parser.parse_args(argv)
    print(json.dumps(identify_file(args.image, args.checkpoint, variants=args.variants), indent=2))


if __name__ == "__main__": main()
