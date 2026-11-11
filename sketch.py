from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import oracle_cnn
import rules

BACKEND = "both"
CNN_WEIGHT = 0.6  # with "both": how much to trust the CNN over CLIP (0-1)

MODEL_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

cnn, cnn_labels = oracle_cnn.load()

TEMPLATES = [
    "a simple sketch of {}",
    "a black and white doodle of {}",
    "a child's drawing of {}",
]

QUICKDRAW = [l.strip() for l in open("quickdraw_categories.txt", encoding="utf-8") if l.strip()]
# Everything the Oracle can recognise: Quick, Draw!'s objects plus the game's fantasy items.
VOCABULARY = list(dict.fromkeys(QUICKDRAW + rules.ACTION_ITEMS))

_text_cache = {}


def _vectors(output):
    """CLIP's feature vectors. Newer versions of transformers wrap them in an
    output object (in pooler_output); older versions return them directly."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def _text_features(labels):
    """CLIP's description of each label, averaged over the phrasings.
    Worked out once per label and remembered, which keeps the game fast."""
    new = [label for label in labels if label not in _text_cache]
    for i in range(0, len(new), 64):
        batch = new[i:i + 64]
        texts = [t.format(label) for label in batch for t in TEMPLATES]
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = _vectors(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.view(len(batch), len(TEMPLATES), -1).mean(dim=1)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for label, f in zip(batch, feats):
            _text_cache[label] = f
    return torch.stack([_text_cache[label] for label in labels])


def clip_scores(image, labels):
    """Return {label: probability} from CLIP, adding up to 1."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img = _vectors(model.get_image_features(**inputs))
    img = img / img.norm(dim=-1, keepdim=True)
    logits = model.logit_scale.exp() * img @ _text_features(labels).T
    probs = logits.softmax(dim=-1)[0]
    return {label: p.item() for label, p in zip(labels, probs)}


def scores(image, extra=(), backend=None, row=None):
    """Combined {label: probability} over the full vocabulary plus any extra
    labels (for example, objects the Dungeon Master has put in the scene).
    row: optionally, the same drawing as a framed Quick, Draw! row for the CNN."""
    backend = backend or BACKEND
    labels = list(dict.fromkeys(VOCABULARY + [e.lower() for e in extra if e]))
    if backend == "cnn" and cnn is None:
        raise RuntimeError("No trained CNN found. Run train_classifier.py first.")
    if backend == "cnn":
        return oracle_cnn.predict(cnn, cnn_labels, image, row)
    clip = clip_scores(image, labels)
    if backend == "clip" or cnn is None:
        return clip
    mine = oracle_cnn.predict(cnn, cnn_labels, image, row)
    blended = {label: CNN_WEIGHT * mine[label] + (1 - CNN_WEIGHT) * p if label in mine else p
               for label, p in clip.items()}
    total = sum(blended.values())
    return {label: p / total for label, p in blended.items()}


def recognise(image, extra=(), top=3, backend=None, row=None):
    """Top guesses as [(label, percent), ...], best first."""
    ranked = sorted(scores(image, extra, backend, row).items(), key=lambda kv: kv[1], reverse=True)
    return [(label, round(p * 100)) for label, p in ranked[:top]]


def classify_hero(image, classes):
    """Decide a hero's class from the items spotted in their drawing.

    Each class has the same number of items, so adding up the item scores per
    class is fair. Returns (class, percent, runner_up, runner_up_percent, top_items_seen).
    """
    labels = list(dict.fromkeys(i for c in classes.values() for i in c["items"]))
    probs = clip_scores(image, labels)
    totals = {name: sum(probs[i] for i in c["items"]) for name, c in classes.items()}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    seen = [label for label, _ in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    (best, best_p), (second, second_p) = ranked[0], ranked[1]
    return best, round(best_p * 100), second, round(second_p * 100), seen


def save_sketch(image, label, folder="data/player_sketches"):
    """Keep a player's sketch with its confirmed label. evaluate.py uses these
    to test the Oracle on real in-game drawings."""
    out = Path(folder) / label.replace("/", "-")
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / f"{len(list(out.glob('*.png'))) + 1:04d}.png")


if __name__ == "__main__":
    print(recognise(Image.open("test_sketch.png").convert("RGB")))
