"""The Oracle: reads drawings with CLIP (PyTorch).

CLIP compares a picture with a list of words and scores how well each fits.
To make it steadier on rough sketches, each word is tried in a few phrasings
("a doodle of...", "a child's drawing of...") and the scores are averaged.
In weeks 6-7 this can be swapped for your own trained classifier: keep the
same function names and outputs and nothing else needs to change.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

TEMPLATES = [
    "a simple sketch of {}",
    "a black and white doodle of {}",
    "a child's drawing of {}",
]


def scores(image, labels):
    """Return {label: probability} for every label, adding up to 1."""
    texts = [t.format(label) for label in labels for t in TEMPLATES]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]  # one score per phrase
    per_label = logits.view(len(labels), len(TEMPLATES)).mean(dim=1)
    probs = per_label.softmax(dim=0)
    return {label: p.item() for label, p in zip(labels, probs)}


def recognise(image, labels, top=3):
    """Top guesses as [(label, percent), ...], best first."""
    ranked = sorted(scores(image, labels).items(), key=lambda kv: kv[1], reverse=True)
    return [(label, round(p * 100)) for label, p in ranked[:top]]


def classify_hero(image, classes):
    """Decide a hero's class from the items spotted in their drawing.

    Each class has the same number of items, so adding up the item scores
    per class is fair. Returns (class, percent, runner_up, runner_up_percent,
    top_items_seen).
    """
    labels = list(dict.fromkeys(i for c in classes.values() for i in c["items"]))
    probs = scores(image, labels)
    totals = {name: sum(probs[i] for i in c["items"]) for name, c in classes.items()}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    seen = [label for label, _ in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    (best, best_p), (second, second_p) = ranked[0], ranked[1]
    return best, round(best_p * 100), second, round(second_p * 100), seen


if __name__ == "__main__":
    import rules
    print(recognise(Image.open("test_sketch.png").convert("RGB"), rules.ACTION_ITEMS))
