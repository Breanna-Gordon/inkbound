import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import sketch

RESULTS = Path("results")
METHODS = ["clip", "cnn", "both"]


def to_sketch_image(row):
    """A 28x28 Quick, Draw! array as a black-on-white picture, like a player's sketch."""
    img = Image.fromarray(255 - row.reshape(28, 28).astype(np.uint8), "L")
    return img.resize((256, 256), Image.LANCZOS).convert("RGB")


def run(method, items):
    """items: list of (image, true_label, row_or_None). Each model gets the drawing in
    its own form: CLIP a picture, the CNN the framed 28x28 row when there is one.
    Returns top1, top3, ms per drawing, mix-ups."""
    top1 = top3 = 0
    mixups = Counter()
    start = time.time()
    for image, truth, row in items:
        guesses = [label for label, _ in sketch.recognise(image, backend=method, row=row)]
        top1 += guesses[0] == truth
        top3 += truth in guesses
        if guesses[0] != truth:
            mixups[(truth, guesses[0])] += 1
    ms = (time.time() - start) * 1000 / max(1, len(items))
    return top1 / max(1, len(items)), top3 / max(1, len(items)), ms, mixups


def main():
    p = argparse.ArgumentParser(description="Compare CLIP, the trained CNN and both on Quick, Draw! test drawings and saved player sketches.")
    p.add_argument("--per-class", type=int, default=10, help="Quick, Draw! test drawings per class")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    methods = METHODS if sketch.cnn is not None else ["clip"]
    if sketch.cnn is None:
        print("No trained CNN found, so only CLIP is tested. Run train_classifier.py first.")

    sets = {}
    test_file = RESULTS / "test_set.npz"
    if test_file.exists():
        with np.load(test_file) as data:  # unpack each array once: reading data["x"] unzips it again
            x, y, labels = data["x"], data["y"], list(data["labels"])
        picked = []
        for i in range(len(labels)):
            idx = np.where(y == i)[0][:args.per_class]
            picked += [(to_sketch_image(x[j]), labels[i], x[j]) for j in idx]
        sets["Quick, Draw! test set"] = picked
    player = Path("data/player_sketches")
    if player.exists():
        items = [(Image.open(f).convert("RGB"), f.parent.name, None)
                 for f in sorted(player.glob("*/*.png"))]
        if items:
            sets["Player sketches"] = items
    if not sets:
        print("Nothing to test yet: train the CNN or play a few turns first.")
        return

    rows = [("test set", "method", "drawings", "top-1 %", "top-3 %", "ms per drawing")]
    confusion_lines = []
    for name, items in sets.items():
        print(f"\n{name}: {len(items)} drawings")
        for method in methods:
            top1, top3, ms, mixups = run(method, items)
            rows.append((name, method, len(items), f"{top1 * 100:.1f}", f"{top3 * 100:.1f}", f"{ms:.0f}"))
            print(f"  {method:5s}  top-1 {top1:6.1%}  top-3 {top3:6.1%}  {ms:5.0f} ms each")
            confusion_lines.append(f"\n{name} · {method}: most common mix-ups (drawn -> guessed)")
            confusion_lines += [f"  {t} -> {g}: {n}" for (t, g), n in mixups.most_common(10)]

    with open(RESULTS / "evaluation.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (RESULTS / "confusions.txt").write_text("\n".join(confusion_lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4), squeeze=False)
        colours = {"clip": "#7a86a8", "cnn": "#8e2a1f", "both": "#b8893a"}
        for ax, name in zip(axes[0], sets):
            data_rows = [r for r in rows[1:] if r[0] == name]
            x = np.arange(len(data_rows))
            ax.bar(x - 0.2, [float(r[3]) for r in data_rows], 0.4, label="top-1",
                   color=[colours[r[1]] for r in data_rows])
            ax.bar(x + 0.2, [float(r[4]) for r in data_rows], 0.4, label="top-3",
                   color=[colours[r[1]] for r in data_rows], alpha=0.45)
            ax.set_xticks(x, [r[1] for r in data_rows])
            ax.set_ylim(0, 100)
            ax.set_ylabel("accuracy (%)")
            ax.set_title(f"{name} ({data_rows[0][2]} drawings)", fontsize=10)
        axes[0][0].legend(["top-1 (solid)", "top-3 (faded)"], fontsize=8)
        fig.suptitle("How well each Oracle reads drawings")
        fig.tight_layout()
        fig.savefig(RESULTS / "evaluation.png", dpi=150)
    except ImportError:
        pass
    print("\nSaved results/evaluation.csv, results/evaluation.png and results/confusions.txt")


if __name__ == "__main__":
    main()
