import argparse
import io
import random
from pathlib import Path

from PIL import Image, ImageDraw

import art
import rules

PICKS = ["inkbound"]

SCENES = {
    "tavern scene": ("medium shot, (a young woman in a patched green hood sliding a crumpled map across a "
                     "tavern table, two guards in red tabards bursting through the door:1.25), "
                     "(candlelit tavern:1.1)", "warm candlelight"),
    "cave scene": ("medium shot, (a barbarian in a fur cloak at the dark mouth of a cave in a snowy cliff, "
                   "a thick plume of smoke pouring out, huge clawed footprints:1.25), (snowy mountain pass:1.1)",
                   "pink dawn light"),
}
CHARACTERS = {
    "hero design": ("a young bard in a red feathered hat and teal cape, carrying a lute", "red and teal"),
    "character design": ("freckled girl with dark hair in two braids, strong defined features and a "
                         "flour-dusted apron", "warm cream and brown"),
}
# Fixed seeds, so every style is compared on the same starting noise and BG's favourite pictures
# (in samples/) are made again exactly. anatomy=False matches how the bard and cave favourites were made.
SEEDS = {"tavern scene": (1017620332, True), "cave scene": (1866876706, False),
         "hero design": (313066362, False), "character design": (2033388415, True)}
CELL = (300, 300)


def fit(png):
    """Shrink a picture to fit a square cell without stretching it."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img.thumbnail(CELL)
    cell = Image.new("RGB", CELL, "white")
    cell.paste(img, ((CELL[0] - img.width) // 2, (CELL[1] - img.height) // 2))
    return cell


def grid(rows, columns, cells, size, path):
    """Save labelled rows x columns of pictures as one image."""
    label_w, label_h = 170, 30
    w, h = size
    sheet = Image.new("RGB", (label_w + w * len(columns), label_h + (h + 8) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        draw.text((label_w + c * w + 8, 8), name, fill="black")
    for r, name in enumerate(rows):
        y = label_h + r * (h + 8)
        draw.text((8, y + h // 2), name, fill="black")
        for c, img in enumerate(cells[r]):
            if img:
                sheet.paste(img, (label_w + c * w, y))
    sheet.save(path)
    print(f"Saved {path}")


def compare_styles(styles, preset, out):
    columns = list(SCENES) + list(CHARACTERS)
    cells = []
    for style in styles:
        row = []
        for name in columns:
            print(f"Painting {style} / {name}...")
            try:
                if name in SCENES:
                    description, light = SCENES[name]
                    png = art.paint(description, light, preset=preset, art_style=style, emphasis=None,
                                    seed=SEEDS[name][0], anatomy=SEEDS[name][1])
                else:
                    look, colours = CHARACTERS[name]
                    seed, anatomy = SEEDS[name]
                    png = art.portrait(look, colours, preset=preset, art_style=style, seed=seed,
                                       anatomy=anatomy)
                (out / f"{style}_{name.replace(' ', '_')}.png").write_bytes(png)
                row.append(fit(png))
            except Exception as e:  # e.g. a missing model file
                print(f"  skipped: {e}")
                row.append(None)
        cells.append(row)
    grid(styles, columns, cells, CELL, out / "style_comparison.png")


def class_sheet(style, preset, out):
    names = list(rules.CLASSES)
    cells = []
    for start in range(0, len(names), 4):
        row = []
        for name in names[start:start + 4]:
            print(f"Painting {name} in {style}...")
            settings = rules.portrait_settings(name)  # the same settings the game uses
            seed = settings.pop("seed", None) or random.randint(0, 2**32 - 1)
            try:
                png = art.portrait(rules.CLASSES[name]["look"], preset=preset, art_style=style, seed=seed,
                                   **settings)
                # the seed is in the file name, so a favourite can be added to rules.PORTRAITS
                (out / f"class_{style}_{name}_{seed}.png").write_bytes(png)
                row.append(fit(png))
            except Exception as e:
                print(f"  skipped: {e}")
                row.append(None)
        cells.append(row)
    rows = [" / ".join(names[i:i + 4]) for i in range(0, len(names), 4)]
    grid(rows, ["", "", "", ""], cells, CELL, out / f"class_designs_{style}.png")


def main():
    p = argparse.ArgumentParser(description="Paint the same scenes and characters in every art style, "
                                            "or all 12 classes in one style.")
    p.add_argument("--styles", nargs="+", default=PICKS,
                   help="art styles to compare (use 'all' for every style)")
    p.add_argument("--preset", default=art.PRESET, help="model preset from art.py")
    p.add_argument("--classes", metavar="STYLE", nargs="?", const=art.CHARACTER_STYLE,
                   help="paint all 12 classes in this art style instead (default: the character style)")
    args = p.parse_args()
    out = Path("results/styles")
    out.mkdir(parents=True, exist_ok=True)
    if args.styles == ["all"]:
        args.styles = list(art.ART_STYLES)
    if args.classes:
        class_sheet(args.classes, args.preset, out)
    else:
        compare_styles(args.styles, args.preset, out)


if __name__ == "__main__":
    main()
