import io
import hashlib
import json
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# STYLE SETTINGS
# ---------------------------------------------------------------------------
# Each preset is a workflow exported from ComfyUI with "Export (API)". In the positive prompt box,
# type INKBOUND_PROMPT before exporting (and INKBOUND_NEGATIVE in the negative box, if there is one).
# The code fills those in, gives every picture a new seed and sets the picture size, so any
# ComfyUI template works without code changes.
PRESETS = {
    # Neta Lumina (Apache 2.0): anime and game illustration, reads full sentences (Gemma text encoder).
    "neta_lumina": dict(workflow="workflows/neta_lumina_api.json", width=1216, height=832),
}
PRESET = "neta_lumina"

PROMPT_MARK = "INKBOUND_PROMPT"
NEGATIVE_MARK = "INKBOUND_NEGATIVE"

ART_STYLES = {
    # Inkbound's house style, from the project mood board: anime fantasy RPG illustration with
    # hand-drawn ink linework and a deep purple, indigo, cream and gold palette.
    "inkbound": dict(
        medium="anime fantasy RPG illustration",
        words="expressive hand-drawn ink linework, cel shading, detailed fantasy outfits with cloaks, "
              "belts and pouches, deep purple, indigo, cream and gold colour palette, ink splash accents, "
              "parchment tones",
        avoid="photo, photorealistic, realistic, 3d render, glossy, blurry, monochrome, greyscale"),
    "graphic_novel": dict(
        medium="fantasy graphic novel illustration",
        words="bold black ink lines, strong flat colours, heavy shadows, dramatic comic panel",
        avoid="anime, chibi, photo, 3d render, soft focus, monochrome, black and white, greyscale"),
    "hand_drawn_animation": dict(
        medium="hand-drawn 2D animation still",
        words="clean character lines, soft painted backgrounds, warm gentle colours",
        avoid="photo, 3d render, cgi, realistic skin, gritty, monochrome, greyscale"),
    "cartoon": dict(
        medium="cartoon illustration",
        words="bold clean outlines, flat bright colours, cel shading, expressive faces",
        avoid="photo, realistic, 3d render, painterly, detailed skin, monochrome, greyscale"),
    "painted_fantasy": dict(
        medium="epic fantasy oil painting",
        words="dramatic cinematic lighting, rich deep colours, detailed brushwork, dungeons and dragons art",
        avoid="anime, cartoon, chibi, 3d render, photo, flat colours, glossy"),
    "dark_fairytale": dict(
        medium="dark fairy-tale illustration",
        words="gothic storybook, moody shadows, muted colours with glowing highlights, intricate linework",
        avoid="anime, cute, chibi, 3d render, photo, glossy"),
    "concept_art": dict(
        medium="painterly fantasy concept art",
        words="loose expressive brushstrokes, atmospheric lighting, video game art",
        avoid="anime, chibi, photo, 3d render, cartoon outlines"),
    "stained_glass": dict(
        medium="stained glass window art",
        words="bold lead outlines, glowing jewel colours, mosaic panels",
        avoid="photo, 3d render, anime, soft shading, realistic"),
    "storybook": dict(
        medium="storybook illustration",
        words="hand-painted gouache and watercolour, soft edges, warm light, whimsical picture book",
        avoid="photo, realistic, 3d render, black and white, monochrome"),
    "pixel_art": dict(
        medium="pixel art",
        words="16-bit fantasy role-playing game, limited colour palette, crisp pixels",
        avoid="photo, realistic, 3d render, smooth shading, painterly, blurry, anime"),
}
ART_STYLE = "inkbound"        # scenes
CHARACTER_STYLE = "inkbound"  # character designs and portraits

# Always avoided, whatever the style.
NEGATIVE_ALWAYS = ("blurry, text, watermark, ugly, deformed, bad anatomy, extra limbs, missing head, "
                   "headless, cropped head")
# The same without the anatomy words: the first inkbound favourites were made with this.
NEGATIVE_BASIC = "blurry, text, watermark, ugly, deformed, extra limbs"
# Added for single-character pictures, so the model draws one person, not several.
NEGATIVE_SOLO = "two people, multiple people, duplicate, twins, crowd, group"

# How scene pictures are made:
#   "layered": a simple background with no people, then each character's saved portrait cut out and placed
#              on it, then the player's drawing inked on top. Characters look the same in every picture.
#   "full":    the whole scene painted in one go from a text description (the first version of the game).
PICTURE_MODE = "layered"
# "scenery" and "no humans" are the tags this kind of model learnt for pictures without people; words like
# "nobody" make it think of people.
BACKGROUND_WORDS = "scenery, no humans, simple uncluttered storybook background, minimal detail, open ground"
# Left out of the style words for backgrounds and animals, which would otherwise be given clothes or people.
OUTFIT_WORDS = "detailed fantasy outfits with cloaks, belts and pouches, "
NEGATIVE_CREATURE = "clothes, clothing, armour, anthropomorphic, humanoid, standing on two legs, person"
NEGATIVE_EMPTY = "people, person, character, figure, face, creature, animal, crowd"
BACKGROUND_SIZE = (1024, 704)  # smaller than the 1216 x 832 of other pictures: about 30% fewer pixels to paint
BACKGROUND_STEPS = 14  # the workflow uses 20; backgrounds don't need that much detail
INK = (59, 31, 74)  # the player's drawing is inked into the scene in this deep purple...
PALE = (247, 236, 208)  # ...with a parchment-coloured edge
# ---------------------------------------------------------------------------

# ComfyUI usually listens on 8188; the Desktop app sometimes uses 8000.
PORTS = [8188, 8000]
_server = None


def server():
    """Find ComfyUI the first time it's needed."""
    global _server
    if _server is None:
        for port in PORTS:
            url = f"http://127.0.0.1:{port}"
            try:
                if requests.get(f"{url}/system_stats", timeout=2).ok:
                    _server = url
                    break
            except requests.RequestException:
                pass
        else:
            raise RuntimeError("Can't reach ComfyUI on port 8188 or 8000. Is the ComfyUI app open?")
    return _server


def plain(text):
    """Remove (words:1.2) weights. Language-model text encoders read plain sentences best."""
    return re.sub(r"\(([^()]*?):\d+(\.\d+)?\)", r"\1", text)


def build_workflow(prompt, negative, preset=None, size=None, seed=None, steps=None):
    """Load the preset's workflow and fill in the prompt, the seed and the picture size.
    seed: a number to repeat an earlier picture exactly; leave it out for a new random picture.
    steps: fewer denoising steps is faster but less detailed; leave it out to use the workflow's own number."""
    p = PRESETS[preset or PRESET]
    path = Path(p["workflow"])
    if not path.exists():
        raise RuntimeError(f"Workflow file {path} not found. Export it from ComfyUI with Export (API).")
    workflow = json.loads(path.read_text(encoding="utf-8"))
    width, height = size or (p["width"], p["height"])
    found_prompt = False
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, str) and PROMPT_MARK in value:
                inputs[key] = value.replace(PROMPT_MARK, prompt)
                found_prompt = True
            elif isinstance(value, str) and NEGATIVE_MARK in value:
                inputs[key] = value.replace(NEGATIVE_MARK, negative)
            elif key in ("seed", "noise_seed") and isinstance(value, int):
                inputs[key] = seed if seed is not None else random.randint(0, 2**32 - 1)
            elif key == "steps" and isinstance(value, int) and steps:
                inputs[key] = steps
        if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
            inputs.update(width=width, height=height)
    if not found_prompt:
        raise RuntimeError(f"No {PROMPT_MARK} found in {path}. Type it in the prompt box in ComfyUI, "
                           "then export the workflow again.")
    return workflow


def paint(description, style="", preset=None, art_style=None, emphasis=None, size=None, solo=False, seed=None,
          anatomy=True, avoid="", outfits=True, steps=None):
    """Make one picture and return it as PNG bytes.
    The prompt is written as plain sentences: the art style's medium first, then the scene,
    then the colours and the style words. solo: one character only (character designs).
    avoid: extra things to keep out of this picture. outfits=False leaves out the clothing style words.
    emphasis is accepted for older callers and ignored: these models don't use weights."""
    a = ART_STYLES[art_style or ART_STYLE]
    scene = plain(description).strip().rstrip(".")
    extra = plain(style).strip()
    prompt = f"{a['medium'][0].upper()}{a['medium'][1:]}. {scene}." + (f" {extra}." if extra else "") \
        + f" Style: {a['words'] if outfits else a['words'].replace(OUTFIT_WORDS, '')}."
    negative = ", ".join([a["avoid"], NEGATIVE_ALWAYS if anatomy else NEGATIVE_BASIC]
                         + ([NEGATIVE_SOLO] if solo else []) + ([avoid] if avoid else []))
    workflow = build_workflow(prompt, negative, preset, size, seed, steps)

    url = server()
    job = requests.post(f"{url}/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}).json()
    if "prompt_id" not in job:  # ComfyUI rejected it, e.g. a model file is missing
        raise RuntimeError(f"ComfyUI refused the job for preset '{preset or PRESET}': {job.get('error', job)}")
    job_id = job["prompt_id"]
    for _ in range(300):  # wait up to 5 minutes
        history = requests.get(f"{url}/history/{job_id}").json()
        if job_id in history:
            break
        time.sleep(1)
    else:
        raise TimeoutError("ComfyUI took too long to make the picture.")

    for output in history[job_id]["outputs"].values():
        for img in output.get("images", []):
            params = {"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]}
            return requests.get(f"{url}/view", params=params).content
    raise RuntimeError("ComfyUI finished but returned no image.")


def portrait(look, colours="", preset=None, art_style=None, seed=None, anatomy=True, creature=False):
    """A full-body character design on a tall canvas, one character only.
    creature=True: an animal or monster, drawn as a real animal with no clothes."""
    p = PRESETS[preset or PRESET]
    tall = (min(p["width"], p["height"]), max(p["width"], p["height"]))
    look = re.sub(r"^(a|an|one)\s+", "", look.strip(), flags=re.IGNORECASE)
    if creature:
        description = (f"A full-body drawing of a single {look}, a real animal standing on all fours, "
                       "no clothes, on a plain parchment background")
        return paint(description, colours, preset, art_style or CHARACTER_STYLE, size=tall, solo=True, seed=seed,
                     avoid=NEGATIVE_CREATURE, outfits=False)
    description = (f"A full-body character design of a single {look}, standing alone and facing the viewer, "
                   "on a plain parchment background")
    return paint(description, colours, preset, art_style or CHARACTER_STYLE, size=tall, solo=True, seed=seed,
                 anatomy=anatomy)


def background(description, style="", preset=None, art_style=None):
    """A simple scene background with no people in it, for the characters to stand on.
    Smaller and with fewer steps than other pictures: backgrounds are softened or partly covered anyway."""
    return paint(f"{plain(description).strip().rstrip('.')}. {BACKGROUND_WORDS}", style, preset, art_style,
                 avoid=NEGATIVE_EMPTY, outfits=False, size=BACKGROUND_SIZE, steps=BACKGROUND_STEPS)


# ---------------------------------------------------------------------------
# Saved pictures: each portrait and location background is painted once and kept in media/, so it is
# reused instead of painted again. prepare_art.py paints them all in advance.
# ---------------------------------------------------------------------------
PORTRAITS = Path("media/portraits")
BACKGROUNDS = Path("media/backgrounds")
PORTRAIT_VERSION = 2  # change this when portraits are painted differently, so old ones aren't reused


def safe_name(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def saved_name(name, description, version=PORTRAIT_VERSION):
    """A file name with a short code made from the description and version, so a changed description gets a
    new picture instead of an old one."""
    return f"{safe_name(name)}-{hashlib.md5(f'{description}|{version}'.encode()).hexdigest()[:6]}"


def saved_portrait(name, look, creature=False, **settings):
    """A character's portrait: loaded if it was painted before, otherwise painted now and saved."""
    path = PORTRAITS / f"{saved_name(name, look)}.png"
    if path.exists():
        return path.read_bytes()
    png = portrait(look, creature=creature, **settings)
    PORTRAITS.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return png


def location_background(location):
    """The opening background of a starting location, painted once and saved."""
    description = location.get("background") or f"{location['name']}, {location['type'].lower()}"
    path = BACKGROUNDS / f"{saved_name(location['id'], description + location['style'])}.png"
    if path.exists():
        return path.read_bytes()
    png = background(description, location["style"])
    BACKGROUNDS.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return png


def _spread(mask):
    """Every pixel touching the mask, plus the mask itself (one step of growing in four directions)."""
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def _grow(seed, allowed):
    """Spread from the seed pixels to touching allowed pixels until nothing changes."""
    reach = seed & allowed
    while True:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= allowed
        if (grown == reach).all():
            return reach
        reach = grown


GAP = 2  # outline gaps up to about twice this many pixels wide (at the working size) are closed


def cut_out(png, work=420):
    """Cut a portrait out of its parchment background. Returns an RGBA image cropped to the character.

    The parchment is found by spreading in from the edges across warm, light, plain pixels (and the darker
    burnt edges of the page). The character's ink outline stops the spreading, so the character is kept."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    small = np.asarray(img.resize((work, int(work * img.height / img.width)), Image.BILINEAR)).astype(np.int16)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    brightest, dullest = small.max(-1), small.min(-1)
    warm = (r >= g - 6) & (g >= b - 6)
    paper = warm & (brightest > 150) & (brightest - dullest < 120)
    h, w = paper.shape
    m = int(min(h, w) * 0.09)
    band = np.zeros_like(paper)
    band[:m] = band[-m:] = True
    band[:, :m] = band[:, -m:] = True
    burnt = band & warm & (brightest > 50)
    edge = np.zeros_like(paper)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    allowed = paper | burnt
    # Close small gaps in the outline first (e.g. between strands of pale hair), so the spreading can't
    # leak through them into a pale face. Then let the background creep back up to the lines.
    walls = ~allowed
    for _ in range(GAP):
        walls = _spread(walls)
    background = _grow(edge, allowed & ~walls)
    for _ in range(GAP):
        background = _spread(background) & allowed
    keep = np.where(background, 0, 255).astype(np.uint8)
    k = max(2, int(min(h, w) * 0.025))  # bits of page corner at the very edge
    keep[:k] = keep[-k:] = 0
    keep[:, :k] = keep[:, -k:] = 0
    alpha = Image.fromarray(keep).resize(img.size, Image.BILINEAR)
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
    sprite = img.convert("RGBA")
    sprite.putalpha(alpha)
    return sprite.crop(alpha.point(lambda v: 255 if v > 20 else 0).getbbox())


def ink(drawing, size):
    """The player's strokes (an RGBA image) as thick ink lines at the picture's size, with a pale edge
    so they show up on dark and light parts of the scene."""
    alpha = drawing.getchannel("A").resize(size, Image.BILINEAR)
    lines = alpha.filter(ImageFilter.MaxFilter(5))
    edge = lines.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1))
    layer = Image.new("RGBA", size, PALE + (0,))
    layer.putalpha(edge.point(lambda v: v * 0.85))
    ink_lines = Image.new("RGBA", size, INK + (0,))
    ink_lines.putalpha(lines)
    layer.alpha_composite(ink_lines)
    return layer


def compose(background_png, sprites, drawing=None):
    """Put the characters on the background (the hero first, on the left) and the drawing on top.
    sprites: RGBA cut-outs from cut_out(). drawing: optional RGBA strokes. Returns PNG bytes."""
    scene = Image.open(io.BytesIO(background_png)).convert("RGBA")
    w, h = scene.size
    if drawing is not None:
        scene.alpha_composite(ink(drawing, (w, h)))
    spots = {1: [0.32], 2: [0.27, 0.73], 3: [0.2, 0.5, 0.8], 4: [0.14, 0.38, 0.62, 0.86]}[min(len(sprites), 4)] \
        if sprites else []
    tall = 0.8 if len(sprites) <= 2 else 0.68
    for sprite, x in zip(sprites[:4], spots):
        scale = min(h * tall / sprite.height, w / len(spots) * 1.05 / sprite.width)
        s = sprite.resize((int(sprite.width * scale), int(sprite.height * scale)), Image.LANCZOS)
        left, top = int(w * x - s.width / 2), int(h * 0.97 - s.height)
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # a soft shadow on the ground under the feet
        ImageDraw.Draw(shadow).ellipse((left + s.width * 0.15, h * 0.97 - h * 0.025, left + s.width * 0.85,
                                        h * 0.97 + h * 0.02), fill=(30, 15, 35, 90))
        scene.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(6)))
        scene.alpha_composite(s, (max(0, left), max(0, top)))
    out = io.BytesIO()
    scene.convert("RGB").save(out, "PNG")
    return out.getvalue()


def _bust(sprite, part=0.58):
    """The top of a full-body cut-out (head to waist), for dialogue close-ups."""
    return sprite.crop((0, 0, sprite.width, int(sprite.height * part)))


def compose_dialogue(background_png, speaker, hero=None, speaker_is_creature=False):
    """A dialogue close-up, like Persona or Baldur's Gate 3: the background softened, the speaker large on the
    left (head to waist), the hero smaller and dimmer on the right. Returns PNG bytes."""
    scene = Image.open(io.BytesIO(background_png)).convert("RGB").filter(ImageFilter.GaussianBlur(3))
    scene = Image.blend(scene, Image.new("RGB", scene.size, (30, 18, 40)), 0.25).convert("RGBA")
    w, h = scene.size
    if hero is not None:
        b = _bust(hero)
        scale = min(h * 0.74 / b.height, w * 0.38 / b.width)
        b = b.resize((int(b.width * scale), int(b.height * scale)), Image.LANCZOS)
        dim = Image.blend(b, Image.new("RGBA", b.size, (40, 25, 50, 0)), 0.3)
        dim.putalpha(b.getchannel("A"))
        scene.alpha_composite(dim, (w - b.width - int(w * 0.03), h - b.height))
    if speaker_is_creature:  # an animal is shown whole, not cut at the waist
        s = speaker
        scale = min(h * 0.8 / s.height, w * 0.6 / s.width)
        bottom = int(h * 0.97)
    else:
        s = _bust(speaker)
        scale = min(h * 1.0 / s.height, w * 0.55 / s.width)
        bottom = h
    s = s.resize((int(s.width * scale), int(s.height * scale)), Image.LANCZOS)
    scene.alpha_composite(s, (int(w * 0.04), bottom - s.height))
    out = io.BytesIO()
    scene.convert("RGB").save(out, "PNG")
    return out.getvalue()


def stamp(background_png, drawing):
    """Ink the player's drawing into the background itself, so it stays in later pictures of this place."""
    scene = Image.open(io.BytesIO(background_png)).convert("RGBA")
    scene.alpha_composite(ink(drawing, scene.size))
    out = io.BytesIO()
    scene.convert("RGB").save(out, "PNG")
    return out.getvalue()


if __name__ == "__main__":
    # Style test: makes one picture with the current settings.
    print("Using preset:", PRESET, "| art style:", ART_STYLE)
    png = paint("A young woman in a patched green hood slides a crumpled map across a tavern table "
                "while two guards in red tabards burst through the door", "warm candlelight")
    Path("media").mkdir(exist_ok=True)
    Path("media/style_test.png").write_bytes(png)
    print("Saved media/style_test.png")
