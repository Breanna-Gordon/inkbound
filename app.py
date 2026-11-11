import base64
import csv
import difflib
import html
import io
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import art
import dm
import rules
import sketch
import voice

CONFIDENT = 50   # below this %, the Oracle asks the player what they drew
W, H = 576, 384  # canvas size: same 3:2 shape as the pictures, sized to fit one page of the book
WINS_PER_GOAL = 2  # successes needed to move on to the next part of the quest
DRAW_EVERY = 3  # at least one page in this many is a drawing page (an obstacle to draw past, nobody talking)
STUCK_AFTER = 3  # pages on one part of the quest before the Dungeon Master is told to move the story on
STRIP = 3  # how many earlier pages show as small panels above the current scene
PAPER = (248, 239, 217)  # the colour of a blank page
FADE = 0.4  # how much the scene fades when you draw on it (0 = not at all, 1 = blank page)
PEN = "#8e2a1f"  # dark red ink, easy to see on the faded scene

st.set_page_config(page_title="Inkbound", layout="wide")

# ---------------------------------------------------------------------------
# Storybook look: an open book on a dark desk. Each screen is a two-page spread,
# pictures sit in inked panel frames with captions underneath, like a picture book.
# ---------------------------------------------------------------------------
INK = "#2b1f17"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cinzel+Decorative:wght@700&family=IM+Fell+English:ital@0;1&display=swap');
.stApp {{ background: radial-gradient(ellipse at 50% 30%, #3b2a1d 0%, #20160f 70%, #140e09 100%); }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1400px; padding-top: 2.2rem; }}

/* the open book: two parchment pages with a shadow down the middle */
.st-key-book {{
  background:
    linear-gradient(to right, rgba(0,0,0,0) 46%, rgba(90,60,30,.28) 49.6%, rgba(40,25,10,.45) 50%,
                    rgba(90,60,30,.28) 50.4%, rgba(0,0,0,0) 54%),
    radial-gradient(ellipse at 25% 50%, #f7ecd0 0%, #efdfb8 65%, #dcc493 100%) left / 50% 100% no-repeat,
    radial-gradient(ellipse at 75% 50%, #f7ecd0 0%, #efdfb8 65%, #dcc493 100%) right / 50% 100% no-repeat;
  border-radius: 6px 6px 10px 10px; padding: 34px 44px 18px;
  box-shadow: 0 18px 40px rgba(0,0,0,.6), 0 2px 0 #b89a63, 0 5px 0 #8b6d42, 0 8px 0 #5e4528;
  color: {INK};
}}
.st-key-book p, .st-key-book li, .st-key-book label, .st-key-book span {{ color: {INK}; }}
.st-key-book h1, .st-key-book h2, .st-key-book h3 {{ font-family: 'Cinzel', serif !important; color: {INK} !important; }}
.st-key-book h1 {{ font-family: 'Cinzel Decorative', serif !important; font-size: 2.3rem !important; padding-top: 0; }}
.st-key-book [data-testid="stMarkdownContainer"] {{ font-family: 'IM Fell English', serif; font-size: 1.08rem; }}

.kicker {{ font-family: 'Cinzel', serif; letter-spacing: .3em; font-size: .78rem; color: #8e2a1f; }}
.folio {{ text-align: center; font-family: 'IM Fell English', serif; font-style: italic; color: #7a6145;
          margin-top: 14px; }}
.ornament {{ text-align: center; color: #8e2a1f; letter-spacing: .6em; margin: 4px 0 8px; }}

/* the narrator's text, with a drop capital like a printed storybook */
.page {{ font-family: 'IM Fell English', serif; font-size: 1.2rem; line-height: 1.6; color: {INK}; margin: 2px 0 10px; }}
.page::first-letter {{ float: left; font-family: 'Cinzel Decorative', serif; font-size: 3.4rem; line-height: .82;
                       padding: 6px 8px 0 0; color: #8e2a1f; }}
.label {{ display: block; font-family: 'Cinzel', serif; font-size: .72rem; letter-spacing: .25em; color: #8e2a1f;
          margin-bottom: 2px; }}

/* inked panel frames with a caption, like the panels of a picture book */
.strip {{ display: grid; grid-template-columns: repeat(var(--n), 1fr); gap: 12px; margin: 4px 0 14px; }}
.panel {{ margin: 0; }}
.panel img {{ width: 100%; display: block; border: 3px solid {INK}; box-shadow: 3px 3px 0 rgba(43,31,23,.35); }}
.panel figcaption {{ font-family: 'IM Fell English', serif; font-style: italic; font-size: .88rem; line-height: 1.25;
                     color: #4a3828; margin-top: 5px; }}
.panel figcaption b {{ font-family: 'Cinzel', serif; font-style: normal; color: #8e2a1f; margin-right: 4px; }}
.st-key-scene, .st-key-hero-canvas {{ width: fit-content !important; border: 4px solid {INK};
  box-shadow: 4px 4px 0 rgba(43,31,23,.35); background: #f8efd9; gap: 0; }}
.st-key-portrait img, .st-key-last-picture img, .st-key-scene-picture img {{ border: 4px solid {INK}; box-shadow: 4px 4px 0 rgba(43,31,23,.35); }}

/* dialogue box under the picture when someone speaks */
.speech {{ background: {INK}; padding: 22px 20px 14px; margin: 14px 0 6px; position: relative;
           font-family: 'IM Fell English', serif; font-size: 1.18rem; line-height: 1.45; color: #f7ecd0 !important;
           box-shadow: 4px 4px 0 rgba(43,31,23,.35); clip-path: polygon(0 0, 100% 0, 99% 100%, 1% 100%); }}
.speech .who {{ position: absolute; top: -2px; left: 14px; background: #8e2a1f; color: #f7ecd0 !important;
                font-family: 'Cinzel', serif; font-size: .78rem; letter-spacing: .2em; padding: 2px 10px;
                text-transform: uppercase; }}

/* a drawing challenge set by the Dungeon Master */
.challenge {{ border: 2px dashed #8e2a1f; padding: 8px 12px; margin: 0 0 10px; font-family: 'IM Fell English', serif;
              font-size: 1.08rem; background: rgba(255,250,235,.5); }}
.challenge b {{ font-family: 'Cinzel', serif; font-size: .75rem; letter-spacing: .18em; color: #8e2a1f; }}

/* character ribbon and sheet */
.ribbon {{ border-top: 2px solid {INK}; border-bottom: 2px solid {INK}; padding: 6px 2px; margin-bottom: 10px;
           font-family: 'Cinzel', serif; font-size: .85rem; display: flex; flex-wrap: wrap; gap: 6px 16px;
           align-items: center; }}
.ribbon .name {{ font-family: 'Cinzel Decorative', serif; font-size: 1.15rem; color: #8e2a1f; }}
.hearts {{ color: #b23a2a; font-size: 1.15rem; letter-spacing: 2px; }}
.insp {{ color: #b8893a; }}
.stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; text-align: center; margin: 10px 0; }}
.stats div {{ border: 1.5px solid {INK}; border-radius: 4px; padding: 4px 0; font-family: 'Cinzel', serif;
              font-size: .75rem; background: rgba(255,250,235,.5); }}
.stats b {{ display: block; font-size: 1.15rem; }}
.tag {{ display: inline-block; border: 1px solid {INK}; border-radius: 3px; padding: 1px 8px; margin: 2px;
        font-family: 'Cinzel', serif; font-size: .76rem; background: rgba(255,250,235,.5); }}
.tag.new {{ background: #8e2a1f; color: #f7ecd0 !important; border-color: #8e2a1f; }}

/* the dice reveal, inked onto the page */
.roll {{ border: 2px solid {INK}; border-radius: 4px; padding: 10px 14px; margin: 4px 0 12px; display: flex; gap: 16px;
         align-items: center; background: rgba(255,250,235,.45); animation: pop .45s ease-out; }}
.roll .d20 {{ font-family: 'Cinzel', serif; font-size: 2.3rem; font-weight: 700; min-width: 60px; text-align: center;
              clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
              background: {INK}; color: #f7ecd0; padding: 10px 6px; }}
.roll .d20.crit {{ background: #b8893a; color: {INK}; }}
.roll .d20.fumble {{ background: #8e2a1f; }}
.roll .what {{ font-family: 'Cinzel', serif; font-size: .76rem; letter-spacing: .14em; color: #6b5640; }}
.roll .maths {{ font-family: 'IM Fell English', serif; font-size: 1.02rem; }}
.roll .verdict {{ font-family: 'Cinzel', serif; font-weight: 700; font-size: 1.08rem; letter-spacing: .08em; }}
.roll .ok {{ color: #3f6b2a; }} .roll .bad {{ color: #8e2a1f; }}
@keyframes pop {{ from {{ transform: scale(.94); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}

/* dialogue choices: a hand-written list, not boxes */
.st-key-choices button {{ background: transparent !important; border: none !important; border-radius: 0 !important;
  border-bottom: 1px dashed #8a6d45 !important; justify-content: flex-start !important; padding: 6px 2px !important; }}
.st-key-choices button > div, .st-key-choices button [data-testid="stMarkdownContainer"] {{
  justify-content: flex-start !important; width: 100%; text-align: left; }}
.st-key-choices button p {{ text-align: left; font-family: 'IM Fell English', serif; font-size: 1.08rem; }}
.st-key-choices button:hover {{ background: rgba(142,42,31,.08) !important; }}
.st-key-choices button:hover p {{ color: #8e2a1f; }}

/* buttons and inputs on the page */
.st-key-book button[kind="primary"] {{ background: {INK}; border-color: {INK}; font-family: 'Cinzel', serif; }}
.st-key-book button[kind="primary"] p {{ color: #f7ecd0; }}
.st-key-book button[kind="secondary"] {{ background: rgba(255,250,235,.4); border-color: {INK}; }}
.st-key-book input {{ background: rgba(255,250,235,.7) !important; font-family: 'IM Fell English', serif; }}
.st-key-book [data-testid="stExpander"] details {{ border-color: rgba(43,31,23,.4); background: rgba(255,250,235,.25); }}

/* table of contents */
.toc-title {{ font-family: 'Cinzel', serif; font-size: 1.15rem; display: flex; align-items: baseline; gap: 8px; }}
.toc-title .num {{ color: #8e2a1f; min-width: 2.2em; }}
.toc-title .dots {{ flex: 1; border-bottom: 2px dotted #8a6d45; transform: translateY(-4px); }}
.toc-sub {{ font-family: 'IM Fell English', serif; font-style: italic; color: #5a4532; margin-left: 2.9em; }}
</style>
""", unsafe_allow_html=True)

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def page(text, label=""):
    """Show the narrator's text as printed storybook text, with a drop capital."""
    if label:
        st.markdown(f"<span class='label'>{html.escape(label)}</span>", unsafe_allow_html=True)
    body = html.escape(text.strip()).replace("\n", "<br>")
    st.markdown(f"<div class='page'>{body}</div>", unsafe_allow_html=True)


def folio(text):
    st.markdown(f"<div class='folio'>~ {html.escape(str(text))} ~</div>", unsafe_allow_html=True)


def stat_grid(stats):
    cells = "".join(f"<div>{s}<b>{stats[s]:+d}</b></div>" for s in rules.STAT_NAMES)
    return f"<div class='stats'>{cells}</div>"


def hearts(hero):
    return "♥" * hero["health"] + "♡" * (hero["max_health"] - hero["health"])


def caption(story, words=14):
    """The first sentence of a page, short enough to sit under a panel."""
    first = re.split(r"(?<=[.!?])\s", story.strip(), maxsplit=1)[0]
    cut = first.split()
    return " ".join(cut[:words]) + ("…" if len(cut) > words else "")


def thumb(p):
    """A small copy of a page's picture for the panel strip, made once and kept with the page."""
    if "thumb" not in p:
        img = to_image(p["picture"])
        img.thumbnail((420, 280))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        p["thumb"] = base64.b64encode(buf.getvalue()).decode()
    return p["thumb"]


def strip(pages, first_number=1, columns=None):
    """A row of inked panels with captions, like a picture book."""
    if not pages:
        return
    figures = "".join(
        f"<figure class='panel'><img src='data:image/jpeg;base64,{thumb(p)}'>"
        f"<figcaption><b>{n}</b>{html.escape(caption(p['story']))}</figcaption></figure>"
        for n, p in enumerate(pages, start=first_number))
    st.markdown(f"<div class='strip' style='--n:{columns or len(pages)}'>{figures}</div>",
                unsafe_allow_html=True)


VERDICTS = {"TRIUMPH": "CRITICAL SUCCESS", "CLEAN SUCCESS": "SUCCESS", "SUCCESS WITH A COST": "SUCCESS, AT A COST",
            "PARTLY WORKS, THINGS GET WORSE": "FAILURE", "FAILS, NEW PROBLEM": "FAILURE",
            "COMIC DISASTER": "CRITICAL FAILURE"}


def roll_panel(o):
    """A Baldur's Gate-style dice reveal for the last roll."""
    if o["power"]:
        dice, maths = "★", "Class power: automatic success"
    else:
        dice = str(o["roll"])
        parts = [f"{o['stat_bonus']:+d} {o['stat']}"]
        if o.get("training"):
            label = "trained" if o["training"] == rules.PROFICIENCY_BONUS else (
                "expertise" if o["training"] > rules.PROFICIENCY_BONUS else "Jack of All Trades")
            parts.append(f"+{o['training']} {label}")
        extra = []
        if len(o["rolls"]) > 1:
            extra.append(f"advantage: rolled {o['rolls'][0]} and {o['rolls'][1]}")
        if o.get("rerolled"):
            extra.append(f"Inspiration re-rolled a {o['rerolled']}")
        maths = f"d20 {o['roll']} {' '.join(parts)} = <b>{o['total']}</b> vs DC {o['difficulty']}"
        if extra:
            maths += f"<br><i>{html.escape('; '.join(extra))}</i>"
    face = "crit" if o["tier"] == "TRIUMPH" else "fumble" if o["tier"] == "COMIC DISASTER" else ""
    tone = "ok" if o["success"] else "bad"
    notes = []
    if o["damage"]:
        notes.append(f"−{o['damage']} health")
    if o["item"]:
        notes.append(f"found {html.escape(o['item'])}")
    if o.get("inspired"):
        notes.append("+1 Inspiration")
    if o.get("clear_drawing"):
        notes.append("clear drawing: advantage")
    return f"""<div class='roll'><div class='d20 {face}'>{dice}</div><div>
        <div class='what'>{html.escape(rules.check_name(o).upper())} CHECK</div>
        <div class='maths'>{maths}</div>
        <div class='verdict {tone}'>{VERDICTS[o['tier']]}</div>
        <div class='maths'>{' · '.join(notes)}</div></div></div>"""


def to_image(png_bytes):
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


@st.cache_data
def load_locations():
    files = sorted(Path("locations").glob("*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def race_label(race):
    r = rules.RACES[race]
    bonus = ", ".join(f"+{v} {k}" for k, v in r["bonus"].items())
    return f"{race} ({bonus}, {r['trait'][0]})"


def find_character(name, cast):
    """Match a name the DM gave to a known character, e.g. 'Brann' -> 'Brann the trapper'."""
    name = name.strip().lower()
    if not name:
        return None
    for known in cast:
        if known.lower() == name or known.lower() in name or name in known.lower():
            return known
    return None


def party_update(extra):
    """After each page: companions react to the hero, and maybe someone offers to join."""
    s = st.session_state
    hero = s.hero
    s.party_news = rules.react(hero, extra.get("reactions", {}))
    name = find_character(extra.get("recruit", ""), s.cast)
    in_party = [c["name"] for c in hero["party"]]
    s.recruit = name if name and name not in in_party and len(in_party) < rules.MAX_PARTY else None


def party_panel(hero):
    """Companions, their approval, help for the next roll, and anyone who wants to join."""
    s = st.session_state
    if not (hero["party"] or s.get("recruit") or s.get("party_news")):
        return
    st.markdown("<span class='label'>YOUR PARTY</span>", unsafe_allow_html=True)
    for c in hero["party"]:
        face, words = st.columns([1, 4])
        if c["name"] in s.get("portraits", {}):
            face.image(character_sprite(c["name"]), width="stretch")
        words.markdown(f"**{html.escape(rules.capital(c['name']))}** · {rules.approval_word(c['approval'])} "
                       f"<span style='color:#7a6145'>({c['approval']}/100)</span>", unsafe_allow_html=True)
        if c["helped"]:
            st.caption("Has helped already in this part of the quest.")
        else:
            st.checkbox(f"Ask {c['name']} to help with your next move (advantage)",
                        key=f"help-{c['name']}-{s.turn_no}")
    for line in s.get("party_news") or []:
        st.caption(line)
    if s.get("recruit") and st.button(f"Invite {s.recruit} to join your party", width="stretch"):
        rules.recruit(hero, s.recruit, s.cast[s.recruit])
        s.party_news = [f"{rules.capital(s.recruit)} joins your party!"]
        s.recruit = None
        st.rerun()


TIMINGS = Path("results/timings.csv")  # how long each step of the game takes, for the report


@contextmanager
def timed(step):
    """Time a slow step (a picture, the Dungeon Master, ...) and add it to results/timings.csv."""
    start = time.time()
    yield
    record_time(step, time.time() - start)


def record_time(step, seconds):
    TIMINGS.parent.mkdir(exist_ok=True)
    new = not TIMINGS.exists()
    with open(TIMINGS, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "step", "seconds"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), step, f"{seconds:.1f}"])


def save_hero_portrait(hero, png):
    """Keep a numbered copy of every hero's portrait in media/portraits/."""
    art.PORTRAITS.mkdir(parents=True, exist_ok=True)
    name = art.safe_name(f"hero-{hero['race']}-{hero['class']}")
    (art.PORTRAITS / f"{name}-{len(list(art.PORTRAITS.glob('hero-*.png'))) + 1}.png").write_bytes(png)


def character_sprite(name):
    """A character cut out of their portrait, ready to stand in a scene. "hero" is the player's hero.
    Other characters are painted the first time they appear, then loaded from media/portraits/ (also in later
    tales, and after prepare_art.py has painted them in advance)."""
    s = st.session_state
    sprites, portraits = s.setdefault("sprites", {}), s.setdefault("portraits", {})
    if name not in sprites:
        if name == "hero":
            portraits["hero"] = s.portrait
        elif name not in portraits:
            look = s.cast[name]
            with st.spinner(f"Drawing {name}..."), timed("character portrait"):
                portraits[name] = art.saved_portrait(name, look, rules.is_creature(look),
                                                     **rules.portrait_settings(None))
        sprites[name] = art.cut_out(portraits[name])
    return sprites[name]


def quotes(story):
    """Every line of speech in the story text, in "double", “curly” or 'single' quote marks."""
    found = re.findall(r'["“]([^"”]{3,}?)["”]', story)
    found += re.findall(r"(?:^|(?<=[\s(]))['‘](.{3,}?)['’](?=[\s.,!?;:)]|$)", story)
    return [q.strip().rstrip(",;") for q in found]


def spoken_line(story, line):
    """The speaker's line as it really appears in the story: the quote most like the line the DM gave.
    Returns "" if the story has no quote, so a line the DM made up is never shown."""
    candidates = quotes(story)
    if not candidates:
        return ""
    return max(candidates, key=lambda q: difflib.SequenceMatcher(None, q.lower(), line.lower()).ratio())


def paint_page(plan, loc, drawing=None, extra=None, story=""):
    """The picture for a new page.
    "layered" (art.PICTURE_MODE): the background is reused while the hero stays in the same place, the
    player's drawing is inked into it, and the saved characters stand on top.
    "full": the whole scene is painted from one description (the first version of the game).
    When a character speaks (extra["speaker"]), the page shows a dialogue close-up of them instead."""
    s = st.session_state
    s.speech = None
    if art.PICTURE_MODE == "full":
        with st.spinner("Painting the page..."):
            return art.paint(plan["prompt"], loc["style"])
    if s.get("background") is None or plan["new_place"]:
        with st.spinner("Painting the place..."), timed("background"):
            s.background = art.background(plan["background"], loc["style"])
    if drawing is not None:
        s.background = art.stamp(s.background, drawing)
    speaker = find_character((extra or {}).get("speaker", ""), s.cast)
    line = spoken_line(story, (extra or {}).get("line", ""))
    if speaker and line and not rules.is_scenery(speaker, s.cast[speaker]):
        s.speech = {"name": rules.capital(speaker), "line": line}
        return art.compose_dialogue(s.background, character_sprite(speaker), character_sprite("hero"),
                                    rules.is_creature(s.cast[speaker]))
    names = [c["name"] for c in s.hero["party"]]
    for name in plan["characters"]:
        known = find_character(name, s.cast)
        if known and known not in names and not rules.is_scenery(known, s.cast[known]):
            names.append(known)
    sprites = [character_sprite("hero")] + [character_sprite(n) for n in names[:3]]
    return art.compose(s.background, sprites)


def new_game():
    st.session_state.clear()
    st.session_state.stage = "hero"
    st.session_state.hero_tries = 0


def spread():
    """The open book: returns the left and right page."""
    book = st.container(key="book")
    return book.columns(2, gap="large")


if "stage" not in st.session_state:
    new_game()


# ---------------------------------------------------------------------------
# Screen 1: draw your hero
# ---------------------------------------------------------------------------
def screen_hero():
    left, right = spread()
    with left:
        st.markdown("<span class='kicker'>PROLOGUE · WHO ARE YOU?</span>", unsafe_allow_html=True)
        st.title("Draw Your Hero")
        st.write("What do they wear? What do they carry? The Oracle will decide who they are.")
        with st.container(key="hero-canvas"):
            canvas = st_canvas(stroke_width=4, stroke_color=INK, background_color="#f8efd9",
                               height=H, width=W, key=f"hero-{st.session_state.hero_tries}",
                               return_image_data=True)
        folio("i")
    with right:
        st.markdown("<span class='label'>WHAT THE ORACLE LOOKS FOR</span>", unsafe_allow_html=True)
        names = list(rules.CLASSES)
        col_a, col_b = st.columns(2)
        for i, name in enumerate(names):
            (col_a if i < len(names) / 2 else col_b).markdown(
                f"**{name}**  \n*{', '.join(rules.CLASSES[name]['items'])}*")
        st.markdown("<div class='ornament'>❦ ❦ ❦</div>", unsafe_allow_html=True)
        people, hair_col = st.columns([3, 2])
        race = people.selectbox("Your people", list(rules.RACES), key="race", format_func=race_label)
        hair = hair_col.selectbox("Hair", rules.HAIR, key="hair")
        st.caption(f"{rules.RACES[race]['trait'][0]}: {rules.RACES[race]['trait'][1]}")
        st.caption("How you draw matters too: bold lines, light lines, tiny details and big "
                   "confident shapes all change your stats.")
        if st.button("Reveal my class", type="primary", width="stretch"):
            if canvas.image_data is None:
                st.warning("Draw your hero first!")
                return
            image = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
            gray = np.array(image.convert("L"))
            if gray.min() >= 128:
                st.warning("Draw your hero first!")
                return
            with st.spinner("The Oracle studies your hero..."):
                cls, pct, second, second_pct, seen = sketch.classify_hero(image, rules.CLASSES)
                bonuses, reasons = rules.drawing_style(gray)
                st.session_state.style_bonuses = (bonuses, reasons, seen)
                hero = rules.make_hero(cls, bonuses, reasons, seen, race, hair)
                with timed("hero portrait"):
                    portrait = art.portrait(hero["look"], **rules.portrait_settings(hero["class"]))
            save_hero_portrait(hero, portrait)
            st.session_state.update(hero=hero, hero_sketch=image, portrait=portrait, sprites={},
                                    oracle=(pct, second, second_pct), stage="class")
            st.rerun()
        folio("ii")


# ---------------------------------------------------------------------------
# Screen 2: class and stats reveal
# ---------------------------------------------------------------------------
def screen_class():
    hero = st.session_state.hero
    c = rules.CLASSES[hero["class"]]
    pct, second, second_pct = st.session_state.oracle
    left, right = spread()
    with left:
        st.markdown("<span class='kicker'>PROLOGUE · THE ORACLE HAS SPOKEN</span>", unsafe_allow_html=True)
        with st.container(key="portrait"):
            st.image(st.session_state.portrait, width="stretch")
        a, b = st.columns([1, 2])
        a.image(st.session_state.hero_sketch, width="stretch")
        b.markdown(f"*Your drawing.* The Oracle is {pct}% sure. It also saw a little {second} "
                   f"({second_pct}%).")
        folio("iii")
    with right:
        st.title(f"You are {rules.with_article(hero['race'] + ' ' + hero['class'])}")
        st.markdown(f"*{c['tagline']}*")
        st.markdown(stat_grid(hero["stats"]), unsafe_allow_html=True)
        st.markdown("**Why the Oracle chose this**")
        lines = [f"Spotted in your drawing: {', '.join(hero['items_seen'])}"] + hero["reasons"]
        st.markdown("\n".join(f"- {line}" for line in lines))
        st.markdown(f"**Hit die:** d{c['hit_die']} &nbsp; **Trained skills (+{rules.PROFICIENCY_BONUS}):** "
                    f"{', '.join(hero['skills'])}", unsafe_allow_html=True)
        trait = rules.RACES[hero["race"]]["trait"]
        st.markdown(f"**{hero['race']} trait · {trait[0]}:** {trait[1]}")
        if c["passive"]:
            st.markdown(f"**Class trait · {c['passive'][0]}:** {c['passive'][1]}")
        st.markdown(f"**Class power · {c['power'][0]}:** {c['power'][1]}")
        st.markdown(f"**Weakness · {c['weakness'][0]}:** {c['weakness'][1]}")
        st.markdown(f"**Health:** <span class='hearts'>{hearts(hero)}</span> &nbsp; "
                    f"**Pack:** {', '.join(hero['pack'])}", unsafe_allow_html=True)
        with st.expander("Not right? Change your class or people"):
            names = list(rules.CLASSES)
            pick = st.selectbox("Class", names, index=names.index(hero["class"]))
            race = st.selectbox("People", list(rules.RACES), index=list(rules.RACES).index(hero["race"]),
                                format_func=race_label)
            hair = st.selectbox("Hair", rules.HAIR, index=rules.HAIR.index(hero["hair"]))
            if st.button(f"Become {rules.with_article(race + ' ' + pick)}"):
                bonuses, reasons, seen = st.session_state.style_bonuses
                new_hero = rules.make_hero(pick, bonuses, reasons, seen, race, hair)
                with st.spinner("Painting your new portrait..."):
                    with timed("hero portrait"):
                        portrait = art.portrait(new_hero["look"], **rules.portrait_settings(new_hero["class"]))
                save_hero_portrait(new_hero, portrait)
                st.session_state.update(hero=new_hero, portrait=portrait, sprites={})
                st.rerun()
        b1, b2 = st.columns(2)
        if b1.button("Redraw my hero", width="stretch"):
            st.session_state.hero_tries += 1
            st.session_state.stage = "hero"
            st.rerun()
        if b2.button("Choose where to begin", type="primary", width="stretch"):
            st.session_state.stage = "start"
            st.rerun()
        folio("iv")


# ---------------------------------------------------------------------------
# Screen 3: choose where to begin (the book's table of contents)
# ---------------------------------------------------------------------------
def start_location(loc):
    hero = st.session_state.hero
    with st.spinner(f"The Dungeon Master sets the scene at {loc['name']}..."):
        cast = dict(loc["cast"])  # characters and how they look, kept for the whole tale
        with timed("dungeon master"):
            story, plan, choices, props, extra = dm.open_scene(hero, loc, cast)
    with st.spinner("Opening the book..."), timed("location background"):
        background = art.location_background(loc)  # painted in advance by prepare_art.py, or now
    st.session_state.update(cast=cast, background=background)
    plan["new_place"] = False  # the tale starts in the location's own background
    picture = paint_page(plan, loc, extra=extra, story=story)
    party_update(extra)
    set_challenge(extra)
    st.session_state.update(
        location=loc, goal_index=0, goal_wins=0, turn_no=0, guesses=None, last=None, goal_pages=0,
        since_challenge=0 if st.session_state.challenge else 1,
        pages=[{"picture": picture, "story": story, "result": "The tale begins."}],
        history=[story], scene=story, picture=picture, choices=choices, props=props, cast=cast,
        home_advantage=hero["class"] in loc["suits"], stage="play",
    )


def chapter_entry(number, loc, hero):
    star = " ★" if hero["class"] in loc["suits"] else ""
    st.markdown(f"<div class='toc-title'><span class='num'>{ROMAN[number]}.</span>"
                f"<span>{html.escape(loc['name'])}{star}</span><span class='dots'></span>"
                f"<span class='kicker'>{html.escape(loc['type'].upper())}</span></div>"
                f"<div class='toc-sub'>{html.escape(loc['opening'])}</div>", unsafe_allow_html=True)
    with st.expander("Quest and who it suits"):
        st.markdown("**Quest:** " + " → ".join(loc["beats"]))
        st.markdown("**Suits:** " + ", ".join(loc["suits"]))
    if st.button(f"Begin at {loc['name']}", key=f"start-{loc['id']}", width="stretch"):
        start_location(loc)
        st.rerun()


def screen_start():
    hero = st.session_state.hero
    locations = load_locations()
    half = (len(locations) + 1) // 2
    left, right = spread()
    with left:
        st.markdown("<span class='kicker'>PROLOGUE · YOUR STORY BEGINS...</span>", unsafe_allow_html=True)
        st.title("Contents")
        st.markdown(f"*Choose the chapter where your tale begins. Chapters marked ★ suit a {hero['class']}: "
                    f"you get advantage on your first roll there.*")
        for i, loc in enumerate(locations[:half]):
            chapter_entry(i, loc, hero)
        folio("v")
    with right:
        for i, loc in enumerate(locations[half:], start=half):
            chapter_entry(i, loc, hero)
        folio("vi")


# ---------------------------------------------------------------------------
# Screen 4: play
# ---------------------------------------------------------------------------
def strokes_only(image_data):
    """Keep just the player's lines, as black on white, for the Oracle."""
    alpha = Image.fromarray(image_data.astype("uint8"), "RGBA").getchannel("A")
    return ImageOps.invert(alpha).convert("RGB")


def play_turn(label=None, typed="", choice=None):
    """One turn. Either a dialogue choice the DM offered, or the player's own move:
    label = what the Oracle saw in the drawing, typed = what the player wrote."""
    s = st.session_state
    turn_start = time.time()
    hero, loc = s.hero, s.location
    goal = loc["beats"][s.goal_index]
    skill = stat = on_class = None
    difficulty = rules.pick_difficulty(s.goal_index)
    if choice:
        action = f'chose to: "{choice["text"]}"'
        skill, difficulty, on_class = choice["skill"], choice["dc"], choice["class_option"]
    elif typed and label:
        action = f'drew a {label} and said: "{typed}"'
    elif typed:
        action = f'said: "{typed}"'
    else:
        action = f"drew a {label}"
    helper = next((c for c in hero["party"] if not c["helped"] and s.get(f"help-{c['name']}-{s.turn_no}")), None)
    if s.get("challenge") and label:
        action += f' (the drawing challenge was: "{s.challenge}")'
    clear_drawing = s.pop("clear_drawing", False)  # a clearly recognised drawing in a challenge: advantage
    if helper:  # a companion helps: advantage on this roll
        helper["helped"] = True
        action += f", with {helper['name']}'s help"
    if not choice and (typed or label not in rules.CHECKS):  # free text: the DM picks the skill
        with st.spinner("The Dungeon Master considers your move..."):
            skill, on_class = dm.choose_skill(action, hero)
    outcome = rules.resolve(hero, label or typed or action, difficulty,
                            advantage=s.home_advantage or helper is not None or clear_drawing, use_power=s.get("use_power", False),
                            loot=loc["loot"], stat=stat, on_class=on_class, skill=skill,
                            use_inspiration=s.get("use_inspiration", False))
    outcome["action"] = action
    outcome["clear_drawing"] = clear_drawing
    s.home_advantage = False  # the home advantage is only for the first roll
    page = "draw" if s.get("since_challenge", 0) >= DRAW_EVERY - 1 else "talk"
    stuck = s.get("goal_pages", 0) >= STUCK_AFTER
    with st.spinner("The page is turning..."):
        with timed("dungeon master"):
            story, plan, choices, props, extra = dm.narrate(hero, loc, goal, s.history, outcome, s.cast,
                                                            page=page, stuck=stuck)
    picture = paint_page(plan, loc, s.pop("pending_strokes", None), extra, story)
    party_update(extra)
    set_challenge(extra)
    record_time("whole turn (after the move)", time.time() - turn_start)
    s.pages.append({"picture": picture, "story": story, "result": rules.describe(outcome)})
    s.history.append(story)
    s.update(scene=story, picture=picture, last=outcome, guesses=None, choices=choices, props=props,
             turn_no=s.turn_no + 1, since_challenge=0 if s.challenge else s.get("since_challenge", 0) + 1,
             goal_pages=s.get("goal_pages", 0) + 1)
    if outcome["success"]:
        s.goal_wins += 1
        if s.goal_wins >= WINS_PER_GOAL:
            s.goal_index += 1
            s.goal_wins = 0
            s.goal_pages = 0
            rules.new_quest_step(hero)  # companions can help again
    if hero["health"] <= 0:
        s.update(stage="end", ending="defeat")
    elif s.goal_index >= len(loc["beats"]):
        s.update(stage="end", ending="victory")


def set_challenge(extra):
    """Show the Dungeon Master's drawing challenge, if there is one, on pages where nobody is talking."""
    challenge = (extra or {}).get("draw_challenge", "").strip()
    st.session_state.challenge = challenge if challenge and not st.session_state.get("speech") else ""


def voice_hint():
    """Names and game words Whisper should expect right now: the place, the characters met, the hero."""
    s = st.session_state
    words = [s.location["name"], *s.cast, s.hero["race"], s.hero["class"], *s.hero["pack"], *s.hero["skills"]]
    return voice.hint_text(words)


def remember_voice(typed):
    """Save the spoken move with what Whisper heard and what the player finally meant, for evaluate_voice.py."""
    clip = st.session_state.pop("voice_clip", None)
    if clip and clip[1] and typed:
        voice.save_clip(clip[0], clip[1], typed, source="play")


def your_move(hero, c):
    """The drawing canvas over the current scene, plus the typed move."""
    s = st.session_state
    canvas = None
    if s.get("speech"):  # a dialogue close-up: answer by voice, typing or a choice (nothing to draw here)
        with st.container(key="scene-picture"):
            st.image(s.picture, width="stretch")
        st.markdown(f"<div class='speech'><span class='who'>{html.escape(s.speech['name'])}</span>"
                    f"{html.escape(s.speech['line'])}</div>", unsafe_allow_html=True)
    else:  # a scene: draw something to use, and say or type what you do with it
        if s.get("challenge"):
            st.markdown(f"<div class='challenge'><b>DRAWING CHALLENGE</b> · {html.escape(s.challenge)}<br>"
                        "<small>A drawing the Oracle recognises clearly gives you advantage.</small></div>",
                        unsafe_allow_html=True)
        scene = to_image(s.picture).resize((W, H))
        mode = st.radio("Draw", ["On the scene", "On a blank page"], horizontal=True, key="draw_mode",
                        label_visibility="collapsed")
        if mode == "On the scene":  # the picture is faded, so dark ink lines stand out
            background = Image.blend(scene, Image.new("RGB", (W, H), PAPER), FADE)
        else:
            with st.container(key="scene-picture"):
                st.image(scene, width="stretch")
            background = Image.new("RGB", (W, H), PAPER)
        with st.container(key="scene"):
            canvas = st_canvas(stroke_width=5, stroke_color=PEN, background_color="rgba(0,0,0,0)",
                               background_image=background, height=H, width=W,
                               key=f"draw-{s.turn_no}-{mode}", return_image_data=True)
    talking = bool(s.get("speech"))
    spoken = st.audio_input(f"Answer {s.speech['name']} out loud" if talking else "Say your move out loud",
                            key=f"voice-{s.turn_no}")
    if spoken is not None and s.get("voice_clip", (None,))[0] != spoken.getvalue():
        data = spoken.getvalue()
        with st.spinner("Listening... (the very first time, Whisper downloads itself)"):
            with timed("speech to text"):
                heard = voice.transcribe(data, hint=voice_hint() if voice.USE_HINT else None)
        s.voice_clip = (data, heard)
        if heard:
            s[f"typed-{s.turn_no}"] = heard  # shown in the box below, so it can be corrected
        else:
            st.warning("No words heard. Try again a little closer to the microphone.")
    typed = st.text_input("Your answer: say it, type it, or pick a choice on the right" if talking else
                          "Your own move: draw on the scene, say it, type it, or a mix",
                          placeholder="e.g. I promise to find your papa" if talking else
                          "e.g. I swing across on the rope and grab the map",
                          key=f"typed-{s.turn_no}").strip()
    if not hero["power_used"]:
        st.checkbox(f"Use {c['power'][0]} (once per tale: this move succeeds automatically)", key="use_power")
    if hero["inspiration"]:
        st.checkbox(f"Spend Inspiration if the roll fails: re-roll once ({hero['inspiration']} left)",
                    key="use_inspiration")
    if st.button("Make your move", type="primary"):
        drew = canvas is not None and canvas.image_data is not None and canvas.image_data[:, :, 3].max() > 0
        remember_voice(typed)
        if not drew and not typed:
            st.warning("Draw on the scene or describe what you do first!")
        elif not drew:
            play_turn(typed=typed)
            st.rerun()
        else:
            s.last_sketch = strokes_only(canvas.image_data)
            s.pending_strokes = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")  # inked into the scene
            with st.spinner("The Oracle reads your drawing..."):
                with timed("oracle reads drawing"):
                    guesses = sketch.recognise(s.last_sketch, extra=s.get("props", []))
            s.clear_drawing = bool(s.get("challenge")) and guesses[0][1] >= CONFIDENT
            if typed:  # the words explain the drawing, so no need to ask
                play_turn(guesses[0][0] if guesses[0][1] >= CONFIDENT else None, typed)
                st.rerun()
            elif guesses[0][1] >= CONFIDENT:
                play_turn(guesses[0][0])
                st.rerun()
            else:
                s.guesses = guesses

    if s.guesses:
        other = "Something else..."
        options = [f"{label} ({pct}%)" for label, pct in s.guesses] + [other]
        st.write("The Oracle isn't sure. What did you draw?")
        choice = st.radio("Pick one", options, label_visibility="collapsed")
        typed = st.text_input("Type what you drew", placeholder="e.g. a grappling hook",
                              key=f"other-{s.turn_no}") if choice == other else ""
        if st.button("That's it"):
            drawing = typed.strip().lower() if choice == other else s.guesses[options.index(choice)][0]
            if drawing:
                sketch.save_sketch(s.last_sketch, drawing)  # a labelled example for evaluate.py
                play_turn(label=drawing)
                st.rerun()

    if "last_sketch" in s:
        with st.expander("What the Oracle saw"):
            st.image(s.last_sketch, width=240)


def character_ribbon(hero, c, loc):
    s = st.session_state
    goal = loc["beats"][min(s.goal_index, len(loc["beats"]) - 1)]
    st.markdown(f"""<div class='ribbon'><span class='name'>{hero['race']} {hero['class']}</span>
        <span class='hearts'>{hearts(hero)}</span><span class='insp'>{'✦' * hero['inspiration']}</span>
        <span>Quest {s.goal_index + 1}/{len(loc['beats'])}: <i>{html.escape(goal)}</i>
        ({s.goal_wins}/{WINS_PER_GOAL})</span></div>""", unsafe_allow_html=True)
    with st.expander("Character sheet"):
        pack = "".join(f"<span class='tag{' new' if s.last and item == s.last['item'] else ''}'>"
                       f"{html.escape(item)}</span>" for item in hero["pack"])
        trait = rules.RACES[hero["race"]]["trait"]
        st.markdown(f"*{html.escape(c['tagline'])}*{stat_grid(hero['stats'])}"
                    f"Trained: {', '.join(hero['skills'])} (+{rules.PROFICIENCY_BONUS})<br>"
                    f"<b>{trait[0]}:</b> {trait[1]}<br>{pack}", unsafe_allow_html=True)
    party_panel(hero)


def screen_play():
    s = st.session_state
    hero, loc = s.hero, s.location
    c = rules.CLASSES[hero["class"]]
    number = len(s.pages)
    chapter = next((i for i, l in enumerate(load_locations()) if l["id"] == loc["id"]), 0)
    left, right = spread()

    with left:
        st.markdown(f"<span class='kicker'>CHAPTER {ROMAN[chapter]} · {html.escape(loc['type'].upper())}</span>",
                    unsafe_allow_html=True)
        st.title(loc["name"])
        earlier = s.pages[:-1][-STRIP:]
        strip(earlier, first_number=number - len(earlier), columns=STRIP)
        your_move(hero, c)
        folio(number * 2 - 1)

    with right:
        character_ribbon(hero, c, loc)
        if s.last:
            st.markdown(f"<span class='label'>YOU {html.escape(s.last['action'].upper())}</span>",
                        unsafe_allow_html=True)
            st.markdown(roll_panel(s.last), unsafe_allow_html=True)
        page(s.scene, f"PAGE {number}")
        if s.get("choices"):
            st.markdown("<span class='label'>WHAT DO YOU DO?</span>", unsafe_allow_html=True)
            with st.container(key="choices"):
                for i, ch in enumerate(s.choices, start=1):
                    tag = f":red[{hero['class'].upper()}] " if ch["class_option"] else ""
                    trained = " ✓" if ch["skill"] in hero["skills"] else ""
                    label = (f"**{i}.** {tag}{ch['text']} :gray[· {ch['skill']}{trained}, DC {ch['dc']}]")
                    if st.button(label, key=f"choice-{s.turn_no}-{i}", width="stretch"):
                        play_turn(choice=ch)
                        st.rerun()
            st.caption(f"Or make your own move on the scene. ✓ = trained skill. "
                       f"{hero['class']} options get advantage.")
        with st.expander(f"Read the tale so far ({number} pages)"):
            for n, p in enumerate(s.pages, start=1):
                st.markdown(f"**Page {n}** · *{p['result']}*")
                st.write(p["story"])
        folio(number * 2)


# ---------------------------------------------------------------------------
# Screen 5: the end
# ---------------------------------------------------------------------------
def screen_end():
    s = st.session_state
    won = s.ending == "victory"
    left, right = spread()
    with left:
        st.markdown(f"<span class='kicker'>{html.escape(s.location['name'].upper())} · THE LAST PAGE</span>",
                    unsafe_allow_html=True)
        st.title("The Tale Is Told" if won else "Your Tale Ends Here")
        with st.container(key="last-picture"):
            st.image(s.picture, width="stretch")
        page(s.scene)
        st.markdown(f"*{s.location['name']} will remember you.*" if won else
                    "*The ink runs dry... but every hero gets another page.*")
        if s.hero["party"]:
            st.markdown("*Travelling with you at the end: " + ", ".join(
                f"{c['name']} ({rules.approval_word(c['approval']).lower()})" for c in s.hero["party"]) + ".*")
        st.markdown("<div class='ornament'>❦ THE END ❦</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<span class='label'>YOUR WHOLE TALE</span>", unsafe_allow_html=True)
        for start in range(0, len(s.pages), 3):
            strip(s.pages[start:start + 3], first_number=start + 1, columns=3)
        if st.button("Start a new tale", type="primary", width="stretch"):
            new_game()
            st.rerun()


SCREENS = {"hero": screen_hero, "class": screen_class, "start": screen_start,
           "play": screen_play, "end": screen_end}
SCREENS[st.session_state.stage]()
