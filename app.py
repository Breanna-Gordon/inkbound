"""Inkbound: a drawn fairy-tale adventure.

Screens, in order: draw your hero -> class reveal -> choose where to begin ->
play (draw on the scene each turn) -> the end.
Run with:  streamlit run app.py   (Ollama and ComfyUI must be running)
"""
import html
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import art
import dm
import rules
import sketch

CONFIDENT = 50   # below this %, the Oracle asks the player what they drew
W, H = 690, 460  # canvas size: same 3:2 shape as the pictures (768 x 512)
WINS_PER_GOAL = 2  # successes needed to move on to the next part of the quest

st.set_page_config(page_title="Inkbound", layout="wide")

# ---------------------------------------------------------------------------
# Storybook look: fonts, parchment pages, character sheet
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cinzel+Decorative:wght@700&family=IM+Fell+English:ital@0;1&display=swap');
h1, h2, h3 { font-family: 'Cinzel', serif !important; color: #e6c27a !important; }
h1 { font-family: 'Cinzel Decorative', serif !important; }
.kicker { font-family: 'Cinzel', serif; letter-spacing: .25em; font-size: .8rem; color: #e6c27a; }
.page { background: #f2e5c4; color: #2b1f17; border-radius: 6px; padding: 18px 22px; margin: 6px 0 12px;
        font-family: 'IM Fell English', serif; font-size: 1.15rem; line-height: 1.55;
        box-shadow: inset 0 0 50px rgba(130,95,50,.3); }
.page .label { display: block; font-family: 'Cinzel', serif; font-size: .75rem; letter-spacing: .2em; color: #8e2a1f; margin-bottom: 4px; }
.sheet { background: #2a1f14; border: 2px solid #b8893a; border-radius: 8px; padding: 14px 16px; color: #f2e5c4; margin-bottom: 12px; }
.sheet .name { font-family: 'Cinzel Decorative', serif; font-size: 1.4rem; color: #f0c674; }
.sheet .small { font-family: 'IM Fell English', serif; font-style: italic; color: #d6c8a8; }
.stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; text-align: center; margin: 10px 0; }
.stats div { border: 1px solid #b8893a; border-radius: 6px; padding: 4px 0; font-family: 'Cinzel', serif; font-size: .75rem; }
.stats b { display: block; font-size: 1.15rem; }
.hearts { color: #e8735a; font-size: 1.3rem; letter-spacing: 2px; }
.tag { display: inline-block; background: #e9d8b0; color: #2b1f17; border-radius: 3px; padding: 2px 8px; margin: 2px;
       font-family: 'Cinzel', serif; font-size: .78rem; }
.tag.new { background: #8e2a1f; color: #f2e5c4; }
.goal { font-family: 'IM Fell English', serif; color: #f2e5c4; }
</style>
""", unsafe_allow_html=True)


def page(text, label=""):
    """Show text on a parchment storybook page."""
    lab = f"<span class='label'>{html.escape(label)}</span>" if label else ""
    st.markdown(f"<div class='page'>{lab}{html.escape(text)}</div>", unsafe_allow_html=True)


def stat_grid(stats):
    cells = "".join(f"<div>{s}<b>{stats[s]:+d}</b></div>" for s in rules.STAT_NAMES)
    return f"<div class='stats'>{cells}</div>"


def hearts(hero):
    return "♥" * hero["health"] + "♡" * (hero["max_health"] - hero["health"])


def to_image(png_bytes):
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


@st.cache_data
def load_locations():
    files = sorted(Path("locations").glob("*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def new_game():
    st.session_state.clear()
    st.session_state.stage = "hero"
    st.session_state.hero_tries = 0


if "stage" not in st.session_state:
    new_game()


# ---------------------------------------------------------------------------
# Screen 1: draw your hero
# ---------------------------------------------------------------------------
def screen_hero():
    st.markdown("<span class='kicker'>PROLOGUE · WHO ARE YOU?</span>", unsafe_allow_html=True)
    st.title("Draw Your Hero")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.write("What do they wear? What do they carry? The Oracle will decide who they are.")
        canvas = st_canvas(stroke_width=4, stroke_color="#1d1611", background_color="#f8efd9",
                           height=H, width=W, key=f"hero-{st.session_state.hero_tries}",
                           return_image_data=True)
    with right:
        st.subheader("What the Oracle looks for")
        for name, c in rules.CLASSES.items():
            st.markdown(f"**{name}**: {', '.join(c['items'])}")
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
                hero = rules.make_hero(cls, bonuses, reasons, seen)
                portrait = art.paint(f"portrait of {hero['look']}, facing the viewer", hero["style"])
            st.session_state.update(hero=hero, hero_sketch=image, portrait=portrait,
                                    oracle=(pct, second, second_pct), stage="class")
            st.rerun()


# ---------------------------------------------------------------------------
# Screen 2: class and stats reveal
# ---------------------------------------------------------------------------
def screen_class():
    hero = st.session_state.hero
    c = rules.CLASSES[hero["class"]]
    pct, second, second_pct = st.session_state.oracle
    st.markdown("<span class='kicker'>PROLOGUE · THE ORACLE HAS SPOKEN</span>", unsafe_allow_html=True)
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.image(st.session_state.portrait, width="stretch")
        st.image(st.session_state.hero_sketch, caption="Your drawing", width=200)
    with right:
        st.title(f"You are a {hero['class']}")
        st.markdown(f"*{c['tagline']}*")
        st.caption(f"The Oracle is {pct}% sure. It also saw a little {second} ({second_pct}%).")
        st.markdown(stat_grid(hero["stats"]), unsafe_allow_html=True)
        st.markdown("**Why the Oracle chose this**")
        lines = [f"Spotted in your drawing: {', '.join(hero['items_seen'])}"] + hero["reasons"]
        st.markdown("\n".join(f"- {line}" for line in lines))
        st.markdown(f"**Class power · {c['power'][0]}:** {c['power'][1]}")
        st.markdown(f"**Weakness · {c['weakness'][0]}:** {c['weakness'][1]}")
        st.markdown(f"**Health:** <span class='hearts'>{hearts(hero)}</span> &nbsp; "
                    f"**Pack:** {', '.join(hero['pack'])}", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Redraw my hero", width="stretch"):
            st.session_state.hero_tries += 1
            st.session_state.stage = "hero"
            st.rerun()
        if b2.button("Choose where to begin", type="primary", width="stretch"):
            st.session_state.stage = "start"
            st.rerun()


# ---------------------------------------------------------------------------
# Screen 3: choose where to begin
# ---------------------------------------------------------------------------
def start_location(loc):
    hero = st.session_state.hero
    with st.spinner(f"Painting {loc['name']}..."):
        picture = art.paint(loc["picture"], f"{hero['look']}, {loc['style']}")
    st.session_state.update(
        location=loc, goal_index=0, goal_wins=0, turn_no=0, guesses=None, last=None,
        pages=[{"picture": picture, "story": loc["opening"], "result": "The tale begins."}],
        history=[loc["opening"]], scene=loc["opening"], picture=picture,
        home_advantage=hero["class"] in loc["suits"], stage="play",
    )


def screen_start():
    hero = st.session_state.hero
    st.markdown("<span class='kicker'>PROLOGUE · YOUR STORY BEGINS...</span>", unsafe_allow_html=True)
    st.title("Choose Where You Begin")
    st.write(f"Places marked ★ suit a {hero['class']}: you get advantage on your first roll there.")
    locations = load_locations()
    cols = st.columns(2, gap="large")
    for i, loc in enumerate(locations):
        with cols[i % 2].container(border=True):
            star = " ★" if hero["class"] in loc["suits"] else ""
            st.subheader(loc["name"] + star)
            st.caption(loc["type"].upper())
            st.markdown(f"*{loc['opening']}*")
            st.markdown("**Quest:** " + " → ".join(loc["beats"]))
            st.markdown("**Suits:** " + ", ".join(loc["suits"]))
            if st.button(f"Begin at {loc['name']}", key=f"start-{loc['id']}", width="stretch"):
                start_location(loc)
                st.rerun()


# ---------------------------------------------------------------------------
# Screen 4: play
# ---------------------------------------------------------------------------
def strokes_only(image_data):
    """Keep just the player's lines, as black on white, for the Oracle."""
    alpha = Image.fromarray(image_data.astype("uint8"), "RGBA").getchannel("A")
    return ImageOps.invert(alpha).convert("RGB")


def play_turn(label=None, typed=""):
    """One turn. label = what the Oracle saw in the drawing (or None);
    typed = what the player wrote (or empty). Either or both can be given."""
    s = st.session_state
    hero, loc = s.hero, s.location
    goal = loc["beats"][s.goal_index]
    if typed and label:
        action = f'drew a {label} and said: "{typed}"'
    elif typed:
        action = f'said: "{typed}"'
    else:
        action = f"drew a {label}"
    stat = on_class = None
    if typed or label not in rules.CHECKS:  # free text: let the DM pick the stat
        with st.spinner("The Dungeon Master considers your move..."):
            stat, on_class = dm.choose_stat(action, hero)
    outcome = rules.resolve(hero, label or typed, rules.pick_difficulty(s.goal_index),
                            advantage=s.home_advantage, use_power=s.get("use_power", False),
                            loot=loc["loot"], stat=stat, on_class=on_class)
    outcome["action"] = action
    s.home_advantage = False  # the home advantage is only for the first roll
    with st.spinner("The page is turning..."):
        story, picture_text = dm.narrate(hero, loc, goal, s.history, outcome)
        picture = art.paint(picture_text, f"{hero['look']}, {loc['style']}")
    s.pages.append({"picture": picture, "story": story, "result": rules.describe(outcome)})
    s.history.append(story)
    s.update(scene=story, picture=picture, last=outcome, guesses=None, turn_no=s.turn_no + 1)
    if outcome["success"]:
        s.goal_wins += 1
        if s.goal_wins >= WINS_PER_GOAL:
            s.goal_index += 1
            s.goal_wins = 0
    if hero["health"] <= 0:
        s.update(stage="end", ending="defeat")
    elif s.goal_index >= len(loc["beats"]):
        s.update(stage="end", ending="victory")


def screen_play():
    s = st.session_state
    hero, loc = s.hero, s.location
    c = rules.CLASSES[hero["class"]]
    st.markdown(f"<span class='kicker'>{html.escape(loc['type'].upper())} · PAGE {len(s.pages)}</span>",
                unsafe_allow_html=True)
    st.title(loc["name"])
    left, right = st.columns([3, 2], gap="large")

    with left:
        background = to_image(s.picture).resize((W, H))
        canvas = st_canvas(stroke_width=5, stroke_color="#ffe9a8", background_color="rgba(0,0,0,0)",
                           background_image=background, height=H, width=W,
                           key=f"draw-{s.turn_no}", return_image_data=True)
        typed = st.text_input("Draw on the scene, describe your move, or both",
                              placeholder="e.g. I swing across on the rope and grab the map",
                              key=f"typed-{s.turn_no}").strip()
        if not hero["power_used"]:
            st.checkbox(f"Use {c['power'][0]} (once per tale: this move succeeds automatically)",
                        key="use_power")
        if st.button("Make your move", type="primary"):
            drew = canvas.image_data is not None and canvas.image_data[:, :, 3].max() > 0
            if not drew and not typed:
                st.warning("Draw on the scene or describe what you do first!")
            elif not drew:
                play_turn(typed=typed)
                st.rerun()
            else:
                s.last_sketch = strokes_only(canvas.image_data)
                with st.spinner("The Oracle reads your drawing..."):
                    guesses = sketch.recognise(s.last_sketch, rules.ACTION_ITEMS)
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
                    play_turn(label=drawing)
                    st.rerun()

        if "last_sketch" in s:
            with st.expander("What the Oracle saw"):
                st.image(s.last_sketch, width=240)

    with right:
        goal = loc["beats"][min(s.goal_index, len(loc["beats"]) - 1)]
        pack = "".join(f"<span class='tag{' new' if s.last and item == s.last['item'] else ''}'>"
                       f"{html.escape(item)}</span>" for item in hero["pack"])
        st.markdown(f"""<div class='sheet'>
            <div class='name'>{hero['class']}</div>
            <div class='small'>{html.escape(c['tagline'])}</div>
            <div class='hearts'>{hearts(hero)}</div>
            {stat_grid(hero['stats'])}
            <div>{pack}</div>
            <p class='goal'><b>Quest {s.goal_index + 1} of {len(loc['beats'])}:</b> {html.escape(goal)}
            ({s.goal_wins}/{WINS_PER_GOAL})</p>
        </div>""", unsafe_allow_html=True)
        if s.last:
            st.caption(f"You {s.last['action']} · {rules.describe(s.last)}")
        page(s.scene, "THE DUNGEON MASTER")

    with st.expander(f"Your storybook so far ({len(s.pages)} pages)"):
        for number, p in enumerate(s.pages, start=1):
            a, b = st.columns([1, 2])
            a.image(p["picture"], width="stretch")
            b.markdown(f"**Page {number}** · {p['result']}")
            b.write(p["story"])


# ---------------------------------------------------------------------------
# Screen 5: the end
# ---------------------------------------------------------------------------
def screen_end():
    s = st.session_state
    won = s.ending == "victory"
    st.title("The Tale Is Told" if won else "Your Tale Ends Here")
    st.write(f"{s.location['name']} will remember you." if won else
             "The ink runs dry... but every hero gets another page.")
    st.image(s.picture, width="stretch")
    page(s.scene, "THE LAST PAGE")
    for number, p in enumerate(s.pages, start=1):
        with st.expander(f"Page {number}"):
            st.image(p["picture"], width="stretch")
            st.write(p["story"])
    if st.button("Start a new tale", type="primary"):
        new_game()
        st.rerun()


SCREENS = {"hero": screen_hero, "class": screen_class, "start": screen_start,
           "play": screen_play, "end": screen_end}
SCREENS[st.session_state.stage]()
