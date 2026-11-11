import json
import random
from pathlib import Path

import streamlit as st

import rules
import voice

# A short recording session for evaluate_voice.py: read game sentences out loud, and each recording is saved
# with the sentence it should be (data/voice/, source "practice"). Run with: streamlit run record_voice.py

LINES = 60  # how many sentences make a full session

TEMPLATES = [
    "I ask {npc} what happened at {place}",
    "I sneak past {npc} and head towards {place}",
    "I roll {skill} to read the old map",
    "As a {race} {cls}, I bow to {npc}",
    "I tell {npc} we will reach {place} together",
    "I cast a spell to light the path through {place}",
    "I show {npc} my {item}",
    "I promise {npc} I will protect them",
    "I try {skill} on {npc}",
    "I draw a rope and throw it over the gap",
    "I hide behind the barrels and listen",
    "I offer {npc} three gold coins for the key",
]
ITEMS = ["lute", "dagger", "spell book", "shield", "lantern", "map", "healing potion", "bow and arrows"]


def characters_and_places():
    names, places = [], []
    for path in sorted(Path("locations").glob("*.json")):
        location = json.loads(path.read_text(encoding="utf-8"))
        places.append(location["name"])
        names += [name.split(",")[0] for name, look in location["cast"].items() if not rules.is_scenery(name, look)]
    return names, places


def practice_sentences():
    """The same list every time (fixed random seed), so a session can carry on where it stopped."""
    rng = random.Random(7)
    names, places = characters_and_places()
    sentences = []
    for i in range(LINES):
        template = TEMPLATES[i % len(TEMPLATES)]
        sentences.append(template.format(
            npc=rng.choice(names), place=rng.choice(places), skill=rng.choice(list(rules.SKILLS)),
            race=rng.choice(list(rules.RACES)), cls=rng.choice(list(rules.CLASSES)), item=rng.choice(ITEMS)))
    return sentences


def practice_done():
    """How many practice lines are already saved."""
    count = 0
    for meta in voice.CLIPS.glob("*.json"):
        count += json.loads(meta.read_text(encoding="utf-8")).get("source") == "practice"
    return count


st.set_page_config(page_title="Inkbound: voice recording", layout="centered")
st.title("Voice recording session")
st.write("Read each sentence out loud in your normal voice. Every recording is saved with the sentence, so "
         "`evaluate_voice.py` can measure how well Whisper hears you. Recordings stay on this computer.")

sentences = practice_sentences()
s = st.session_state
s.setdefault("line", practice_done())
if s.line >= len(sentences):
    st.success(f"All {len(sentences)} lines recorded. Now run: python evaluate_voice.py")
    st.stop()

st.progress(s.line / len(sentences), text=f"Line {s.line + 1} of {len(sentences)}")
sentence = sentences[s.line]
st.markdown(f"### “{sentence}”")
recording = st.audio_input("Record", key=f"line-{s.line}")
if recording is not None:
    data = recording.getvalue()
    if s.get("heard_for") != (s.line, data):
        with st.spinner("Listening..."):
            hint = voice.hint_text(voice.game_words()) if voice.USE_HINT else None
            s.heard = voice.transcribe(data, hint=hint)
        s.heard_for = (s.line, data)
    st.write(f"Whisper heard: *{s.heard or '(nothing)'}*")
    said = st.text_input("What you actually said (change it if you said any words differently)", value=sentence,
                         key=f"said-{s.line}")
    if st.button("Save and next", type="primary"):
        voice.save_clip(data, s.heard, said.strip() or sentence, source="practice")
        s.line += 1
        st.rerun()
if st.button("Skip this line"):
    s.line += 1
    st.rerun()
