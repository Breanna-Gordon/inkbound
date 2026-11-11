import difflib
import json
import random
import re

import ollama

import rules

MODEL = "llama3.2:3b"  # any model installed with "ollama pull"

NARRATOR = """You are the narrator of a fairy-tale role-playing game, in the style of a classic
tabletop Dungeon Master: an intimate, slightly wry voice speaking straight to the player.
- Present tense, second person ("you"). Vivid, specific, never generic.
- Sometimes describe what the hero feels or notices, as if you can hear their thoughts.
- A little dry humour is welcome, especially when things go wrong.
- Each turn, set up a SITUATION: someone or something in the way, what they want,
  what is at stake, and things in the scene the hero could use.
- Characters may speak one short line of dialogue in quotes.
- Companions travelling with the hero have their own opinions and sometimes speak up or help.
- People react to the hero's race now and then, as described.
- Never decide what the hero does next. End by handing the moment back to the player."""

SKILL_LIST = ", ".join(rules.SKILLS)

CHOICES_FORMAT = """Reply in JSON with exactly eight keys:
"story": 3 or 4 sentences of narration.
"choices": exactly 3 things the hero could do next, like dialogue options in a role-playing game.
Each is an object with:
  "text": under 9 words, starting with a verb (e.g. "Convince the guard you're his cousin");
  "skill": one of: {skills};
  "dc": the difficulty, a number from 8 (easy) to 18 (very hard);
  "class_option": true for exactly ONE choice that only a {cls} would think of, else false.
Use three DIFFERENT skills.
"props": up to 5 objects visible in this scene that the hero could draw or use (1 to 3 words each,
lowercase, e.g. "chandelier", "ale barrel").
"recruit": the name of ONE character on this page who now trusts the hero enough to join them on their
quest, only if that truly fits the story (most pages: ""). Never someone already travelling with the hero.
"reactions": how each companion feels about what the hero just did, e.g. {{"Pip": "approves"}}.
Use "approves", "disapproves" or "neutral". Use {{}} if there are no companions.
"speaker": the name of the character who speaks out loud on this page, exactly as written in the list of
characters (not the hero), or "" if nobody speaks.
"line": what the speaker says, copied word for word from the story (without quote marks), or "".
"draw_challenge": {challenge_rule}"""

CHALLENGE_SOMETIMES = """only on about one page in four, and only when nobody is speaking: a short task where
the hero could DRAW an object to get past the situation, starting with "Draw", e.g. "Draw something to get
across the river". Otherwise ""."""

CHALLENGE_NOW = """REQUIRED on this page: a short task where the hero could DRAW an object to get past the
obstacle you just set up, starting with "Draw", e.g. "Draw something to get across the river"."""

# Pages the Dungeon Master is told to write, so the tale has a rhythm instead of one long conversation
TALK_PAGE = ""
DRAW_PAGE = """THIS PAGE IS A DRAWING PAGE: nobody speaks out loud ("speaker" and "line" are ""). Put a physical
obstacle in the hero's way (a gap, a locked door, a fire, a fall, a beast, a trap) that the hero could
get past with an object they draw."""
MOVE_ON = """The story has stayed in one moment too long. Move it on NOW: the scene changes (a way forward
opens, someone new arrives, the ground gives way, a discovery). Do not describe the same character
staring, drifting or whispering again."""


def _clean_choices(raw, hero):
    """Make sure the choices are usable, whatever the LLM sent back."""
    choices = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict) or not str(c.get("text", "")).strip():
            continue
        skill = str(c.get("skill", "")).strip().title()
        if skill not in rules.SKILLS:
            skill = random.choice(list(rules.SKILLS))
        try:
            dc = int(c.get("dc", 12))
        except (TypeError, ValueError):
            dc = 12
        choices.append({"text": str(c["text"]).strip().rstrip("."), "skill": skill,
                        "dc": max(8, min(18, dc)), "class_option": bool(c.get("class_option"))})
    return choices[:3]


def _parse(text):
    """Read the LLM's JSON reply. Returns a dict with a story, or None if it can't be read."""
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)  # the JSON part, if extra text surrounds it
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and str(data.get("story", "")).strip():
            return data
    return None


def _story_from_broken_reply(text):
    """Rescue just the story text from a reply that isn't valid JSON."""
    match = re.search(r'"story"\s*:\s*"(.*?)"\s*(,|\})', text, re.DOTALL)
    return match.group(1).replace('\\"', '"').strip() if match else ""


def _backup_choices(hero):
    """Three sensible choices for when the LLM doesn't give any."""
    return [
        {"text": "Look around carefully", "skill": "Perception", "dc": 12, "class_option": False},
        {"text": "Talk your way forward", "skill": "Persuasion", "dc": 13, "class_option": False},
        {"text": f"Trust your {hero['class']} training", "skill": hero["skills"][0], "dc": 12,
         "class_option": True},
    ]


def repeated(story, history, limit=0.6):
    """The sentences of a new page that are (almost) the same as a sentence already in the tale."""
    old = [s.lower() for page in history for s in re.split(r"(?<=[.!?])\s+", page) if len(s) > 20]
    return [s for s in re.split(r"(?<=[.!?])\s+", story) if len(s) > 20 and
            any(difflib.SequenceMatcher(None, s.lower(), o).ratio() > limit for o in old)]


def _ask(prompt, fallback_story, hero, attempts=2, challenge_now=False):
    """Send a prompt to the LLM and unpack the story, choices and scene props.
    If the reply can't be read, it asks once more, then falls back to safe defaults."""
    rescued = ""
    rule = CHALLENGE_NOW if challenge_now else CHALLENGE_SOMETIMES
    for _ in range(attempts):
        reply = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": NARRATOR},
                      {"role": "user", "content": prompt + "\n\n" +
                       CHOICES_FORMAT.format(skills=SKILL_LIST, cls=hero["class"], challenge_rule=rule)}],
            format="json",
            options={"temperature": 0.9},  # a bit more creative, less repetitive
        )
        text = reply["message"]["content"]
        data = _parse(text)
        rescued = rescued or _story_from_broken_reply(text)
        if data:
            props = data.get("props") if isinstance(data.get("props"), list) else []
            props = [str(p).lower().strip() for p in props if str(p).strip()][:5]
            choices = _clean_choices(data.get("choices"), hero) or _backup_choices(hero)
            reactions = data.get("reactions") if isinstance(data.get("reactions"), dict) else {}
            extra = {"recruit": str(data.get("recruit") or "").strip(), "reactions": reactions,
                     "speaker": str(data.get("speaker") or "").strip(), "line": str(data.get("line") or "").strip(),
                     "draw_challenge": str(data.get("draw_challenge") or "").strip()}
            return str(data["story"]).strip(), choices, props, extra
    story = rescued or fallback_story or "The moment hangs in the air. What do you do?"
    return story, _backup_choices(hero), [], {"recruit": "", "reactions": {}, "speaker": "", "line": "",
                                              "draw_challenge": ""}


ILLUSTRATOR = """You are the illustrator of a fairy-tale picture storybook. You read one page of the
story and plan the single picture that goes with it, for a painter.
1. Decide the FOCUS: the one thing this picture must show, taken from the most important moment on the
   page. If a character speaks or is met, the focus is the hero facing that character, both clearly
   visible. If the page is about a place or object (a cave mouth, a note, a door), make it big and
   central, with the hero next to it.
2. Pick the SHOT: "close-up" for a small object or a face, "medium shot" for characters and actions
   (the usual choice), "wide shot" only when the page is about a whole landscape.
3. Describe people and creatures by how they LOOK, never by name. Use the looks given to you.
Show the hero and up to three companions or other characters (four people at most), each drawn whole
(head and body visible) and each looking clearly different. A bigger group or crowd goes in the
background as simple shapes.
Only things that can be seen. Turn feelings into visible things: "being watched" becomes "glowing eyes
peering from the shadows". Never sounds, smells or echoes (not "a sneeze echoing", not "the smell of
smoke"): show what makes them instead, or leave them out. No words or writing in the picture."""

ILLUSTRATOR_FORMAT = """Reply in JSON with these keys:
"shot": "close-up", "medium shot" or "wide shot".
"focus": under 25 words: the main subject and what is happening, e.g. "a hooded barbarian standing at
  the dark mouth of a cave in a snowy cliff, a thick plume of smoke pouring out of it".
"setting": under 10 words: where it is, e.g. "snowy mountain cliff".
"details": a list of up to 3 short supporting details from the page.
"light": under 8 words.
"new_characters": a list of characters who physically appear on this page and are not in the known list,
  each {{"name": "...", "look": "under 15 words describing how they look"}}. Never someone who is only
  mentioned, remembered or heard (a whispered name is not a character in the picture).
"background": under 20 words: ONLY the place and the objects in it, with NO people or creatures, e.g.
  "a candlelit tavern with long wooden tables, ale barrels and a crackling fireplace".
"in_picture": a list of the names of the characters who are physically here in this moment (known or new),
  at most 3. Only people and animals: not lights, objects or places. Not someone only mentioned or heard.
  Don't list the hero: they are always in the picture.
"new_place": true if the hero has moved somewhere that looks different from the last picture, else false."""


def _cast_lines(cast):
    return "\n".join(f"- {name}: {look}" for name, look in cast.items()) or "- (none yet)"


def _hero_line(hero):
    """e.g. 'an Elf Bard (young bard in ..., an elf with long pointed ears)', plus how people see them."""
    race = hero.get("race", "Human")
    who = rules.with_article(f"{race} {hero['class']}")
    return f"{who} ({hero['look']}). How people see them: {rules.RACES[race]['people']}"


def _party_lines(hero):
    party = hero.get("party", [])
    return "\n".join(f"- {c['name']} ({c['look']}), feels {rules.approval_word(c['approval']).lower()} "
                     f"about the hero" for c in party) or "- (nobody yet: the hero travels alone)"


def illustrate(story, hero, location, cast, fallback=""):
    """Second step: read the finished page and plan the picture that matches it.
    Adds any new characters to `cast` so they look the same in later pictures. Returns a plan:
    "prompt": the whole scene as one description (for art.PICTURE_MODE "full"),
    "background": the place with nobody in it, "characters": who stands in it, "new_place": True/False
    (for "layered")."""
    plan = {"prompt": fallback or f"({' '.join(story.split()[:30])}:1.3)",
            "background": f"{location['name']}, {location['type'].lower()}", "characters": [], "new_place": False}
    prompt = (f"The hero: {hero['look']}.\nCompanions with the hero (show them when they are part of the "
              f"moment):\n{_party_lines(hero)}\nPlace: {location['name']}, {location['type'].lower()}.\n"
              f"Known characters and how they look:\n{_cast_lines(cast)}\n\nThe page:\n{story}\n\n"
              + ILLUSTRATOR_FORMAT)
    try:
        reply = ollama.chat(model=MODEL, messages=[{"role": "system", "content": ILLUSTRATOR},
                                                   {"role": "user", "content": prompt}],
                            format="json", options={"temperature": 0.4})
        data = json.loads(reply["message"]["content"])
        focus = str(data.get("focus", "")).strip()
        if focus:
            for c in data.get("new_characters") or []:
                vague = re.search(r"\b(unknown|implied|unseen|unclear|not described)\b", str(c.get("look", "")).lower()) \
                    if isinstance(c, dict) else True
                if isinstance(c, dict) and c.get("name") and c.get("look") and c["name"] not in cast and not vague:
                    cast[str(c["name"])] = str(c["look"])
            shot = str(data.get("shot", "medium shot")).strip() or "medium shot"
            details = data.get("details") if isinstance(data.get("details"), list) else []
            # the hero's look is added word for word, so they look the same in every picture
            parts = [shot, f"({focus}:1.25)", f"the hero is {hero['look']}", f"({data.get('setting', '')}:1.1)",
                     *[str(d) for d in details[:3]], str(data.get("light", ""))]
            plan["prompt"] = ", ".join(p for p in parts if p and p != "(:1.1)")
            background = str(data.get("background") or data.get("setting") or "").strip()
            if background:
                plan["background"] = f"{background}, {data.get('light', '')}".strip(", ")
            names = data.get("in_picture") if isinstance(data.get("in_picture"), list) else []
            plan["characters"] = [str(n) for n in names][:3]
            plan["new_place"] = bool(data.get("new_place"))
    except Exception:  # if the illustrator fails, the game carries on with a simpler picture
        pass
    return plan


def open_scene(hero, location, cast):
    """The first page: drop the hero into the opening moment."""
    prompt = f"""Start the adventure.
Hero: {_hero_line(hero)}.
Trained in: {', '.join(hero['skills'])}. Pack: {', '.join(hero['pack'])}.
Place: {location['name']}, {location['type'].lower()}.
What is happening as the hero arrives: {location['opening']}
Characters here: {_cast_lines(cast)}
First goal: {location['beats'][0]}
Hidden secret of this place (do NOT reveal it yet, only hint): {location['secret']}

Put the hero right in the middle of this moment. Keep the character from the opening in the scene
and let them speak one line. Make the first challenge clear and urgent, and mention one detail that a
{hero['class']} would notice that others wouldn't."""
    story, choices, props, extra = _ask(prompt, location["opening"], hero)
    picture = illustrate(story, hero, location, cast, location["picture"].replace("the hero", hero["look"]))
    return story, picture, choices, props, extra


def narrate(hero, location, goal, history, outcome, cast, page="talk", stuck=False):
    """A normal turn: tell the result of the dice, then set up what comes next.
    page: "talk" (anything can happen) or "draw" (no speech, an obstacle to draw past).
    stuck: True when the tale has stayed on the same part of the quest for a while."""
    so_far = "\n".join(f"- {page_}" for page_ in history[-3:])
    prompt = f"""Hero: {_hero_line(hero)}.
Health {hero['health']}/{hero['max_health']}. Pack: {', '.join(hero['pack'])}.
Companions travelling with the hero:
{_party_lines(hero)}
Place: {location['name']}, {location['type'].lower()}.
Current goal: {goal}
Hidden secret of this place (reveal it slowly, one clue at a time): {location['secret']}
Characters met so far (bring them back when it makes sense; new ones are welcome):
{_cast_lines(cast)}

Story so far:
{so_far}

This turn the hero {outcome['action']}.
Dice result (already decided, you must follow it): {rules.describe(outcome)}

Narrate:
1. In 1 or 2 sentences, what that move caused. Follow the dice honestly. A natural 20 is a
   glorious moment; a natural 1 is a funny disaster; a failure must go wrong in an interesting
   way; a success must clearly work. If the hero found an item, show them finding it.
2. Then set up the NEXT situation that moves towards the current goal.
Do NOT repeat events, objects or phrases from the story so far. Bring in something new.
{DRAW_PAGE if page == "draw" else TALK_PAGE}
{MOVE_ON if stuck else ""}"""
    story, choices, props, extra = _ask(prompt, "", hero, challenge_now=page == "draw")
    copied = repeated(story, history)
    if copied:  # a small model often copies earlier sentences: ask once more, naming what not to reuse
        again = prompt + "\n\nThese sentences were already used. Write completely different ones:\n" + \
            "\n".join(f"- {s}" for s in copied)
        story2, choices2, props2, extra2 = _ask(again, "", hero, challenge_now=page == "draw")
        if len(repeated(story2, history)) < len(copied):
            story, choices, props, extra = story2, choices2, props2, extra2
    if page == "draw":
        extra.update(speaker="", line="")
        if not extra["draw_challenge"].lower().startswith("draw"):
            extra["draw_challenge"] = f"Draw something that helps you {goal[0].lower() + goal[1:]}"
    return story, illustrate(story, hero, location, cast), choices, props, extra


_skill_cache = {}


def choose_skill(action, hero):
    """For a typed or drawn-and-typed move: which skill does it test, and does it suit the class?

    Returns (skill, suits_class). Answers are remembered, so the same drawing
    doesn't need asking twice. Falls back to word matching if the LLM gives an odd answer.
    """
    key = (action, hero["class"])
    if key in _skill_cache:
        return _skill_cache[key]
    prompt = f"""A {hero['class']} in a fairy-tale role-playing game tries this: {action}
Which ONE skill does it test? Choose from: {SKILL_LIST}.
Does it play to a {hero['class']}'s strengths?
Reply in JSON like {{"skill": "Stealth", "suits_class": true}}"""
    try:
        reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                            format="json", options={"temperature": 0})
        data = json.loads(reply["message"]["content"])
        skill = str(data["skill"]).strip().title()
        if skill in rules.SKILLS:
            _skill_cache[key] = (skill, bool(data.get("suits_class", False)))
            return _skill_cache[key]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    for word, stat in rules.CHECKS.items():  # backup: look for a known word
        if word in action.lower():
            return next(s for s, st_ in rules.SKILLS.items() if st_ == stat), False
    return "Perception", False


if __name__ == "__main__":
    hero = rules.make_hero("Bard", {}, [], [])
    location = json.load(open("locations/1-gilded-tankard.json"))
    story, picture, choices, props, extra = open_scene(hero, location, dict(location["cast"]))
    print("OPENING:", story)
    print("PICTURE:", picture)
    print("PROPS:  ", props)
    print("RECRUIT:", extra["recruit"] or "-")
    for c in choices:
        tag = f"[{hero['class'].upper()}] " if c["class_option"] else ""
        print(f"  {tag}[{c['skill'].upper()} DC {c['dc']}] {c['text']}")
