"""The Dungeon Master: a local LLM through Ollama.

The dice have already been rolled in rules.py before this is called. The DM's
job is only to turn that result into story, so it can't cheat or forget the
rules. It is given the recent story, the current quest goal and the place's
hidden secret, and told not to repeat itself, so each turn moves forward.
"""
import json

import ollama

import rules

MODEL = "llama3.2:3b"  # for richer stories try "llama3.1:8b" (ollama pull llama3.1:8b)


def narrate(hero, location, goal, history, outcome):
    so_far = "\n".join(f"- {page}" for page in history[-3:])
    prompt = f"""You are the Dungeon Master of a fairy-tale adventure, told like a picture storybook.

Hero: a {hero['class']} ({hero['look']}). Health {hero['health']}/{hero['max_health']}. Pack: {', '.join(hero['pack'])}.
Place: {location['name']}, {location['type'].lower()}.
Current goal: {goal}
Hidden secret of this place (reveal it slowly, a clue at a time): {location['secret']}

Story so far:
{so_far}

This turn the hero {outcome['action']}.
Dice result (already decided, you must follow it): {rules.describe(outcome)}

Write what happens next:
- 2 or 3 sentences, second person ("you").
- Follow the dice result honestly. A failure must go wrong; a success must work.
- Do NOT repeat events, objects or phrases from the story so far. Bring in something new each time: a character, a place detail, a clue or a twist.
- If the hero found an item, show them finding it.
- End with a hook that points towards the current goal.
- No headings, labels or lists.

Reply in JSON with exactly two keys:
"story": the text.
"picture": under 15 words for an illustrator: ONE clear moment from this turn. Start with the main action or creature, then the place. Use concrete things that can be painted (no feelings, no names, at most two characters)."""
    reply = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.9},  # a bit more creative, less repetitive
    )
    text = reply["message"]["content"]
    try:
        data = json.loads(text)
        return str(data["story"]), str(data["picture"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return text, location["picture"]  # backup if the LLM breaks the format


def choose_stat(action, hero):
    """For a typed action: which stat does it test, and does it suit the class?

    Returns (stat, suits_class). Falls back to simple word matching if the LLM
    gives an odd answer.
    """
    prompt = f"""A {hero['class']} in a fairy-tale adventure tries this: {action}
Which ability does it test most?
STR = force. DEX = agility, aim, sneaking, climbing. CON = endurance, toughness.
INT = knowledge, building, puzzles. WIS = noticing, nature, healing. CHA = talking, music, magic, persuading.
Does it play to a {hero['class']}'s strengths?
Reply in JSON like {{"stat": "DEX", "suits_class": true}}"""
    try:
        reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                            format="json", options={"temperature": 0})
        data = json.loads(reply["message"]["content"])
        stat = str(data["stat"]).upper()[:3]
        if stat in rules.STAT_NAMES:
            return stat, bool(data.get("suits_class", False))
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    for word, stat in rules.CHECKS.items():  # backup: look for a known word
        if word in action.lower():
            return stat, False
    return "WIS", False


if __name__ == "__main__":
    hero = rules.make_hero("Bard", {}, [], [])
    location = json.load(open("locations/1-gilded-tankard.json"))
    outcome = rules.resolve(hero, "rope", 12, loot=location["loot"])
    story, picture = narrate(hero, location, location["beats"][0], [location["opening"]], outcome)
    print("RESULT: ", rules.describe(outcome))
    print("STORY:  ", story)
    print("PICTURE:", picture)
