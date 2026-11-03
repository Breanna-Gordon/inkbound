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

This turn the hero drew: {outcome['drawing']}.
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
"picture": under 15 words for an illustrator, describing only what can be seen now (the place, the hero, the action, the light)."""
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


if __name__ == "__main__":
    hero = rules.make_hero("Bard", {}, [], [])
    location = json.load(open("locations/1-gilded-tankard.json"))
    outcome = rules.resolve(hero, "rope", 12, loot=location["loot"])
    story, picture = narrate(hero, location, location["beats"][0], [location["opening"]], outcome)
    print("RESULT: ", rules.describe(outcome))
    print("STORY:  ", story)
    print("PICTURE:", picture)
