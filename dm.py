import json
import ollama

MODEL = "llama3.2:3b"

def narrate(hero, scene, drawing, result):
    prompt = (
        "You are the Dungeon Master of a dark fairy-tale adventure.\n"
        f"Hero: {hero['class']}. Stats: {hero['stats']}. Health: {hero['health']}.\n"
        f"Current scene: {scene}\n"
        f"The player drew: {drawing}\n"
        f"Dice result: {result}\n\n"
        "Reply in JSON with exactly two keys:\n"
        '"story": 2-3 sentences in second person showing what happens because of the dice result. '
        "End with a new danger woven into the story. No headings, labels or lists.\n"
        '"picture": one short sentence (under 15 words) describing what the scene looks like now, '
        "for an illustrator. Only visible things: the place, the hero, the action, the light."
    )
    reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format="json")
    text = reply["message"]["content"]
    try:
        data = json.loads(text)
        return data["story"], data["picture"]
    except (json.JSONDecodeError, KeyError):
        return text, "a hero in a dark fantasy cavern"  # backup if the LLM breaks the format

if __name__ == "__main__":
    hero = {"class": "Bard", "stats": "DEX +2, CHA +3", "health": "4/5"}
    story, picture = narrate(hero, "the Windy Chasm", "rope and hook", "SUCCESS WITH A COST")
    print("STORY:", story)
    print("PICTURE:", picture)