import ollama

MODEL = "llama3.2:3b"

def narrate(hero, scene, drawing, result):
    prompt = (
        "You are the Dungeon Master of a dark fairy-tale.\n"
        f"Hero: {hero['class']}. Stats: {hero['stats']}. Health: {hero['health']}.\n"
        f"Scene: {scene}\n"
        f"Player drew: {drawing}\n"
        f"Result: {result}\n"
        "Write 2-3 sentences in second person. Show the result. End with a new danger."
    )
    reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return reply["message"]["content"]

if __name__ == "__main__":
    hero = {"class": "Bard", "stats": "DEX +2, CHA +3", "health": "4/5"}
    print(narrate(hero, "the Windy Chasm", "rope and hook", "SUCCESS WITH A COST"))