import random
import re

import numpy as np

STAT_NAMES = ["STR", "DEX", "CON", "INT", "WIS", "CHA"]

# ---------------------------------------------------------------------------
# Classes
# "items" = what the Oracle looks for in a hero drawing (6 per class, so the
#           scores stay fair between classes).
# "affinity" = drawings that suit the class during play: they get advantage.
# "weak" = the stat whose failures cost an extra point of health.
# ---------------------------------------------------------------------------
def _cls(items, stats, hit_die, tagline, power, weak, weakness, skills, affinity, look, style, pack,
         passive=None, expertise=()):
    """One class. Health comes from the class's hit die, as in Baldur's Gate 3."""
    return dict(items=items, stats=dict(zip(STAT_NAMES, stats)), hit_die=hit_die,
                health={6: 5, 8: 6, 10: 7, 12: 8}[hit_die], tagline=tagline, power=power, weak=weak,
                weakness=weakness, skills=skills, affinity=affinity, look=look, style=style, pack=pack,
                passive=passive, expertise=list(expertise))


# The 12 classes of Baldur's Gate 3 (D&D 5e). Stats are in the order STR, DEX, CON, INT, WIS, CHA.
# items: what the Oracle looks for in a hero drawing (6 per class, so the scores stay fair).
# affinity: drawings that suit the class during play (they get advantage).
# weak: the stat whose failures cost 1 extra health. skills: trained skills (+2).
CLASSES = {
    "Barbarian": _cls(
        ["battle axe", "fur cloak", "war paint", "bone necklace", "bear", "flexed muscles"],
        [3, 1, 2, -1, 0, 0], 12, "Fury first, questions later. Much later.",
        ("Rage", "Once per tale, your next move succeeds automatically."),
        "INT", ("Hot-Headed", "Failed Intelligence checks cost 1 extra health."),
        ["Athletics", "Intimidation", "Survival"],
        ["axe", "hammer", "rock", "spear", "fire", "sword"],
        "a wild barbarian in a fur cloak with war paint and a huge battle axe",
        "fiery reds, snowy wilds", ["battle axe", "fur cloak", "hunting horn"]),
    "Bard": _cls(
        ["lute", "flute", "harp", "music notes", "feathered hat", "cape"],
        [-1, 2, 1, 0, 0, 3], 8, "Silver-tongued wanderer. Talks first, fights last.",
        ("Bardic Inspiration", "Once per tale, your next move succeeds automatically."),
        "STR", ("Delicate", "Failed Strength checks cost 1 extra health."),
        ["Performance", "Persuasion", "Deception"],
        ["music notes", "speech bubble", "coins", "crown", "heart", "flower"],
        "a young bard in a red feathered hat and teal cape, carrying a lute",
        "warm candlelight, gold and amber", ["lute", "feathered hat", "3 gold coins"],
        passive=("Jack of All Trades", "+1 on every skill you are not trained in.")),
    "Cleric": _cls(
        ["holy symbol", "candle", "prayer book", "halo", "mace", "sun symbol"],
        [0, -1, 2, 0, 3, 1], 8, "Carries a light for anyone lost in the dark.",
        ("Channel Divinity", "Once per tale, your next move succeeds automatically."),
        "DEX", ("Heavy Robes", "Failed Dexterity checks cost 1 extra health."),
        ["Medicine", "Insight", "Religion"],
        ["lantern", "torch", "sun", "heart", "potion", "candle"],
        "a cleric in white and gold robes holding a glowing holy symbol",
        "soft golden holy light", ["mace", "holy symbol", "healing herbs"]),
    "Druid": _cls(
        ["antlers", "leaves", "crescent moon", "vine-wrapped branch", "flowers", "mushroom"],
        [0, 1, 2, 1, 3, -1], 8, "Speaks for the wild, and sometimes becomes it.",
        ("Wild Shape", "Once per tale, your next move succeeds automatically."),
        "CHA", ("Feral", "Failed Charisma checks cost 1 extra health."),
        ["Nature", "Animal Handling", "Survival"],
        ["tree", "flower", "wolf", "bird", "moon", "leaf"],
        "a druid with antlers woven into their hair, wrapped in leaves and vines",
        "mossy greens, dappled sunlight", ["wooden staff", "pouch of seeds", "herbal salve"]),
    "Fighter": _cls(
        ["sword", "round shield", "knight armour", "helmet", "spear", "banner"],
        [3, 1, 2, 0, 1, -1], 10, "Stands in front so everyone else can stand behind.",
        ("Second Wind", "Once per tale, your next move succeeds automatically."),
        "CHA", ("Blunt", "Failed Charisma checks cost 1 extra health."),
        ["Athletics", "Perception", "Intimidation"],
        ["sword", "shield", "axe", "hammer", "spear", "helmet"],
        "a sturdy fighter in dented armour with a sword and round shield",
        "bright red banners, polished steel", ["sword", "round shield", "travel rations"]),
    "Monk": _cls(
        ["prayer beads", "robe", "bamboo staff", "headband", "bare fists", "sandals"],
        [0, 3, 1, 0, 2, -1], 8, "Calm as still water. Fast as lightning.",
        ("Flurry of Blows", "Once per tale, your next move succeeds automatically."),
        "CHA", ("Few Words", "Failed Charisma checks cost 1 extra health."),
        ["Acrobatics", "Insight", "Stealth"],
        ["rope", "ladder", "wings", "rock", "bird", "cloud"],
        "a barefoot monk in a simple orange robe with a bamboo staff",
        "calm dawn light, soft orange", ["bamboo staff", "prayer beads", "rice cakes"]),
    "Paladin": _cls(
        ["shield with a cross", "glowing sword", "angel wings", "white horse", "plate armour with a cape",
         "lion emblem"],
        [3, -1, 1, 0, 0, 2], 10, "Sworn to an oath, and stubborn about keeping it.",
        ("Lay on Hands", "Once per tale, your next move succeeds automatically."),
        "DEX", ("Clanking Armour", "Failed Dexterity checks cost 1 extra health."),
        ["Athletics", "Persuasion", "Religion"],
        ["sword", "shield", "sun", "heart", "crown", "horse"],
        "a paladin in shining plate armour with a cape and a shield bearing a sun",
        "radiant gold and white", ["longsword", "oath shield", "holy water"]),
    "Ranger": _cls(
        ["bow and arrow", "quiver of arrows", "wolf", "hawk", "paw print", "pine tree"],
        [1, 3, 1, 0, 2, -1], 10, "Knows every path, and every creature on it.",
        ("Hunter's Mark", "Once per tale, your next move succeeds automatically."),
        "CHA", ("Wild at Heart", "Failed Charisma checks cost 1 extra health."),
        ["Survival", "Stealth", "Perception", "Animal Handling"],
        ["bow and arrow", "wolf", "bird", "tree", "horse", "boat"],
        "a ranger in a green hooded cloak with a longbow",
        "misty teal forest light", ["longbow", "quiver of arrows", "hunting knife"]),
    "Rogue": _cls(
        ["hood", "dagger", "mask", "bag of coins", "key", "lockpick"],
        [-1, 3, 0, 2, 1, 0], 8, "Quick hands, quicker feet, and a plan for everything.",
        ("Sneak Attack", "Once per tale, your next move succeeds automatically."),
        "STR", ("Light Build", "Failed Strength checks cost 1 extra health."),
        ["Stealth", "Sleight of Hand", "Deception", "Investigation"],
        ["rope", "hook", "key", "door", "net", "map", "dagger"],
        "a hooded rogue in a dark cloak with a dagger at the belt",
        "moonlit blue shadows", ["dagger", "lockpicks", "dark cloak"],
        passive=("Expertise", "Double training bonus (+4) on Stealth and Sleight of Hand."),
        expertise=["Stealth", "Sleight of Hand"]),
    "Sorcerer": _cls(
        ["lightning bolt", "fire in hands", "swirls", "dragon", "sparkles", "glowing eyes"],
        [-1, 1, 2, 0, 0, 3], 6, "Born with magic in the blood. It doesn't always listen.",
        ("Wild Magic Surge", "Once per tale, your next move succeeds automatically."),
        "STR", ("Glass Arms", "Failed Strength checks cost 1 extra health."),
        ["Arcana", "Intimidation", "Persuasion"],
        ["lightning bolt", "fire", "star", "cloud", "magic wand", "dragon"],
        "a young sorcerer in violet robes with crackling magic in their hands",
        "glowing violet light, arcane sparks", ["glowing crystal", "silk robes", "dragon scale"]),
    "Warlock": _cls(
        ["eye symbol", "tentacles", "chains", "purple flames", "demon horns", "pact scroll"],
        [-1, 1, 1, 1, 0, 3], 8, "Made a deal for power. The small print is still coming.",
        ("Eldritch Blast", "Once per tale, your next move succeeds automatically."),
        "STR", ("Bookish Grip", "Failed Strength checks cost 1 extra health."),
        ["Deception", "Arcana", "Intimidation"],
        ["eye", "fire", "key", "door", "moon", "star"],
        "a warlock in a dark coat with a glowing eye symbol and purple flames",
        "eerie purple and green glow", ["pact tome", "black candle", "strange coin"]),
    "Wizard": _cls(
        ["pointy wizard hat", "spell book", "magic wand", "long beard", "magic staff", "scroll"],
        [-1, 1, 1, 3, 2, -1], 6, "Has read a book about this. Possibly several.",
        ("Arcane Recovery", "Once per tale, your next move succeeds automatically."),
        "CHA", ("Absent-Minded", "Failed Charisma checks cost 1 extra health."),
        ["Arcana", "History", "Investigation"],
        ["book", "map", "magic wand", "clock", "key", "star"],
        "an old wizard in a starry blue robe and pointy hat, holding a spell book",
        "deep blue and silver starlight", ["spell book", "magic staff", "ink and quill"]),
}

# ---------------------------------------------------------------------------
# Things a player might draw during the game, and the stat each one tests.
# Add new items here: the Oracle reads this list.
# ---------------------------------------------------------------------------
CHECKS = {
    # Dexterity: moving, climbing, aiming
    "rope": "DEX", "ladder": "DEX", "wings": "DEX", "hook": "DEX", "net": "DEX",
    "boat": "DEX", "horse": "DEX", "bow and arrow": "DEX", "dagger": "DEX",
    # Strength: force
    "sword": "STR", "axe": "STR", "hammer": "STR", "rock": "STR", "shield": "STR", "spear": "STR",
    # Intelligence: building, solving
    "bridge": "INT", "map": "INT", "key": "INT", "door": "INT", "book": "INT", "clock": "INT",
    # Wisdom: nature, light, noticing
    "tree": "WIS", "flower": "WIS", "moon": "WIS", "sun": "WIS", "eye": "WIS",
    "bird": "WIS", "wolf": "WIS", "lantern": "WIS", "torch": "WIS",
    # Charisma: people and magic
    "music notes": "CHA", "speech bubble": "CHA", "coins": "CHA", "crown": "CHA",
    "heart": "CHA", "magic wand": "CHA", "lightning bolt": "CHA", "star": "CHA",
    "fire": "CHA", "cloud": "CHA",
    # Constitution: enduring
    "potion": "CON", "food": "CON", "blanket": "CON",
}
ACTION_ITEMS = list(CHECKS)

# D&D-style skills: each one tests a stat. Choices from the DM name a skill.
SKILLS = {
    "Athletics": "STR",
    "Acrobatics": "DEX", "Sleight of Hand": "DEX", "Stealth": "DEX",
    "Arcana": "INT", "History": "INT", "Investigation": "INT", "Nature": "INT", "Religion": "INT",
    "Animal Handling": "WIS", "Insight": "WIS", "Medicine": "WIS", "Perception": "WIS", "Survival": "WIS",
    "Deception": "CHA", "Intimidation": "CHA", "Performance": "CHA", "Persuasion": "CHA",
    "Endurance": "CON",  # not an official D&D skill, but useful for toughing things out
}
# Trained skills add this bonus (each class lists its trained skills above).
PROFICIENCY_BONUS = 2
MAX_INSPIRATION = 4

TIERS_SUCCESS = {"TRIUMPH", "CLEAN SUCCESS", "SUCCESS WITH A COST"}


def drawing_style(gray):
    """Turn HOW a hero was drawn into stat bonuses.

    gray: 2D numpy array, 0 = black ink, 255 = paper.
    Returns (bonuses, reasons).
    """
    ink = gray < 128
    if not ink.any():
        return {}, []
    coverage = ink.mean()  # how much of the page is ink
    # how often a line is crossed, per row: more crossings = more detail
    complexity = np.abs(np.diff(ink.astype(np.int8), axis=1)).sum() / ink.shape[0]
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    spread = ((rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)) / ink.size

    bonuses, reasons = {}, []
    if coverage > 0.06:
        bonuses["STR"] = 1
        reasons.append("Bold, heavy lines: +1 Strength")
    elif coverage < 0.02:
        bonuses["DEX"] = 1
        reasons.append("Light, quick lines: +1 Dexterity")
    if complexity > 6:
        bonuses["INT"] = 1
        reasons.append("Lots of small details: +1 Intelligence")
    if spread > 0.45:
        bonuses["CHA"] = 1
        reasons.append("A big, confident drawing: +1 Charisma")
    if not bonuses:
        bonuses["WIS"] = 1
        reasons.append("Steady, balanced lines: +1 Wisdom")
    return bonuses, reasons



# ---------------------------------------------------------------------------
# Races (the playable races of Baldur's Gate 3, kept simple)
# bonus: added to the stats. skill: an extra trained skill. trait: (name, what it does).
# look: features added to the hero's description. people: a note for the Dungeon Master.
# ---------------------------------------------------------------------------
RACES = {
    "Human": dict(bonus={"CHA": 1, "WIS": 1}, skill=None, trait=("Resourceful", "Start with 1 extra Inspiration."),
                  look="", people="humans are everywhere and few people look twice at one"),
    "Elf": dict(bonus={"DEX": 2}, skill="Perception", trait=("Keen Senses", "Trained in Perception."),
                look="long pointed ears", people="elves are admired, and thought a little aloof"),
    "Dwarf": dict(bonus={"CON": 2}, skill=None, trait=("Dwarven Toughness", "+1 maximum health."),
                  look="short and stocky, thick braided hair",
                  people="dwarves are trusted as honest craftsfolk, and known for being stubborn"),
    "Halfling": dict(bonus={"DEX": 2}, skill=None, trait=("Lucky", "A natural 1 is re-rolled once."),
                     look="small and short, curly hair, bare feet",
                     people="halflings are welcomed as cheerful folk, and often underestimated"),
    "Gnome": dict(bonus={"INT": 2}, skill="Arcana", trait=("Gnome Cunning", "Trained in Arcana."),
                  look="tiny, big bright eyes, a large nose",
                  people="gnomes are known as clever tinkerers who talk too fast"),
    "Tiefling": dict(bonus={"CHA": 2}, skill="Intimidation", trait=("Infernal Legacy", "Trained in Intimidation."),
                     look="curved horns, a long tail, purple skin",
                     people="some people distrust tieflings at first because of their horns"),
    "Half-Orc": dict(bonus={"STR": 2}, skill=None,
                     trait=("Relentless Endurance", "Once per tale, drop to 1 health instead of 0."),
                     look="grey-green skin, small tusks",
                     people="half-orcs are feared at first, and respected once they prove themselves"),
    "Dragonborn": dict(bonus={"STR": 2}, skill="Athletics", trait=("Dragon Blood", "Trained in Athletics."),
                       look="a dragon's head, bronze scales",
                       people="dragonborn are rare, proud, and stared at wherever they go"),
}


def with_article(words):
    """'an Elf Bard', 'a Dwarf Bard'."""
    return ("an " if words[:1].upper() in "AEIOU" else "a ") + words


# ---------------------------------------------------------------------------
# Party: companions who travel with the hero, like in Baldur's Gate 3
# ---------------------------------------------------------------------------
MAX_PARTY = 3
APPROVAL_START = 50
APPROVAL_STEP = 10


def recruit(hero, name, look):
    """Add a companion to the party. Returns True if they joined."""
    if len(hero["party"]) >= MAX_PARTY or any(c["name"] == name for c in hero["party"]):
        return False
    hero["party"].append({"name": name, "look": look, "approval": APPROVAL_START, "helped": False})
    return True


def capital(name):
    """'the hooded stranger' -> 'The hooded stranger', for the start of a sentence."""
    return name[:1].upper() + name[1:]


def approval_word(approval):
    for limit, word in [(80, "Devoted"), (60, "Friendly"), (40, "Neutral"), (20, "Wary")]:
        if approval >= limit:
            return word
    return "About to leave"


def react(hero, reactions):
    """Change each companion's approval: {"Pip": "approves", ...}.
    Companions whose approval reaches 0 leave. Returns short news lines for the screen."""
    news = []
    for companion in list(hero["party"]):
        name = companion["name"].lower()  # the DM may write the name differently, e.g. "The Stranger"
        feeling = next((str(f).lower() for n, f in reactions.items()
                        if str(n).lower() == name or str(n).lower() in name or name in str(n).lower()), "")
        change = APPROVAL_STEP if feeling.startswith("approv") else -APPROVAL_STEP if feeling.startswith("disapprov") else 0
        if not change:
            continue
        companion["approval"] = max(0, min(100, companion["approval"] + change))
        if companion["approval"] == 0:
            hero["party"].remove(companion)
            news.append(f"{capital(companion['name'])} has had enough and leaves your party.")
        else:
            news.append(f"{capital(companion['name'])} {'approves' if change > 0 else 'disapproves'}.")
    return news


def new_quest_step(hero):
    """Each companion can help once per part of the quest."""
    for companion in hero["party"]:
        companion["helped"] = False


# The hero's appearance is written down once and used word for word in the portrait AND in every
# scene, so the painter draws the same person each time.
# CLASS_COLOURS: each class's outfit colours. HAIR: the hair colours a player can pick.
CLASS_COLOURS = {
    "Barbarian": "brown fur and dark red", "Bard": "deep red and dusky teal", "Cleric": "white and gold",
    "Druid": "moss green and brown", "Fighter": "steel grey and dark blue", "Monk": "saffron orange",
    "Paladin": "silver and royal blue", "Ranger": "forest green and brown", "Rogue": "charcoal grey and black",
    "Sorcerer": "violet and silver", "Warlock": "black and sickly green", "Wizard": "midnight blue and silver",
}
HAIR = ["black", "dark brown", "light brown", "blonde", "red", "white", "silver", "blue", "pink"]

# How portraits are painted. Every portrait gets PORTRAIT_TONE, so the colours stay soft and inky like
# the rest of the book. A class can also have a seed here to repeat one portrait exactly.
PORTRAITS = {}
PORTRAIT_TONE = "muted earthy ink colours, soft faded tones"


def hero_look(class_name, race, hair):
    """e.g. 'a young elf bard in a red feathered hat ..., wearing deep red and dusky teal, long pointed ears,
    black hair'. The race goes in front of the class ("a young elf bard"), so the painter draws one person."""
    look = CLASSES[class_name]["look"]
    if race != "Human":
        look = re.sub(rf"\b{class_name.lower()}\b", f"{race.lower()} {class_name.lower()}", look, count=1)
        look = re.sub(r"^a (?=[aeiou])", "an ", look)  # "a elf paladin" -> "an elf paladin"
    parts = [look, f"wearing {CLASS_COLOURS[class_name]}", RACES[race]["look"], f"{hair} hair"]
    return ", ".join(p for p in parts if p)


# Characters the painter should draw as animals (no clothes), and things that are scenery, not characters.
CREATURE_WORDS = ("wolf", "dragon", "bear", "fox", "horse", "pony", "stag", "deer", "boar", "cat", "dog", "hound",
                  "owl", "raven", "crow", "hawk", "eagle", "spider", "rat", "snake", "serpent", "frog", "toad",
                  "goat", "bird", "bat", "beast", "griffin", "unicorn", "boar")
SCENERY = r"\b(wisps?|orbs?|fireflies|sparks|mist|fog|smoke)\b|\bballs? of\b[^,]*\blight\b"


def is_creature(look):
    """True for an animal or monster (it gets painted without clothes)."""
    first = re.split(r"\b(with|in|wearing|carrying)\b", look.lower())[0]  # 'a great grey wolf' from 'a great grey wolf with ...'
    return any(re.search(rf"\b{w}s?\b", first) for w in CREATURE_WORDS)


def is_scenery(name, look):
    """True for lights, mist and similar: part of the background, not a character to cut out."""
    return bool(re.search(SCENERY, f"{name} {look}".lower())) and not is_creature(look)


def portrait_settings(class_name):
    """Settings for art.portrait() for this class, e.g. {"colours": ...}."""
    settings = dict(PORTRAITS.get(class_name, {}))
    settings["colours"] = ", ".join(c for c in (settings.get("colours", ""), PORTRAIT_TONE) if c)
    return settings


def make_hero(class_name, bonuses, reasons, items_seen, race="Human", hair="black"):
    """Build the hero sheet from the class, the race, the hair colour and the drawing."""
    c = CLASSES[class_name]
    r = RACES[race]
    stats = {s: c["stats"][s] + bonuses.get(s, 0) + r["bonus"].get(s, 0) for s in STAT_NAMES}
    skills = list(c["skills"]) + ([r["skill"]] if r["skill"] and r["skill"] not in c["skills"] else [])
    health = c["health"] + (1 if race == "Dwarf" else 0)
    return {
        "class": class_name,
        "race": race,
        "stats": stats,
        "health": health,
        "max_health": health,
        "pack": list(c["pack"]),
        "power_used": False,
        "endurance_used": False,
        "hair": hair,
        "look": hero_look(class_name, race, hair),
        "style": c["style"],
        "reasons": reasons,
        "items_seen": items_seen,
        "skills": skills,
        "inspiration": 2 if race == "Human" else 1,  # spend one to re-roll a failed check
        "party": [],
    }


def pick_difficulty(beat):
    """Each part of the quest gets a little harder."""
    return random.choice([8, 10, 12]) + beat


def resolve(hero, drawing, difficulty, advantage=False, use_power=False, loot=None,
            stat=None, on_class=None, skill=None, use_inspiration=False):
    """Roll the dice for one turn and apply the result to the hero.

    - A drawing: the stat and class fit come from CHECKS and the class's affinity list.
    - A choice or typed move: the DM names a skill (e.g. "Persuasion"), which sets the stat.
    Trained skills add +2. Class-suited moves get advantage (roll two d20s, keep the higher).
    Spending Inspiration re-rolls a failed check once.
    Returns a dict describing what happened, for the screen and the DM.
    """
    c = CLASSES[hero["class"]]
    if skill in SKILLS:
        stat = SKILLS[skill]
    stat = stat or CHECKS.get(drawing, "WIS")
    on_class = (drawing in c["affinity"]) if on_class is None else on_class
    adv = advantage or on_class
    proficient = skill in hero.get("skills", [])
    training = PROFICIENCY_BONUS if proficient else 0
    if proficient and skill in c["expertise"]:
        training *= 2  # Rogue: Expertise
    elif skill and not proficient and c["passive"] and c["passive"][0] == "Jack of All Trades":
        training = 1   # Bard: Jack of All Trades
    bonus = hero["stats"].get(stat, 0) + training

    def roll_d20():
        dice = [random.randint(1, 20)] + ([random.randint(1, 20)] if adv else [])
        return dice, max(dice)

    rolls, roll = roll_d20()
    lucky = None
    if roll == 1 and hero.get("race") == "Halfling":  # Halfling: Lucky
        lucky = roll
        rolls, roll = roll_d20()
    rerolled = None
    if (use_inspiration and hero.get("inspiration", 0) > 0 and not use_power
            and roll != 20 and roll + bonus < difficulty):
        hero["inspiration"] -= 1
        rerolled = roll
        rolls, roll = roll_d20()
    total = roll + bonus

    if use_power and not hero["power_used"]:
        hero["power_used"] = True
        tier = "CLEAN SUCCESS"
    elif roll == 20:
        tier = "TRIUMPH"
    elif roll == 1:
        tier = "COMIC DISASTER"
    elif total >= difficulty + 5:
        tier = "CLEAN SUCCESS"
    elif total >= difficulty:
        tier = "SUCCESS WITH A COST"
    elif total >= difficulty - 4:
        tier = "PARTLY WORKS, THINGS GET WORSE"
    else:
        tier = "FAILS, NEW PROBLEM"
    success = tier in TIERS_SUCCESS

    # "Success with a cost" costs something in the story, not health.
    damage = {"PARTLY WORKS, THINGS GET WORSE": 1, "FAILS, NEW PROBLEM": 1,
              "COMIC DISASTER": 2}.get(tier, 0)
    if not success and stat == c["weak"]:
        damage += 1
    hero["health"] = max(0, hero["health"] - damage)
    endured = hero["health"] == 0 and hero.get("race") == "Half-Orc" and not hero.get("endurance_used")
    if endured:  # Half-Orc: Relentless Endurance
        hero["health"], hero["endurance_used"] = 1, True

    item = None
    if tier == "TRIUMPH":  # a natural 20 heals 1 health and finds treasure
        hero["health"] = min(hero["max_health"], hero["health"] + 1)
    if tier == "TRIUMPH" and loot:
        options = [i for i in loot if i not in hero["pack"]] or loot
        item = random.choice(options)
        hero["pack"].append(item)

    # Playing to your class earns Inspiration, like in Baldur's Gate 3.
    inspired = success and on_class and hero.get("inspiration", 0) < MAX_INSPIRATION
    if inspired:
        hero["inspiration"] += 1

    return {
        "drawing": drawing, "action": f"drew a {drawing}", "skill": skill, "stat": stat,
        "proficient": proficient, "training": training, "rolls": rolls, "roll": roll, "rerolled": rerolled,
        "bonus": bonus, "stat_bonus": hero["stats"].get(stat, 0), "total": total,
        "difficulty": difficulty, "tier": tier, "damage": damage, "item": item,
        "advantage": adv, "power": use_power, "success": success, "inspired": inspired,
        "lucky": lucky, "endured": endured,
    }


def check_name(o):
    """e.g. 'Persuasion (CHA)' or 'DEX'."""
    return f"{o['skill']} ({o['stat']})" if o.get("skill") else o["stat"]


def describe(o):
    """One line for the screen and the DM, e.g.
    'Persuasion (CHA) check, DC 12: rolled 13 +5 = 18. SUCCESS WITH A COST.'"""
    if o["power"]:
        return f"Class power used: automatic success. {o['tier']}."
    if len(o["rolls"]) > 1:
        roll = f"rolled {o['rolls'][0]} and {o['rolls'][1]} with advantage, kept {o['roll']}"
    else:
        roll = f"rolled {o['roll']}"
    if o.get("rerolled"):
        roll = f"re-rolled a {o['rerolled']} with Inspiration, then {roll}"
    if o.get("lucky"):
        roll = f"rolled a 1, re-rolled it with Halfling luck, then {roll}"
    text = f"{check_name(o)} check, DC {o['difficulty']}: {roll} {o['bonus']:+d} = {o['total']}. {o['tier']}."
    if o["damage"]:
        text += f" Lost {o['damage']} health."
    if o["item"]:
        text += f" Found: {o['item']}."
    if o.get("inspired"):
        text += " Gained Inspiration."
    if o.get("endured"):
        text += " Should have fallen, but Relentless Endurance kept the hero standing on 1 health."
    return text
