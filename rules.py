"""Inkbound game rules: classes, stats, dice and outcomes.

Everything here is plain Python, not AI. The dice and rules decide what
happens; the language model only turns the result into story. That keeps the
game fair and consistent from turn to turn.
"""
import random

import numpy as np

STAT_NAMES = ["STR", "DEX", "CON", "INT", "WIS", "CHA"]

# ---------------------------------------------------------------------------
# Classes
# "items" = what the Oracle looks for in a hero drawing (6 per class, so the
#           scores stay fair between classes).
# "affinity" = drawings that suit the class during play: they get advantage.
# "weak" = the stat whose failures cost an extra point of health.
# ---------------------------------------------------------------------------
CLASSES = {
    "Bard": {
        "items": ["lute", "guitar", "flute", "music notes", "feathered hat", "cape"],
        "stats": {"STR": -1, "DEX": 2, "CON": 1, "INT": 0, "WIS": 0, "CHA": 3},
        "health": 6,
        "tagline": "Silver-tongued wanderer. Talks first, fights last.",
        "power": ("Song of Courage", "Once per tale, your next drawing succeeds automatically."),
        "weak": "STR",
        "weakness": ("Delicate", "Failed Strength checks cost 1 extra health."),
        "affinity": ["music notes", "speech bubble", "coins", "crown", "heart", "flower"],
        "look": "a young bard in a red feathered hat and teal cape, carrying a lute",
        "style": "warm candlelight, gold and amber",
        "pack": ["lute", "feathered hat", "3 gold coins"],
    },
    "Sorcerer": {
        "items": ["magic staff", "magic wand", "stars", "swirls", "crystal ball", "lightning bolt"],
        "stats": {"STR": -1, "DEX": 1, "CON": 0, "INT": 2, "WIS": 0, "CHA": 3},
        "health": 5,
        "tagline": "Born with magic in the blood. It doesn't always listen.",
        "power": ("Arcane Surge", "Once per tale, your next drawing succeeds automatically."),
        "weak": "CON",
        "weakness": ("Glass Bones", "Failed Constitution checks cost 1 extra health."),
        "affinity": ["magic wand", "lightning bolt", "star", "fire", "potion", "cloud"],
        "look": "a young sorcerer in violet robes holding a glowing staff",
        "style": "glowing violet light, arcane runes",
        "pack": ["oak staff", "spell book", "glowing crystal"],
    },
    "Rogue": {
        "items": ["hood", "dagger", "mask", "bag of coins", "key", "cloak"],
        "stats": {"STR": 0, "DEX": 3, "CON": 0, "INT": 2, "WIS": 1, "CHA": 0},
        "health": 6,
        "tagline": "Quick hands, quicker feet, and a plan for everything.",
        "power": ("Shadow Step", "Once per tale, your next drawing succeeds automatically."),
        "weak": "CHA",
        "weakness": ("Untrusted", "Failed Charisma checks cost 1 extra health."),
        "affinity": ["rope", "hook", "key", "door", "net", "map", "dagger"],
        "look": "a hooded rogue in a dark cloak with a dagger at the belt",
        "style": "moonlit blue shadows",
        "pack": ["dagger", "lockpicks", "dark cloak"],
    },
    "Fighter": {
        "items": ["sword", "shield", "armour", "axe", "helmet", "spear"],
        "stats": {"STR": 3, "DEX": 0, "CON": 2, "INT": -1, "WIS": 0, "CHA": 0},
        "health": 8,
        "tagline": "Stands in front so everyone else can stand behind.",
        "power": ("Second Wind", "Once per tale, your next drawing succeeds automatically."),
        "weak": "INT",
        "weakness": ("Headstrong", "Failed Intelligence checks cost 1 extra health."),
        "affinity": ["sword", "shield", "axe", "hammer", "rock", "spear"],
        "look": "a sturdy fighter in dented armour with a sword and round shield",
        "style": "bright red banners, polished steel",
        "pack": ["sword", "round shield", "travel rations"],
    },
    "Ranger": {
        "items": ["bow and arrow", "wolf", "bird", "tree", "leaf", "boots"],
        "stats": {"STR": 1, "DEX": 3, "CON": 1, "INT": 0, "WIS": 2, "CHA": -1},
        "health": 7,
        "tagline": "Knows every path, and every creature on it.",
        "power": ("Beast Friend", "Once per tale, your next drawing succeeds automatically."),
        "weak": "CHA",
        "weakness": ("Wild at Heart", "Failed Charisma checks cost 1 extra health."),
        "affinity": ["bow and arrow", "wolf", "bird", "tree", "horse", "boat"],
        "look": "a ranger in a green hooded cloak with a longbow",
        "style": "misty teal forest light",
        "pack": ["longbow", "quiver of arrows", "hunting knife"],
    },
    "Cleric": {
        "items": ["candle", "book", "sun symbol", "lantern", "potion bottle", "halo"],
        "stats": {"STR": 0, "DEX": -1, "CON": 2, "INT": 0, "WIS": 3, "CHA": 1},
        "health": 7,
        "tagline": "Carries a light for anyone lost in the dark.",
        "power": ("Healing Light", "Once per tale, your next drawing succeeds automatically."),
        "weak": "DEX",
        "weakness": ("Heavy Robes", "Failed Dexterity checks cost 1 extra health."),
        "affinity": ["lantern", "torch", "sun", "heart", "potion", "flower"],
        "look": "a cleric in white and gold robes holding a glowing lantern",
        "style": "soft golden holy light",
        "pack": ["lantern", "prayer book", "healing herbs"],
    },
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

TIERS_SUCCESS = {"TRIUMPH", "CLEAN SUCCESS", "SUCCESS WITH A COST"}


def drawing_style(gray):
    """Turn HOW a hero was drawn into stat bonuses.

    gray: 2D numpy array, 0 = black ink, 255 = paper.
    Returns (bonuses, reasons). The thresholds are first guesses: test with
    friends' drawings and adjust them.
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


def make_hero(class_name, bonuses, reasons, items_seen):
    """Build the hero sheet from the class and the drawing."""
    c = CLASSES[class_name]
    stats = {s: c["stats"][s] + bonuses.get(s, 0) for s in STAT_NAMES}
    return {
        "class": class_name,
        "stats": stats,
        "health": c["health"],
        "max_health": c["health"],
        "pack": list(c["pack"]),
        "power_used": False,
        "look": c["look"],
        "style": c["style"],
        "reasons": reasons,
        "items_seen": items_seen,
    }


def pick_difficulty(beat):
    """Each part of the quest gets a little harder."""
    return random.choice([8, 10, 12]) + beat


def resolve(hero, drawing, difficulty, advantage=False, use_power=False, loot=None):
    """Roll the dice for one turn and apply the result to the hero.

    Returns a dict describing what happened, for the screen and the DM.
    """
    c = CLASSES[hero["class"]]
    stat = CHECKS.get(drawing, "WIS")
    on_class = drawing in c["affinity"]
    adv = advantage or on_class
    rolls = [random.randint(1, 20)] + ([random.randint(1, 20)] if adv else [])
    roll = max(rolls)  # advantage: roll twice, keep the higher
    bonus = hero["stats"].get(stat, 0)
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

    # "Success with a cost" costs something in the story, not health.
    damage = {"PARTLY WORKS, THINGS GET WORSE": 1, "FAILS, NEW PROBLEM": 1,
              "COMIC DISASTER": 2}.get(tier, 0)
    if tier not in TIERS_SUCCESS and stat == c["weak"]:
        damage += 1
    hero["health"] = max(0, hero["health"] - damage)

    item = None
    if tier == "TRIUMPH":  # a natural 20 heals 1 health and finds treasure
        hero["health"] = min(hero["max_health"], hero["health"] + 1)
    if tier == "TRIUMPH" and loot:
        options = [i for i in loot if i not in hero["pack"]] or loot
        item = random.choice(options)
        hero["pack"].append(item)

    return {
        "drawing": drawing, "stat": stat, "rolls": rolls, "roll": roll, "bonus": bonus,
        "total": total, "difficulty": difficulty, "tier": tier, "damage": damage,
        "item": item, "advantage": adv, "power": use_power, "success": tier in TIERS_SUCCESS,
    }


def describe(o):
    """One line for the screen and the DM, e.g.
    'DEX check: rolled 13 (+2) = 15 vs 12. SUCCESS WITH A COST. Lost 1 health.'"""
    if o["power"]:
        return f"Class power used: automatic success. {o['tier']}."
    roll = f"rolled {o['rolls'][0]} and {o['rolls'][1]}, kept {o['roll']}" if len(o["rolls"]) > 1 else f"rolled {o['roll']}"
    text = f"{o['stat']} check: {roll} ({o['bonus']:+d}) = {o['total']} vs {o['difficulty']}. {o['tier']}."
    if o["damage"]:
        text += f" Lost {o['damage']} health."
    if o["item"]:
        text += f" Found: {o['item']}."
    return text
