import random

CHECKS = {"rope": "DEX", "wings": "DEX", "bridge": "INT", "sword": "STR", "music notes": "CHA"}

def resolve(hero, drawing, difficulty=12):
    stat = CHECKS.get(drawing, "WIS")
    roll = random.randint(1, 20)
    total = roll + hero["stats"].get(stat, 0)
    if roll == 20:
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
    return f"{stat} check: rolled {roll}, total {total} vs {difficulty}. {tier}"