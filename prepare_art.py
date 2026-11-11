import json
import time
from pathlib import Path

import art
import rules

# Paints every starting location's background and every character who lives there, once, and saves them in
# media/. After this, starting a tale doesn't need to wait for any painting. Already painted pictures are
# skipped, so it is safe to run again (for example after adding a location).


def main():
    start = time.time()
    for path in sorted(Path("locations").glob("*.json")):
        location = json.loads(path.read_text(encoding="utf-8"))
        t = time.time()
        art.location_background(location)
        print(f"{location['name']}: background ({time.time() - t:.0f}s)")
        for name, look in location["cast"].items():
            if rules.is_scenery(name, look):
                continue
            t = time.time()
            art.saved_portrait(name, look, rules.is_creature(look), **rules.portrait_settings(None))
            print(f"  {name} ({time.time() - t:.0f}s)")
    print(f"Done in {(time.time() - start) / 60:.1f} minutes. Pictures are in media/backgrounds and media/portraits.")


if __name__ == "__main__":
    main()
