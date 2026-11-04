"""The illustrator: sends prompts to ComfyUI (Stable Diffusion, PyTorch).

ComfyUI runs as its own program. This file loads the saved workflow, fills in
the prompt and the style settings, sends it to ComfyUI, waits, and returns the
finished PNG.

STYLE SETTINGS: everything that controls the look of Inkbound is in the block
below, so you can experiment without touching the rest of the code.
"""
import json
import random
import time
import uuid

import requests

# ---------------------------------------------------------------------------
# STYLE SETTINGS
# ---------------------------------------------------------------------------
# Pick a look by changing PRESET to one of the names below. Each preset's
# model files must be in ComfyUI's models/checkpoints (and models/loras) folders.
PRESETS = {
    # The plain test model you started with.
    "sd15": dict(checkpoint="v1-5-pruned-emaonly-fp16.safetensors",
                 width=768, height=512, steps=25, cfg=7, sampler="dpmpp_2m", scheduler="karras",
                 lora=None, lora_strength=0.8, trigger=""),
    # DreamShaper 8: more painterly, same size and speed.
    "dreamshaper8": dict(checkpoint="DreamShaper_8_pruned.safetensors",
                         width=768, height=512, steps=25, cfg=7, sampler="dpmpp_2m", scheduler="karras",
                         lora=None, lora_strength=0.8, trigger=""),
    # The real picture-book look: DreamShaper XL Turbo + StorybookRedmond LoRA.
    # Turbo models need very few steps and a low CFG.
    "storybook_xl": dict(checkpoint="DreamShaperXL_Turbo_dpmppSdeKarras_half_pruned_6.safetensors",
                         width=1216, height=832, steps=6, cfg=2, sampler="dpmpp_sde", scheduler="karras",
                         lora="StorybookRedmondUnbound-KidsRedmAF.safetensors", lora_strength=0.8,
                         trigger="KidsRedmAF"),
}
PRESET = "dreamshaper8"

# The art style, added to every picture. Pick one with ART_STYLE, or add your
# own. Most important words go first: the model only reads about 75 words.
ART_STYLES = {
    "storybook": dict(
        words="storybook illustration, hand-painted gouache and watercolour, soft edges, "
              "warm light, whimsical fairy-tale picture book",
        avoid="photo, realistic, 3d render, black and white, monochrome, harsh lines"),
    "cartoon": dict(
        words="cartoon illustration, bold clean outlines, flat bright colours, cel shading, "
              "cute rounded characters, expressive faces, animated fairy-tale",
        avoid="photo, realistic, 3d render, painterly, watercolour texture, detailed skin, "
              "black and white, monochrome, gritty"),
    "comic": dict(
        words="comic book illustration, thick ink outlines, vibrant flat colours, "
              "graphic novel panel, dynamic pose",
        avoid="photo, realistic, 3d render, blurry, soft focus, black and white"),
}
ART_STYLE = "cartoon"

# Always avoided, whatever the style.
NEGATIVE_ALWAYS = "blurry, text, watermark, ugly, deformed, extra limbs"
# ---------------------------------------------------------------------------

# ComfyUI usually listens on 8188; the Desktop app sometimes uses 8000.
PORTS = [8188, 8000]
_server = None


def server():
    """Find ComfyUI the first time it's needed."""
    global _server
    if _server is None:
        for port in PORTS:
            url = f"http://127.0.0.1:{port}"
            try:
                if requests.get(f"{url}/system_stats", timeout=2).ok:
                    _server = url
                    break
            except requests.RequestException:
                pass
        else:
            raise RuntimeError("Can't reach ComfyUI on port 8188 or 8000. Is the ComfyUI app open?")
    return _server


def build_workflow(prompt, workflow_file="workflows/scene_api.json", preset=None):
    """Load the saved workflow and apply the chosen preset to it."""
    p = PRESETS[preset or PRESET]
    workflow = json.load(open(workflow_file))
    workflow["4"]["inputs"]["ckpt_name"] = p["checkpoint"]
    workflow["5"]["inputs"].update(width=p["width"], height=p["height"])
    workflow["6"]["inputs"]["text"] = prompt                         # what to draw
    workflow["7"]["inputs"]["text"] = f"{ART_STYLES[ART_STYLE]['avoid']}, {NEGATIVE_ALWAYS}"
    sampler = workflow["3"]["inputs"]
    sampler.update(seed=random.randint(0, 2**32 - 1), steps=p["steps"], cfg=p["cfg"],
                   sampler_name=p["sampler"], scheduler=p["scheduler"])
    if p["lora"]:
        # Slot a LoRA loader between the model and everything that uses it.
        workflow["10"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": p["lora"], "strength_model": p["lora_strength"],
            "strength_clip": p["lora_strength"], "model": ["4", 0], "clip": ["4", 1]}}
        sampler["model"] = ["10", 0]
        workflow["6"]["inputs"]["clip"] = ["10", 1]
        workflow["7"]["inputs"]["clip"] = ["10", 1]
    return workflow


def paint(description, style="", workflow_file="workflows/scene_api.json"):
    """Make one picture and return it as PNG bytes."""
    # (words:1.3) tells Stable Diffusion these words matter most, so the scene
    # itself wins over the hero and the style words.
    parts = [PRESETS[PRESET]["trigger"], f"({description}:1.3)", style, ART_STYLES[ART_STYLE]["words"]]
    prompt = ", ".join(part for part in parts if part)
    workflow = build_workflow(prompt, workflow_file)

    url = server()
    job = requests.post(f"{url}/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}).json()
    if "prompt_id" not in job:  # ComfyUI rejected it, e.g. a model file name is wrong
        raise RuntimeError(f"ComfyUI refused the job (check the model file names for preset "
                           f"'{PRESET}' in art.py): {job.get('error', job)}")
    job_id = job["prompt_id"]
    for _ in range(300):  # wait up to 5 minutes
        history = requests.get(f"{url}/history/{job_id}").json()
        if job_id in history:
            break
        time.sleep(1)
    else:
        raise TimeoutError("ComfyUI took too long to make the picture.")

    for output in history[job_id]["outputs"].values():
        for img in output.get("images", []):
            params = {"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]}
            return requests.get(f"{url}/view", params=params).content
    raise RuntimeError("ComfyUI finished but returned no image.")


if __name__ == "__main__":
    # Style test: run  python art.py  and compare media/style_test.png after each change.
    print("Using preset:", PRESET, "| art style:", ART_STYLE)
    png = paint("a hooded woman sliding an old map across a wooden tavern table",
                "a young bard in a red feathered hat, warm candlelight")
    open("media/style_test.png", "wb").write(png)
    print("Saved media/style_test.png")
