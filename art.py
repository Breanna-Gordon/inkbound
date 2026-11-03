import json
import random
import time
import uuid
 
import requests
 
# ---------------------------------------------------------------------------
# STYLE SETTINGS
# ---------------------------------------------------------------------------
# 1. The base model (a file in ComfyUI's models/checkpoints folder).
#    Plain SD 1.5:  "v1-5-pruned-emaonly-fp16.safetensors"
#    DreamShaper 8: "DreamShaper_8_pruned.safetensors"  (more painterly, recommended)
CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"
 
# 2. Optional style LoRA (a file in ComfyUI's models/loras folder).
#    Must be made for SD 1.5. Leave as None to switch it off.
LORA = None                 # e.g. "storybook_style_sd15.safetensors"
LORA_STRENGTH = 0.8         # 0.5 = a hint of the style, 1.0 = full strength
LORA_TRIGGER = ""           # some LoRAs need a trigger word: copy it from the LoRA's page
 
# 3. Words added to every picture. Most important words go first, because
#    Stable Diffusion 1.5 only reads about 75 words.
BASE_STYLE = ("storybook illustration, hand-painted gouache and watercolour, soft edges, "
              "warm light, whimsical fairy-tale picture book")
NEGATIVE = ("photo, realistic, 3d render, black and white, monochrome, harsh lines, "
            "blurry, text, watermark, ugly, deformed, extra limbs")
 
# 4. How the picture is made. These suit DreamShaper and SD 1.5.
STEPS = 25                  # more = more detail but slower
CFG = 7                     # how strictly to follow the prompt (5-9 is normal)
SAMPLER = "dpmpp_2m"
SCHEDULER = "karras"
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
 
 
def build_workflow(prompt, workflow_file="workflows/scene_api.json"):
    """Load the saved workflow and apply the style settings to it."""
    workflow = json.load(open(workflow_file))
    workflow["4"]["inputs"]["ckpt_name"] = CHECKPOINT
    workflow["6"]["inputs"]["text"] = prompt                         # what to draw
    workflow["7"]["inputs"]["text"] = NEGATIVE                       # what to avoid
    sampler = workflow["3"]["inputs"]
    sampler.update(seed=random.randint(0, 2**32 - 1), steps=STEPS, cfg=CFG,
                   sampler_name=SAMPLER, scheduler=SCHEDULER)
    if LORA:
        # Slot a LoRA loader between the model and everything that uses it.
        workflow["10"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA, "strength_model": LORA_STRENGTH, "strength_clip": LORA_STRENGTH,
            "model": ["4", 0], "clip": ["4", 1]}}
        sampler["model"] = ["10", 0]
        workflow["6"]["inputs"]["clip"] = ["10", 1]
        workflow["7"]["inputs"]["clip"] = ["10", 1]
    return workflow
 
 
def paint(description, style="", workflow_file="workflows/scene_api.json"):
    """Make one picture and return it as PNG bytes."""
    # (words:1.3) tells Stable Diffusion these words matter most, so the scene
    # itself wins over the hero and the style words.
    parts = [LORA_TRIGGER if LORA else "", f"({description}:1.3)", style, BASE_STYLE]
    prompt = ", ".join(part for part in parts if part)
    workflow = build_workflow(prompt, workflow_file)
 
    url = server()
    job = requests.post(f"{url}/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}).json()
    if "prompt_id" not in job:  # ComfyUI rejected it, e.g. a model file name is wrong
        raise RuntimeError(f"ComfyUI refused the job: {job.get('error', job)}")
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
    png = paint("a hooded woman sliding an old map across a wooden tavern table",
                "a young bard in a red feathered hat, warm candlelight")
    open("media/style_test.png", "wb").write(png)
    print("Saved media/style_test.png")