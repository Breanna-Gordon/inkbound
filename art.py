import json, random, time, uuid, requests

COMFY = "http://127.0.0.1:8188"
STYLE = "storybook watercolour illustration, warm golden colours, soft light, detailed fantasy art"
NEGATIVE = "black and white, monochrome, pencil sketch, blurry, text, watermark, ugly"

def paint(description, workflow_file="workflows/scene_api.json"):
    workflow = json.load(open(workflow_file))
    workflow["6"]["inputs"]["text"] = f"{description}, {STYLE}"    # what to draw
    workflow["7"]["inputs"]["text"] = NEGATIVE                      # what to avoid
    workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)  # new picture each time
    job = requests.post(f"{COMFY}/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}).json()
    job_id = job["prompt_id"]
    while True:  # wait until ComfyUI finishes
        history = requests.get(f"{COMFY}/history/{job_id}").json()
        if job_id in history:
            break
        time.sleep(1)
    for output in history[job_id]["outputs"].values():
        for img in output.get("images", []):
            params = {"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]}
            return requests.get(f"{COMFY}/view", params=params).content  # the PNG bytes