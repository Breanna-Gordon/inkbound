import json, time, uuid, requests

COMFY = "http://127.0.0.1:8188"

def paint(prompt, workflow_file="workflows/scene_api.json", prompt_node="6"):
    workflow = json.load(open(workflow_file))
    workflow[prompt_node]["inputs"]["text"] = prompt
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