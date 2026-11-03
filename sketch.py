import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

LABELS = ["rope", "sword", "bridge", "wings", "music notes", "key", "fire", "shield"]

def recognise(image):
    texts = [f"a simple sketch of {label}" for label in LABELS]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    top = probs.topk(3)
    return [(LABELS[i], round(p.item() * 100)) for p, i in zip(top.values, top.indices)]

if __name__ == "__main__":
    print(recognise(Image.open("test_sketch.png").convert("RGB")))