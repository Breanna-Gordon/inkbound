from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps

MODEL_PATH = Path("models/sketch_cnn.pt")
SIZE = 28


class SketchCNN(nn.Module):
    """Three convolution blocks, then two fully connected layers.

    28x28 -> 14x14 -> 7x7 -> 3x3 feature maps, then a guess for every class.
    """

    def __init__(self, n_classes):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def frame(gray, thicken):
    """Crop a drawing (white strokes on black) to its ink, centre it in a square
    with a small margin, and shrink it to 28x28. Training drawings and player
    sketches both go through this, so the CNN always sees drawings framed the same way."""
    box = gray.point(lambda v: 255 if v > 40 else 0).getbbox()
    if box is None:
        return np.zeros((SIZE, SIZE), dtype=np.float32)
    gray = gray.crop(box)
    w, h = gray.size
    side = int(max(w, h) * 1.15) + 2
    square = Image.new("L", (side, side), 0)
    square.paste(gray, ((side - w) // 2, (side - h) // 2))
    if thicken:  # player lines are thin compared with the drawing size: widen them
        square = square.filter(ImageFilter.MaxFilter(max(3, int(side / SIZE * 1.5) | 1)))
    small = square.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(small, dtype=np.float32) / 255.0


def preprocess(image):
    """A player sketch (dark lines on a light background) as a framed 28x28 array."""
    return frame(ImageOps.invert(image.convert("L")), thicken=True)


def prepare_rows(rows):
    """Quick, Draw! rows (N x 784, white on black) framed the same way, as uint8."""
    out = np.empty((len(rows), SIZE * SIZE), dtype=np.uint8)
    for i, row in enumerate(rows):
        img = Image.fromarray(row.reshape(SIZE, SIZE).astype(np.uint8), "L")
        out[i] = (frame(img, thicken=False) * 255).round().astype(np.uint8).reshape(-1)
    return out


def device():
    """Use the GPU if PyTorch can see one, otherwise the CPU."""
    if torch.cuda.is_available():  # NVIDIA, or AMD through ROCm
        return torch.device("cuda")
    try:
        import torch_directml  # AMD on Windows (optional: pip install torch-directml)
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def save(model, labels, path=MODEL_PATH, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, **info}, path)


def load(path=MODEL_PATH):
    """Load the trained model, or return (None, None) if it hasn't been trained yet."""
    if not Path(path).exists():
        return None, None
    data = torch.load(path, map_location="cpu")
    model = SketchCNN(len(data["labels"]))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data["labels"]


def predict(model, labels, image=None, row=None):
    """Return {label: probability} for every class the CNN knows.
    Give either a player sketch (image) or an already framed Quick, Draw! row."""
    x = preprocess(image) if row is None else row.reshape(SIZE, SIZE).astype(np.float32) / 255.0
    with torch.no_grad():
        probs = model(torch.from_numpy(x)[None, None]).softmax(dim=1)[0]
    return {label: p.item() for label, p in zip(labels, probs)}
