import argparse
import ast
import csv
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import torch
import torch.nn.functional as F

import oracle_cnn

URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
DATA = Path("data/quickdraw")
RESULTS = Path("results")
RUNS = RESULTS / "runs"  # every training run keeps its own log, chart and model here
PIXELS = 28 * 28


def download_class(name, n):
    """Download the first n drawings of one class, without fetching the whole file.

    Each .npy file is a header followed by one row of 784 bytes per drawing,
    so we read the header, then stop as soon as we have n rows.
    """
    out = DATA / f"{name}.npy"
    if out.exists() and len(np.load(out, mmap_mode="r")) >= n:
        return np.load(out)[:n]
    with requests.get(URL.format(quote(name)), stream=True, timeout=60) as r:
        r.raise_for_status()
        raw = bytearray()
        need = None
        for chunk in r.iter_content(1 << 16):
            raw += chunk
            if need is None and len(raw) >= 12:
                major = raw[6]
                size_len = 2 if major == 1 else 4
                header_len = int.from_bytes(raw[8:8 + size_len], "little")
                start = 8 + size_len + header_len
                if len(raw) >= start:
                    header = ast.literal_eval(raw[8 + size_len:start].decode("latin1"))
                    total = header["shape"][0]
                    n = min(n, total)
                    need = start + n * PIXELS
            if need is not None and len(raw) >= need:
                break
    images = np.frombuffer(bytes(raw[start:start + n * PIXELS]), dtype=np.uint8).reshape(n, PIXELS)
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(out, images)
    return images


def load_data(labels, per_class):
    xs, ys = [], []
    for i, name in enumerate(labels):
        images = oracle_cnn.prepare_rows(download_class(name, per_class))
        xs.append(images)
        ys.append(np.full(len(images), i, dtype=np.int64))
        print(f"\r  data: {i + 1}/{len(labels)} classes ({name})          ", end="", flush=True)
    print()
    return np.concatenate(xs), np.concatenate(ys)


def augment(x):
    """Random small rotations, shifts and size changes, a different one per drawing."""
    n = x.shape[0]
    angle = (torch.rand(n, device=x.device) - 0.5) * 0.35          # about +/- 10 degrees
    scale = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.25      # +/- 12%
    shift = (torch.rand(n, 2, device=x.device) - 0.5) * 0.2        # +/- 10%
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def evaluate(model, x, y, dev, batch=2048):
    model.eval()
    correct = top3 = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).float().div(255).view(-1, 1, 28, 28).to(dev)
            yb = torch.from_numpy(y[i:i + batch]).to(dev)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            top3 += (logits.topk(3, 1).indices == yb[:, None]).any(1).sum().item()
    return correct / len(x), top3 / len(x)


def plot_curve(history, path=RESULTS / "training_curve.png"):
    """Save the training chart: loss and validation accuracy per epoch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for the training chart: pip install matplotlib")
        return
    e = [h[0] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(e, [h[1] for h in history], color="#8e2a1f", label="training loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(e, [h[2] * 100 for h in history], color="#1f4e6e", label="validation accuracy")
    ax2.plot(e, [h[3] * 100 for h in history], color="#1f4e6e", linestyle="--", label="top-3 accuracy")
    ax2.set_ylabel("accuracy (%)")
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Inkbound sketch classifier: training")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.as_posix()}")


def read_log(path):
    with open(path, newline="") as f:
        return [(int(r[0]), *map(float, r[1:])) for r in list(csv.reader(f))[1:]]


def plot_comparison():
    """One chart with the validation accuracy of every run, to show what each change did."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    runs = sorted((d for d in RUNS.glob("*") if (d / "training_log.csv").exists()),
                  key=lambda d: int(d.name.split("-")[0]) if d.name.split("-")[0].isdigit() else 0)
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for d in runs:
        h = read_log(d / "training_log.csv")
        line, = ax.plot([e[0] for e in h], [e[2] * 100 for e in h], label=f"{d.name} (top-1)")
        ax.plot([e[0] for e in h], [e[3] * 100 for e in h], linestyle="--", color=line.get_color(), alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Sketch classifier: every training run (dashed = top-3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUNS / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {(RUNS / 'comparison.png').as_posix()}")


def best_so_far():
    """The best validation accuracy of any earlier run (0 if this is the first)."""
    summary = RUNS / "summary.csv"
    if not summary.exists():
        return 0.0
    with open(summary, newline="") as f:
        return max((float(r["best_val_acc"]) / 100 for r in csv.DictReader(f)), default=0.0)


def record_run(row):
    """Add one line to results/runs/summary.csv: the settings and the result of a run."""
    summary = RUNS / "summary.csv"
    new = not summary.exists()
    with open(summary, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Train Inkbound's sketch classifier on Google's Quick, Draw! dataset.")
    p.add_argument("--per-class", type=int, default=3000, help="drawings per class")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--classes", default="quickdraw_categories.txt", help="file with one class per line")
    p.add_argument("--name", default="run", help="a short name for this run, e.g. more-data")
    p.add_argument("--note", default="", help="what you changed in this run, for the summary")
    p.add_argument("--plot-only", action="store_true",
                   help="just redraw the charts from the saved logs, without training")
    args = p.parse_args()
    if args.plot_only:
        plot_curve(read_log(RESULTS / "training_log.csv"))
        plot_comparison()
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / f"{len([d for d in RUNS.iterdir() if d.is_dir()]) + 1}-{args.name}"
    run_dir.mkdir()
    print(f"This run is saved in {run_dir.as_posix()}")
    began = time.time()

    labels = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
    print(f"Classes: {len(labels)}, drawings per class: {args.per_class}")
    x, y = load_data(labels, args.per_class)

    # Shuffle once with a fixed seed, so the test set is the same every run.
    order = np.random.default_rng(0).permutation(len(x))
    x, y = x[order], y[order]
    n_val = n_test = len(x) // 10
    x_test, y_test = x[:n_test], y[:n_test]
    x_val, y_val = x[n_test:n_test + n_val], y[n_test:n_test + n_val]
    x_train, y_train = x[n_test + n_val:], y[n_test + n_val:]
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "test_set.npz", x=x_test, y=y_test, labels=np.array(labels))
    print(f"Train {len(x_train)}, validation {len(x_val)}, test {len(x_test)}")

    dev = oracle_cnn.device()
    print("Training on:", dev)
    model = oracle_cnn.SketchCNN(len(labels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(x_train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)

    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss, batches = time.time(), 0.0, 0
        perm = np.random.permutation(len(x_train))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(x_train[idx]).float().div(255).view(-1, 1, 28, 28).to(dev)
            yb = torch.from_numpy(y_train[idx]).to(dev)
            loss = F.cross_entropy(model(augment(xb)), yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
            batches += 1
        val_acc, val_top3 = evaluate(model, x_val, y_val, dev)
        history.append((epoch, total_loss / batches, val_acc, val_top3))
        print(f"Epoch {epoch}/{args.epochs}: loss {total_loss / batches:.3f}, "
              f"validation accuracy {val_acc:.1%} (top-3 {val_top3:.1%}), {time.time() - start:.0f}s")
        if val_acc > best:
            best = val_acc
            oracle_cnn.save(model.cpu(), labels, run_dir / "sketch_cnn.pt", per_class=args.per_class,
                            val_acc=val_acc)
            model.to(dev)

    for folder in (RESULTS, run_dir):  # results/ always holds the latest run
        with open(folder / "training_log.csv", "w", newline="") as f:
            csv.writer(f).writerows([("epoch", "loss", "val_acc", "val_top3"), *history])
    plot_curve(history)
    plot_curve(history, run_dir / "training_curve.png")

    previous = best_so_far()
    best_top3 = max(h[3] for h in history if h[2] == best)
    record_run({"run": run_dir.name, "date": time.strftime("%Y-%m-%d %H:%M"), "per_class": args.per_class,
                "epochs": args.epochs, "batch": args.batch, "best_val_acc": f"{best * 100:.1f}",
                "best_val_top3": f"{best_top3 * 100:.1f}", "minutes": f"{(time.time() - began) / 60:.0f}",
                "note": args.note})
    plot_comparison()
    print(f"Best validation accuracy {best:.1%} (earlier best: {previous:.1%}).")
    if best > previous:
        oracle_cnn.MODEL_PATH.parent.mkdir(exist_ok=True)
        shutil.copy(run_dir / "sketch_cnn.pt", oracle_cnn.MODEL_PATH)
        print(f"This is the best run so far, so the game now uses it ({oracle_cnn.MODEL_PATH.as_posix()}).")
    else:
        print(f"An earlier run was better, so the game keeps using it. This run's model is in {run_dir.as_posix()}.")

if __name__ == "__main__":
    main()
