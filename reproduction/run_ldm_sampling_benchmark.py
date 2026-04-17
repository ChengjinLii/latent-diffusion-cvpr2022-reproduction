#!/usr/bin/env python3
"""Benchmark DDIM sampling with an official pretrained LDM.

This experiment complements the first-stage autoencoder reproduction by testing
the paper's main generative component: sampling in latent space with different
DDIM step counts. It uses the official unconditional CelebA-HQ 256 checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
STUB_DIR = ROOT / "reproduction" / "third_party" / "taming_stub"
RESULTS_DIR = ROOT / "reproduction" / "results"
FIG_DIR = ROOT / "reports" / "FIG"
SAMPLE_DIR = ROOT / "reproduction" / "samples" / "celeba_ddim"

sys.path.insert(0, str(STUB_DIR))
sys.path.insert(0, str(SRC_DIR))

from ldm.models.diffusion.ddim import DDIMSampler  # noqa: E402
from ldm.util import instantiate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(SRC_DIR / "models" / "ldm" / "celeba256"))
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 50, 100, 200])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-steps", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "model.ckpt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}. Download and unzip the official CelebA-HQ LDM first."
        )
    config = OmegaConf.load(str(config_path))
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded {ckpt_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval().to(device)
    return model


def to_uint_grid(tensor: torch.Tensor, nrow: int) -> torch.Tensor:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    return make_grid(tensor, nrow=nrow, padding=4, pad_value=1.0)


def add_row_labels(image_path: Path, labels: List[str], row_height: int) -> None:
    image = Image.open(image_path).convert("RGB")
    label_width = 115
    canvas = Image.new("RGB", (image.width + label_width, image.height), "white")
    canvas.paste(image, (label_width, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for idx, label in enumerate(labels):
        y = int(idx * row_height + row_height / 2 - 10)
        draw.text((12, y), label, fill=(0, 0, 0), font=font)
    canvas.save(image_path)


def sample_once(
    model: torch.nn.Module,
    sampler: DDIMSampler,
    steps: int,
    batch_size: int,
    eta: float,
    x_t: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    shape = (model.channels, model.image_size, model.image_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        samples_z, _ = sampler.sample(
            S=steps,
            batch_size=batch_size,
            shape=shape,
            conditioning=None,
            eta=eta,
            verbose=False,
            x_T=x_t,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    denoise_sec = time.perf_counter() - start

    start_decode = time.perf_counter()
    with torch.no_grad():
        decoded = model.decode_first_stage(samples_z)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_sec = time.perf_counter() - start_decode
    return {
        "samples": decoded,
        "denoise_sec": denoise_sec,
        "decode_sec": decode_sec,
        "total_sec": denoise_sec + decode_sec,
    }


def write_metrics(rows: List[Dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ldm_sampling_benchmark.csv"
    json_path = RESULTS_DIR / "ldm_sampling_benchmark.json"
    fieldnames = [k for k in rows[0].keys() if k != "sample_path"]
    fieldnames.append("sample_path")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_runtime_plot(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [int(r["ddim_steps"]) for r in rows]
    sec = [float(r["sec_per_image"]) for r in rows]
    ips = [float(r["images_per_second"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax1.plot(xs, sec, marker="o", linewidth=2, label="seconds / image")
    ax1.set_xlabel("DDIM steps")
    ax1.set_ylabel("Seconds per image")
    ax1.grid(True, linewidth=0.4, alpha=0.4)
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    ax2.plot(xs, ips, marker="s", linewidth=2, color="#2a9d8f", label="images / second")
    ax2.set_ylabel("Images per second")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ddim_sampling_runtime.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(model_dir, device)
    sampler = DDIMSampler(model)
    shape = (args.batch_size, model.channels, model.image_size, model.image_size)
    torch.manual_seed(args.seed)
    x_t = torch.randn(shape, device=device)

    ema_context = model.ema_scope("Sampling") if getattr(model, "use_ema", False) else nullcontext()
    with ema_context:
        if args.warmup_steps > 0:
            print(f"Running {args.warmup_steps}-step warmup batch.")
            _ = sample_once(model, sampler, args.warmup_steps, args.batch_size, args.eta, x_t, device)

        rows: List[Dict[str, object]] = []
        sample_rows = []
        for steps in args.steps:
            result = sample_once(model, sampler, steps, args.batch_size, args.eta, x_t, device)
            samples = result["samples"]
            sample_path = SAMPLE_DIR / f"celeba_ddim_{steps:03d}.png"
            save_image(((samples.detach().cpu().clamp(-1, 1) + 1) / 2), str(sample_path), nrow=args.batch_size)
            sample_rows.append(samples.detach().cpu())
            total_sec = float(result["total_sec"])
            row = {
                "model": "celeba256",
                "ddim_steps": steps,
                "batch_size": args.batch_size,
                "eta": args.eta,
                "seed": args.seed,
                "latent_shape": f"{model.channels}x{model.image_size}x{model.image_size}",
                "denoise_sec": float(result["denoise_sec"]),
                "decode_sec": float(result["decode_sec"]),
                "total_sec": total_sec,
                "sec_per_image": total_sec / args.batch_size,
                "images_per_second": args.batch_size / total_sec,
                "sample_path": str(sample_path.relative_to(ROOT)),
            }
            rows.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

    write_metrics(rows)
    grid = to_uint_grid(torch.cat(sample_rows, dim=0), nrow=args.batch_size)
    grid_path = FIG_DIR / "ddim_steps_samples.png"
    save_image(grid, str(grid_path))
    row_height = grid.shape[1] // len(args.steps)
    add_row_labels(grid_path, [f"{s} steps" for s in args.steps], row_height)
    save_runtime_plot(rows)
    print(f"Wrote {RESULTS_DIR / 'ldm_sampling_benchmark.csv'}")
    print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
