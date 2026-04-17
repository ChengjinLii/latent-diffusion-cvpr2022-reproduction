#!/usr/bin/env python3
"""Reproduce the LDM paper's first-stage compression tradeoff.

The script evaluates official CompVis KL autoencoders with downsampling factors
f=4, f=8 and f=16 on a small natural-image subset. It writes numerical metrics
and figures used by the LaTeX report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
STUB_DIR = ROOT / "reproduction" / "third_party" / "taming_stub"
RESULTS_DIR = ROOT / "reproduction" / "results"
FIG_DIR = ROOT / "reports" / "FIG"
DATA_DIR = ROOT / "reproduction" / "data"

sys.path.insert(0, str(STUB_DIR))
sys.path.insert(0, str(SRC_DIR))

from ldm.util import instantiate_from_config  # noqa: E402


PAPER_KL_RESULTS = {
    "kl-f4": {"paper_rfid": 0.27, "paper_psnr": 27.53, "paper_psim": 0.55},
    "kl-f8": {"paper_rfid": 0.90, "paper_psnr": 24.19, "paper_psim": 1.02},
    "kl-f16": {"paper_rfid": 0.87, "paper_psnr": 24.08, "paper_psim": 1.07},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["kl-f4", "kl-f8", "kl-f16"])
    parser.add_argument("--dataset", choices=["assets", "flowers102"], default="assets")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260417)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_asset_dataset(num_images: int, image_size: int) -> torch.Tensor:
    asset_paths = sorted(
        [
            p
            for p in (SRC_DIR / "assets").iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and p.name
            not in {
                "modelfigure.png",
                "txt2img-preview.png",
                "txt2img-convsample.png",
            }
        ]
    )
    if not asset_paths:
        raise FileNotFoundError("No usable images found in src/assets")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    images = []
    for path in asset_paths[:num_images]:
        with Image.open(path) as img:
            images.append(transform(img.convert("RGB")))
    return torch.stack(images, dim=0)


def load_flowers102_dataset(num_images: int, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.Flowers102(
        root=str(DATA_DIR),
        split="test",
        transform=transform,
        download=True,
    )
    count = min(num_images, len(dataset))
    images = [dataset[i][0] for i in range(count)]
    return torch.stack(images, dim=0)


def load_dataset(name: str, num_images: int, image_size: int) -> torch.Tensor:
    if name == "assets":
        return load_asset_dataset(num_images, image_size)
    if name == "flowers102":
        return load_flowers102_dataset(num_images, image_size)
    raise ValueError(f"Unsupported dataset: {name}")


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model_dir = SRC_DIR / "models" / "first_stage_models" / model_name
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "model.ckpt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}. Download and unzip the official first-stage model first."
        )
    config = OmegaConf.load(str(config_path))
    config.model.params.ckpt_path = str(ckpt_path)
    config.model.params.lossconfig = {"target": "torch.nn.Identity"}
    model = instantiate_from_config(config.model)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def batches(x: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, x.shape[0], batch_size):
        yield x[start : start + batch_size]


def psnr_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_single(x: np.ndarray, y: np.ndarray) -> float:
    # Global SSIM variant, sufficient for a lightweight reproduction.
    c1 = 0.01**2
    c2 = 0.03**2
    ux, uy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cxy = float(((x - ux) * (y - uy)).mean())
    return ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))


def ssim_batch(x: torch.Tensor, y: torch.Tensor) -> List[float]:
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    vals = []
    for xi, yi in zip(x_np, y_np):
        vals.append(ssim_single(np.moveaxis(xi, 0, -1), np.moveaxis(yi, 0, -1)))
    return vals


def model_factor(model_name: str) -> int:
    return int(model_name.split("-f")[-1])


def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor]:
    recon_chunks = []
    psnr_vals: List[float] = []
    mse_vals: List[float] = []
    ssim_vals: List[float] = []
    latent_shapes = []
    timings = []
    n_seen = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_all = time.perf_counter()

    with torch.inference_mode():
        for batch_cpu in batches(images, batch_size):
            batch = batch_cpu.to(device)
            batch_l = batch * 2.0 - 1.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            posterior = model.encode(batch_l)
            latent = posterior.mode()
            recon_l = model.decode(latent)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            recon = ((recon_l.clamp(-1, 1) + 1.0) / 2.0).cpu()
            recon_chunks.append(recon)
            timings.append(elapsed)
            latent_shapes.append(tuple(latent.shape[1:]))
            psnr_vals.extend(psnr_per_image(batch_cpu, recon).tolist())
            mse_vals.extend(F.mse_loss(batch_cpu, recon, reduction="none").flatten(1).mean(dim=1).tolist())
            ssim_vals.extend(ssim_batch(batch_cpu, recon))
            n_seen += batch_cpu.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_elapsed = time.perf_counter() - start_all

    reconstructions = torch.cat(recon_chunks, dim=0)
    c, h, w = latent_shapes[0]
    factor = model_factor(model_name)
    metrics = {
        "model": model_name,
        "downsample_factor": factor,
        "latent_channels": c,
        "latent_height": h,
        "latent_width": w,
        "latent_values_per_image": c * h * w,
        "pixel_values_per_image": 3 * images.shape[-2] * images.shape[-1],
        "representation_ratio": (c * h * w) / float(3 * images.shape[-2] * images.shape[-1]),
        "mse_mean": float(np.mean(mse_vals)),
        "mse_std": float(np.std(mse_vals, ddof=1)),
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals, ddof=1)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals, ddof=1)),
        "batch_runtime_sec_mean": float(np.mean(timings)),
        "images_per_second": float(n_seen / total_elapsed),
        "num_images": int(n_seen),
    }
    metrics.update(PAPER_KL_RESULTS.get(model_name, {}))
    return metrics, reconstructions


def write_metrics(rows: List[Dict[str, float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "autoencoder_metrics.csv"
    json_path = RESULTS_DIR / "autoencoder_metrics.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_reconstruction_grid(images: torch.Tensor, recons_by_model: Dict[str, torch.Tensor]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = min(6, images.shape[0])
    rows = [images[:n]]
    labels = ["input"]
    for name, rec in recons_by_model.items():
        rows.append(rec[:n])
        labels.append(name)
    grid_tensor = torch.cat(rows, dim=0)
    grid = make_grid(grid_tensor, nrow=n, padding=4, pad_value=1.0)
    tmp_path = FIG_DIR / "reconstruction_grid_raw.png"
    out_path = FIG_DIR / "reconstruction_grid.png"
    save_image(grid, str(tmp_path))

    image = Image.open(tmp_path).convert("RGB")
    label_h = 34
    canvas = Image.new("RGB", (image.width + 120, image.height), "white")
    canvas.paste(image, (120, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    cell_h = image.height / len(labels)
    for idx, label in enumerate(labels):
        y = int(idx * cell_h + cell_h / 2 - label_h / 2)
        draw.text((12, y), label, fill=(0, 0, 0), font=font)
    canvas.save(out_path)
    tmp_path.unlink(missing_ok=True)


def save_metric_plot(rows: List[Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["downsample_factor"])
    x = [r["downsample_factor"] for r in rows]
    psnr = [r["psnr_mean"] for r in rows]
    ssim = [r["ssim_mean"] for r in rows]
    ratio = [100.0 * r["representation_ratio"] for r in rows]
    paper_psnr = [r.get("paper_psnr", math.nan) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax1.plot(x, psnr, marker="o", linewidth=2, label="PSNR (ours)")
    ax1.plot(x, paper_psnr, marker="s", linewidth=2, linestyle="--", label="PSNR (paper, KL)")
    ax1.set_xlabel("Downsampling factor f")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_xticks(x)
    ax1.grid(True, linewidth=0.4, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, ratio, marker="^", linewidth=2, color="#2a9d8f", label="latent / pixel (%)")
    ax2.set_ylabel("Latent representation size (%)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_tradeoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(x, ssim, marker="o", linewidth=2, color="#264653")
    ax.set_xlabel("Downsampling factor f")
    ax.set_ylabel("SSIM (ours)")
    ax.set_xticks(x)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ssim_tradeoff.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    images = load_dataset(args.dataset, args.num_images, args.image_size)

    rows: List[Dict[str, float]] = []
    recons_by_model: Dict[str, torch.Tensor] = {}
    for model_name in args.models:
        model = load_model(model_name, device)
        metrics, recon = evaluate_model(model_name, model, images, args.batch_size, device)
        rows.append(metrics)
        recons_by_model[model_name] = recon
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_metrics(rows)
    save_reconstruction_grid(images, recons_by_model)
    save_metric_plot(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
