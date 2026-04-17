# latent-diffusion-cvpr2022-reproduction

本项目用于复现 Rombach 等人在 CVPR 2022 论文 **High-Resolution Image Synthesis with Latent Diffusion Models** 中的一个核心结论：先用感知自编码器把图像压缩到 latent 空间，可以在显著降低表示维度的同时保持较好的图像重建质量，从而为后续 latent diffusion model 降低训练和采样成本。

本次复现重点放在论文的 first-stage compression tradeoff，而不是完整训练大规模扩散模型。实验使用官方 CompVis 代码和官方预训练 KL autoencoder checkpoint，对比 `kl-f4`、`kl-f8`、`kl-f16` 三种压缩率的重建质量、latent 表示大小和推理速度。

## 目录说明

```text
.
├── README.md
├── Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf
├── src/
├── reproduction/
└── reports/
```

### `Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf`

论文原文 PDF。报告中的实验目标、论文对照数值和结论解释都来自这篇论文及官方仓库说明。

### `src/`

官方源码，来自：

```text
https://github.com/CompVis/latent-diffusion
```

本目录按普通源码文件纳入本仓库，来源快照对应官方上游 commit `a506df5756472e2ebaf9078affdde2c4f1502cd4`。

其中比较重要的内容包括：

- `src/README.md`：官方项目说明，包含模型 zoo、下载链接和训练/采样命令。
- `src/environment.yaml`：官方推荐环境配置。
- `src/ldm/`：Latent Diffusion Models 的核心 Python 模块。
- `src/configs/`：autoencoder 和 latent diffusion 的官方配置文件。
- `src/models/first_stage_models/`：first-stage autoencoder 配置与 checkpoint。
- `src/scripts/`：官方采样、下载、inpainting、text-to-image 等脚本。
- `src/assets/`：官方示例图片；本次快速复现实验默认使用这里的示例图作为输入。

本次复现实验下载并解压了三个官方 KL autoencoder checkpoint：

```text
src/models/first_stage_models/kl-f4/model.ckpt
src/models/first_stage_models/kl-f8/model.ckpt
src/models/first_stage_models/kl-f16/model.ckpt
```

### `reproduction/`

本次复现代码和数值结果目录。

- `reproduction/run_autoencoder_tradeoff.py`：主复现实验脚本。加载官方 KL autoencoder，重建同一批图像，计算 MSE、PSNR、SSIM、latent 表示大小和吞吐量，并生成报告图片。
- `reproduction/README.md`：复现实验的简要运行说明。
- `reproduction/results/autoencoder_metrics.csv`：实验结果表格，便于直接查看和导入分析。
- `reproduction/results/autoencoder_metrics.json`：同一批实验结果的 JSON 格式。
- `reproduction/third_party/taming_stub/`：一个最小兼容 stub。官方 `ldm.models.autoencoder` 在导入时会引用 `taming` 的 VQ 模块；本次只运行 KL autoencoder，不使用 VQ 模型，因此用该 stub 满足导入依赖。

复现实验命令：

```bash
cd /data/chengjin/latent-diffusion-cvpr2022-reproduction
conda run -n DDCSR python reproduction/run_autoencoder_tradeoff.py --device cuda:1 --batch-size 4 --num-images 12
```

默认实验使用 `src/assets` 中的 12 张官方示例图。脚本也保留了 `--dataset flowers102` 选项，可用于扩大自然图像样本，但本次报告采用的是默认设置。

### `reports/`

中文复现报告和图表目录。

- `reports/ldm_reproduction_report.tex`：中文 LaTeX 报告源文件。
- `reports/ldm_reproduction_report.pdf`：最终编译得到的复现报告 PDF。
- `reports/FIG/metrics_tradeoff.png`：PSNR 与 latent 表示比例的权衡图。
- `reports/FIG/reconstruction_grid.png`：输入图像与 `kl-f4`、`kl-f8`、`kl-f16` 重建图的对比拼图。
- `reports/FIG/ssim_tradeoff.png`：SSIM 随压缩率变化的图。

报告编译命令：

```bash
cd /data/chengjin/latent-diffusion-cvpr2022-reproduction/reports
tectonic ldm_reproduction_report.tex
```

## 本次复现结果摘要

本次实验在 12 张官方示例图上得到：

| 模型 | latent/像素 | PSNR | SSIM | 吞吐量 |
|---|---:|---:|---:|---:|
| `kl-f4` | 6.25% | 32.07 | 0.995 | 14.58 images/s |
| `kl-f8` | 2.08% | 28.33 | 0.985 | 65.95 images/s |
| `kl-f16` | 2.08% | 27.48 | 0.982 | 90.26 images/s |

结论与论文主张一致：

- `f=4` 的 mild compression 保真度最好，同时已经把表示规模压缩到像素空间的 6.25%。
- 更大的下采样因子会进一步提高速度、降低 latent 空间计算量，但会牺牲纹理和边缘细节。
- 这支持论文采用 latent space 训练 diffusion model 的核心理由：保留主要感知内容，同时显著降低后续生成模型的计算成本。

## 复现范围和限制

本项目完成的是 first-stage autoencoder 的数值复现，不是整篇论文的完整大规模复现。

已完成：

- 拉取官方源码。
- 下载官方 `kl-f4`、`kl-f8`、`kl-f16` autoencoder checkpoint。
- 编写并运行复现实验代码。
- 生成 CSV/JSON 数值结果。
- 生成报告图片。
- 编写中文 LaTeX 复现报告并编译为 PDF。

未完成或未纳入本次范围：

- 未训练完整 latent diffusion prior。
- 未复现 ImageNet class-conditional 的 2M steps 训练。
- 未复现 text-to-image 的 LAION-400M 大规模训练。
- 未计算论文级 ImageNet-val rFID；本次使用小样本示例图，因此数值用于验证趋势，而不是严格同分布对比。

如果后续要做更严格的论文级复现，建议下一步使用 ImageNet-val 或 OpenImages validation 子集，扩大到数千张图像，补充 rFID/LPIPS，并进一步复现 `LDM-4` 或 `LDM-8` 的 DDIM 采样 FID。
