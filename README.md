# latent-diffusion-cvpr2022-reproduction

本项目用于复现 Rombach 等人在 CVPR 2022 论文 **High-Resolution Image Synthesis with Latent Diffusion Models** 中的核心结论：把扩散模型从像素空间迁移到感知压缩后的 latent 空间，可以在较低计算成本下保持较好的视觉质量。

当前复现包含两部分：

- **first-stage compression tradeoff**：使用官方预训练 KL autoencoder，对比 `kl-f4`、`kl-f8`、`kl-f16` 的重建质量、latent 表示大小和吞吐量。
- **LDM DDIM sampling tradeoff**：使用官方 CelebA-HQ 256 无条件 LDM checkpoint，对比 DDIM steps 为 10、20、50、100、200 时的采样速度和生成样本。

这不是整篇论文的大规模训练复现；它是基于官方源码和官方 checkpoint 的小规模、端到端数值验证。

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

重要内容包括：

- `src/README.md`：官方项目说明，包含模型 zoo、下载链接和训练/采样命令。
- `src/environment.yaml`：官方推荐环境配置。
- `src/ldm/`：Latent Diffusion Models 的核心 Python 模块。
- `src/configs/`：autoencoder 和 latent diffusion 的官方配置文件。
- `src/models/first_stage_models/`：first-stage autoencoder 配置与 checkpoint 放置位置。
- `src/models/ldm/celeba256/`：CelebA-HQ 256 无条件 LDM 配置与 checkpoint 放置位置。
- `src/scripts/`：官方采样、下载、inpainting、text-to-image 等脚本。
- `src/assets/`：官方示例图片；autoencoder 快速复现实验默认使用这里的 12 张图片。

本次实验本地下载并解压了以下官方 checkpoint：

```text
src/models/first_stage_models/kl-f4/model.ckpt
src/models/first_stage_models/kl-f8/model.ckpt
src/models/first_stage_models/kl-f16/model.ckpt
src/models/ldm/celeba256/model.ckpt
```

这些 checkpoint 及其 zip 包体积较大，已通过 `.gitignore` 排除，不会上传到 GitHub。

### `reproduction/`

复现代码和数值结果目录。

- `reproduction/run_autoencoder_tradeoff.py`：加载官方 KL autoencoder，重建同一批图像，计算 MSE、PSNR、SSIM、latent 表示比例和吞吐量，并生成报告图片。
- `reproduction/run_ldm_sampling_benchmark.py`：加载官方 CelebA-HQ 256 LDM checkpoint，比较不同 DDIM steps 下的采样耗时、吞吐量和生成样本。
- `reproduction/README.md`：复现实验运行说明。
- `reproduction/results/autoencoder_metrics.csv`：autoencoder 压缩--重建实验结果。
- `reproduction/results/autoencoder_metrics.json`：autoencoder 实验 JSON 结果。
- `reproduction/results/ldm_sampling_benchmark.csv`：LDM DDIM 采样速度实验结果。
- `reproduction/results/ldm_sampling_benchmark.json`：LDM DDIM 实验 JSON 结果。
- `reproduction/third_party/taming_stub/`：最小兼容 stub。官方 `ldm.models.autoencoder` 在导入时会引用 `taming` 的 VQ 模块；本次只运行 KL autoencoder，因此用该 stub 满足导入依赖。
- `reproduction/samples/`：采样脚本生成的单张样本输出目录，已在 `.gitignore` 中排除。

autoencoder 复现实验命令：

```bash
cd /data/chengjin/latent-diffusion-cvpr2022-reproduction
conda run -n DDCSR python reproduction/run_autoencoder_tradeoff.py \
  --device cuda:1 --batch-size 4 --num-images 12
```

LDM DDIM 采样实验命令：

```bash
cd /data/chengjin/latent-diffusion-cvpr2022-reproduction
CUDA_VISIBLE_DEVICES=1 conda run -n DDCSR \
  python reproduction/run_ldm_sampling_benchmark.py \
  --steps 10 20 50 100 200 --batch-size 4 --eta 0.0 --device cuda
```

这里使用 `CUDA_VISIBLE_DEVICES=1` 是因为官方 DDIM sampler 内部默认使用 `cuda` 设备；该写法把物理 GPU 1 映射为进程内的 `cuda:0`。
采样脚本会在推理时切换到 checkpoint 中的 EMA 权重。

### `reports/`

中文复现报告和图表目录。

- `reports/ldm_reproduction_report.tex`：中文 LaTeX 报告源文件。
- `reports/ldm_reproduction_report.pdf`：最终编译得到的复现报告 PDF。
- `reports/FIG/metrics_tradeoff.png`：PSNR 与 latent 表示比例的权衡图。
- `reports/FIG/reconstruction_grid.png`：输入图像与 `kl-f4`、`kl-f8`、`kl-f16` 重建图的对比拼图。
- `reports/FIG/ssim_tradeoff.png`：SSIM 随压缩率变化的图。
- `reports/FIG/ddim_sampling_runtime.png`：DDIM steps 与采样速度关系图。
- `reports/FIG/ddim_steps_samples.png`：同一初始噪声下不同 DDIM steps 的 CelebA-HQ 生成样本对比。

报告编译命令：

```bash
cd /data/chengjin/latent-diffusion-cvpr2022-reproduction/reports
tectonic ldm_reproduction_report.tex
```

## 本次复现结果摘要

autoencoder 实验在 12 张官方示例图上得到：

| 模型 | latent/像素 | PSNR | SSIM | 吞吐量 |
|---|---:|---:|---:|---:|
| `kl-f4` | 6.25% | 32.07 | 0.995 | 14.58 images/s |
| `kl-f8` | 2.08% | 28.33 | 0.985 | 65.95 images/s |
| `kl-f16` | 2.08% | 27.48 | 0.982 | 90.26 images/s |

LDM DDIM 采样实验每组生成 4 张图，计时包含 latent denoising 和 first-stage decode：

| DDIM steps | 总时间(s) | 单图时间(s/img) | 吞吐量(img/s) | decode 时间(s) |
|---:|---:|---:|---:|---:|
| 10 | 0.282 | 0.071 | 14.17 | 0.041 |
| 20 | 0.474 | 0.119 | 8.44 | 0.041 |
| 50 | 1.122 | 0.280 | 3.57 | 0.041 |
| 100 | 2.274 | 0.568 | 1.76 | 0.041 |
| 200 | 4.389 | 1.097 | 0.91 | 0.041 |

结论与论文主张一致：

- `f=4` 的 mild compression 保真度最好，同时已经把表示规模压缩到像素空间的 6.25%。
- 更大的下采样因子会进一步提高 autoencoder 前向吞吐量、降低 latent 空间表示成本，但会牺牲纹理和边缘细节。
- DDIM steps 越多，单图采样时间近似线性上升；步数较少时速度快，但视觉结构和局部纹理更不稳定。
- 这支持论文采用 latent space 训练 diffusion model 的核心理由：保留主要感知内容，同时显著降低后续生成模型的计算成本，并允许通过采样步数调节速度--质量折中。

## 复现范围和限制

已完成：

- 拉取官方源码。
- 下载官方 `kl-f4`、`kl-f8`、`kl-f16` autoencoder checkpoint。
- 下载官方 CelebA-HQ 256 LDM checkpoint。
- 编写并运行 autoencoder 压缩--重建实验。
- 编写并运行 LDM DDIM 采样步数基准实验。
- 生成 CSV/JSON 数值结果和报告图片。
- 编写中文 LaTeX 复现报告并编译为 PDF。
- 更新 `.gitignore`，排除 checkpoint、zip、缓存和采样中间图。

未完成或未纳入本次范围：

- 未训练完整 latent diffusion prior。
- 未复现 ImageNet class-conditional 的 2M steps 训练。
- 未复现 text-to-image 的 LAION-400M 大规模训练。
- 未计算论文级 ImageNet-val reconstruction rFID。
- 未对 CelebA-HQ 生成样本计算 5k/50k FID；当前 DDIM 实验主要验证采样速度趋势并给出定性样本。

后续若要做更严格的论文级复现，建议使用 ImageNet-val、OpenImages validation 或 CelebA-HQ 验证集，扩大到数千张图像，补充 rFID/LPIPS/FID，并进一步复现 `LDM-4` 或 `LDM-8` 的完整采样质量曲线。
