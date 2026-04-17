# LDM CVPR 2022 Core Reproduction

本目录包含复现报告中实际运行的实验代码。当前实现覆盖论文的两个核心机制：

- first-stage autoencoder 的 latent 压缩--重建权衡；
- 官方 CelebA-HQ 256 LDM 的 DDIM steps 采样速度--质量权衡。

## 运行环境

从仓库根目录运行，默认使用已经配置好的 `DDCSR` conda 环境：

```bash
cd /data/chengjin/latent-diffusion-cvpr2022-reproduction
```

需要存在以下官方 checkpoint：

```text
src/models/first_stage_models/kl-f4/model.ckpt
src/models/first_stage_models/kl-f8/model.ckpt
src/models/first_stage_models/kl-f16/model.ckpt
src/models/ldm/celeba256/model.ckpt
```

checkpoint 和 zip 包不纳入 Git 跟踪。

## 实验 1：Autoencoder 压缩--重建权衡

运行命令：

```bash
conda run -n DDCSR python reproduction/run_autoencoder_tradeoff.py \
  --device cuda:1 --batch-size 4 --num-images 12
```

脚本会加载官方 `kl-f4`、`kl-f8`、`kl-f16` KL autoencoder，在 `src/assets` 的 12 张官方示例图上计算：

- MSE
- PSNR
- SSIM
- latent 表示大小
- latent/像素比例
- images/s

主要输出：

- `reproduction/results/autoencoder_metrics.csv`
- `reproduction/results/autoencoder_metrics.json`
- `reports/FIG/reconstruction_grid.png`
- `reports/FIG/metrics_tradeoff.png`
- `reports/FIG/ssim_tradeoff.png`

## 实验 2：LDM DDIM 采样步数基准

运行命令：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n DDCSR \
  python reproduction/run_ldm_sampling_benchmark.py \
  --steps 10 20 50 100 200 --batch-size 4 --eta 0.0 --device cuda
```

脚本会加载官方 CelebA-HQ 256 无条件 LDM checkpoint，在推理时切换到 EMA 权重，固定随机种子和初始噪声，分别用 10、20、50、100、200 个 DDIM steps 生成样本，并统计：

- latent denoising 时间
- first-stage decode 时间
- 总采样时间
- 单图时间
- images/s

主要输出：

- `reproduction/results/ldm_sampling_benchmark.csv`
- `reproduction/results/ldm_sampling_benchmark.json`
- `reports/FIG/ddim_sampling_runtime.png`
- `reports/FIG/ddim_steps_samples.png`
- `reproduction/samples/celeba_ddim/`，该目录保存单组样本图，但不上传 GitHub

## 当前结论

autoencoder 实验验证了论文中 first-stage compression 的核心趋势：`kl-f4` 的重建质量最好，`kl-f8` 和 `kl-f16` 使用更小 latent 表示并获得更高前向吞吐量，但重建细节下降。

DDIM 采样实验验证了另一个直接可复现的核心趋势：在同一官方 LDM checkpoint 上，采样时间随 DDIM steps 近似线性上升。10 steps 单图约 0.071 s，200 steps 单图约 1.097 s；更多 steps 通常带来更充分的去噪过程，但计算成本同步增加。
