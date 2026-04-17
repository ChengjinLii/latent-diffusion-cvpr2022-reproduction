# LDM CVPR 2022 Core Reproduction

This directory contains the executable reproduction used for the Chinese report
in `../reports`.

The implemented experiment focuses on the paper's core first-stage conclusion:
moderate latent compression preserves image fidelity while substantially
reducing the representation size. The script evaluates the official CompVis
KL autoencoders (`kl-f4`, `kl-f8`, `kl-f16`) on a small natural-image test set
and records reconstruction quality and runtime.

Run from the repository root:

```bash
conda run -n DDCSR python reproduction/run_autoencoder_tradeoff.py
```

Main outputs:

- `reproduction/results/autoencoder_metrics.csv`
- `reproduction/results/autoencoder_metrics.json`
- `reports/FIG/reconstruction_grid.png`
- `reports/FIG/metrics_tradeoff.png`

