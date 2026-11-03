# PHYAE: Short-Packet Communications over Fading ISI Channels

[![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/) [![W&B](https://img.shields.io/badge/logging-W%26B-ffbe00)](https://wandb.ai/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental reproduction of selected ideas and results from:

> **Learning Joint Detection, Equalization and Decoding for Short-Packet Communications**<br>
> Sebastian Dörner, Jannis Clausius, Sebastian Cammerer, Stephan ten Brink, 2022.<br>
> [arXiv:2207.05699](https://arxiv.org/abs/2207.05699)

This repository uses modified channel, receiver, and training settings; it is not an exact reproduction of the paper's experiments.

## Contents

- [1. Overview](#1-overview)
- [2. Installation](#2-installation)
- [3. Repository Structure](#3-repository-structure)
- [4. Evaluating Models and Baselines](#4-evaluating-models-and-baselines)
- [5. Training from Scratch](#5-training-from-scratch)
- [6. License](#6-license)

## 1. Overview

- **Encoder:** serial outer/inner CNNs with straight-through binarization and an interleaver, mapping 64 information bits to 64 complex symbols.
- **Channel:** random five-tap Proakis-C-weighted complex fading, integer timing shifts, raised-cosine filtering with fractional sampling offsets, and AWGN.
- **Receiver:** six iterations of CNN equalization and decoding with separate weights and extrinsic-information exchange.
- **References:** TensorFlow/Sionna implementations of coded modulation with BCJR equalization and Zadoff–Chu preambles, under perfect or estimated synchronization/CSI.

### Differences from the paper

The active autoencoder directly decodes an observation window. The separate detector/no-message classifier is inactive, and training uses BCE rather than the paper's joint BCE + detection loss. An even-parity bit is enabled, with a least-confident-bit correction during inference. Tap scaling, timing windows, and training schedules also differ. Detection/FAR experiments and over-the-air evaluation are not reproduced.

### Results

Runs report BER and BLER versus SNR through the console and W&B. Comparisons with the paper require matching the channel, timing, normalization, and SNR conventions. No pretrained checkpoints are bundled.

## 2. Installation

Use Python 3.10+ and a CUDA-capable NVIDIA GPU; the current trainer selects a CUDA device directly. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy wandb
wandb login
```

Use a PyTorch build compatible with your CUDA environment. Baselines additionally require TensorFlow, a TensorFlow-based Sionna version exposing `sionna.phy`, and `loguru`. Dependencies are not pinned, and no lockfile is provided.

## 3. Repository Structure

```text
src/
├── models/
│   ├── SerialTurbo_AutoEncoder_key2.py     # Active encoder and iterative receiver
│   └── components/                        # CNN block and straight-through estimator
├── channel/
│   ├── Proakis_type_C_channel_new_debug.py # Fading, timing shifts, and noise
│   └── pulse_shaping_pytorch.py            # Raised-cosine filtering
├── train/
│   ├── train_PHY_AE.py                     # Entry point and configuration
│   └── trainer.py                          # Training, validation, and checkpoints
├── inference/inference_utils_TurboAE.py     # BER/BLER evaluation and parity correction
├── features/build_pytorch_dataset2.py      # Random information-bit dataset
└── baseline/                               # Sionna coded-modulation/BCJR references
jobscript-*.sh                              # Site-specific Slurm launch examples
LICENSE
```

## 4. Evaluating Models and Baselines

Settings live in the `config` dictionary at the bottom of [train_PHY_AE.py](src/train/train_PHY_AE.py). To evaluate your own checkpoint, set `pretrained` to its `.pth` path and `epochs=0` to skip training. Keep the model and interleaver settings consistent with the checkpoint, then run from the repository root:

```bash
python src/train/train_PHY_AE.py
```

Evaluation uses one million random packets per SNR point, from 0 to 20 dB in 2 dB steps by default. The packet count is set in `make()`; the SNR bounds are in `config`.

For the conventional references:

```bash
# Perfect synchronization and channel state information; LDPC by default
python -m src.baseline.baseline_Dorner_ldpc_bcjr_zadoff_perfSync_perfCSI

# Estimated synchronization/CSI with pulse shaping and interference
python -m src.baseline.baseline_Dorner_ldpc_bcjr_Imperf_ISI_paper_matched_filter
```

Edit each script's simulation settings before running; both currently evaluate only 12 dB. Defaults use 64 information bits, a 20-symbol preamble, and 44 QPSK payload symbols. The perfect-CSI script also supports Polar codes. Baseline assumptions differ from the autoencoder's, so align them before comparing curves.

## 5. Training from Scratch

### 5.1 Running

Leave `pretrained=None`, set a positive `epochs` value, and run:

```bash
python src/train/train_PHY_AE.py
```

For multiple GPUs, set `use_ddp=True` in `config`:

```bash
torchrun --standalone --nproc_per_node=2 src/train/train_PHY_AE.py
```

The Slurm scripts contain cluster-specific paths and resource settings that need adapting. Use `channel="proakis_c_channel"` and `is_delay=True` for the active implementation; the remaining AWGN configuration branches are not supported by the current model.

### 5.2 Outputs

- Checkpoints: `models/Turbo_AutoEncoder_<timestamp>/Turbo_AutoEncoder_<timestamp>_epoch<N>.pth`, every `save_every` epochs (default: 1000).
- W&B: training/validation loss, validation BER/BLER, and final BER/BLER-versus-SNR plots in the project configured in `model_pipeline()`.

### 5.3 Configuration reference

Current defaults in `train_PHY_AE.py`:

| Setting | Field | Default |
| --- | --- | --- |
| Information bits / complex symbols | `k`, `n` | 64 / 64 |
| Even-parity bit | `add_parity_check_bit` | `True` |
| Feature streams / decoder iterations | `F`, `num_iteration` | 10 / 6 |
| Channel taps / receiver padding parameter | `num_taps`, `n_M` | 5 / 5 |
| Fractional timing-offset setting | `max_sampling_time_offset` | 0.49 |
| Encoder / decoder training SNR | `train_enc_snr_*`, `train_dec_snr_*` | 15 / 10–15 dB |
| Encoder / decoder passes per epoch | `num_train_enc`, `num_train_dec` | 1 / 5 |
| Initial samples / batch size | `num_samples`, `batch_size` | 500 / 500 |
| Epochs / Adam learning rate / loss | `epochs`, `learning_rate`, `loss_function` | 1000 / 1e-4 / BCE |
| Interleaver / random seed | `p_array_seed`, `seed` | 2 / 1 |

On validation-loss plateaus, the trainer increases the batch size and sample count, then reduces the learning rate.

## 6. License

Released under the [MIT License](LICENSE).
