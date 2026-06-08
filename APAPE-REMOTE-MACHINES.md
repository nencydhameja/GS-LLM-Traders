# Machine: promaxgb10

## Hardware

| Component | Details |
|-----------|---------|
| **Hostname** | `promaxgb10-4ae4` |
| **Vendor / Model** | Dell Pro Max with GB10 FCM1253 |
| **Architecture** | ARM64 (aarch64) |
| **OS** | Ubuntu 24.04.4 LTS |
| **Kernel** | Linux 6.17.0-1021-nvidia |
| **Firmware** | 5.36_2.0.0 (2025-09-23) |

### CPU

Two clusters of Arm cores (big.LITTLE arrangement):

| Cluster | Core | Count | Max freq |
|---------|------|-------|----------|
| Performance | Cortex-X925 | 10 | 3.9 GHz |
| Efficiency | Cortex-A725 | 10 | 2.8 GHz |

**Total logical CPUs:** 20 (no SMT / 1 thread per core)

### Memory

| | |
|---|---|
| RAM | 121 GiB |
| Swap | 15 GiB |

### GPU

| | |
|---|---|
| Model | NVIDIA GB10 |
| VRAM | Unified / shared (reports as N/A in nvidia-smi — integrated NVLink topology) |
| Driver | 580.159.03 |
| CUDA | 13.0 |

> The GB10 is Nvidia's Grace-Blackwell SoC. System and GPU memory are unified via NVLink-C2C; the ~41 GiB shown in `nvidia-smi` at query time was occupied by a running Python inference process.

### Storage

| Device | Size | Model |
|--------|------|-------|
| nvme0n1 | 1.9 TB | KIOXIA EG6 2048 GB NVMe |

---

## Installed LLM Models

All models are served via [Ollama](https://ollama.com) and stored in `/usr/share/ollama/.ollama/models/`.

| Model | Tag | Parameters | Quantization | Context | Size on disk | Capabilities |
|-------|-----|-----------|--------------|---------|-------------|--------------|
| **Qwen2.5-VL** | `qwen2.5vl:7b` | 8.3 B | Q4_K_M | 128 k tokens | 6.0 GB | text completion, vision |
| **Qwen2.5** | `qwen2.5:0.5b` | 494 M | Q4_K_M | 32 k tokens | 397 MB | text completion, tool use |

Both models are released under the Apache 2.0 license by Alibaba Cloud.

### Model notes

- **`qwen2.5vl:7b`** — multimodal (vision + language) variant of Qwen 2.5; configured with `temperature=0.0001` (near-deterministic). Used in this project for structured LLM trading experiments.
- **`qwen2.5:0.5b`** — tiny text-only model; useful for fast local smoke tests and prototyping where GPU memory pressure matters.

---

## Relevant running processes (at time of writing, 2026-06-08)

A Python inference process was consuming ~41 GiB of GPU memory, consistent with active model inference via the `.venv` virtual environment in this repo.
