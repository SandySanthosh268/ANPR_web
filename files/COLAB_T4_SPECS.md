# Google Colab T4 GPU — Specs & Tier Limits

Reference notes for testing the ANPR pipeline on Colab (see `colab_full_pipeline_test.ipynb`,
`colab_gpu_benchmark.ipynb`). Compiled from public sources as of July 2026 — Google doesn't
publish these as a fixed contract, so treat exact numbers (compute-unit burn rate, RAM) as
approximate and re-check the live [Colab pricing page](https://colab.research.google.com/signup)
if planning anything cost-sensitive.

## T4 GPU hardware specs (same hardware on every tier)

| Spec | Value |
|---|---|
| VRAM | 15GB GDDR6 |
| CUDA Cores | 2,560 |
| Tensor Cores | 320 |
| FP32 performance | ~8.1 TFLOPS |
| FP16 (with sparsity) | ~130 TFLOPS |
| Architecture | Turing — optimized for **inference**, not training |

Good fit for this project's workload (YOLO plate detection + OCR inference), since both stages
are inference-only.

## Free vs Pro vs Pro+

| | **Free** | **Pro (~$11.99/mo)** | **Pro+ (~$49.99/mo)** |
|---|---|---|---|
| GPU access | T4 — lottery basis, may be unavailable at peak demand | T4 priority, occasional L4/A100 | T4/L4/A100, best priority |
| System RAM | Standard (~12.7GB) | High-RAM option available (compute-unit dependent) | High-RAM, higher priority |
| Max session length | 12 hours | 12 hours | Up to 24 hours (with sufficient compute units) |
| Idle timeout | ~90 min of inactivity | Same | No idle timeout while code is actively executing |
| Background execution | No | Limited | Supported |
| Compute units | None (free lottery only) | Monthly quota | Monthly quota (larger) |
| Availability guarantee | None | Subject to availability (better priority) | Subject to availability (best priority, still not guaranteed) |

## Compute units (paid tiers)

- T4 burns roughly **1.76 compute units/hour**.
- A100 burns roughly **15 compute units/hour**.
- Pay-as-you-go: **$9.99 = 100 units** → ~57 hours on a T4, ~7 hours on an A100.
- When a paid plan's units run out, the session drops back to free-tier restrictions.

## Practical notes for this project

1. **Colab is a testing/benchmarking tool here, not a production target.** Session length caps
   (12-24h), idle disconnects, and Colab's terms restricting long-running automated/non-interactive
   workloads mean a real 5-camera, always-on ANPR deployment should run on a dedicated cloud GPU VM
   (AWS EC2 g4dn/g5, GCP Compute Engine with a T4/L4, or on-prem GPU server) — not a Colab notebook.
2. **No GPU availability guarantee on the free tier.** If a session fails to get a GPU at all
   during a busy period, that's Google's allocation "lottery," not a bug in the notebook/pipeline.
3. Running multiple concurrent camera streams / detection models on **one** GPU means they share
   its fixed compute + VRAM — FPS per stream drops as more models run concurrently (see the
   multi-camera GPU-sharing discussion from this project's testing session).

## Sources

- [Google Colab GPU: free access, limits, and alternatives — Hivenet](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)
- [Google Colab official FAQ](https://research.google.com/colaboratory/faq.html)
- [Google Colab Pro vs Colab Pro+ — Technical Overview — Bison Knowledgebase](https://knowledgebase.bison.co.in/view_article.php?id=690)
