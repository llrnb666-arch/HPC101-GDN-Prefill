# HPC101-GDN-Prefill

> TileLang-based GPU kernel optimization for Gated Delta Network (GDN) prefill forward pass on NVIDIA H800 MIG.

## Overview

This project optimizes the forward pass of Gated Delta Networks using TileLang, running on an H800 MIG 1g.10gb partition (14 SMs, 10GB VRAM). All 8 public benchmark cases exceed the 100-point baseline (fastest open-source implementation), achieving bonus scores.

| Case | Core Time (ms) | 100pt Baseline (ms) | Speedup |
|------|---------------|---------------------|---------|
| short_tail_state | 0.083 | 0.277 | 3.34x |
| chain_equal | 0.394 | 0.460 | 1.17x |
| parallel_equal | 0.271 | 0.469 | 1.73x |
| parallel_gva | 0.229 | 0.442 | 1.93x |
| long_low_gva | 1.478 | 1.767 | 1.20x |
| batch_split_gva | 1.188 | 1.416 | 1.19x |
| wide_gva_state | 2.057 | 2.286 | 1.11x |
| deep_gva_state | 2.387 | 2.650 | 1.11x |

**8/8 cases above 100-point baseline.**

## Problem Statement

The GDN prefill forward computes a gated delta rule attention mechanism: for each token, it maintains a state matrix V = A @ W where A is updated via delta rule (gate-gated additive updates). The challenge is mapping this sequential-dependency computation onto GPU tensor cores efficiently within the MIG's 14 SMs and 10GB constraint.

## Key Optimizations

The final implementation (V33) builds on a FlashQLA-style producer-consumer architecture with four key improvements:

### 1. Removed false bar_3 dependency

The consumer computing Vd = A @ W was unnecessarily waiting for the other consumer to finish. Removing this synchronization point let both consumers run independently, reducing pipeline stalls.

### 2. h_shared switched to bf16

Matching FlashQLA's approach, storing the shared state in bf16 instead of fp32 lets the GEMM read state values through the tensor core path, doubling effective memory bandwidth for state loads.

### 3. o_shared double buffering

Output storage uses a double buffer so that writing results from the current tile overlaps with computation of the next tile, hiding memory latency behind compute.

### 4. Forced block_DV=128

The critical breakthrough: using a wider DV tile (128 instead of 64) trades fewer concurrent blocks for much higher per-block tensor core utilization. Although fewer blocks fit per SM, the single-block efficiency improvement more than compensates, especially on the long_low_gva case where occupancy was already sufficient.

## File Structure

```text
+- run.py                          # Test entry point and result output
+- evaluation/                     # Correctness checking and timing
|   +- cases.csv                   # Public benchmark case configs
|   +- support.py                  # Input generation, correctness, timing
+- preprocessing/                  # Out-of-timing-region preprocessing
|   +- tilelang_cumsum.py          # Block-wise prefix sum of gates
|   +- tilelang_kkt_solve.py       # Block-wise KKT triangular solve
+- references/                     # Correctness baseline and reference impls
|   +- torch_gdr.py                # PyTorch high-precision reference
|   +- official/                   # Official high-perf implementations
|       +- fla.py                  # FLA adapter
|       +- flash_qla.py            # FlashQLA adapter
|       +- flashinfer.py           # FlashInfer adapter
+- student/                        # Student implementation (collected at grading)
    +- tilelang_fwd.py             # Optimized GDN prefill forward kernel
```

## How to Run

```bash
# On a ZJU HPC DevPod with lab3 partition access
hpc submit -p lab3 "python run.py"
```

## References

- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
- [FlashQLA: High-Performance Linear Attention Kernel Library built on TileLang](https://github.com/QwenLM/FlashQLA)
