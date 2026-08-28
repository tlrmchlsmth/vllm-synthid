# vLLM data-plane benchmark `3b3a0f539a7bd46e719faab8`

**Status:** succeeded  
**Started:** 2026-08-28T03:40:14.083Z  
**Completed:** 2026-08-28T03:40:25.719Z  
**Cases:** 11 succeeded, 0 failed

## Run configuration

| Field | Value |
| --- | --- |
| Model | `google/gemma-4-26B-A4B-it` |
| Model revision | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| vLLM image | `quay.io/tms/vllm-synthid@sha256:7cfdbe8550a32173793aa980cda20358403485bbe356d4154e80d514d4c353be` |
| vLLM runtime | `image`; environment=`none` |
| Worker plugin image | `quay.io/tms/vllm-forward-bench-plugin@sha256:2aec33ebd414640c7a0ade0fe81f1aa28254abaa0f35625a1e01f10623caa557` |
| Worker bundle | `e66820790b1c7f3d9ab2e1785c061ae97731e6b7415b841e00ba64b0e694f888` |
| vLLM | `0.28.0` |
| Python | `3.12.3`; `/usr/bin/python3` |
| PyTorch / CUDA | `2.13.0+cu130` / `13.0` |
| GPU | `NVIDIA H200` |
| Parallelism | DP=1, PP=1, TP=1, EP=False |
| EngineCore | `InprocClient`; VLLM_ENABLE_V1_MULTIPROCESSING=`0` |
| Execution | `vllm-synthetic-scheduler`; pipeline depth=1 |
| Target | `h200` on queue `synthid-h200` |
| Resolved spec SHA-256 | `e0fbaf54e1181eafe9e6dab889076168da8253d447e24bf7eb7981a53685fa2b` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-b1-t512` | prefill B=1, new tokens=512, history=0×1 | 15.508 | 15.638 | 15.670 | 0.421 | 32740.2 |
| `prefill-b1-t1024` | prefill B=1, new tokens=1024, history=0×1 | 23.294 | 23.449 | 23.582 | 0.528 | 43669.7 |
| `prefill-b1-t2048` | prefill B=1, new tokens=2048, history=0×1 | 38.303 | 38.695 | 39.173 | 1.463 | 52926.2 |
| `prefill-b1-t4096` | prefill B=1, new tokens=4096, history=0×1 | 72.394 | 73.628 | 76.254 | 2.646 | 55631.1 |
| `prefill-b1-t8192` | prefill B=1, new tokens=8192, history=0×1 | 149.036 | 150.044 | 157.780 | 6.037 | 54597.2 |
| `decode-b1` | decode B=1, history=2048×1 | 7.326 | 7.411 | 7.472 | 0.370 | 539.7 |
| `decode-b2` | decode B=2, history=512×1, 2048×1 | 7.516 | 7.629 | 7.837 | 0.431 | 1048.6 |
| `decode-b4` | decode B=4, history=512×1, 2048×2, 8192×1 | 8.489 | 8.630 | 8.932 | 0.464 | 1853.9 |
| `decode-b8` | decode B=8, history=512×2, 2048×4, 8192×2 | 9.708 | 9.830 | 10.002 | 0.417 | 3255.5 |
| `decode-b16` | decode B=16, history=512×4, 2048×8, 8192×4 | 11.374 | 11.533 | 11.619 | 0.544 | 5549.3 |
| `decode-b32` | decode B=32, history=512×8, 2048×16, 8192×8 | 14.290 | 14.524 | 14.717 | 0.816 | 8813.1 |

## Memory

- KV cache: 74.38 GiB across 121858 blocks
- KV block (all cache groups): 640.00 KiB
- Nominal KV per token: 220.00 KiB
- Model weights (aggregate engine footprint): 49.33 GiB
- Effective maximum-context KV per token (sliding/local limits applied): 220.20 KiB

## Interpretation

No automatic reporting warnings were detected.

## Profiles

- `decode` / `decode-b8` (1 iterations): [3b3a0f539a7bd46e719faab8-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888424265644846.pt.trace.json.gz](traces/3b3a0f539a7bd46e719faab8-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888424265644846.pt.trace.json.gz)
- `prefill` / `prefill-b1-t2048` (1 iterations): [3b3a0f539a7bd46e719faab8-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888424938340775.pt.trace.json.gz](traces/3b3a0f539a7bd46e719faab8-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888424938340775.pt.trace.json.gz)

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).

For causal and recurrent runners, the benchmark owns deterministic request state while vLLM's KVCacheManager allocates and recycles each cache group's blocks; history construction occurs before the sustained window. Encoder-only pooling runners instead replay each complete logical sequence because non-causal attention has no incremental KV state. A bounded ring reuses request slots after their prior completion.
