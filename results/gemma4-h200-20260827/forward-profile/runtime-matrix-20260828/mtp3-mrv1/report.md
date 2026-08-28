# vLLM data-plane benchmark `d414c0314efc28b6d7754b45`

**Status:** succeeded  
**Started:** 2026-08-28T04:32:19.442Z  
**Completed:** 2026-08-28T04:32:30.959Z  
**Cases:** 11 succeeded, 0 failed

## Run configuration

| Field | Value |
| --- | --- |
| Model | `google/gemma-4-26B-A4B-it` |
| Model revision | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| vLLM image | `quay.io/tms/vllm-synthid@sha256:7cfdbe8550a32173793aa980cda20358403485bbe356d4154e80d514d4c353be` |
| vLLM runtime | `image`; environment=`none` |
| Worker plugin image | `quay.io/tms/vllm-forward-bench-plugin@sha256:a825baf134507cf78d05b7b647f6981e3f635b0986cafdd465066d0edce11961` |
| Worker bundle | `82e66ce491d8239469159ce778d75822d8ef429c5160531da47251461c5a0783` |
| vLLM | `0.28.0` |
| Python | `3.12.3`; `/usr/bin/python3` |
| PyTorch / CUDA | `2.13.0+cu130` / `13.0` |
| GPU | `NVIDIA H200` |
| Parallelism | DP=1, PP=1, TP=1, EP=False |
| EngineCore | `InprocClient`; VLLM_ENABLE_V1_MULTIPROCESSING=`0` |
| Execution | `vllm-synthetic-scheduler`; pipeline depth=1 |
| Target | `h200` on queue `synthid-h200` |
| Resolved spec SHA-256 | `237c990ff0d482d8572b78ada222a011c23211243383ffcbebb8192085e14e97` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-b1-t512` | prefill B=1, new tokens=512, history=0×1 | 15.844 | 15.880 | 16.006 | 0.113 | 32241.8 |
| `prefill-b1-t1024` | prefill B=1, new tokens=1024, history=0×1 | 24.224 | 24.142 | 24.300 | 0.179 | 42416.1 |
| `prefill-b1-t2048` | prefill B=1, new tokens=2048, history=0×1 | 39.330 | 39.529 | 39.875 | 0.837 | 51810.0 |
| `prefill-b1-t4096` | prefill B=1, new tokens=4096, history=0×1 | 73.978 | 74.354 | 76.668 | 1.584 | 55087.6 |
| `prefill-b1-t8192` | prefill B=1, new tokens=8192, history=0×1 | 151.938 | 151.946 | 155.730 | 2.618 | 53914.0 |
| `decode-b1` | decode B=1, history=2048×1 | 7.442 | 7.447 | 7.515 | 0.097 | 537.1 |
| `decode-b2` | decode B=2, history=512×1, 2048×1 | 7.596 | 7.608 | 7.734 | 0.106 | 1051.5 |
| `decode-b4` | decode B=4, history=512×1, 2048×2, 8192×1 | 8.834 | 8.860 | 9.072 | 0.188 | 1805.8 |
| `decode-b8` | decode B=8, history=512×2, 2048×4, 8192×2 | 10.322 | 10.319 | 10.577 | 0.208 | 3101.0 |
| `decode-b16` | decode B=16, history=512×4, 2048×8, 8192×4 | 12.408 | 12.526 | 12.919 | 0.321 | 5109.5 |
| `decode-b32` | decode B=32, history=512×8, 2048×16, 8192×8 | 15.702 | 15.724 | 15.976 | 0.186 | 8140.6 |

## Memory

- KV cache: 69.00 GiB across 113053 blocks
- KV block (all cache groups): 640.00 KiB
- Nominal KV per token: 220.00 KiB
- Model weights (aggregate engine footprint): 49.33 GiB
- Effective maximum-context KV per token (sliding/local limits applied): 220.20 KiB

## Interpretation

No automatic reporting warnings were detected.

## Profiles

- `decode` / `decode-b8` (1 iterations): [d414c0314efc28b6d7754b45-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787891549591190342.pt.trace.json.gz](traces/d414c0314efc28b6d7754b45-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787891549591190342.pt.trace.json.gz)
- `prefill` / `prefill-b1-t2048` (1 iterations): [d414c0314efc28b6d7754b45-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787891550199086926.pt.trace.json.gz](traces/d414c0314efc28b6d7754b45-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787891550199086926.pt.trace.json.gz)

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).

For causal and recurrent runners, the benchmark owns deterministic request state while vLLM's KVCacheManager allocates and recycles each cache group's blocks; history construction occurs before the sustained window. Encoder-only pooling runners instead replay each complete logical sequence because non-causal attention has no incremental KV state. A bounded ring reuses request slots after their prior completion.
