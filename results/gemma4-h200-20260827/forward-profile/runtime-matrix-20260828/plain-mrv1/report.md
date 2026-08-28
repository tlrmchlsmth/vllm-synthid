# vLLM data-plane benchmark `8484b6ec8cd0f46b4fc17c68`

**Status:** succeeded  
**Started:** 2026-08-28T03:39:49.147Z  
**Completed:** 2026-08-28T03:39:59.490Z  
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
| Resolved spec SHA-256 | `2f56e1d1e57e4789320414209c48885968ab196e8122411b50dde0772738cf95` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-b1-t512` | prefill B=1, new tokens=512, history=0×1 | 15.323 | 15.349 | 15.384 | 0.081 | 33357.1 |
| `prefill-b1-t1024` | prefill B=1, new tokens=1024, history=0×1 | 22.900 | 22.873 | 23.042 | 0.169 | 44769.7 |
| `prefill-b1-t2048` | prefill B=1, new tokens=2048, history=0×1 | 37.478 | 37.491 | 37.681 | 0.174 | 54626.5 |
| `prefill-b1-t4096` | prefill B=1, new tokens=4096, history=0×1 | 69.958 | 70.311 | 72.102 | 1.223 | 58255.3 |
| `prefill-b1-t8192` | prefill B=1, new tokens=8192, history=0×1 | 140.944 | 142.208 | 145.384 | 3.342 | 57605.7 |
| `decode-b1` | decode B=1, history=2048×1 | 5.225 | 5.236 | 5.298 | 0.055 | 191.0 |
| `decode-b2` | decode B=2, history=512×1, 2048×1 | 5.832 | 5.831 | 5.869 | 0.039 | 343.0 |
| `decode-b4` | decode B=4, history=512×1, 2048×2, 8192×1 | 7.292 | 7.337 | 7.536 | 0.129 | 545.2 |
| `decode-b8` | decode B=8, history=512×2, 2048×4, 8192×2 | 7.763 | 7.754 | 7.854 | 0.122 | 1031.7 |
| `decode-b16` | decode B=16, history=512×4, 2048×8, 8192×4 | 9.213 | 9.284 | 9.385 | 0.292 | 1723.4 |
| `decode-b32` | decode B=32, history=512×8, 2048×16, 8192×8 | 11.826 | 11.827 | 12.005 | 0.142 | 2705.7 |

## Memory

- KV cache: 73.86 GiB across 121018 blocks
- KV block (all cache groups): 640.00 KiB
- Nominal KV per token: 220.00 KiB
- Model weights (aggregate engine footprint): 48.54 GiB
- Effective maximum-context KV per token (sliding/local limits applied): 220.20 KiB

## Interpretation

No automatic reporting warnings were detected.

## Profiles

- `decode` / `decode-b8` (1 iterations): [8484b6ec8cd0f46b4fc17c68-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888398241453635.pt.trace.json.gz](traces/8484b6ec8cd0f46b4fc17c68-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888398241453635.pt.trace.json.gz)
- `prefill` / `prefill-b1-t2048` (1 iterations): [8484b6ec8cd0f46b4fc17c68-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888398793066908.pt.trace.json.gz](traces/8484b6ec8cd0f46b4fc17c68-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888398793066908.pt.trace.json.gz)

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).

For causal and recurrent runners, the benchmark owns deterministic request state while vLLM's KVCacheManager allocates and recycles each cache group's blocks; history construction occurs before the sustained window. Encoder-only pooling runners instead replay each complete logical sequence because non-causal attention has no incremental KV state. A bounded ring reuses request slots after their prior completion.
