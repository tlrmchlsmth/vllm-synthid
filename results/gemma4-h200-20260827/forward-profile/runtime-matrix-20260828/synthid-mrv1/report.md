# vLLM data-plane benchmark `d331a9a53768d897a7614039`

**Status:** succeeded  
**Started:** 2026-08-28T03:40:03.646Z  
**Completed:** 2026-08-28T03:40:15.469Z  
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
| Resolved spec SHA-256 | `ca3ac54b1d795af9d43d414871c94f3b8e78abdeeb7b451065075441015b7051` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-b1-t512` | prefill B=1, new tokens=512, history=0×1 | 15.798 | 15.836 | 15.915 | 0.129 | 32332.2 |
| `prefill-b1-t1024` | prefill B=1, new tokens=1024, history=0×1 | 23.456 | 23.562 | 23.945 | 0.276 | 43459.1 |
| `prefill-b1-t2048` | prefill B=1, new tokens=2048, history=0×1 | 37.927 | 38.101 | 38.378 | 0.657 | 53751.5 |
| `prefill-b1-t4096` | prefill B=1, new tokens=4096, history=0×1 | 70.345 | 71.236 | 73.890 | 1.815 | 57498.7 |
| `prefill-b1-t8192` | prefill B=1, new tokens=8192, history=0×1 | 141.749 | 141.704 | 144.795 | 2.573 | 57810.5 |
| `decode-b1` | decode B=1, history=2048×1 | 5.937 | 5.966 | 6.109 | 0.089 | 167.6 |
| `decode-b2` | decode B=2, history=512×1, 2048×1 | 7.344 | 7.668 | 8.467 | 0.515 | 260.8 |
| `decode-b4` | decode B=4, history=512×1, 2048×2, 8192×1 | 10.222 | 10.248 | 10.395 | 0.143 | 390.3 |
| `decode-b8` | decode B=8, history=512×2, 2048×4, 8192×2 | 13.482 | 13.535 | 13.954 | 0.247 | 591.1 |
| `decode-b16` | decode B=16, history=512×4, 2048×8, 8192×4 | 20.637 | 20.709 | 20.884 | 0.255 | 772.6 |
| `decode-b32` | decode B=32, history=512×8, 2048×16, 8192×8 | 34.903 | 34.952 | 35.233 | 0.271 | 915.5 |

## Memory

- KV cache: 73.86 GiB across 121018 blocks
- KV block (all cache groups): 640.00 KiB
- Nominal KV per token: 220.00 KiB
- Model weights (aggregate engine footprint): 48.54 GiB
- Effective maximum-context KV per token (sliding/local limits applied): 220.20 KiB

## Interpretation

No automatic reporting warnings were detected.

## Profiles

- `decode` / `decode-b8` (1 iterations): [d331a9a53768d897a7614039-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888413564888755.pt.trace.json.gz](traces/d331a9a53768d897a7614039-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888413564888755.pt.trace.json.gz)
- `prefill` / `prefill-b1-t2048` (1 iterations): [d331a9a53768d897a7614039-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888414651775893.pt.trace.json.gz](traces/d331a9a53768d897a7614039-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888414651775893.pt.trace.json.gz)

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).

For causal and recurrent runners, the benchmark owns deterministic request state while vLLM's KVCacheManager allocates and recycles each cache group's blocks; history construction occurs before the sustained window. Encoder-only pooling runners instead replay each complete logical sequence because non-causal attention has no incremental KV state. A bounded ring reuses request slots after their prior completion.
