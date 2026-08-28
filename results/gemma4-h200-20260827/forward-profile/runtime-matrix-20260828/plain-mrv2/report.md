# vLLM data-plane benchmark `46558510f068084b37835c4a`

**Status:** succeeded  
**Started:** 2026-08-28T03:41:54.088Z  
**Completed:** 2026-08-28T03:42:04.517Z  
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
| Resolved spec SHA-256 | `574fc4456bff63176a57f99956f47e0563379fd11d5d04e488c71108712567c6` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-b1-t512` | prefill B=1, new tokens=512, history=0×1 | 15.260 | 15.302 | 15.412 | 0.104 | 33460.2 |
| `prefill-b1-t1024` | prefill B=1, new tokens=1024, history=0×1 | 23.130 | 23.129 | 23.301 | 0.198 | 44273.2 |
| `prefill-b1-t2048` | prefill B=1, new tokens=2048, history=0×1 | 37.792 | 37.722 | 37.931 | 0.248 | 54292.6 |
| `prefill-b1-t4096` | prefill B=1, new tokens=4096, history=0×1 | 70.845 | 71.011 | 72.795 | 1.258 | 57681.4 |
| `prefill-b1-t8192` | prefill B=1, new tokens=8192, history=0×1 | 143.127 | 143.212 | 148.460 | 3.687 | 57202.0 |
| `decode-b1` | decode B=1, history=2048×1 | 4.786 | 4.808 | 4.829 | 0.081 | 208.0 |
| `decode-b2` | decode B=2, history=512×1, 2048×1 | 5.438 | 5.452 | 5.483 | 0.052 | 366.9 |
| `decode-b4` | decode B=4, history=512×1, 2048×2, 8192×1 | 7.129 | 7.165 | 7.349 | 0.163 | 558.3 |
| `decode-b8` | decode B=8, history=512×2, 2048×4, 8192×2 | 7.361 | 7.359 | 7.446 | 0.092 | 1087.1 |
| `decode-b16` | decode B=16, history=512×4, 2048×8, 8192×4 | 8.908 | 8.998 | 9.466 | 0.348 | 1778.2 |
| `decode-b32` | decode B=32, history=512×8, 2048×16, 8192×8 | 11.313 | 11.357 | 11.535 | 0.157 | 2817.7 |

## Memory

- KV cache: 76.03 GiB across 124575 blocks
- KV block (all cache groups): 640.00 KiB
- Nominal KV per token: 220.00 KiB
- Model weights (aggregate engine footprint): 48.54 GiB
- Effective maximum-context KV per token (sliding/local limits applied): 220.20 KiB

## Interpretation

No automatic reporting warnings were detected.

## Profiles

- `decode` / `decode-b8` (1 iterations): [46558510f068084b37835c4a-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888523192512434.pt.trace.json.gz](traces/46558510f068084b37835c4a-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787888523192512434.pt.trace.json.gz)
- `prefill` / `prefill-b1-t2048` (1 iterations): [46558510f068084b37835c4a-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888523796609166.pt.trace.json.gz](traces/46558510f068084b37835c4a-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787888523796609166.pt.trace.json.gz)

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).

For causal and recurrent runners, the benchmark owns deterministic request state while vLLM's KVCacheManager allocates and recycles each cache group's blocks; history construction occurs before the sustained window. Encoder-only pooling runners instead replay each complete logical sequence because non-causal attention has no incremental KV state. A bounded ring reuses request slots after their prior completion.
