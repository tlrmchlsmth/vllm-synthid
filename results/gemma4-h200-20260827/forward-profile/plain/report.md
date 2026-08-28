# vLLM data-plane benchmark `a83db8ced07a2be3e95fbdc6`

**Status:** succeeded  
**Started:** 2026-08-27T22:38:02.355Z  
**Completed:** 2026-08-27T22:38:17.883Z  
**Cases:** 11 succeeded, 0 failed

## Run configuration

| Field | Value |
| --- | --- |
| Model | `google/gemma-4-26B-A4B-it` |
| Model revision | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| vLLM image | `quay.io/tms/vllm-synthid@sha256:7cfdbe8550a32173793aa980cda20358403485bbe356d4154e80d514d4c353be` |
| vLLM runtime | `image`; environment=`none` |
| Worker plugin image | `quay.io/tms/vllm-forward-bench-plugin@sha256:8cb9489dabaaa557c67ee591fcd48d31d9702c11fa3f89994b2714dd0eb80752` |
| Worker bundle | `b653928cd548fa4205a453cc0169dd2a5640104873e940e5d2e31fb8e43e6610` |
| vLLM | `0.28.0` |
| Python | `3.12.3`; `/usr/bin/python3` |
| PyTorch / CUDA | `2.13.0+cu130` / `13.0` |
| GPU | `NVIDIA H200` |
| Parallelism | DP=1, PP=1, TP=1, EP=False |
| EngineCore | `InprocClient`; VLLM_ENABLE_V1_MULTIPROCESSING=`0` |
| Execution | `vllm-synthetic-scheduler`; pipeline depth=2 |
| Target | `h200` on queue `synthid-h200` |
| Resolved spec SHA-256 | `8e8d5e2f2e05c29574da7b1f5796b94f422a16b18ed7498551fe55125c6504a6` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-b1-t512` | prefill B=1, new tokens=512, history=0×1 | 14.030 | 14.008 | 14.114 | 0.105 | 36550.6 |
| `prefill-b1-t1024` | prefill B=1, new tokens=1024, history=0×1 | 21.956 | 21.929 | 22.084 | 0.133 | 46695.1 |
| `prefill-b1-t2048` | prefill B=1, new tokens=2048, history=0×1 | 38.077 | 38.235 | 39.655 | 1.073 | 53564.1 |
| `prefill-b1-t4096` | prefill B=1, new tokens=4096, history=0×1 | 70.021 | 70.525 | 73.238 | 1.614 | 58078.7 |
| `prefill-b1-t8192` | prefill B=1, new tokens=8192, history=0×1 | 143.053 | 143.607 | 148.286 | 3.729 | 57044.6 |
| `decode-b1` | decode B=1, history=2048×1 | 4.051 | 4.055 | 4.094 | 0.035 | 246.6 |
| `decode-b2` | decode B=2, history=512×1, 2048×1 | 4.646 | 4.635 | 4.688 | 0.041 | 431.5 |
| `decode-b4` | decode B=4, history=512×1, 2048×2, 8192×1 | 6.092 | 6.106 | 6.213 | 0.093 | 655.1 |
| `decode-b8` | decode B=8, history=512×2, 2048×4, 8192×2 | 6.492 | 6.454 | 6.722 | 0.270 | 1239.5 |
| `decode-b16` | decode B=16, history=512×4, 2048×8, 8192×4 | 7.855 | 7.858 | 7.963 | 0.146 | 2036.2 |
| `decode-b32` | decode B=32, history=512×8, 2048×16, 8192×8 | 9.876 | 9.905 | 10.038 | 0.105 | 3230.7 |

## Memory

- KV cache: 76.03 GiB across 124575 blocks
- KV block (all cache groups): 640.00 KiB
- Nominal KV per token: 220.00 KiB
- Model weights (aggregate engine footprint): 48.54 GiB
- Effective maximum-context KV per token (sliding/local limits applied): 220.20 KiB

## Interpretation

No automatic reporting warnings were detected.

## Profiles

- `decode` / `decode-b8` (1 iterations): [a83db8ced07a2be3e95fbdc6-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787870296342463413.pt.trace.json.gz](traces/a83db8ced07a2be3e95fbdc6-decode_dp0_pp0_tp0_dcp0_ep0_rank0.1787870296342463413.pt.trace.json.gz)
- `prefill` / `prefill-b1-t2048` (1 iterations): [a83db8ced07a2be3e95fbdc6-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787870296847443767.pt.trace.json.gz](traces/a83db8ced07a2be3e95fbdc6-prefill_dp0_pp0_tp0_dcp0_ep0_rank0.1787870296847443767.pt.trace.json.gz)

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).

For causal and recurrent runners, the benchmark owns deterministic request state while vLLM's KVCacheManager allocates and recycles each cache group's blocks; history construction occurs before the sustained window. Encoder-only pooling runners instead replay each complete logical sequence because non-causal attention has no incremental KV state. A bounded ring reuses request slots after their prior completion.
