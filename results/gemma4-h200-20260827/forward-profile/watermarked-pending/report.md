# vLLM data-plane benchmark `6702f82fd8e6ab821c008677`

**Status:** failed  
**Started:** 2026-08-27T23:58:30.723Z  
**Completed:** 2026-08-28T00:02:31.810Z  
**Cases:** 0 succeeded, 0 failed

## Run configuration

| Field | Value |
| --- | --- |
| Model | `unknown` |
| Model revision | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| vLLM image | `quay.io/tms/vllm-synthid@sha256:7cfdbe8550a32173793aa980cda20358403485bbe356d4154e80d514d4c353be` |
| vLLM runtime | `image`; environment=`none` |
| Worker plugin image | `quay.io/tms/vllm-forward-bench-plugin@sha256:0a6bba3e77acf33ac2a66c5b500ddba7b0042d1e98775c68d19236f05dc513d3` |
| Worker bundle | `41c19ef9b96acff0cb6c741e3c24d3f1cb5f99c8f6deff06591fa61f2e984217` |
| vLLM | `unknown` |
| Python | `unknown`; `unknown` |
| PyTorch / CUDA | `unknown` / `unknown` |
| GPU | `unknown` |
| Parallelism | DP=1, PP=1, TP=1, EP=False |
| EngineCore | `unknown`; VLLM_ENABLE_V1_MULTIPROCESSING=`unknown` |
| Execution | `unknown`; pipeline depth=unknown |
| Target | `h200` on queue `synthid-h200` |
| Resolved spec SHA-256 | `a960dcae34dbd8b19f142999dc626a9bd2358dd4ffde6f9a745aa6544d47563c` |

## Results

| Case | Shape | p50 interval (ms) | Mean interval (ms) | p90 interval (ms) | Stdev (ms) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

## Memory

This report predates run-level memory profiling.

## Interpretation

No automatic reporting warnings were detected.

## Profiles

Profiling was disabled.

## Failure

- RuntimeError: synthetic execution requires Model Runner V2; set VLLM_USE_V2_MODEL_RUNNER=1 or use a supported configuration

## Methodology

Each result measures sustained synthetic vLLM EngineCore data-plane throughput. Cases used 3 warmup iterations and 10 measured iterations. Completion intervals, the sustained measurement window, and exact synthetic shapes are in [`report.json`](report.json). The graphs are in [`report.html`](report.html). Immutable artifact hashes are in [`artifact-manifest.json`](artifact-manifest.json).
