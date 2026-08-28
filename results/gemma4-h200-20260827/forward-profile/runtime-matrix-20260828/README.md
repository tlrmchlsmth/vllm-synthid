# Gemma 4 forward-pass runtime matrix

This directory contains the corrected H200 forward-pass benchmark matrix for
`google/gemma-4-26B-A4B-it` on vLLM 0.28.0. All four runs use synchronous
synthetic execution. Every representative decode and prefill profile contains
real GPU events; the service's zero-GPU-work validity gate passed.

## Runtime coverage

| Configuration | Job | Runner | Decode-b8 p50 | Decode throughput | Decode GPU events |
| --- | --- | --- | ---: | ---: | ---: |
| Plain | `8484b6ec8cd0f46b4fc17c68` | MRV1 | 7.76 ms | 1,031.7 tok/s | 700 |
| SynthID | `d331a9a53768d897a7614039` | MRV1 | 13.48 ms | 591.1 tok/s | 700 |
| Plain | `46558510f068084b37835c4a` | MRV2 | 7.36 ms | 1,087.1 tok/s | 675 |
| MTP, 3 draft tokens | `3b3a0f539a7bd46e719faab8` | MRV2 | 9.71 ms | 3,255.5 verification tok/s | 675 |

MTP throughput counts four target-side scheduled tokens per request step: one
ordinary target token plus three speculative verification tokens. It is not
accepted output-token throughput and should not be compared directly with the
other rows.

SynthID plus MTP is not included because vLLM 0.28 rejects the combination of
custom logits processors and speculative decoding before the benchmark driver
runs.

## Controlled MRV1 comparison

The plain and SynthID rows below use the same runner, image, model revision,
H200 resource shape, benchmark cases, measurement settings, and worker bundle.

| Case | Plain tok/s | SynthID tok/s | SynthID vs plain |
| --- | ---: | ---: | ---: |
| prefill-b1-t512 | 33,357.1 | 32,332.2 | -3.1% |
| prefill-b1-t1024 | 44,769.7 | 43,459.1 | -2.9% |
| prefill-b1-t2048 | 54,626.5 | 53,751.5 | -1.6% |
| prefill-b1-t4096 | 58,255.3 | 57,498.7 | -1.3% |
| prefill-b1-t8192 | 57,605.7 | 57,810.5 | +0.4% |
| decode-b1 | 191.0 | 167.6 | -12.2% |
| decode-b2 | 343.0 | 260.8 | -24.0% |
| decode-b4 | 545.2 | 390.3 | -28.4% |
| decode-b8 | 1,031.7 | 591.1 | -42.7% |
| decode-b16 | 1,723.4 | 772.6 | -55.2% |
| decode-b32 | 2,705.7 | 915.5 | -66.2% |

For the representative decode-b8 profile, plain and SynthID recorded almost
identical GPU execution time: 6.421 ms and 6.419 ms respectively. The remaining
iteration-latency gap is therefore outside the captured model GPU forward path,
consistent with logits-processing, sampling, and host-control overhead. It is
not explained by comparing MRV1 with MRV2.

## Provenance

- Accelerator: NVIDIA H200, one GPU per Job
- Kueue LocalQueue: `synthid-h200`
- Model revision: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- vLLM image: `quay.io/tms/vllm-synthid@sha256:7cfdbe8550a32173793aa980cda20358403485bbe356d4154e80d514d4c353be`
- Benchmark worker image: `quay.io/tms/vllm-forward-bench-plugin@sha256:2aec33ebd414640c7a0ade0fe81f1aa28254abaa0f35625a1e01f10623caa557`
- Worker bundle SHA-256: `e66820790b1c7f3d9ab2e1785c061ae97731e6b7415b841e00ba64b0e694f888`
- Benchmark-service change: [PR #27](https://github.com/tlrmchlsmth/vllm-forward-pass-benchmark-service/pull/27)

Each configuration directory contains the service-produced HTML, Markdown, and
JSON reports; submitted and resolved specifications; rendered Pod manifest;
artifact manifest; compact Perfetto-compatible trace; and human-readable trace
summary. The worker zipapp and verbose profiler console table are omitted to
avoid duplicating code and bulky logs.
