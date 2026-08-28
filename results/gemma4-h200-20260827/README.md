# Gemma 4 SynthID results on H200

Run ID: `synthid-gemma4-v028-r2-20260827t220400z`

This directory records the plain-versus-SynthID run of
`google/gemma-4-26B-A4B-it` on one NVIDIA H200 using vLLM 0.28.0. The serving
image, model revision, rendered deployments, queue placement, seeds, and tool
versions are retained under [`provenance/`](provenance/).

## Summary

| Check | Plain | SynthID | Result |
| --- | ---: | ---: | --- |
| GSM8K 5-shot exact match, 1,319 samples | 94.769% | 94.541% | Pass; 99.76% of baseline |
| Watermark detector ROC AUC, 32 + 32 evaluation samples | — | 0.9980 | Pass |
| Detector TPR at calibrated threshold | 3.125% FPR | 100% TPR | Pass |
| Peak ShareGPT throughput | 1,608.9 tok/s | 565.9 tok/s | SynthID overhead is substantial |
| Forward profile | Valid MRV2 profile | Pending | MRV1 decode attempt was invalid |

The accuracy requirement was at least 98% of the plain baseline. The observed
SynthID score was 0.945413 versus 0.947688 plain, above the required 0.928734.
The detector requirement was AUC at least 0.9, TPR at least 0.8, and FPR at
most 0.1.

## ShareGPT performance sweep

Each stage ran for two minutes. All recorded requests succeeded.

| Concurrency | Plain tok/s | SynthID tok/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 220.4 | 150.6 | -31.7% |
| 2 | 356.2 | 232.3 | -34.8% |
| 4 | 516.9 | 318.2 | -38.4% |
| 8 | 782.2 | 423.6 | -45.8% |
| 16 | 1,058.0 | 513.1 | -51.5% |
| 32 | 1,400.9 | 565.9 | -59.6% |
| 64 | 1,608.8 | 471.6 | -70.7% |
| 128 | 1,608.9 | 182.7 | -88.6% |
| 256 | 1,284.9 | 47.5 | -96.3% |

The aggregate nyann outputs, including TTFT and inter-token latency
percentiles, are in [`performance/`](performance/). Per-request JSONL logs are
intentionally omitted.

## Accuracy and watermark detection

[`accuracy/`](accuracy/) contains the nm-hard-tools/lm-eval reports, effective
configurations, raw aggregate lm-eval results, and baseline comparison. GSM8K
sample JSONL files are intentionally omitted.

[`detection/`](detection/) contains the generated texts, per-sample detector
scores, and calibrated comparison. At threshold `0.5184263`, the detector
obtained 0.998047 ROC AUC, 100% TPR, and 3.125% FPR.

## Forward-pass profiling status

The unwatermarked MRV2 run is valid and retained under
[`forward-profile/plain/`](forward-profile/plain/), including the website,
Markdown and JSON reports, compact PyTorch traces, trace summaries, and exact
resolved specification.

The original watermarked MRV1 decode result is excluded because its trace had
zero GPU events and its throughput was invalid. The fail-closed rerun is
retained under
[`forward-profile/watermarked-pending/`](forward-profile/watermarked-pending/):
it contains no results or profiles and records the current driver rejection.
A valid watermarked profile remains pending MRV1 support in the benchmark
service.

## Artifact policy

This checkout intentionally excludes GSM8K sample JSONL, nyann per-request
logs, Prometheus exports, and the invalid watermarked trace.
[`SHA256SUMS`](SHA256SUMS) covers every retained artifact except the checksum
file itself.
