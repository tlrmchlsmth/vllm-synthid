# SynthID Gemma 4 H200 sweep

The model role explicitly renders as a one-pod LeaderWorkerSet. Waldorf already
enables Kueue's LeaderWorkerSet integration, so the model is admitted through
the `synthid-h200` LocalQueue without a cluster-wide Deployment integration
change.

This Waldorf sweep compares plain and SynthID-watermarked serving on
`google/gemma-4-26B-A4B-it`. Each cell runs full five-shot GSM8K, a ShareGPT
Nyann saturation sweep, a vLLM forward-pass profile, and a 64-sample watermark
detection evaluation. The detector calibrates its threshold on half of the
plain samples, then reports held-out false-positive rate, watermarked
true-positive rate, and ROC AUC. The watermarked case fails unless AUC is at
least 0.90, TPR is at least 0.80, and FPR is at most 0.10.

Forward-pass profiles measure the model execution graph and sampler. The
watermarked case selects the profiler service's `synthid` runtime profile and
explicitly loads `SynthIDTextLogitsProcessor`; the profile mounts the same
server-side Secret without exposing its keys in the benchmark specification.
Nyann and Prometheus continue to measure end-to-end serving overhead.

The cluster profile expects these namespace resources:

- LocalQueue `synthid-h200` targeting the Waldorf H200 ClusterQueue;
- RWX PVCs `synthid-bench-cache` and `synthid-bench-results`;
- Secret `hf-secret` with key `HF_TOKEN`;
- Secret `synthid-watermark-config` with key `synthid.json`;
- pull Secret `quay-tms` for the private nm-hard-tools image.

Apply `infrastructure.yaml`, then create the three Secrets without writing
their values into Git:

```console
kubectl --context waldorf -n tysmith-dev create secret generic hf-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN"
kubectl --context waldorf -n tysmith-dev create secret generic \
  synthid-watermark-config \
  --from-file=synthid.json=/path/to/synthid.json
kubectl --context waldorf -n tysmith-dev create secret generic \
  synthid-lm-eval-token --from-literal=token='<random bearer token>'
```

Install the Kueue-enabled nm-hard-tools evaluation service using
`lm-eval-values.yaml`. The sweep controller talks directly to Waldorf's
Prometheus service and stores the exact query window, selected pod identities,
raw vLLM series, and pod-attributed DCGM series before tearing down each case.
