# vLLM SynthID

`vllm-synthid` is a multi-tenant SynthID Text watermarking plugin for vLLM.
It wraps Transformers' `SynthIDTextWatermarkLogitsProcessor`, keeps watermark
keys in server-side configuration, and lets requests select named profiles
without receiving the key material.

## Installation

Install the package into the same environment as vLLM:

```bash
uv pip install .
```

Installation alone does not register an automatic entry point or change vLLM
behavior. Load the processor explicitly only for servers that need SynthID.
This avoids vLLM treating every process in a shared environment as having an
active custom logits processor.

For immutable cluster deployments, `Containerfile` layers the plugin onto a
digest-pinned vLLM image. Supply `VLLM_IMAGE` as a build argument; the resulting
image still requires explicit `--logits-processors` and `SYNTHID_CONFIG` to
enable watermarking.

When the processor is explicitly loaded but `SYNTHID_CONFIG` is absent, it is
disabled and does not change generation. A request that selects a profile in
that state is rejected rather than silently generating unwatermarked text.

## Configuration

Create a JSON file readable by every vLLM process:

```json
{
  "default_profile": "provider",
  "profiles": {
    "provider": {
      "keys": [654, 400, 836, 123, 340, 443, 597, 160]
    },
    "tenant-a": {
      "keys": [91, 782, 633, 204, 515, 326, 47, 918],
      "ngram_len": 5,
      "sampling_table_size": 65536,
      "sampling_table_seed": 0,
      "context_history_size": 1024
    }
  }
}
```

The bundled keys are examples only. Generate independent secret keys for a
real deployment and store the file as a secret. Configuration is cached;
restart vLLM after rotating keys.

Enable watermarking before starting vLLM:

```bash
export SYNTHID_CONFIG=/run/secrets/synthid.json
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --logits-processors \
  vllm_synthid.processor:SynthIDTextLogitsProcessor
```

The default profile is applied to requests that do not select one. An
authenticated gateway can choose a tenant profile using `vllm_xargs`:

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt": "Explain continuous batching",
  "temperature": 0.8,
  "max_tokens": 256,
  "vllm_xargs": {"synthid_watermark_profile": "tenant-a"}
}
```

Profile selection is not authorization. Restrict `vllm_xargs` at the gateway
when tenants must not select one another's profiles.

## Detection

The detector uses the same configuration, tokenizer, and profile:

```bash
vllm-synthid-detect \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
  --profile tenant-a \
  --text "Generated text to inspect"
```

It returns a mean watermark score rather than a boolean verdict. Calibrate a
threshold for the desired false-positive rate, output length, tokenizer, and
deployment distribution.

## Limitations

- Custom logits processors are incompatible with speculative decoding in the
  currently supported vLLM release.
- The plugin follows vLLM's custom logits-processor placement, before
  penalties, temperature, top-k, and top-p.
- The Transformers processor is stateful, and state restoration currently
  relies on implementation details that may require updates across releases.

## Development

```bash
uv pip install -e '.[test]'
.venv/bin/python -m pytest
```

Licensed under Apache-2.0.
