# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from transformers.generation import SynthIDTextWatermarkLogitsProcessor
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality

from vllm_synthid.detect import score_token_ids
from vllm_synthid.processor import (
    CONFIG_ENV_VAR,
    PROFILE_ARG,
    SynthIDTextLogitsProcessor,
    _load_config_file,
    build_transformers_processor,
    load_config,
)

PROFILE_A_KEYS = [654, 400, 836, 123, 340, 443, 597, 160]
PROFILE_B_KEYS = [91, 782, 633, 204, 515, 326, 47, 918]


@pytest.fixture
def synthid_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    config_path = tmp_path / "synthid.json"
    config_path.write_text(
        json.dumps(
            {
                "default_profile": "tenant-a",
                "profiles": {
                    "tenant-a": {
                        "keys": PROFILE_A_KEYS,
                        "ngram_len": 5,
                        "sampling_table_size": 256,
                        "context_history_size": 128,
                    },
                    "tenant-b": {
                        "keys": PROFILE_B_KEYS,
                        "ngram_len": 5,
                        "sampling_table_size": 256,
                        "context_history_size": 128,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
    _load_config_file.cache_clear()
    yield config_path
    _load_config_file.cache_clear()


def test_load_config_and_defaults(synthid_config: Path) -> None:
    config = load_config()

    assert config.default_profile == "tenant-a"
    assert config.get_profile(None).profile_id == "tenant-a"
    assert config.get_profile("tenant-b").keys == tuple(PROFILE_B_KEYS)
    assert config.get_profile("tenant-b").sampling_table_seed == 0


def test_plugin_is_disabled_without_server_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    _load_config_file.cache_clear()

    SynthIDTextLogitsProcessor.validate_params(SamplingParams())
    processor = SynthIDTextLogitsProcessor(
        None,
        torch.device("cpu"),
        False,  # type: ignore[arg-type]
    )

    assert processor.new_req_logits_processor(SamplingParams()) is None
    with pytest.raises(ValueError, match=f"requires.*{CONFIG_ENV_VAR}"):
        processor.validate_params(SamplingParams(extra_args={PROFILE_ARG: "tenant-a"}))


@pytest.mark.parametrize(
    "config, match",
    [
        ({"default_profile": "a", "profiles": {}}, "non-empty profiles"),
        (
            {"default_profile": "missing", "profiles": {"a": {"keys": [1]}}},
            "is not defined",
        ),
        (
            {"default_profile": "a", "profiles": {"a": {"keys": []}}},
            "non-empty list",
        ),
        (
            {
                "default_profile": "a",
                "profiles": {"a": {"keys": [1], "ngram_len": 1}},
            },
            "at least 2",
        ),
    ],
)
def test_rejects_invalid_config(tmp_path: Path, config: dict, match: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    _load_config_file.cache_clear()

    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_profile_selection_and_validation(synthid_config: Path) -> None:
    processor = SynthIDTextLogitsProcessor(None, torch.device("cpu"), False)  # type: ignore[arg-type]

    default = processor.new_req_logits_processor(SamplingParams())
    tenant_b = processor.new_req_logits_processor(
        SamplingParams(extra_args={PROFILE_ARG: "tenant-b"})
    )

    assert default.profile_id == "tenant-a"  # type: ignore[attr-defined]
    assert tenant_b.profile_id == "tenant-b"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Unknown SynthID watermark profile"):
        processor.validate_params(SamplingParams(extra_args={PROFILE_ARG: "missing"}))
    with pytest.raises(ValueError, match="must be a non-empty string"):
        processor.validate_params(SamplingParams(extra_args={PROFILE_ARG: 1}))


def test_matches_transformers_processor(synthid_config: Path) -> None:
    profile = load_config().get_profile("tenant-a")
    adapter = SynthIDTextLogitsProcessor(None, torch.device("cpu"), False)  # type: ignore[arg-type]
    wrapped = adapter.new_req_logits_processor(SamplingParams())
    reference = build_transformers_processor(profile, torch.device("cpu"))
    output_ids: list[int] = []
    generator = torch.Generator().manual_seed(11)

    for token_id in [7, 19, 3, 42, 8, 31]:
        logits = torch.randn(64, generator=generator)
        actual = wrapped(output_ids, logits.clone())
        input_ids = torch.tensor([output_ids], dtype=torch.long)
        expected = reference(input_ids, logits.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(actual, expected)
        output_ids.append(token_id)


def test_batch_state_is_isolated_and_tracks_moves(synthid_config: Path) -> None:
    processor = SynthIDTextLogitsProcessor(None, torch.device("cpu"), False)  # type: ignore[arg-type]
    output_a: list[int] = []
    output_b: list[int] = []
    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[
                (0, SamplingParams(), [], output_a),
                (
                    1,
                    SamplingParams(extra_args={PROFILE_ARG: "tenant-b"}),
                    [],
                    output_b,
                ),
            ],
            moved=[],
        )
    )

    state_a = processor.req_info[0].func
    state_b = processor.req_info[1].func
    assert state_a is not state_b
    assert state_a.processor is not state_b.processor
    assert state_a.processor.sampling_table is state_b.processor.sampling_table
    assert state_a.profile_id == "tenant-a"
    assert state_b.profile_id == "tenant-b"

    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[],
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert processor.req_info[0].func is state_b
    assert processor.req_info[1].func is state_a

    processor.update_state(BatchUpdate(batch_size=1, removed=[1], added=[], moved=[]))
    assert list(processor.req_info) == [0]


def test_placeholder_token_id_fails_closed(synthid_config: Path) -> None:
    processor = SynthIDTextLogitsProcessor(None, torch.device("cpu"), False)  # type: ignore[arg-type]
    request_processor = processor.new_req_logits_processor(SamplingParams())

    with pytest.raises(RuntimeError, match="unresolved output token ID placeholder"):
        request_processor([-1], torch.zeros(32))


def test_resumed_request_restores_transformers_state(synthid_config: Path) -> None:
    profile = load_config().get_profile("tenant-a")
    adapter = SynthIDTextLogitsProcessor(
        None,
        torch.device("cpu"),
        False,  # type: ignore[arg-type]
    )
    resumed = adapter.new_req_logits_processor(SamplingParams())
    uninterrupted = build_transformers_processor(profile, torch.device("cpu"))
    output_ids = [7, 19, 3, 42, 7, 19, 3, 42, 7]
    generator = torch.Generator().manual_seed(22)

    for num_outputs in range(len(output_ids)):
        logits = torch.randn(64, generator=generator)
        uninterrupted(
            torch.tensor([output_ids[:num_outputs]], dtype=torch.long),
            logits.unsqueeze(0),
        )

    current_logits = torch.randn(64, generator=generator)
    actual = resumed(output_ids, current_logits.clone())
    expected = uninterrupted(
        torch.tensor([output_ids], dtype=torch.long),
        current_logits.unsqueeze(0),
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    assert resumed.processor.state is not None  # type: ignore[attr-defined]
    assert uninterrupted.state is not None
    torch.testing.assert_close(
        resumed.processor.state.context,  # type: ignore[attr-defined]
        uninterrupted.state.context,
    )
    torch.testing.assert_close(
        resumed.processor.state.context_history,  # type: ignore[attr-defined]
        uninterrupted.state.context_history,
    )


def _generate_tokens(
    request_processor, *, steps: int = 160, vocab_size: int = 128
) -> list[int]:
    output_ids: list[int] = []
    logits_generator = torch.Generator().manual_seed(123)
    sample_generator = torch.Generator().manual_seed(456)
    for _ in range(steps):
        logits = torch.randn(vocab_size, generator=logits_generator) * 2
        if request_processor is not None:
            logits = request_processor(output_ids, logits)
        probabilities = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(
            probabilities, 1, generator=sample_generator
        ).item()
        output_ids.append(token_id)
    return output_ids


def test_profile_specific_detection_round_trip(synthid_config: Path) -> None:
    processor = SynthIDTextLogitsProcessor(None, torch.device("cpu"), False)  # type: ignore[arg-type]
    tenant_a_processor = processor.new_req_logits_processor(SamplingParams())
    tenant_a_tokens = _generate_tokens(tenant_a_processor)
    plain_tokens = _generate_tokens(None)

    matching = score_token_ids(tenant_a_tokens, "tenant-a").mean_score
    other_profile = score_token_ids(tenant_a_tokens, "tenant-b").mean_score
    plain = score_token_ids(plain_tokens, "tenant-a").mean_score

    assert matching > 0.6
    assert matching > other_profile + 0.08
    assert matching > plain + 0.08


def test_sampling_table_matches_transformers_cpu_rng(synthid_config: Path) -> None:
    profile = load_config().get_profile("tenant-a")
    processor = build_transformers_processor(profile, torch.device("cpu"))
    reference = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=profile.ngram_len,
        keys=list(profile.keys),
        sampling_table_size=profile.sampling_table_size,
        sampling_table_seed=profile.sampling_table_seed,
        context_history_size=profile.context_history_size,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(processor.sampling_table, reference.sampling_table)
