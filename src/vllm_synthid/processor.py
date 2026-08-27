# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-tenant SynthID Text watermarking plugin for vLLM.

This custom logits processor wraps Transformers' production
``SynthIDTextWatermarkLogitsProcessor``. Watermark profiles, including their
secret keys, are configured by the server. Requests may select a named profile
without receiving the underlying key material.
"""

from __future__ import annotations

import json
import os
from copy import copy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers.generation import SynthIDTextWatermarkLogitsProcessor
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

CONFIG_ENV_VAR = "SYNTHID_CONFIG"
PROFILE_ARG = "synthid_watermark_profile"
_INT64_MAX = torch.iinfo(torch.int64).max


@dataclass(frozen=True)
class SynthIDProfile:
    """Configuration for one independently detectable watermark."""

    profile_id: str
    keys: tuple[int, ...]
    ngram_len: int = 5
    sampling_table_size: int = 65536
    sampling_table_seed: int = 0
    context_history_size: int = 1024


@dataclass(frozen=True)
class SynthIDConfig:
    """Validated server-side watermark configuration."""

    default_profile: str
    profiles: dict[str, SynthIDProfile]

    def get_profile(self, profile_id: str | None) -> SynthIDProfile:
        selected = profile_id or self.default_profile
        try:
            return self.profiles[selected]
        except KeyError as exc:
            raise ValueError(
                f"Unknown SynthID watermark profile: {selected!r}"
            ) from exc


def _require_positive_int(value: Any, field: str, profile_id: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"SynthID profile {profile_id!r} field {field!r} must be a positive integer"
        )
    return value


def _parse_profile(profile_id: str, raw: Any) -> SynthIDProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"SynthID profile {profile_id!r} must be an object")

    keys = raw.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError(
            f"SynthID profile {profile_id!r} field 'keys' must be a non-empty list"
        )
    if any(
        not isinstance(key, int) or isinstance(key, bool) or not 0 <= key <= _INT64_MAX
        for key in keys
    ):
        raise ValueError(
            f"SynthID profile {profile_id!r} keys must be int64-compatible "
            "non-negative integers"
        )

    ngram_len = _require_positive_int(raw.get("ngram_len", 5), "ngram_len", profile_id)
    if ngram_len < 2:
        raise ValueError(
            f"SynthID profile {profile_id!r} field 'ngram_len' must be at least 2"
        )

    seed = raw.get("sampling_table_seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(
            f"SynthID profile {profile_id!r} field 'sampling_table_seed' "
            "must be an integer"
        )

    return SynthIDProfile(
        profile_id=profile_id,
        keys=tuple(keys),
        ngram_len=ngram_len,
        sampling_table_size=_require_positive_int(
            raw.get("sampling_table_size", 65536),
            "sampling_table_size",
            profile_id,
        ),
        sampling_table_seed=seed,
        context_history_size=_require_positive_int(
            raw.get("context_history_size", 1024),
            "context_history_size",
            profile_id,
        ),
    )


@cache
def _load_config_file(path: str) -> SynthIDConfig:
    try:
        with open(path, encoding="utf-8") as config_file:
            raw = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to load SynthID configuration {path!r}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError("SynthID configuration must be a JSON object")
    default_profile = raw.get("default_profile")
    if not isinstance(default_profile, str) or not default_profile:
        raise ValueError("SynthID configuration requires a non-empty default_profile")
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("SynthID configuration requires a non-empty profiles object")

    profiles: dict[str, SynthIDProfile] = {}
    for profile_id, profile in raw_profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("SynthID profile IDs must be non-empty strings")
        profiles[profile_id] = _parse_profile(profile_id, profile)

    if default_profile not in profiles:
        raise ValueError(
            f"SynthID default profile {default_profile!r} is not defined in profiles"
        )
    return SynthIDConfig(default_profile=default_profile, profiles=profiles)


def load_config(path: str | os.PathLike[str] | None = None) -> SynthIDConfig:
    """Load and validate the configured SynthID watermark profiles."""
    config_path = (
        os.fspath(path) if path is not None else os.environ.get(CONFIG_ENV_VAR)
    )
    if not config_path:
        raise ValueError(f"{CONFIG_ENV_VAR} must point to a SynthID JSON config file")
    return _load_config_file(str(Path(config_path).expanduser().resolve()))


def load_config_if_configured() -> SynthIDConfig | None:
    """Load server configuration, or return ``None`` when disabled."""
    config_path = os.environ.get(CONFIG_ENV_VAR)
    return load_config(config_path) if config_path else None


@cache
def _canonical_sampling_table(size: int, seed: int) -> torch.Tensor:
    """Build the sampling table with the CPU RNG used by detection."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, (size,), generator=generator, device="cpu")


def build_transformers_processor(
    profile: SynthIDProfile,
    device: torch.device,
    sampling_table: torch.Tensor | None = None,
) -> SynthIDTextWatermarkLogitsProcessor:
    """Create a Transformers processor with a device-independent table."""
    processor = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=profile.ngram_len,
        keys=list(profile.keys),
        sampling_table_size=profile.sampling_table_size,
        sampling_table_seed=profile.sampling_table_seed,
        context_history_size=profile.context_history_size,
        device=device,
    )
    canonical = sampling_table
    if canonical is None:
        canonical = _canonical_sampling_table(
            profile.sampling_table_size, profile.sampling_table_seed
        ).to(device)
    processor.sampling_table = canonical
    return processor


class _PerRequestSynthIDProcessor:
    """Request-local adapter around Transformers' stateful processor."""

    def __init__(
        self,
        profile: SynthIDProfile,
        processor: SynthIDTextWatermarkLogitsProcessor,
    ) -> None:
        self.profile = profile
        self.profile_id = profile.profile_id
        self.device = processor.device
        self.initialized = False
        self.processor = copy(processor)
        self.processor.state = None

    def _restore_state(self, output_ids: list[int]) -> None:
        """Rebuild Transformers state when vLLM resumes a removed request."""
        self.processor._init_state(1)
        state = self.processor.state
        assert state is not None

        context_len = self.profile.ngram_len - 1
        preceding_tokens = [0] * context_len + output_ids[:-1]
        state.context = torch.tensor(
            [preceding_tokens[-context_len:]],
            dtype=torch.long,
            device=self.device,
        )

        history_len = min(len(output_ids), self.profile.context_history_size)
        first_context = len(output_ids) - history_len
        contexts = []
        padded_tokens = [0] * context_len + output_ids
        for num_previous_tokens in range(first_context, len(output_ids)):
            end = context_len + num_previous_tokens
            contexts.append(padded_tokens[end - context_len : end])
        if contexts:
            context_tensor = torch.tensor(
                contexts, dtype=torch.long, device=self.device
            )
            hashes = self.processor.accumulate_hash(
                torch.ones(len(contexts), dtype=torch.long, device=self.device),
                context_tensor,
            )
            state.context_history[0, :history_len] = hashes.flip(0)
        state.num_calls = len(output_ids)

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        has_placeholder = (
            any(token_id < 0 for token_id in output_ids)
            if not self.initialized
            else bool(output_ids and output_ids[-1] < 0)
        )
        if has_placeholder:
            raise RuntimeError(
                "SynthID received an unresolved output token ID placeholder; "
                "disable async scheduling or use a vLLM version that repairs "
                "custom logits processor token IDs"
            )
        if not self.initialized and output_ids:
            self._restore_state(output_ids)
        input_ids = torch.tensor([output_ids], dtype=torch.long, device=self.device)
        updated = self.processor(input_ids, logits.unsqueeze(0))
        self.initialized = True
        return updated.squeeze(0)


class SynthIDTextLogitsProcessor(AdapterLogitsProcessor):
    """Apply server-managed SynthID profiles when the plugin is configured."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        profile_id = params.extra_args and params.extra_args.get(PROFILE_ARG)
        if profile_id is not None and (
            not isinstance(profile_id, str) or not profile_id
        ):
            raise ValueError(f"{PROFILE_ARG} must be a non-empty string")
        config = load_config_if_configured()
        if config is None:
            if profile_id is not None:
                raise ValueError(
                    f"{PROFILE_ARG} requires the server to configure {CONFIG_ENV_VAR}"
                )
            return
        config.get_profile(profile_id)

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        self.config = load_config_if_configured()
        self.device = device
        self.sampling_tables: dict[tuple[int, int], torch.Tensor] = {}
        self.processor_templates: dict[str, SynthIDTextWatermarkLogitsProcessor] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _sampling_table(self, profile: SynthIDProfile) -> torch.Tensor:
        key = (profile.sampling_table_size, profile.sampling_table_seed)
        table = self.sampling_tables.get(key)
        if table is None:
            table = _canonical_sampling_table(*key).to(self.device)
            self.sampling_tables[key] = table
        return table

    def _processor_template(
        self, profile: SynthIDProfile
    ) -> SynthIDTextWatermarkLogitsProcessor:
        processor = self.processor_templates.get(profile.profile_id)
        if processor is None:
            processor = build_transformers_processor(
                profile,
                self.device,
                sampling_table=self._sampling_table(profile),
            )
            self.processor_templates[profile.profile_id] = processor
        return processor

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None:
        self.validate_params(params)
        if self.config is None:
            return None
        profile_id = params.extra_args and params.extra_args.get(PROFILE_ARG)
        profile = self.config.get_profile(profile_id)
        return _PerRequestSynthIDProcessor(profile, self._processor_template(profile))
