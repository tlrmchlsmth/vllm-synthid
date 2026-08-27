# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score text against a configured SynthID watermark profile."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from transformers import AutoTokenizer

from vllm_synthid.processor import (
    build_transformers_processor,
    load_config,
)


@dataclass(frozen=True)
class DetectionScore:
    """Mean-score detector result without a deployment-specific threshold."""

    profile: str
    num_tokens: int
    num_scored_ngrams: int
    mean_score: float


def score_token_ids(
    token_ids: list[int], profile_id: str | None, eos_token_id: int | None = None
) -> DetectionScore:
    """Compute the reference mean score for one token sequence."""
    profile = load_config().get_profile(profile_id)
    if len(token_ids) < profile.ngram_len:
        raise ValueError(
            f"Detection requires at least {profile.ngram_len} tokens, "
            f"but received {len(token_ids)}"
        )

    processor = build_transformers_processor(profile, torch.device("cpu"))
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    g_values = processor.compute_g_values(input_ids)
    mask = processor.compute_context_repetition_mask(input_ids)
    if eos_token_id is not None:
        eos_mask = processor.compute_eos_token_mask(input_ids, eos_token_id)
        mask = mask & eos_mask[:, profile.ngram_len - 1 :]

    num_scored_ngrams = int(mask.sum().item())
    if num_scored_ngrams == 0:
        raise ValueError("Detection found no non-repeated n-grams to score")
    weighted_sum = (g_values * mask.unsqueeze(-1)).sum()
    mean_score = weighted_sum / (num_scored_ngrams * len(profile.keys))
    return DetectionScore(
        profile=profile.profile_id,
        num_tokens=len(token_ids),
        num_scored_ngrams=num_scored_ngrams,
        mean_score=float(mean_score.item()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--text", required=True, help="Text to score")
    parser.add_argument(
        "--profile", help="Configured profile ID; defaults to default_profile"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids = tokenizer.encode(args.text, add_special_tokens=False)
    score = score_token_ids(token_ids, args.profile, tokenizer.eos_token_id)
    print(json.dumps(asdict(score), sort_keys=True))


if __name__ == "__main__":
    main()
