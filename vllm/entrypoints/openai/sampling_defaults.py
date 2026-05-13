# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Default sampling-parameter overrides applied across all generation endpoints.

Values here mirror the hard-coded defaults used by the ``/v1/responses``
endpoint and are applied after per-request defaults are resolved.
"""

from vllm.sampling_params import SamplingParams

# ---------------------------------------------------------------------------
# Override values
# ---------------------------------------------------------------------------

DEFAULT_THINKING_TOKEN_BUDGET = 8192
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 25
DEFAULT_MIN_P = 0.2
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_PRESENCE_PENALTY = 0.1


def apply_sampling_defaults(params: SamplingParams) -> None:
    """Mutate *params* in-place with the configured defaults."""
    params.thinking_token_budget = DEFAULT_THINKING_TOKEN_BUDGET
    params.temperature = DEFAULT_TEMPERATURE
    params.top_p = DEFAULT_TOP_P
    params.top_k = DEFAULT_TOP_K
    params.min_p = DEFAULT_MIN_P
    params.repetition_penalty = DEFAULT_REPETITION_PENALTY
    params.presence_penalty = DEFAULT_PRESENCE_PENALTY
    if params.max_tokens == 64000:
        params.max_tokens = 128000
