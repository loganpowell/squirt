"""
Squirt Transforms Module

Reusable transform functions for common metrics.
Each transform takes (inputs: Dict, output: Any) -> metric_value.

Usage:
    from squirt.transforms import runtime_transform, token_cost_transform
    from squirt.transforms.validation import json_valid, has_required_fields
    from squirt.transforms.similarity import expected_match, compute_similarity

    @track(metrics=[
        Metric("runtime_ms", runtime_transform, AggregationType.SUM),
        Metric("cost", token_cost_transform, AggregationType.SUM),
    ])
    def my_component(text: str) -> dict:
        ...
"""

from .cost import (
    create_token_cost_transform,
    token_cost_transform,
)
from .performance import (
    cpu_usage_transform,
    latency_p95_transform,
    memory_usage_transform,
    throughput_transform,
)
from .runtime import (
    runtime_from_metadata,
    runtime_transform,
)
from .similarity import (
    compute_similarity,
    create_expected_match_transform,
    expected_match_transform,
)
from .tokens import (
    completion_tokens_transform,
    prompt_tokens_transform,
    token_count_transform,
    total_tokens_transform,
)
from .utility import (
    always_fail_transform,
    always_pass_transform,
    create_threshold_transform,
    output_length_transform,
)
from .validation import (
    create_pydantic_validation_transform,
    error_free_transform,
    has_required_fields_transform,
    json_valid_transform,
)

__all__ = [
    # Runtime
    "runtime_transform",
    "runtime_from_metadata",
    # Tokens
    "token_count_transform",
    "prompt_tokens_transform",
    "completion_tokens_transform",
    "total_tokens_transform",
    # Cost
    "token_cost_transform",
    "create_token_cost_transform",
    # Validation
    "json_valid_transform",
    "has_required_fields_transform",
    "error_free_transform",
    "create_pydantic_validation_transform",
    # Similarity
    "expected_match_transform",
    "compute_similarity",
    "create_expected_match_transform",
    # Performance
    "memory_usage_transform",
    "cpu_usage_transform",
    "throughput_transform",
    "latency_p95_transform",
    # Utility
    "always_pass_transform",
    "always_fail_transform",
    "output_length_transform",
    "create_threshold_transform",
]
