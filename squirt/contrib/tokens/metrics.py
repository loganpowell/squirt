"""
Token and Cost Metrics for LLM Operations.

Provides character-based token estimation (3.5 chars per token)
and cost calculation with configurable pricing.
"""

from typing import Any, Callable, Dict

from ...categories.system import SystemMetric
from ...core.types import Metric
from ...plugins.base import MetricNamespace


def _extract_field(data: Dict[str, Any], path: str) -> str:
    """
    Extract field value from dict using dot notation.

    Args:
        data: Dictionary to extract from
        path: Dot-notation path (e.g., "data.taxRule")

    Returns:
        Field value as string, empty string if not found
    """
    value = data
    for key in path.split("."):
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return ""

    # Convert to string
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from character count.

    Uses 3.5 characters per token approximation (model-agnostic).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return int(len(text) / 3.5)


def create_token_estimator(
    input_path: str,
    output_path: str,
    system_prompt: str = "",
) -> Callable[[Dict[str, Any], Any], int]:
    """
    Factory to create a token counting transform.

    Args:
        input_path: Path to input field in inputs dict
        output_path: Path to output field in output dict
        system_prompt: System prompt text to include in count

    Returns:
        Transform function that estimates token count
    """

    def transform(inputs: Dict[str, Any], output: Any) -> int:
        """Estimate total tokens from input, system prompt, and output."""
        # Extract input text
        input_text = _extract_field(inputs, input_path)
        input_tokens = _estimate_tokens(input_text)

        # Extract output text
        if not isinstance(output, dict):
            output = {"result": output}
        output_text = _extract_field(output, output_path)
        output_tokens = _estimate_tokens(output_text)

        # System prompt tokens
        system_tokens = _estimate_tokens(system_prompt)

        return input_tokens + system_tokens + output_tokens

    return transform


def create_token_cost_transform(
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    input_path: str,
    output_path: str,
    system_prompt: str = "",
) -> Callable[[Dict[str, Any], Any], float]:
    """
    Factory to create a cost calculation transform.

    Args:
        input_cost_per_1m: Cost per 1M input tokens
        output_cost_per_1m: Cost per 1M output tokens
        input_path: Path to input field in inputs dict
        output_path: Path to output field in output dict
        system_prompt: System prompt text to include in input cost

    Returns:
        Transform function that calculates cost in USD
    """

    def transform(inputs: Dict[str, Any], output: Any) -> float:
        """Calculate API cost from token usage."""
        # Extract input text
        input_text = _extract_field(inputs, input_path)
        input_tokens = _estimate_tokens(input_text)

        # Extract output text
        if not isinstance(output, dict):
            output = {"result": output}
        output_text = _extract_field(output, output_path)
        output_tokens = _estimate_tokens(output_text)

        # System prompt counts as input
        system_tokens = _estimate_tokens(system_prompt)
        total_input_tokens = input_tokens + system_tokens

        # Calculate costs
        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m

        return input_cost + output_cost

    return transform


class TokensMetrics(MetricNamespace):
    """
    Token and cost tracking metrics namespace.

    Provides methods for estimating token usage and calculating API costs
    based on character counts (3.5 chars per token approximation).

    Use this for:
    - Token counting with configurable field paths
    - Cost calculation with custom pricing
    - Tracking LLM API expenses

    Examples:
        # Count tokens from specific fields
        tokens.count(input_path="description", output_path="bullets")

        # Include system prompt in token count
        tokens.count(
            input_path="description",
            output_path="bullets",
            system_prompt="You are a helpful assistant."
        )

        # Calculate cost with custom pricing
        tokens.cost(
            input_cost_per_1m=5.0,
            output_cost_per_1m=15.0,
            input_path="description",
            output_path="bullets",
            system_prompt="You are a helpful assistant."
        )
    """

    def count(
        self,
        input_path: str,
        output_path: str,
        system_prompt: str = "",
    ) -> Metric:
        """
        Estimate token count from specific input/output fields and system prompt.

        Uses character-based approximation (3.5 chars per token).
        Supports dot notation for nested fields (e.g., "data.taxRule").

        Args:
            input_path: Field path in inputs to count (e.g., "description")
            output_path: Field path in output to count (e.g., "bullets")
            system_prompt: System prompt text to include in token count (default: "")

        Returns:
            Metric configured to estimate token count

        Examples:
            # Count user input + system prompt + output
            tokens.count(
                input_path="description",
                output_path="bullets",
                system_prompt="You are a helpful assistant."
            )

            # Count just input + output (no system prompt)
            tokens.count(input_path="description", output_path="bullets")

            # Count nested fields
            tokens.count(input_path="query", output_path="data.taxAssistRule")
        """
        estimator = create_token_estimator(
            input_path=input_path, output_path=output_path, system_prompt=system_prompt
        )

        parts = [input_path, output_path]
        if system_prompt:
            parts.insert(1, "system_prompt")

        return self._define(
            "total_tokens",
            system_metric=SystemMetric.TOTAL_TOKENS,
            description=f"Total tokens from {' + '.join(parts)}",
        ).compute(estimator)

    def cost(
        self,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
        input_path: str,
        output_path: str,
        system_prompt: str = "",
    ) -> Metric:
        """
        Calculate API cost from token usage with custom pricing.

        Uses character-based token estimation (3.5 chars per token).
        Supports dot notation for nested fields.

        Args:
            input_cost_per_1m: Cost per 1M input tokens
            output_cost_per_1m: Cost per 1M output tokens
            input_path: Field path in inputs to count (e.g., "description")
            output_path: Field path in output to count (e.g., "bullets")
            system_prompt: System prompt text to include in input cost (default: "")

        Returns:
            Metric configured to calculate cost in USD

        Examples:
            # GPT-4o pricing
            tokens.cost(
                input_cost_per_1m=5.0,
                output_cost_per_1m=15.0,
                input_path="description",
                output_path="bullets",
                system_prompt="You are a helpful assistant."
            )

            # Custom model pricing
            tokens.cost(
                input_cost_per_1m=2.5,
                output_cost_per_1m=10.0,
                input_path="query",
                output_path="response"
            )
        """
        cost_transform = create_token_cost_transform(
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
            input_path=input_path,
            output_path=output_path,
            system_prompt=system_prompt,
        )

        parts = [input_path, output_path]
        if system_prompt:
            parts.insert(1, "system_prompt")

        return self._define(
            "cost_usd",
            system_metric=SystemMetric.COST_USD,
            description=f"API cost (${input_cost_per_1m}/{output_cost_per_1m} per 1M) from {' + '.join(parts)}",
        ).compute(cost_transform)


__all__ = [
    "TokensMetrics",
    "create_token_estimator",
    "create_token_cost_transform",
]
