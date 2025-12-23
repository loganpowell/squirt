"""
Cost Transform Functions

Calculate LLM API costs from token usage.
Supports multiple providers with model-specific pricing.
"""

from typing import Any, Callable, Dict, Optional


# Model-specific pricing (per million tokens) - Updated December 2024
MODEL_PRICING = {
    # OpenAI GPT-4 family
    "gpt-4": (30.0, 60.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    # OpenAI GPT-3.5
    "gpt-3.5-turbo": (1.50, 2.00),
    # Anthropic Claude 3
    "claude-3-opus": (15.0, 75.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    # Anthropic Claude 3.5
    "claude-3.5-sonnet": (3.0, 15.0),
    "claude-3.5-haiku": (0.80, 4.0),
    # Google Gemini
    "gemini-pro": (0.50, 1.50),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
}


def get_model_pricing(model: str) -> tuple[float, float]:
    """
    Get pricing for a model name.

    Args:
        model: Model name (can be versioned like "gpt-4-0613")

    Returns:
        Tuple of (input_price, output_price) per million tokens
    """
    model_lower = model.lower()

    for model_key, prices in MODEL_PRICING.items():
        if model_key in model_lower:
            return prices

    # Default to GPT-4 pricing if unknown
    return (30.0, 60.0)


def token_cost_transform(inputs: Dict[str, Any], output: Any) -> float:
    """
    Calculate LLM API cost from token usage.

    Supports multiple SDK response formats:
    - OpenAI: output["usage"]["prompt_tokens"], output["usage"]["completion_tokens"]
    - Azure OpenAI: Same as OpenAI
    - Anthropic: output["usage"]["input_tokens"], output["usage"]["output_tokens"]

    Uses model-specific per-million-token pricing.

    Returns:
        Cost in USD (float), 0.0 if usage data missing
    """
    if not isinstance(output, dict):
        return 0.0

    usage = output.get("usage", {})
    model = output.get("model", "gpt-4")  # Default to GPT-4

    # Try OpenAI/Azure format first, then Anthropic
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    # Get pricing for model
    input_price, output_price = get_model_pricing(model)

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return round(input_cost + output_cost, 6)


def create_token_cost_transform(
    input_cost_per_1k: float = 0.00003,
    output_cost_per_1k: float = 0.00006,
    model: Optional[str] = None,
) -> Callable[[Dict[str, Any], Any], float]:
    """
    Factory to create a token cost transform with custom pricing.

    Args:
        input_cost_per_1k: Cost per 1000 input tokens (legacy pricing format)
        output_cost_per_1k: Cost per 1000 output tokens (legacy pricing format)
        model: Optional model name for auto-detection

    Returns:
        Transform function that calculates cost

    Example:
        # Custom pricing
        custom_cost = create_token_cost_transform(
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03
        )

        # Model-based pricing
        gpt4_cost = create_token_cost_transform(model="gpt-4o")
    """

    def transform(inputs: Dict[str, Any], output: Any) -> float:
        if not isinstance(output, dict):
            return 0.0

        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = (
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )

        # Try to get model from output or use provided model
        output_model = output.get("model", model)

        # Get pricing
        if output_model:
            in_price, out_price = get_model_pricing(output_model)
            # Convert from per-million to per-thousand
            in_cost = in_price / 1000
            out_cost = out_price / 1000
        else:
            in_cost = input_cost_per_1k
            out_cost = output_cost_per_1k

        total_cost = (prompt_tokens / 1000 * in_cost) + (
            completion_tokens / 1000 * out_cost
        )
        return round(total_cost, 6)

    return transform


__all__ = [
    "token_cost_transform",
    "create_token_cost_transform",
    "get_model_pricing",
    "MODEL_PRICING",
]
