"""
Token Transform Functions

Extracts token usage metrics from LLM responses.
Supports multiple SDK formats: OpenAI, Azure OpenAI, Anthropic.
"""

from typing import Any, Dict


def token_count_transform(inputs: Dict[str, Any], output: Any) -> int:
    """
    Extract total token count from output.

    Expects output to have: output["usage"]["total_tokens"]

    Returns:
        Total token count (int), 0 if not available
    """
    if isinstance(output, dict):
        return output.get("usage", {}).get("total_tokens", 0)
    return 0


def prompt_tokens_transform(inputs: Dict[str, Any], output: Any) -> int:
    """
    Extract prompt/input token count.

    Supports:
    - OpenAI/Azure: output["usage"]["prompt_tokens"]
    - Anthropic: output["usage"]["input_tokens"]

    Returns:
        Prompt token count (int), 0 if not available
    """
    if isinstance(output, dict):
        usage = output.get("usage", {})
        return usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    return 0


def completion_tokens_transform(inputs: Dict[str, Any], output: Any) -> int:
    """
    Extract completion/output token count.

    Supports:
    - OpenAI/Azure: output["usage"]["completion_tokens"]
    - Anthropic: output["usage"]["output_tokens"]

    Returns:
        Completion token count (int), 0 if not available
    """
    if isinstance(output, dict):
        usage = output.get("usage", {})
        return usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return 0


def total_tokens_transform(inputs: Dict[str, Any], output: Any) -> int:
    """
    Extract total token count from LLM response.

    Supports multiple SDK formats:
    - OpenAI/Azure: output["usage"]["total_tokens"]
    - Anthropic: sum of input_tokens + output_tokens

    Returns:
        Total token count (int), 0 if usage data missing
    """
    if not isinstance(output, dict):
        return 0

    usage = output.get("usage", {})

    # Try total_tokens first (OpenAI/Azure)
    total = usage.get("total_tokens")
    if total is not None:
        return total

    # Fall back to sum of input/output (Anthropic)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return input_tokens + output_tokens


__all__ = [
    "token_count_transform",
    "prompt_tokens_transform",
    "completion_tokens_transform",
    "total_tokens_transform",
]
