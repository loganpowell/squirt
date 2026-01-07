"""
Utility Transform Functions

General-purpose transforms for testing and custom logic.
"""

from collections.abc import Callable
from typing import Any


def always_pass_transform(inputs: dict[str, Any], output: Any) -> bool:
    """
    Always returns True.

    Useful as a placeholder or for testing.
    """
    return True


def always_fail_transform(inputs: dict[str, Any], output: Any) -> bool:
    """
    Always returns False.

    Useful for testing failure paths.
    """
    return False


def output_length_transform(inputs: dict[str, Any], output: Any) -> int:
    """
    Return length of output.

    - Strings: character count
    - Lists/Dicts: item count
    - Other: 0

    Returns:
        Length of output (int)
    """
    if isinstance(output, str):
        return len(output)
    if isinstance(output, (list, dict)):
        return len(output)
    return 0


def create_threshold_transform(
    value_extractor: Callable[[dict[str, Any], Any], float],
    threshold: float,
    above: bool = True,
) -> Callable[[dict[str, Any], Any], bool]:
    """
    Factory to create a threshold-based transform.

    Args:
        value_extractor: Function to extract numeric value from (inputs, output)
        threshold: Threshold value to compare against
        above: If True, passes when value >= threshold; else when value <= threshold

    Returns:
        Transform returning True/False based on threshold

    Example:
        # Pass if runtime is under 1000ms
        fast_enough = create_threshold_transform(
            value_extractor=lambda i, o: o.get("metadata", {}).get("runtime_ms", 0),
            threshold=1000,
            above=False  # Pass when value <= threshold
        )
    """

    def transform(inputs: dict[str, Any], output: Any) -> bool:
        value = value_extractor(inputs, output)
        if above:
            return value >= threshold
        return value <= threshold

    return transform


__all__ = [
    "always_pass_transform",
    "always_fail_transform",
    "output_length_transform",
    "create_threshold_transform",
]
