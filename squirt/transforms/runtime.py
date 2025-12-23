"""
Runtime Transform Functions

Extracts runtime metrics from component outputs.
"""

from typing import Any, Dict


def runtime_transform(inputs: Dict[str, Any], output: Any) -> int:
    """
    Extract runtime in milliseconds from output metadata.

    Expects output to have: output["metadata"]["runtime_ms"]

    Returns:
        Runtime in milliseconds (int), 0 if not available
    """
    if isinstance(output, dict):
        return output.get("metadata", {}).get("runtime_ms", 0)
    return 0


def runtime_from_metadata(inputs: Dict[str, Any], output: Any) -> int:
    """
    Alias for runtime_transform.

    Extracts runtime from output["metadata"]["runtime_ms"].
    """
    return runtime_transform(inputs, output)


__all__ = ["runtime_transform", "runtime_from_metadata"]
