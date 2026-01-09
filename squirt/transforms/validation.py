"""
Validation Transform Functions

Check output validity against schemas and constraints.

This module provides reference implementations for common validation patterns.
Most users should use MetricBuilder methods instead:
    m.error_free.compute(error_free_transform)
    m.structure_valid.compute(your_custom_validator)
"""

from typing import Any


def error_free_transform(inputs: dict[str, Any], output: Any) -> bool:
    """
    Check if output contains no error indicators.

    Returns False if output contains common error keys:
    - "error", "errors", "exception", "traceback", "fault"

    Returns:
        True if no error indicators found, False otherwise

    Example:
        @track(metrics=[m.error_free.compute(error_free_transform)])
        def my_component():
            return {"result": "success"}  # Returns True
            # return {"error": "failed"}  # Returns False
    """
    if not isinstance(output, dict):
        return True

    error_indicators = ["error", "errors", "exception", "traceback", "fault"]
    return not any(key in output for key in error_indicators)


__all__ = ["error_free_transform"]
