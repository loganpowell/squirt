"""
Squirt Transforms Module

Reusable transform functions for common metrics.
Each transform takes (inputs: Dict, output: Any) -> metric_value.

Usage:
    from squirt.transforms import error_free_transform

    @track(metrics=[
        m.error_free.compute(error_free_transform),
    ])
    def my_component(text: str) -> dict:
        ...

Note: Most users should use MetricBuilder methods like .from_output() and .compute()
instead of manually creating transforms. This module provides reference implementations
for common patterns.
"""

from .validation import error_free_transform

__all__ = [
    "error_free_transform",
]
