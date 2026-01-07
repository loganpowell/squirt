"""
Built-in Metrics Module

Convenience re-export of metrics from squirt.metrics with a shorter name.

Usage:
    from squirt import m

    @track(metrics=[
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.memory_mb.from_output("metadata.memory_mb"),
    ])
    def my_component(text: str) -> dict:
        ...
"""

from .builtins import BuiltinMetrics

# Singleton instance - the main entry point
m: BuiltinMetrics = BuiltinMetrics()

__all__ = ["m"]
