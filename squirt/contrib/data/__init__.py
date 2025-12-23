"""
Data Structure Metrics Plugin for Squirt

Generic metrics for structured data analysis.

Usage:
    from squirt.contrib.data import data

    @track(metrics=[
        data.field_count.from_output("metadata.field_count"),
        data.nesting_depth.from_output("metadata.depth"),
    ])
    def my_component(...):
        ...
"""

from .metrics import DataMetrics, data

__all__ = ["DataMetrics", "data"]
