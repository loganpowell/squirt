"""
Data Structure Metrics for Squirt

Generic metrics for analyzing data structure characteristics - nesting depth,
field counts, validation results, etc.

Usage:
    from squirt.contrib.data import data

    @track(metrics=[
        data.field_count.from_output("metadata.field_count"),
        data.nesting_depth.from_output("metadata.max_depth"),
        data.structure_valid.from_output("metadata.valid"),
    ])
    def extract_fields(text: str) -> dict:
        ...
"""

from squirt.plugins import MetricBuilder, AggregationType, SystemMetric


class DataMetrics:
    """
    Metrics for analyzing structured data output.

    Generic metrics for detecting structural characteristics like:
    - Field counts and presence
    - Nesting depth
    - Validation status
    - Match accuracy

    All attributes have explicit type annotations for IDE autocomplete support.
    """

    field_count: MetricBuilder = MetricBuilder(
        "field_count",
        AggregationType.SUM,
        description="Number of fields/keys in output structure",
    )

    nesting_depth: MetricBuilder = MetricBuilder(
        "nesting_depth",
        AggregationType.MAX,
        description="Maximum nesting depth of output structure",
    )

    node_count: MetricBuilder = MetricBuilder(
        "node_count",
        AggregationType.SUM,
        description="Number of nodes in tree structure",
    )

    # Validation/accuracy metrics (implementation-specific)
    # Projects should define domain-specific validation metrics:
    # - has_required_fields, structure_valid (schema validation)
    # - match_accuracy, field_accuracy (comparison)
    # - expression_valid (domain-specific validation)
    # These depend on your data schema and validation rules


# Singleton instance
data = DataMetrics()

__all__ = ["DataMetrics", "data"]
