"""
LLM Metrics for Sleuth

Metrics for LLM API calls - tokens, costs, latency, and quality evaluations.

Usage:
    from sleuth.contrib.llm import llm

    @track(metrics=[
        llm.total_tokens.from_output("usage.total_tokens"),
        llm.cost.from_output("usage.cost_usd"),
        llm.latency.from_output("metadata.response_ms"),
    ])
    def call_llm(prompt: str) -> dict:
        ...
"""

from sleuth.plugins import MetricBuilder, AggregationType, SystemMetric


class LLMMetrics:
    """
    Metrics for LLM API calls.

    Tracks token usage, costs, latency, and quality evaluation scores.
    All attributes have explicit type annotations for IDE autocomplete support.
    """

    # Token metrics
    total_tokens: MetricBuilder = MetricBuilder(
        "total_tokens",
        AggregationType.SUM,
        SystemMetric.TOTAL_TOKENS,
        description="Total tokens consumed (input + output)",
    )

    input_tokens: MetricBuilder = MetricBuilder(
        "input_tokens",
        AggregationType.SUM,
        SystemMetric.TOTAL_TOKENS,
        description="Input/prompt tokens consumed",
    )

    output_tokens: MetricBuilder = MetricBuilder(
        "output_tokens",
        AggregationType.SUM,
        SystemMetric.TOTAL_TOKENS,
        description="Output/completion tokens generated",
    )

    # Cost metrics
    cost: MetricBuilder = MetricBuilder(
        "cost_usd",
        AggregationType.SUM,
        SystemMetric.COST_USD,
        description="API call cost in USD",
    )

    # Latency metrics
    latency: MetricBuilder = MetricBuilder(
        "llm_latency_ms",
        AggregationType.AVERAGE,
        SystemMetric.LATENCY_P95,
        description="Response latency in milliseconds",
    )

    time_to_first_token: MetricBuilder = MetricBuilder(
        "time_to_first_token_ms",
        AggregationType.AVERAGE,
        SystemMetric.LATENCY_P95,
        description="Time to first token in milliseconds (streaming)",
    )

    tokens_per_second: MetricBuilder = MetricBuilder(
        "tokens_per_second",
        AggregationType.AVERAGE,
        SystemMetric.THROUGHPUT,
        description="Token generation rate",
    )

    # Quality/validation metrics (implementation-specific)
    # Projects should define domain-specific quality metrics:
    # - relevance, coherence, groundedness, fluency, completeness (Azure AI style)
    # - schema_valid, parse_success (validation)
    # - retry_count (reliability)
    # These vary by use case and should be in project-level metrics


# Singleton instance
llm = LLMMetrics()

__all__ = ["LLMMetrics", "llm"]
