"""
LLM Metrics for Squirt

Metrics for LLM API calls - tokens, costs, latency, and quality evaluations.

Usage:
    from squirt.contrib.llm import llm

    @track(metrics=[
        llm.total_tokens.from_output("usage.total_tokens"),
        llm.cost.from_output("usage.cost_usd"),
        llm.latency.from_output("metadata.response_ms"),
    ])
    def call_llm(prompt: str) -> dict:
        ...
"""

from squirt.plugins import AggregationType, MetricBuilder, MetricNamespace, SystemMetric


class LLMMetrics(MetricNamespace):
    """
    Metrics for LLM API calls.

    Tracks token usage, costs, latency, and quality evaluation scores.
    """

    @property
    def total_tokens(self) -> MetricBuilder:
        """Total tokens consumed (input + output)."""
        return self._define(
            "total_tokens",
            AggregationType.SUM,
            SystemMetric.TOTAL_TOKENS,
            description="Total tokens consumed (input + output)",
        )

    @property
    def input_tokens(self) -> MetricBuilder:
        """Input/prompt tokens consumed."""
        return self._define(
            "input_tokens",
            AggregationType.SUM,
            SystemMetric.TOTAL_TOKENS,
            description="Input/prompt tokens consumed",
        )

    @property
    def output_tokens(self) -> MetricBuilder:
        """Output/completion tokens generated."""
        return self._define(
            "output_tokens",
            AggregationType.SUM,
            SystemMetric.TOTAL_TOKENS,
            description="Output/completion tokens generated",
        )

    @property
    def cost(self) -> MetricBuilder:
        """API call cost in USD."""
        return self._define(
            "cost_usd",
            AggregationType.SUM,
            SystemMetric.COST_USD,
            description="API call cost in USD",
        )

    @property
    def latency(self) -> MetricBuilder:
        """Response latency in milliseconds."""
        return self._define(
            "llm_latency_ms",
            AggregationType.AVERAGE,
            SystemMetric.LATENCY_P95,
            description="Response latency in milliseconds",
        )

    @property
    def time_to_first_token(self) -> MetricBuilder:
        """Time to first token in milliseconds (streaming)."""
        return self._define(
            "time_to_first_token_ms",
            AggregationType.AVERAGE,
            SystemMetric.LATENCY_P95,
            description="Time to first token in milliseconds (streaming)",
        )

    @property
    def tokens_per_second(self) -> MetricBuilder:
        """Token generation rate."""
        return self._define(
            "tokens_per_second",
            AggregationType.AVERAGE,
            SystemMetric.THROUGHPUT,
            description="Token generation rate",
        )


# Singleton instance
llm = LLMMetrics()

__all__ = ["LLMMetrics", "llm"]
