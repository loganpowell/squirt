"""
LLM Metrics Plugin for Sleuth

Metrics for LLM API calls - tokens, costs, latency, and quality evaluations.

Usage:
    from sleuth.contrib.llm import llm

    @track(metrics=[
        llm.total_tokens.from_output("usage.total_tokens"),
        llm.cost.from_output("usage.cost_usd"),
    ])
    def my_llm_call(...):
        ...
"""

from .metrics import LLMMetrics, llm

__all__ = ["LLMMetrics", "llm"]
