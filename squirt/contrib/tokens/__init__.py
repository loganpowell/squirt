"""
Token and cost tracking metrics contrib module.

Provides namespace for estimating token usage and calculating API costs
based on character counts (model-agnostic, no external dependencies).

Usage:
    from squirt.contrib.tokens import tokens

    @track(metrics=[
        tokens.count(input_path="query", output_path="response"),
        tokens.cost(
            input_cost_per_1m=5.0,
            output_cost_per_1m=15.0,
            input_path="query",
            output_path="response"
        )
    ])
    def my_llm_component(query: str) -> dict:
        ...
"""

from .metrics import TokensMetrics

# Singleton instance
tokens = TokensMetrics()

__all__ = ["tokens", "TokensMetrics"]
