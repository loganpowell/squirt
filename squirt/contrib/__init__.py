"""Squirt contrib plugins for domain-specific metrics."""

from squirt.contrib.echo import echo, EchoMetrics
from squirt.contrib.tokens import tokens, TokensMetrics

__all__ = ["echo", "EchoMetrics", "tokens", "TokensMetrics"]
