"""
Echo Metrics - Input/Output Logging and Storage

Provides debugging metrics for capturing and displaying inputs/outputs:
- save: Persist input, expected, and actual to file
- print: Log input, expected, and actual to console (survives pytest suppression)
"""

from squirt.contrib.echo.metrics import EchoMetrics, echo

__all__ = ["EchoMetrics", "echo"]
