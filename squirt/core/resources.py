"""
Sleuth Resource Tracking Utilities

Provides utilities for tracking runtime, memory, and CPU usage.
These are used by the @track decorator to auto-inject metrics.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ResourceMetrics:
    """Captured resource metrics from a function execution."""

    runtime_ms: int
    memory_mb: float
    cpu_percent: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for injection into output."""
        return {
            "runtime_ms": self.runtime_ms,
            "peak_memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
        }


class ResourceTracker:
    """
    Context manager for tracking resource usage during component execution.

    Usage:
        with ResourceTracker() as tracker:
            result = perform_work()

        print(tracker.metrics.runtime_ms)
        print(tracker.metrics.memory_mb)
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.start_memory_mb: float = 0.0
        self.start_cpu_times: Any = None
        self.metrics: ResourceMetrics = ResourceMetrics(
            runtime_ms=0, memory_mb=0.0, cpu_percent=0.0
        )

        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def __enter__(self) -> "ResourceTracker":
        """Start tracking resources."""
        self.start_time = time.time()

        if self.process:
            mem_info = self.process.memory_info()
            self.start_memory_mb = mem_info.rss / (1024 * 1024)
            # Get CPU times for calculating usage
            self.start_cpu_times = self.process.cpu_times()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Calculate final metrics on exit."""
        end_time = time.time()

        # Runtime in milliseconds
        runtime_ms = int((end_time - self.start_time) * 1000)

        # Memory and CPU
        memory_mb = 0.0
        cpu_percent = 0.0

        if self.process:
            mem_info = self.process.memory_info()
            memory_mb = mem_info.rss / (1024 * 1024)

            # Calculate CPU percent based on actual CPU time used
            end_cpu_times = self.process.cpu_times()
            if self.start_cpu_times:
                cpu_time_used = (end_cpu_times.user - self.start_cpu_times.user) + (
                    end_cpu_times.system - self.start_cpu_times.system
                )
                wall_time = end_time - self.start_time
                if wall_time > 0:
                    cpu_percent = (cpu_time_used / wall_time) * 100

        self.metrics = ResourceMetrics(
            runtime_ms=runtime_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
        )

        return False  # Don't suppress exceptions


def inject_metrics_into_output(output: Any, metrics: ResourceMetrics) -> Any:
    """
    Inject resource metrics into the output dict's metadata.

    If output is a dict, adds/updates the 'metadata' key with runtime_ms,
    peak_memory_mb, and cpu_percent. Does not overwrite existing values
    unless they are 0 or None.

    Args:
        output: The function's return value
        metrics: ResourceMetrics to inject

    Returns:
        The output with metrics injected (if dict), otherwise unchanged
    """
    if not isinstance(output, dict):
        return output

    # Ensure metadata exists
    if "metadata" not in output:
        output["metadata"] = {}

    metadata = output["metadata"]
    if not isinstance(metadata, dict):
        return output

    # Only inject if not already set (or set to 0/None)
    if not metadata.get("runtime_ms"):
        metadata["runtime_ms"] = metrics.runtime_ms

    if not metadata.get("peak_memory_mb"):
        metadata["peak_memory_mb"] = metrics.memory_mb

    if not metadata.get("cpu_percent"):
        metadata["cpu_percent"] = metrics.cpu_percent

    return output


__all__ = [
    "ResourceMetrics",
    "ResourceTracker",
    "inject_metrics_into_output",
    "PSUTIL_AVAILABLE",
]
