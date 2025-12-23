"""
Performance Transform Functions

Extract system performance metrics from component outputs.
"""

from typing import Any, Dict


def memory_usage_transform(inputs: Dict[str, Any], output: Any) -> float:
    """
    Extract peak memory usage in MB from output metadata.

    Expects: output["metadata"]["peak_memory_mb"]

    Returns:
        Peak memory in MB (float), 0.0 if not available
    """
    if isinstance(output, dict):
        return output.get("metadata", {}).get("peak_memory_mb", 0.0)
    return 0.0


def cpu_usage_transform(inputs: Dict[str, Any], output: Any) -> float:
    """
    Extract average CPU usage percentage from output metadata.

    Expects: output["metadata"]["avg_cpu_percent"]

    Returns:
        Average CPU percentage (float), 0.0 if not available
    """
    if isinstance(output, dict):
        return output.get("metadata", {}).get("avg_cpu_percent", 0.0)
    return 0.0


def throughput_transform(inputs: Dict[str, Any], output: Any) -> float:
    """
    Calculate throughput (items per second) from output metadata.

    Expects:
        - output["metadata"]["item_count"]: Number of items processed
        - output["metadata"]["runtime_ms"]: Runtime in milliseconds

    Returns:
        Items per second (float), 0.0 if data missing or runtime is 0
    """
    if not isinstance(output, dict):
        return 0.0

    metadata = output.get("metadata", {})
    item_count = metadata.get("item_count", 0)
    runtime_ms = metadata.get("runtime_ms", 0)

    if runtime_ms == 0:
        return 0.0

    # Convert ms to seconds
    runtime_sec = runtime_ms / 1000.0
    return item_count / runtime_sec


def latency_p95_transform(inputs: Dict[str, Any], output: Any) -> float:
    """
    Extract 95th percentile latency from output metadata.

    Expects: output["metadata"]["latency_p95_ms"]

    Returns:
        P95 latency in ms (float), 0.0 if not available
    """
    if isinstance(output, dict):
        return output.get("metadata", {}).get("latency_p95_ms", 0.0)
    return 0.0


__all__ = [
    "memory_usage_transform",
    "cpu_usage_transform",
    "throughput_transform",
    "latency_p95_transform",
]
