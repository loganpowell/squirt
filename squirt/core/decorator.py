"""
Track Decorator for Squirt

The @track decorator instruments components to collect metrics.
It automatically tracks and records runtime_ms, memory_mb, and cpu_percent -
no need to add these to the metrics list!

Usage:
    from squirt import m, track, Expects

    @track(
        expects=Expects(input_key="description", output_key="bullets"),
        metrics=[
            # No need to add runtime_ms or memory_mb - they're automatic!
            m.expected_match.compare_to_expected("bullets", "bullets"),
            m.structure_valid.compute(my_transform),
        ],
    )
    def my_component(text: str) -> dict:
        # Just return your result
        return {"result": process(text)}
"""

from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from .types import Metric, MetricResult
from .resources import ResourceTracker, inject_metrics_into_output
from ..plugins.base import MetricBuilder
from ..builtins import BuiltinMetrics


# =============================================================================
# Expectations Configuration
# =============================================================================

_expectations_config: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "expectations_config",
    default={"path": None, "data": []},
)


def configure_expectations(
    path: Optional[Union[str, Path]] = None,
    data: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Configure the expectations source for all tracked components.

    Args:
        path: Path to expectations.json file
        data: Pre-loaded expectations data (alternative to path)
    """
    config = _expectations_config.get().copy()

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Expectations file not found: {path}")
        with open(path) as f:
            config["data"] = json.load(f)
        config["path"] = str(path)
    elif data is not None:
        config["data"] = data
        config["path"] = "<in-memory>"

    _expectations_config.set(config)


def get_expectations() -> List[Dict[str, Any]]:
    """Get the currently configured expectations data."""
    return _expectations_config.get().get("data", [])


# =============================================================================
# Test Context
# =============================================================================

_test_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "test_context",
    default={"test_case_id": "", "expectations": {}},
)


def set_test_context(
    test_case_id: str, expectations: Optional[Dict[str, Any]] = None
) -> None:
    """
    Set the context for the current test case.

    This should be called at the start of each test to provide
    the expected data for metric transforms.

    Args:
        test_case_id: Unique identifier for the test case
        expectations: Expected values dict for this test case
    """
    _test_context.set(
        {
            "test_case_id": test_case_id,
            "expectations": expectations or {},
        }
    )


def get_test_context() -> Dict[str, Any]:
    """Get the current test context."""
    return _test_context.get()


# =============================================================================
# Expects Contract
# =============================================================================
# Component Execution Stack
# =============================================================================

_component_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "component_stack", default=[]
)


def is_child_execution() -> bool:
    """
    Check if current execution is a child of another tracked component.

    Returns True only if BOTH conditions are met:
    1. Runtime: Component is nested in another component's execution (stack > 1)
    2. AST: Component has that parent in the static dependency graph

    This prevents false positives where a component CAN be a child (has parents
    in the codebase) but IS NOT currently being called as a child.
    """
    stack = _component_stack.get()

    # Must be nested at runtime
    if len(stack) <= 1:
        return False

    # Try AST-based validation
    try:
        from ..pytest import get_dependency_graph

        graph = get_dependency_graph()
        if graph is not None:
            current_component = stack[-1]
            parent_component = stack[-2]

            # Verify the parent->child relationship exists in the graph
            parents = graph.get_parents(current_component)
            return parent_component in parents
    except (ImportError, AttributeError):
        pass

    # Fallback: if no graph available, trust the runtime stack
    return True  # We're nested, so assume it's a child


def get_parent_component() -> Optional[str]:
    """Get the name of the parent component if executing as child."""
    stack = _component_stack.get()
    return stack[-1] if len(stack) > 0 else None


@contextmanager
def component_context(name: str):
    """Context manager to track component execution stack."""
    stack = _component_stack.get().copy()
    stack.append(name)
    token = _component_stack.set(stack)
    try:
        yield
    finally:
        _component_stack.reset(token)


# =============================================================================
# Metrics Results Storage
# =============================================================================

_metrics_results: contextvars.ContextVar[List[MetricResult]] = contextvars.ContextVar(
    "metrics_results", default=[]
)


def record_result(result: MetricResult) -> None:
    """
    Record a metric result.

    Records to both the local context variable storage AND the global
    MetricsClient (if one has been configured via configure_metrics).
    """
    # Local storage (for get_results())
    results = _metrics_results.get().copy()
    results.append(result)
    _metrics_results.set(results)

    # Also record to global MetricsClient if available
    try:
        from squirt import _metrics_client

        client = _metrics_client.get()
        if client is not None:
            client.record_result(result)
    except Exception:
        pass  # No global client configured


def get_results() -> List[MetricResult]:
    """Get all recorded metric results."""
    return _metrics_results.get()


def clear_results() -> None:
    """Clear all recorded results."""
    _metrics_results.set([])


# =============================================================================
# Track Decorator
# =============================================================================


def track(
    metrics: Sequence[Metric | MetricBuilder | BuiltinMetrics] = [],
    expects: Optional[str] = None,
    source: Optional[str] = None,
    record_when_child: bool = False,
    auto_inject_resources: bool = True,
) -> Callable:
    """
    Track metrics for a component.

    Automatically tracks and RECORDS runtime_ms, memory_mb, and cpu_percent.
    These are injected into output.metadata AND added to the collected metrics
    automatically - no need to add them to the metrics list!

    Args:
        expects: Key in expectations containing input to feed into component
        source: Path to custom expectations/data file (overrides global config).
            Behavior depends on file extension:
            - .json with array ([]): Each item passed one at a time (multi-run via test parameterization)
            - .json with object ({}): Entire payload passed once (single run)
            - Other extensions: File path string passed to function using expects key
        metrics: List of metrics to collect (resource metrics added automatically)
        record_when_child: If True, record metrics even when called by parent
        auto_inject_resources: If True (default), automatically inject AND record
            runtime_ms, memory_mb, and cpu_percent

    Returns:
        Decorated function with metrics collection

    Example:
        @track(
            expects="description",
            metrics=[
                # No need to add runtime_ms or memory_mb - automatic!
                m.expected_match.compare_to_expected("data", "result"),
                m.structure_valid.compute(my_transform),
            ]
        )
        def my_component(input_text: str) -> dict:
            # No need to track time or memory - it's automatic!
            return {"result": process(input_text)}
    """

    def decorator(func: Callable) -> Callable:
        # Store contracts on function for introspection
        func._expects = expects
        func._source = source
        func._metrics = metrics or []
        func._record_when_child = record_when_child

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            component_name = func.__name__

            with component_context(component_name):
                # NOW check if we're a child (after adding to stack)
                should_record = record_when_child or not is_child_execution()

                # Use ResourceTracker to capture runtime, memory, CPU
                with ResourceTracker() as tracker:
                    # Execute the component
                    result = func(*args, **kwargs)

                # Auto-inject resource metrics into output.metadata
                if auto_inject_resources:
                    result = inject_metrics_into_output(result, tracker.metrics)

                if should_record:
                    # Build inputs from expects mapping AND test context expectations
                    inputs = {}

                    # First, add test context expectations (ground truth for compare_to_expected)
                    test_ctx = get_test_context()

                    # Load data from custom source if specified
                    if source:
                        import json
                        from pathlib import Path

                        source_path = Path(source)
                        if source_path.exists():
                            # Check file extension to determine how to load
                            if source_path.suffix.lower() == ".json":
                                # JSON file: load and parse
                                with open(source_path) as f:
                                    source_data = json.load(f)

                                # Handle array vs object differently
                                if isinstance(source_data, list):
                                    # Array: each item should be passed one at a time (multiple runs)
                                    # Note: For now, we'll merge all items - proper iteration
                                    # should be handled by test parameterization
                                    inputs["_source_array"] = source_data
                                    # Also merge first item for immediate use
                                    if source_data:
                                        inputs.update(
                                            source_data[0]
                                            if isinstance(source_data[0], dict)
                                            else {}
                                        )
                                elif isinstance(source_data, dict):
                                    # Object: pass entire payload (single run)
                                    inputs.update(source_data)
                            else:
                                # Non-JSON file: pass file path to function
                                # Store it with a key that matches the expects parameter
                                if expects:
                                    inputs[expects] = str(source_path)
                    elif test_ctx.get("expectations"):
                        inputs.update(test_ctx["expectations"])

                    # Then add function arguments (overwrites if same key)
                    if expects and args:
                        inputs[expects] = args[0]
                    inputs.update(kwargs)

                    # Collect metrics
                    collected_metrics = {}
                    aggregation_types = {}
                    system_metric_map = {}

                    # Auto-add resource metrics first (they're always recorded)
                    if auto_inject_resources:
                        collected_metrics["runtime_ms"] = tracker.metrics.runtime_ms
                        collected_metrics["memory_mb"] = tracker.metrics.memory_mb
                        collected_metrics["cpu_percent"] = tracker.metrics.cpu_percent
                        # Resource metrics aggregation: sum runtime, max memory, avg cpu
                        aggregation_types["runtime_ms"] = "sum"
                        aggregation_types["memory_mb"] = "max"
                        aggregation_types["cpu_percent"] = "average"
                        # Map to system metrics
                        from ..categories.system import SystemMetric

                        system_metric_map["runtime_ms"] = SystemMetric.RUNTIME_MS.value
                        system_metric_map["memory_mb"] = SystemMetric.MEMORY_MB.value
                        system_metric_map["cpu_percent"] = (
                            SystemMetric.CPU_PERCENT.value
                        )

                    # Then collect user-defined metrics
                    assertion_failures = []
                    for metric in metrics:
                        if isinstance(metric, Metric):
                            try:
                                value = metric.transform(inputs, result)
                                collected_metrics[metric.name] = value
                                aggregation_types[metric.name] = metric.agg.value

                                # Extract system_metric from metric definition if available
                                # Metric objects don't have system_metric, but the MetricBuilder does
                                # We need to check if this was built from a MetricBuilder
                                if (
                                    hasattr(metric, "system_metric")
                                    and metric.system_metric
                                ):
                                    system_metric_map[metric.name] = (
                                        metric.system_metric.value
                                    )

                                # Check assertions: fail test if metric is assertion and value fails threshold
                                if metric.is_assertion:
                                    threshold = (
                                        metric.failure_threshold
                                        if metric.failure_threshold is not None
                                        else 0.0
                                    )
                                    if value <= threshold:
                                        assertion_failures.append(
                                            f"{metric.name}={value} (threshold: >{threshold})"
                                        )
                            except Exception as e:
                                collected_metrics[metric.name] = 0
                                # If it's an assertion metric, the exception also fails the test
                                if metric.is_assertion:
                                    assertion_failures.append(
                                        f"{metric.name}: {str(e)}"
                                    )

                    # Record the result
                    test_case_id = test_ctx.get("test_case_id", "")
                    metric_result = MetricResult(
                        component=component_name,
                        test_case_id=test_case_id,
                        metrics=collected_metrics,
                        aggregation_types=aggregation_types,
                        inputs=inputs,
                        output=result,
                        timestamp=time.time(),
                        system_metric_map=system_metric_map,
                    )
                    record_result(metric_result)

                    # Raise assertion errors AFTER recording (so results are preserved)
                    if assertion_failures:
                        failure_msg = (
                            f"Assertion metric(s) failed in {component_name}:\n  "
                            + "\n  ".join(assertion_failures)
                        )
                        raise AssertionError(failure_msg)

                return result

        return wrapper

    return decorator


__all__ = [
    "track",
    "configure_expectations",
    "get_expectations",
    "set_test_context",
    "get_test_context",
    "record_result",
    "get_results",
    "clear_results",
    "is_child_execution",
    "get_parent_component",
]
