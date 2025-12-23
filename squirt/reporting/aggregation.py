"""
Sleuth Aggregation Engine

Aggregates metrics across components and generates reports.
Supports both flat and graph-based hierarchical aggregation.

Usage:
    from sleuth.reporting import aggregate_results, generate_heartbeat

    # Simple: from list of results
    results = get_results()
    heartbeat = generate_heartbeat(results)
    print(heartbeat.to_json())

    # Advanced: with dependency graph
    from sleuth.analysis import analyze_codebase
    graph = analyze_codebase("./src")
    heartbeat = generate_heartbeat_from_graph(graph, results_dir="./results")
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Union

from ..core.types import AggregationType, MetricResult

if TYPE_CHECKING:
    from ..analysis.graph_builder import DependencyGraph


def aggregate_values(
    values: Sequence[Union[float, int, bool]], agg_type: Union[AggregationType, str]
) -> Union[float, int]:
    """
    Apply aggregation to a list of values.

    Args:
        values: List of metric values
        agg_type: Aggregation type (enum or string)

    Returns:
        Aggregated value
    """
    if isinstance(agg_type, str):
        agg_type = AggregationType(agg_type)

    numeric_values = [
        int(v) if isinstance(v, bool) else v for v in values if v is not None
    ]

    if not numeric_values:
        return 0

    if agg_type == AggregationType.AVERAGE:
        return sum(numeric_values) / len(numeric_values)
    elif agg_type == AggregationType.SUM:
        return sum(numeric_values)
    elif agg_type == AggregationType.MAX:
        return max(numeric_values)
    elif agg_type == AggregationType.MIN:
        return min(numeric_values)
    elif agg_type == AggregationType.COUNT:
        return len(numeric_values)
    elif agg_type == AggregationType.FAILURE:
        return sum(1 for v in numeric_values if v == 0 or v is False)
    elif agg_type == AggregationType.P95:
        sorted_vals = sorted(numeric_values)
        idx = int(len(sorted_vals) * 0.95) - 1
        return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]
    elif agg_type == AggregationType.P99:
        sorted_vals = sorted(numeric_values)
        idx = int(len(sorted_vals) * 0.99) - 1
        return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]

    return 0


def _get_agg_suffix(agg_type: Union[AggregationType, str]) -> str:
    """Get suffix for aggregated metric name based on aggregation type.

    Uses dot notation (e.g., .sum, .max) so metrics can be easily parsed
    by splitting on '.': metric_name.split('.')[-1] gives aggregation type.
    """
    if isinstance(agg_type, str):
        agg_type_str = agg_type.lower()
    else:
        agg_type_str = agg_type.value.lower()

    # Map aggregation types to suffixes using dot notation
    suffix_map = {
        "sum": ".sum",
        "max": ".max",
        "min": ".min",
        "average": ".avg",
        "count": ".count",
        "failure": ".failures",
        "p95": ".p95",
        "p99": ".p99",
    }
    return suffix_map.get(agg_type_str, "")


@dataclass
class ComponentReport:
    """Report for a single component."""

    component: str
    metrics: Dict[str, Union[float, int, bool]]
    aggregation_types: Dict[str, str]
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "parent": self.parent,
            "metrics": self.metrics,
            "children": self.children,
            "timestamp": self.timestamp,
        }


@dataclass
class SystemHeartbeat:
    """
    System-wide health report.

    Contains two tiers of aggregated metrics:
    1. metrics: Component-level metrics aggregated across components (detailed)
    2. system_metrics: High-level system health metrics (accuracy, error_rate, etc.)
    """

    timestamp: float
    metrics: Dict[str, Union[float, int]]
    component_count: int
    system_metrics: Dict[str, Union[float, int]] = field(default_factory=dict)
    components: List[ComponentReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "system_metrics": self.system_metrics,
            "component_count": self.component_count,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def aggregate_results(results: List[MetricResult]) -> Dict[str, Union[float, int]]:
    """
    Aggregate metrics across all results.

    Args:
        results: List of MetricResult from tracked components

    Returns:
        Dictionary of aggregated metric values
    """
    if not results:
        return {}

    # Group values by metric name
    metric_values: Dict[str, List] = {}
    metric_agg_types: Dict[str, AggregationType] = {}

    for result in results:
        for name, value in result.metrics.items():
            if name not in metric_values:
                metric_values[name] = []
            metric_values[name].append(value)

            # Get aggregation type
            if name not in metric_agg_types:
                agg_str = result.aggregation_types.get(name, "average")
                metric_agg_types[name] = AggregationType(agg_str)

    # Aggregate each metric
    aggregated = {}
    for name, values in metric_values.items():
        agg_type = metric_agg_types.get(name, AggregationType.AVERAGE)
        aggregated[name] = aggregate_values(values, agg_type)

    return aggregated


def generate_heartbeat(results: List[MetricResult]) -> SystemHeartbeat:
    """
    Generate a system heartbeat from metric results.

    Args:
        results: List of MetricResult from tracked components

    Returns:
        SystemHeartbeat with aggregated metrics
    """
    # Group results by component and average within each component
    component_metrics: Dict[str, Dict[str, List]] = {}
    aggregation_types: Dict[str, str] = {}
    system_metric_maps: Dict[str, Dict[str, str]] = (
        {}
    )  # Store system_metric_map per component

    for result in results:
        comp = result.component
        if comp not in component_metrics:
            component_metrics[comp] = {}
            system_metric_maps[comp] = result.system_metric_map  # Store the map

        # Collect values per metric per component
        for metric_name, value in result.metrics.items():
            if metric_name not in component_metrics[comp]:
                component_metrics[comp][metric_name] = []
            component_metrics[comp][metric_name].append(value)

            # Store aggregation type from first result that has it
            if metric_name not in aggregation_types:
                agg_type = result.aggregation_types.get(metric_name, "average")
                aggregation_types[metric_name] = agg_type

    # Average within each component (across test cases)
    component_averaged: Dict[str, Dict[str, float]] = {}
    for comp, metrics in component_metrics.items():
        component_averaged[comp] = {}
        for metric_name, values in metrics.items():
            component_averaged[comp][metric_name] = aggregate_values(values, "average")

    # Aggregate across components using defined aggregation_types
    system_metrics: Dict[str, float] = {}

    # Get all unique metrics across all components
    all_metric_names = set()
    for metrics in component_averaged.values():
        all_metric_names.update(metrics.keys())

    # For each metric, collect values from all components and apply aggregation_type
    for metric_name in all_metric_names:
        component_values = [
            comp_metrics[metric_name]
            for comp_metrics in component_averaged.values()
            if metric_name in comp_metrics
        ]

        if component_values:
            agg_type = aggregation_types.get(metric_name, "average")
            aggregated_value = aggregate_values(component_values, agg_type)

            # Add suffix to system metric name based on aggregation type
            suffix = _get_agg_suffix(agg_type)
            system_metric_name = f"{metric_name}{suffix}"
            system_metrics[system_metric_name] = aggregated_value

    # Build component reports
    components = []
    seen = set()
    for result in results:
        if result.component not in seen:
            seen.add(result.component)
            components.append(
                ComponentReport(
                    component=result.component,
                    metrics=component_averaged[result.component],
                    aggregation_types=result.aggregation_types,
                    timestamp=result.timestamp,
                )
            )

    # Compute system-level metrics (second tier of aggregation)
    # Convert component_averaged to format expected by aggregate_by_system_metrics
    component_results_for_system = {
        comp: {
            "metrics": metrics,
            "system_metric_map": system_metric_maps.get(comp, {}),
        }
        for comp, metrics in component_averaged.items()
    }
    system_level_metrics = aggregate_by_system_metrics(component_results_for_system)

    return SystemHeartbeat(
        timestamp=time.time(),
        metrics=system_metrics,
        system_metrics=system_level_metrics,
        component_count=len(components),
        components=components,
    )


# =============================================================================
# Graph-Based Aggregation
# =============================================================================


def _aggregate_node_recursive(
    node_name: str,
    graph,
    component_results: Dict[str, Dict],
    node_metrics: Dict[str, Dict],
    include_components: Optional[Set[str]] = None,
    parent_name: Optional[str] = None,
    visited: Optional[Set[str]] = None,
) -> Dict:
    """
    Recursively aggregate metrics from a node and all its children.

    Args:
        node_name: Name of the current node
        graph: Dependency graph (NetworkX or dict)
        component_results: Results loaded from file system
        node_metrics: Accumulator for computed metrics
        include_components: Optional filter for which components to include
        parent_name: Parent node name for hierarchy tracking
        visited: Set of visited nodes to prevent cycles

    Returns:
        Node metrics dict with aggregated values
    """
    # Initialize visited set if not provided
    if visited is None:
        visited = set()

    # Check for cycles
    if node_name in visited:
        return {
            "metrics": {},
            "aggregation_types": {},
            "parent": parent_name,
            "children": [],
        }

    # Return cached result if already processed
    if node_name in node_metrics:
        return node_metrics[node_name]

    # Mark as visited
    visited = visited | {node_name}

    # Get children from graph (supports DependencyGraph or dict)
    if hasattr(graph, "get_children"):
        children = graph.get_children(node_name)
    else:
        # Dict graph fallback
        children = []
        for src, dst in graph.get("edges", []):
            if src == node_name:
                children.append(dst)

    # Filter children if needed
    if include_components is not None:
        children = [c for c in children if c in include_components]

    # Leaf node case
    if not children:
        if node_name in component_results:
            node_metrics[node_name] = {
                "metrics": component_results[node_name]["metrics"],
                "aggregation_types": component_results[node_name].get(
                    "aggregation_types", {}
                ),
                "parent": parent_name,
                "children": [],
            }
        else:
            node_metrics[node_name] = {
                "metrics": {},
                "aggregation_types": {},
                "parent": parent_name,
                "children": [],
            }
        return node_metrics[node_name]

    # Has children: recursively aggregate
    metric_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"values": [], "agg": None}
    )

    # Include this node's own metrics if it has them
    if node_name in component_results:
        own_metrics = component_results[node_name]["metrics"]
        own_agg_types = component_results[node_name].get("aggregation_types", {})

        for metric_name, value in own_metrics.items():
            if value is not None:
                metric_data[metric_name]["values"].append(value)
                metric_data[metric_name]["agg"] = own_agg_types.get(metric_name)

    # Recursively aggregate each child
    for child_name in children:
        child_data = _aggregate_node_recursive(
            node_name=child_name,
            graph=graph,
            component_results=component_results,
            node_metrics=node_metrics,
            include_components=include_components,
            parent_name=node_name,
            visited=visited,
        )

        child_metrics = child_data["metrics"]
        child_agg_types = child_data.get("aggregation_types", {})

        for metric_name, value in child_metrics.items():
            if value is not None:
                metric_data[metric_name]["values"].append(value)
                if not metric_data[metric_name]["agg"]:
                    metric_data[metric_name]["agg"] = child_agg_types.get(metric_name)

    # Calculate aggregated metrics
    aggregated_metrics = {}
    aggregation_types = {}

    for metric_name, data in metric_data.items():
        values = data["values"] or []
        agg_type = data["agg"] or "average"

        aggregated_metrics[metric_name] = aggregate_values(values, agg_type)
        aggregation_types[metric_name] = (
            agg_type if isinstance(agg_type, str) else agg_type.value
        )

    node_metrics[node_name] = {
        "metrics": aggregated_metrics,
        "aggregation_types": aggregation_types,
        "parent": parent_name,
        "children": children,
    }

    return node_metrics[node_name]


def aggregate_metrics_from_graph(
    graph,
    results_dir: Union[str, Path],
    include_components: Optional[Set[str]] = None,
    save_reports: bool = True,
) -> Dict[str, Any]:
    """
    Recursively aggregate metrics from leaf components up to root using a dependency graph.

    Args:
        graph: Dependency graph from AST analysis (NetworkX DiGraph or dict)
        results_dir: Directory containing *_latest.json result files
        include_components: Optional set of component names to include
        save_reports: Whether to save hierarchical reports

    Returns:
        Aggregated metrics from root nodes
    """
    results_dir = Path(results_dir)

    # Load all component results from file system
    component_results = {}
    for result_file in results_dir.glob("*_results.json"):
        with open(result_file) as f:
            data = json.load(f)

        component_name = data["component"]
        if include_components is None or component_name in include_components:
            # Average metrics across test cases at component level
            # aggregation_types are only used for system-level aggregation
            averaged_metrics = {}
            for metric_name, values in data["metrics"].items():
                # Values are in list format (one per test case)
                if isinstance(values, list):
                    # Always average at component level
                    averaged_metrics[metric_name] = aggregate_values(values, "average")
                else:
                    # Backward compatibility: single value
                    averaged_metrics[metric_name] = values

            component_results[component_name] = {
                "component": component_name,
                "metrics": averaged_metrics,
                "aggregation_types": data.get("aggregation_types", {}),
                "result_count": len(data.get("test_case_ids", [])),
            }

    # Find root nodes (supports DependencyGraph or dict)
    if hasattr(graph, "get_roots"):
        root_nodes = graph.get_roots()
    else:
        called_funcs = set()
        for src, dst in graph.get("edges", []):
            called_funcs.add(dst)
        root_nodes = [n for n in graph.get("nodes", {}) if n not in called_funcs]

    # Filter roots if needed
    if include_components is not None:
        root_nodes = [n for n in root_nodes if n in include_components]

    # If no roots found but we have results, use all result components as roots
    if not root_nodes and component_results:
        root_nodes = list(component_results.keys())

    # Recursively aggregate from each root
    node_metrics: Dict[str, Dict] = {}
    for root in root_nodes:
        _aggregate_node_recursive(
            node_name=root,
            graph=graph,
            component_results=component_results,
            node_metrics=node_metrics,
            include_components=include_components,
        )

    # Save hierarchical reports if requested
    if save_reports:
        save_hierarchical_reports(node_metrics, results_dir)

    # Return system-level heartbeat
    if len(root_nodes) == 1:
        return node_metrics.get(root_nodes[0], {}).get("metrics", {})
    elif len(root_nodes) > 1:
        return _aggregate_multiple_roots(root_nodes, node_metrics)
    else:
        return {}


def _aggregate_multiple_roots(
    root_nodes: List[str], node_metrics: Dict[str, Dict]
) -> Dict[str, Union[float, int]]:
    """Aggregate metrics from multiple root nodes."""
    metric_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"values": [], "agg": None}
    )

    for root_name in root_nodes:
        if root_name not in node_metrics:
            continue

        root_data = node_metrics[root_name]
        metrics = root_data["metrics"]
        agg_types = root_data.get("aggregation_types", {})

        for metric_name, value in metrics.items():
            if value is not None:
                metric_data[metric_name]["values"].append(value)
                if not metric_data[metric_name]["agg"]:
                    metric_data[metric_name]["agg"] = agg_types.get(metric_name)

    heartbeat: Dict[str, Union[float, int]] = {}
    for metric_name, data in metric_data.items():
        values = data["values"] or []
        agg_type = data["agg"] or "average"
        heartbeat[metric_name] = aggregate_values(values, agg_type)

    return heartbeat


def save_hierarchical_reports(node_metrics: Dict[str, Dict], results_dir: Path) -> None:
    """
    Save all component reports as a single flat list.

    Each component includes its parent reference for hierarchy reconstruction.
    """
    reports = []
    timestamp = time.time()

    for node_name, node_data in node_metrics.items():
        report = {
            "component": node_name,
            "parent": node_data.get("parent"),
            "metrics": node_data["metrics"],
            "children": node_data.get("children", []),
            "timestamp": timestamp,
        }
        reports.append(report)

    report_file = results_dir / "hierarchical_report.json"
    with open(report_file, "w") as f:
        json.dump(reports, f, indent=2)

    print(f"📊 Saved {len(reports)} component reports to {report_file}")


def generate_heartbeat_from_graph(
    graph,
    results_dir: Union[str, Path],
    include_components: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a system heartbeat using graph-based hierarchical aggregation.

    This is the main entry point for getting aggregated metrics after a test run
    when you have a dependency graph available.

    Args:
        graph: Dependency graph from AST analysis
        results_dir: Directory containing result files
        include_components: Optional filter for which components to include

    Returns:
        Heartbeat dict with aggregated metrics and metadata
    """
    results_dir = Path(results_dir)

    # Load component results and average within each component (across test cases)
    component_averaged: Dict[str, Dict[str, float]] = {}
    aggregation_types: Dict[str, str] = {}
    system_metric_maps: Dict[str, Dict[str, str]] = (
        {}
    )  # Store system_metric_map per component

    for result_file in results_dir.glob("*_results.json"):
        with open(result_file) as f:
            data = json.load(f)

        component_name = data["component"]
        if include_components is not None and component_name not in include_components:
            continue

        # Average metrics within this component (across test cases)
        averaged_metrics = {}
        for metric_name, values in data.get("metrics", {}).items():
            if isinstance(values, list):
                averaged_metrics[metric_name] = aggregate_values(values, "average")
            else:
                averaged_metrics[metric_name] = values

        component_averaged[component_name] = averaged_metrics
        system_metric_maps[component_name] = data.get("system_metric_map", {})

        # Collect aggregation types
        for metric_name, agg_type in data.get("aggregation_types", {}).items():
            if metric_name not in aggregation_types:
                aggregation_types[metric_name] = agg_type

    # Aggregate across components using defined aggregation_types
    system_metrics: Dict[str, float] = {}

    # Get all unique metrics across all components
    all_metric_names = set()
    for metrics in component_averaged.values():
        all_metric_names.update(metrics.keys())

    # For each metric, collect values from all components and apply aggregation_type
    for metric_name in all_metric_names:
        component_values = [
            comp_metrics[metric_name]
            for comp_metrics in component_averaged.values()
            if metric_name in comp_metrics
        ]

        if component_values:
            agg_type = aggregation_types.get(metric_name, "average")
            aggregated_value = aggregate_values(component_values, agg_type)

            # Add suffix to system metric name based on aggregation type
            suffix = _get_agg_suffix(agg_type)
            system_metric_name = f"{metric_name}{suffix}"
            system_metrics[system_metric_name] = aggregated_value

    # Compute system-level metrics (second tier of aggregation)
    component_results_for_system = {
        comp: {
            "metrics": metrics,
            "system_metric_map": system_metric_maps.get(comp, {}),
        }
        for comp, metrics in component_averaged.items()
    }
    system_level_metrics = aggregate_by_system_metrics(component_results_for_system)

    heartbeat = {
        "timestamp": time.time(),
        "metrics": system_metrics,
        "system_metrics": system_level_metrics,
        "component_count": len(component_averaged),
    }

    # Save heartbeat
    heartbeat_file = results_dir / "system_heartbeat.json"
    with open(heartbeat_file, "w") as f:
        json.dump(heartbeat, f, indent=2)

    return heartbeat


# =============================================================================
# Analysis Utilities
# =============================================================================


def find_bottlenecks(
    results_dir: Union[str, Path], metric_name: str = "runtime_ms", top_n: int = 5
) -> List[Dict]:
    """
    Find the top N components with highest values for a metric.

    Useful for identifying runtime bottlenecks or cost hotspots.

    Args:
        results_dir: Directory containing result files
        metric_name: Metric to analyze (default: runtime_ms)
        top_n: Number of top components to return

    Returns:
        List of components sorted by metric value (highest first)
    """
    results_dir = Path(results_dir)
    components = []

    report_file = results_dir / "hierarchical_report.json"
    if report_file.exists():
        with open(report_file) as f:
            all_reports = json.load(f)

        for report in all_reports:
            value = report["metrics"].get(metric_name, 0)
            if value > 0:
                components.append(
                    {"component": report["component"], metric_name: value}
                )
    else:
        # Fall back to individual result files
        for result_file in results_dir.glob("*_latest.json"):
            with open(result_file) as f:
                result = json.load(f)

            value = result["metrics"].get(metric_name, 0)
            if value > 0:
                components.append(
                    {"component": result["component"], metric_name: value}
                )

    # Sort by value descending
    components.sort(key=lambda x: x[metric_name], reverse=True)

    return components[:top_n]


def find_underperforming_components(
    results_dir: Union[str, Path], metric_name: str, threshold: float
) -> List[Dict]:
    """
    Find all components below a performance threshold.

    Args:
        results_dir: Directory containing result files
        metric_name: Name of the metric to check
        threshold: Threshold value

    Returns:
        List of underperforming components sorted by value (worst first)
    """
    results_dir = Path(results_dir)
    problematic = []

    report_file = results_dir / "hierarchical_report.json"
    if not report_file.exists():
        return []

    with open(report_file) as f:
        all_reports = json.load(f)

    for report in all_reports:
        metric_value = report["metrics"].get(metric_name)

        if metric_value is not None and metric_value < threshold:
            problematic.append(
                {
                    "component": report["component"],
                    "value": metric_value,
                    "threshold": threshold,
                    "children": report.get("children", []),
                }
            )

    # Sort by value (worst first)
    problematic.sort(key=lambda x: x["value"])

    return problematic


# =============================================================================
# System Metric Aggregation
# =============================================================================

# Aggregation types for system metrics by name
SYSTEM_METRIC_AGG_BY_NAME = {
    "accuracy": AggregationType.AVERAGE,
    "runtime_ms": AggregationType.SUM,
    "memory_mb": AggregationType.MAX,
    "cpu_percent": AggregationType.AVERAGE,
    "cost_usd": AggregationType.SUM,
    "total_tokens": AggregationType.SUM,
    "throughput": AggregationType.AVERAGE,
    "error_rate": AggregationType.AVERAGE,
    "latency_p95": AggregationType.P95,
}


def aggregate_by_system_metrics(
    component_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Union[float, int]]:
    """
    Aggregate component-level metrics into system-level metrics.

    Maps component metrics (e.g., field_accuracy, runtime_ms) to their
    canonical system metrics (e.g., accuracy, runtime_ms) and aggregates
    using the appropriate aggregation type.

    Args:
        component_results: Dict mapping component names to result dicts.
            Each result should have:
            - "metrics": Dict of metric_name -> value
            - "system_metric_map" (optional): Dict of metric_name -> system_metric_name

    Returns:
        Dict of system_metric_name -> aggregated_value

    Example:
        >>> results = {
        ...     "chunk_pdf": {"metrics": {"runtime_ms": 1500}},
        ...     "extract_json": {"metrics": {"runtime_ms": 2000}},
        ... }
        >>> aggregate_by_system_metrics(results)
        {"runtime_ms": 3500}  # SUM aggregation
    """
    from collections import defaultdict
    from ..categories import get_aggregation_type, should_invert

    system_metric_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"values": [], "agg": None}
    )

    for component_name, result in component_results.items():
        metrics = result.get("metrics", {})
        system_metric_map = result.get("system_metric_map", {})

        for metric_name, value in metrics.items():
            if value is None:
                continue

            # Get system metric from the explicit map
            sys_metric_name = system_metric_map.get(metric_name)
            if not sys_metric_name:
                continue

            # Get aggregation type for this system metric
            from ..categories import SystemMetric

            try:
                sys_metric_enum = SystemMetric(sys_metric_name)
                agg_type = get_aggregation_type(sys_metric_enum)
            except ValueError:
                # If not a valid SystemMetric, skip it
                continue

            # Invert if needed (e.g., error_free → error_rate)
            if should_invert(metric_name):
                value = 1.0 - value

            # Store value and aggregation type
            system_metric_data[sys_metric_name]["values"].append(value)
            if system_metric_data[sys_metric_name]["agg"] is None:
                system_metric_data[sys_metric_name]["agg"] = agg_type

    # Aggregate using correct aggregation type
    system_metrics: Dict[str, Union[float, int]] = {}
    for name, data in system_metric_data.items():
        values = data["values"]
        agg_type = data["agg"]
        if values:
            system_metrics[name] = aggregate_values(
                values, agg_type or AggregationType.AVERAGE
            )

    return system_metrics


__all__ = [
    # Core aggregation
    "aggregate_values",
    "aggregate_results",
    "generate_heartbeat",
    # Graph-based aggregation
    "aggregate_metrics_from_graph",
    "generate_heartbeat_from_graph",
    "save_hierarchical_reports",
    # System metric aggregation
    "aggregate_by_system_metrics",
    # Data classes
    "ComponentReport",
    "SystemHeartbeat",
    # Analysis
    "find_bottlenecks",
    "find_underperforming_components",
]
