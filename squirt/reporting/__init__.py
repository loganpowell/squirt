"""
Squirt Reporting Module

Provides aggregation, heartbeat generation, and insights analysis.

Usage:
    from squirt.reporting import (
        generate_heartbeat,
        aggregate_results,
        InsightGenerator,
        generate_insight_report,
    )

    # Simple: Generate heartbeat from results
    results = get_results()
    heartbeat = generate_heartbeat(results)
    heartbeat.save("reports/heartbeat.json")

    # Advanced: With dependency graph
    from squirt.analysis import analyze_codebase
    graph = analyze_codebase("./src")
    heartbeat = generate_heartbeat_from_graph(graph, "./results")

    # Generate insights
    report = generate_insight_report(heartbeat)
    print(report)
"""

from .aggregation import (
    # Data classes
    ComponentReport,
    SystemHeartbeat,
    aggregate_by_system_metrics,
    # Graph-based aggregation
    aggregate_metrics_from_graph,
    aggregate_results,
    # Core aggregation
    aggregate_values,
    # Analysis
    find_bottlenecks,
    find_underperforming_components,
    generate_heartbeat,
    generate_heartbeat_from_graph,
    save_hierarchical_reports,
)
from .insights import (
    Insight,
    InsightGenerator,
    Severity,
    generate_insight_report,
)
from .reporter import MetricsReporter

__all__ = [
    # Aggregation
    "aggregate_values",
    "aggregate_results",
    "generate_heartbeat",
    "aggregate_metrics_from_graph",
    "generate_heartbeat_from_graph",
    "save_hierarchical_reports",
    "aggregate_by_system_metrics",
    "find_bottlenecks",
    "find_underperforming_components",
    "ComponentReport",
    "SystemHeartbeat",
    # Insights
    "Severity",
    "Insight",
    "InsightGenerator",
    "generate_insight_report",
    # Reporting
    "MetricsReporter",
]
