"""
Sleuth Reporting Module

Provides aggregation, heartbeat generation, and insights analysis.

Usage:
    from sleuth.reporting import (
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
    from sleuth.analysis import analyze_codebase
    graph = analyze_codebase("./src")
    heartbeat = generate_heartbeat_from_graph(graph, "./results")

    # Generate insights
    report = generate_insight_report(heartbeat)
    print(report)
"""

from .aggregation import (
    # Core aggregation
    aggregate_values,
    aggregate_results,
    generate_heartbeat,
    # Graph-based aggregation
    aggregate_metrics_from_graph,
    generate_heartbeat_from_graph,
    save_hierarchical_reports,
    aggregate_by_system_metrics,
    # Data classes
    ComponentReport,
    SystemHeartbeat,
    # Analysis
    find_bottlenecks,
    find_underperforming_components,
)

from .insights import (
    Severity,
    Insight,
    InsightGenerator,
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
