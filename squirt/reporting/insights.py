"""
Squirt Insights Engine

Analyzes metrics to generate actionable insights.

Usage:
    from squirt.reporting import InsightGenerator, generate_heartbeat

    heartbeat = generate_heartbeat(results)
    generator = InsightGenerator(heartbeat)
    insights = generator.analyze()

    for insight in insights:
        print(insight.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .aggregation import SystemHeartbeat


class Severity(Enum):
    """Severity levels for insights."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Should be addressed soon
    MEDIUM = "medium"  # Worth investigating
    LOW = "low"  # Nice to know
    INFO = "info"  # Informational only


@dataclass
class Insight:
    """
    An actionable insight from metrics analysis.

    Each insight tells the developer:
    - What happened (title, description)
    - Why it matters (severity)
    - What likely caused it (likely_cause)
    - What to do about it (suggested_actions)
    """

    title: str
    severity: Severity
    description: str
    component: Optional[str] = None
    likely_cause: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    related_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "severity": self.severity.value,
            "description": self.description,
            "component": self.component,
            "likely_cause": self.likely_cause,
            "suggested_actions": self.suggested_actions,
            "related_metrics": self.related_metrics,
        }

    def to_markdown(self) -> str:
        """Render insight as markdown."""
        icon = {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🔵",
            Severity.INFO: "ℹ️",
        }

        lines = [
            f"### {icon[self.severity]} {self.title}",
            f"**Severity:** {self.severity.value.title()}",
        ]

        if self.component:
            lines.append(f"**Component:** `{self.component}`")

        lines.append(f"\n{self.description}")

        if self.likely_cause:
            lines.append(f"\n**Likely Cause:** {self.likely_cause}")

        if self.suggested_actions:
            lines.append("\n**Suggested Actions:**")
            for action in self.suggested_actions:
                lines.append(f"- {action}")

        return "\n".join(lines)


class InsightGenerator:
    """
    Generate actionable insights from metrics data.

    Analyzes current metrics against thresholds and historical data
    to identify issues that need attention.
    """

    # Default thresholds
    ACCURACY_THRESHOLD = 0.8
    ERROR_RATE_THRESHOLD = 0.1
    RUNTIME_THRESHOLD_MS = 60000  # 1 minute
    MEMORY_THRESHOLD_MB = 2048  # 2GB

    def __init__(
        self,
        heartbeat: SystemHeartbeat,
        history: Optional[List[Dict[str, Any]]] = None,
        hierarchical_report: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the insight generator.

        Args:
            heartbeat: Current system heartbeat
            history: Optional list of historical metrics for trend analysis
            hierarchical_report: Optional component-level report data
        """
        self.heartbeat = heartbeat
        self.history = history or []
        self.hierarchical_report = hierarchical_report or []

    def analyze(self) -> List[Insight]:
        """
        Analyze metrics and generate insights.

        Returns:
            List of Insight objects
        """
        insights = []

        # Check accuracy
        insights.extend(self._check_accuracy())

        # Check error rate
        insights.extend(self._check_error_rate())

        # Check performance
        insights.extend(self._check_performance())

        # Check memory
        insights.extend(self._check_memory())

        # Check component-level issues
        insights.extend(self._check_components())

        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        insights.sort(key=lambda i: severity_order[i.severity])

        return insights

    def _check_accuracy(self) -> List[Insight]:
        """Check accuracy metrics."""
        insights = []

        # Look for 'accuracy' in system_metrics first (system-level), then fall back to component-level
        # system_metrics contains the high-level accuracy metric (0.0-1.0)
        accuracy = (
            self.heartbeat.system_metrics.get("accuracy")
            or self.heartbeat.metrics.get("accuracy.avg")
            or self.heartbeat.metrics.get("accuracy")
            or self.heartbeat.metrics.get("expected_match.avg")
            or self.heartbeat.metrics.get("expected_match")
        )
        if accuracy is None:
            accuracy = 1.0  # Default to healthy if no accuracy metric found

        if accuracy < 0.5:
            insights.append(
                Insight(
                    title="Critical Accuracy Drop",
                    severity=Severity.CRITICAL,
                    description=f"System accuracy is at {accuracy:.1%}, below 50%.",
                    likely_cause="Major changes to extraction logic or data format",
                    suggested_actions=[
                        "Review recent code changes",
                        "Check if test data has changed",
                        "Validate LLM prompts are intact",
                    ],
                    related_metrics={"accuracy": accuracy},
                )
            )
        elif accuracy < self.ACCURACY_THRESHOLD:
            insights.append(
                Insight(
                    title="Accuracy Below Threshold",
                    severity=Severity.HIGH,
                    description=f"System accuracy is {accuracy:.1%}, below {self.ACCURACY_THRESHOLD:.0%} threshold.",
                    likely_cause="Potential regression in extraction quality",
                    suggested_actions=[
                        "Check component-level accuracy metrics",
                        "Review expected_match values",
                    ],
                    related_metrics={"accuracy": accuracy},
                )
            )

        return insights

    def _check_error_rate(self) -> List[Insight]:
        """Check error rate metrics."""
        insights = []
        # Note: Metrics now have suffixes (e.g., error_rate.avg)
        # Also check for error_free (which should be inverted to error_rate)
        error_rate = self.heartbeat.metrics.get(
            "error_rate.avg"
        ) or self.heartbeat.metrics.get("error_rate")

        # If no error_rate, check for error_free (inverted)
        if error_rate is None:
            error_free = self.heartbeat.metrics.get(
                "error_free.avg"
            ) or self.heartbeat.metrics.get("error_free")
            if error_free is not None:
                error_rate = 1.0 - error_free
            else:
                error_rate = 0.0

        if error_rate > 0.5:
            insights.append(
                Insight(
                    title="High Error Rate",
                    severity=Severity.CRITICAL,
                    description=f"Error rate is {error_rate:.1%}, over 50% of operations failing.",
                    likely_cause="System-wide failure or configuration issue",
                    suggested_actions=[
                        "Check service health",
                        "Review error logs",
                        "Validate environment configuration",
                    ],
                    related_metrics={"error_rate": error_rate},
                )
            )
        elif error_rate > self.ERROR_RATE_THRESHOLD:
            insights.append(
                Insight(
                    title="Elevated Error Rate",
                    severity=Severity.MEDIUM,
                    description=f"Error rate is {error_rate:.1%}, above {self.ERROR_RATE_THRESHOLD:.0%} threshold.",
                    likely_cause="Some components experiencing failures",
                    suggested_actions=[
                        "Check error_free metrics per component",
                        "Review structure_valid failures",
                    ],
                    related_metrics={"error_rate": error_rate},
                )
            )

        return insights

    def _check_performance(self) -> List[Insight]:
        """Check performance metrics."""
        insights = []
        # Note: Metrics now have suffixes (e.g., runtime_ms.sum)
        runtime_ms = self.heartbeat.metrics.get(
            "runtime_ms.sum"
        ) or self.heartbeat.metrics.get("runtime_ms", 0)

        if runtime_ms > self.RUNTIME_THRESHOLD_MS * 2:
            insights.append(
                Insight(
                    title="Severe Performance Degradation",
                    severity=Severity.HIGH,
                    description=f"Total runtime is {runtime_ms/1000:.1f}s, over 2x threshold.",
                    likely_cause="Slow component or blocking operation",
                    suggested_actions=[
                        "Profile slowest components",
                        "Check for N+1 queries",
                        "Consider parallel execution",
                    ],
                    related_metrics={"runtime_ms": runtime_ms},
                )
            )
        elif runtime_ms > self.RUNTIME_THRESHOLD_MS:
            insights.append(
                Insight(
                    title="Runtime Above Threshold",
                    severity=Severity.MEDIUM,
                    description=f"Total runtime is {runtime_ms/1000:.1f}s, above {self.RUNTIME_THRESHOLD_MS/1000:.0f}s threshold.",
                    likely_cause="One or more slow components",
                    suggested_actions=[
                        "Review runtime_ms per component",
                        "Identify bottlenecks",
                    ],
                    related_metrics={"runtime_ms": runtime_ms},
                )
            )

        return insights

    def _check_memory(self) -> List[Insight]:
        """Check memory usage metrics."""
        insights = []
        # Note: Metrics now have suffixes (e.g., memory_mb.max)
        memory_mb = self.heartbeat.metrics.get(
            "memory_mb.max"
        ) or self.heartbeat.metrics.get("memory_mb", 0)

        if memory_mb > self.MEMORY_THRESHOLD_MB:
            insights.append(
                Insight(
                    title="High Memory Usage",
                    severity=Severity.MEDIUM,
                    description=f"Peak memory is {memory_mb:.0f}MB, above {self.MEMORY_THRESHOLD_MB}MB threshold.",
                    likely_cause="Large data structures or memory leaks",
                    suggested_actions=[
                        "Profile memory usage",
                        "Check for large object retention",
                        "Consider streaming processing",
                    ],
                    related_metrics={"memory_mb": memory_mb},
                )
            )

        return insights

    def _check_components(self) -> List[Insight]:
        """Check component-level metrics for issues."""
        insights = []

        for component in self.heartbeat.components:
            # Check individual component accuracy
            expected_match = component.metrics.get("expected_match", 1.0)
            if expected_match < 0.5:
                insights.append(
                    Insight(
                        title="Low Component Accuracy",
                        severity=Severity.HIGH,
                        description=f"Component has {expected_match:.1%} accuracy.",
                        component=component.component,
                        likely_cause="Component-specific extraction issue",
                        suggested_actions=[
                            f"Review {component.component} implementation",
                            "Check input data quality",
                        ],
                        related_metrics={"expected_match": expected_match},
                    )
                )

            # Check for component errors
            error_free = component.metrics.get("error_free", 1.0)
            if error_free < 1.0:
                insights.append(
                    Insight(
                        title="Component Errors Detected",
                        severity=Severity.MEDIUM,
                        description=f"Component has {(1-error_free)*100:.0f}% error rate.",
                        component=component.component,
                        likely_cause="Validation or processing failures",
                        suggested_actions=[
                            f"Check {component.component} error logs",
                            "Review structure_valid metric",
                        ],
                        related_metrics={"error_free": error_free},
                    )
                )

        return insights


def generate_insight_report(
    heartbeat: SystemHeartbeat,
    history: Optional[List[Dict[str, Any]]] = None,
    hierarchical_report: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generate a markdown insight report from a heartbeat.

    Args:
        heartbeat: System heartbeat to analyze (or dict)
        history: Historical metrics for comparison
        hierarchical_report: Component-level metrics

    Returns:
        Markdown formatted report
    """
    # Handle dict input (from CLI)
    if isinstance(heartbeat, dict):
        from .aggregation import SystemHeartbeat

        heartbeat = SystemHeartbeat(
            timestamp=heartbeat.get("timestamp", 0),
            metrics=heartbeat.get("metrics", {}),
            component_count=heartbeat.get("component_count", 0),
        )

    generator = InsightGenerator(heartbeat, history, hierarchical_report)
    insights = generator.analyze()

    if not insights:
        return "## ✅ All Systems Healthy\n\nNo issues detected."

    lines = ["## 📊 Metrics Insights Report\n"]

    for insight in insights:
        lines.append(insight.to_markdown())
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "Severity",
    "Insight",
    "InsightGenerator",
    "generate_insight_report",
]
