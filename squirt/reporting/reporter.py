"""
Generate markdown metrics reports from test results.

This module provides the MetricsReporter class for creating:
1. Concise PR comment format reports
2. Detailed job summary format reports

Both include historical trend analysis using sparklines and comparison arrows.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .aggregation import SystemHeartbeat
from .insights import InsightGenerator


class MetricsReporter:
    """Generate markdown reports from metrics data."""

    # Unicode block characters for sparklines (▁▂▃▄▅▆▇█)
    BLOCKS = "▁▂▃▄▅▆▇█"

    def __init__(
        self,
        results_dir: Path,
        history_dir: Path,
        git_hash: Optional[str] = None,
    ):
        """
        Initialize the metrics reporter.

        Args:
            results_dir: Directory containing current test results
            history_dir: Directory for historical snapshots
            git_hash: Git commit hash (auto-detected if None)
        """
        self.results_dir = Path(results_dir)
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.git_hash = git_hash or self._get_git_hash()
        self.timestamp = datetime.now().isoformat()

    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _get_git_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _load_json(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file, return None if not found."""
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return json.load(f)

    def _load_component_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all *_results.json component results."""
        from .aggregation import aggregate_values

        components = {}
        for result_file in self.results_dir.glob("*_results.json"):
            if result_file.name in [
                "system_heartbeat.json",
                "hierarchical_report.json",
            ]:
                continue
            component_name = result_file.stem.replace("_results", "")
            data = self._load_json(result_file)
            if data:
                # Aggregate metrics using their defined aggregation types
                aggregation_types = data.get("aggregation_types", {})
                aggregated_metrics = {}
                for metric_name, values in data.get("metrics", {}).items():
                    if isinstance(values, list):
                        # Use component-level aggregation: always average across test cases
                        # System-level aggregation (SUM/MAX) happens when combining components
                        aggregated_metrics[metric_name] = aggregate_values(
                            values, "average"
                        )
                    else:
                        # Backward compatibility: single value
                        aggregated_metrics[metric_name] = values

                components[component_name] = {
                    "component": component_name,
                    "metrics": aggregated_metrics,
                    "aggregation_types": aggregation_types,
                    "result_count": len(data.get("test_case_ids", [])),
                }
        return components

    def _load_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Load historical metrics from jsonl file.

        Args:
            limit: Maximum number of historical entries to return

        Returns:
            List of historical metric snapshots, most recent first
        """
        history_file = self.history_dir / "metrics_history.jsonl"
        if not history_file.exists():
            return []

        history = []
        with open(history_file) as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))

        # Return most recent entries first
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def _create_sparkline(self, values: List[float]) -> str:
        """
        Create a sparkline from a list of values.

        Args:
            values: List of numeric values

        Returns:
            String of Unicode block characters
        """
        if not values or len(values) < 2:
            return "N/A"

        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero
        if max_val == min_val:
            return self.BLOCKS[4] * len(values)  # Use middle block

        # Normalize values to 0-7 range (for 8 block characters)
        normalized = [int(((v - min_val) / (max_val - min_val)) * 7) for v in values]

        return "".join(self.BLOCKS[n] for n in normalized)

    def _compare_values(
        self, current: float, previous: float, threshold: float = 0.01
    ) -> str:
        """
        Compare current vs previous value and return arrow indicator.

        Args:
            current: Current value
            previous: Previous value
            threshold: Percentage threshold for "no change" (default 1%)

        Returns:
            Arrow indicator: ↑ (up), ↓ (down), → (no change)
        """
        if previous == 0:
            return "→"

        change_pct = abs(current - previous) / previous

        if change_pct < threshold:
            return "→"
        elif current > previous:
            return "↑"
        else:
            return "↓"

    def _format_duration(self, ms: float) -> str:
        """Format duration in milliseconds to human-readable string."""
        if ms < 1000:
            return f"{int(ms)}ms"
        elif ms < 60000:
            return f"{ms/1000:.1f}s"
        else:
            return f"{ms/60000:.1f}m"

    def _format_percentage(self, value: float) -> str:
        """Format value as percentage."""
        return f"{value * 100:.1f}%"

    def _format_system_metric(self, metric_name: str, value: float) -> str:
        """
        Format system metric based on category.

        Args:
            metric_name: Name of the system metric
            value: Metric value

        Returns:
            Formatted string appropriate for the metric type
        """
        if metric_name in ["accuracy", "error_rate"]:
            return self._format_percentage(value)
        elif metric_name in ["runtime_ms", "latency_p95"]:
            return self._format_duration(value)
        elif metric_name == "memory_mb":
            return f"{value:.1f} MB"
        elif metric_name == "cpu_percent":
            return f"{value:.1f}%"
        elif metric_name == "cost_usd":
            return f"${value:.4f}"
        elif metric_name == "total_tokens":
            return f"{int(value):,} tokens"
        elif metric_name == "throughput":
            return f"{value:.1f} items/s"
        else:
            return f"{value:.2f}"

    def save_historical_snapshot(self) -> None:
        """Save current results to history with git hash."""
        # Load current results
        heartbeat = self._load_json(self.results_dir / "system_heartbeat.json")
        hierarchical = self._load_json(self.results_dir / "hierarchical_report.json")

        if not heartbeat:
            print("Warning: No system_heartbeat.json found, skipping snapshot")
            return

        # Save heartbeat with hash
        heartbeat_file = self.history_dir / f"system_heartbeat.{self.git_hash}.json"
        with open(heartbeat_file, "w") as f:
            json.dump(heartbeat, f, indent=2)

        # Save hierarchical report with hash
        if hierarchical:
            hierarchical_file = (
                self.history_dir / f"hierarchical_report.{self.git_hash}.json"
            )
            with open(hierarchical_file, "w") as f:
                json.dump(hierarchical, f, indent=2)

        # Update metrics history (overwrite if same commit exists)
        history_entry = {
            "timestamp": self.timestamp,
            "commit": self.git_hash,
            "branch": self._get_git_branch(),
            "metrics": heartbeat.get("metrics", {}),  # Component-level
            "system_metrics": heartbeat.get("system_metrics", {}),  # System-level
        }

        history_file = self.history_dir / "metrics_history.jsonl"

        # Read existing entries and filter out same commit
        existing_entries = []
        if history_file.exists():
            with open(history_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Keep entries from different commits
                        if entry.get("commit") != self.git_hash:
                            existing_entries.append(entry)

        # Write back all entries plus the new/updated one
        with open(history_file, "w") as f:
            for entry in existing_entries:
                f.write(json.dumps(entry) + "\n")
            f.write(json.dumps(history_entry) + "\n")

        print(f"Saved snapshot for commit {self.git_hash}")

    def generate_pr_comment(self) -> str:
        """Generate concise PR comment format with system metrics focus."""
        heartbeat = self._load_json(self.results_dir / "system_heartbeat.json")
        hierarchical = self._load_json(self.results_dir / "hierarchical_report.json")
        components = self._load_component_results()
        history = self._load_history(limit=2)  # Current + previous

        if not heartbeat:
            return "No metrics data available."

        lines = [
            f"# Metrics Summary - {self.git_hash}",
            "",
            f"**Branch:** {self._get_git_branch()}",
            f"**Timestamp:** {self.timestamp}",
            "",
        ]

        # Add insights summary at the top of PR comment
        insights_summary = self._generate_insights_summary(
            heartbeat, history, hierarchical
        )
        if insights_summary:
            lines.append(insights_summary)
            lines.append("")

        # Show ONLY system metrics in PR (high-level view)
        system_metrics = heartbeat.get("system_metrics", {})

        if system_metrics:
            lines.extend(
                [
                    "## 📊 System Metrics",
                    "",
                    "| Metric | Current | Δ |",
                    "|--------|---------|---|",
                ]
            )

            # Priority order for PR display
            PRIORITY_METRICS = [
                "accuracy",
                "error_rate",
                "runtime_ms",
                "memory_mb",
                "cost_usd",
                "total_tokens",
            ]

            prev_system_metrics = {}
            if history:
                prev_system_metrics = history[0].get("system_metrics", {})

            for metric_name in PRIORITY_METRICS:
                if metric_name in system_metrics:
                    current = system_metrics[metric_name]
                    formatted = self._format_system_metric(metric_name, current)

                    # Arrow comparison
                    prev = prev_system_metrics.get(metric_name)
                    if prev is not None:
                        arrow = self._compare_values(current, prev)
                    else:
                        arrow = "-"

                    lines.append(f"| {metric_name} | {formatted} | {arrow} |")
        else:
            # Fallback to component metrics if no system metrics
            metrics = heartbeat.get("metrics", {})
            lines.extend(
                [
                    "## System Metrics",
                    "",
                    "| Metric | Current | Δ |",
                    "|--------|---------|---|",
                ]
            )

            for metric_name, current_value in sorted(metrics.items())[:10]:
                formatted = (
                    self._format_duration(current_value)
                    if "runtime" in metric_name
                    else f"{current_value:.2f}"
                )
                lines.append(f"| {metric_name} | {formatted} | - |")

        lines.extend(["", "---", ""])

        # Component breakdown (collapsible)
        if components:
            component_list = [
                (name, data.get("metrics", {}).get("runtime_ms", 0))
                for name, data in components.items()
                if "runtime_ms" in data.get("metrics", {})
            ]
            component_list.sort(key=lambda x: x[1], reverse=True)

            lines.extend(
                [
                    "<details>",
                    "<summary>📦 Component Breakdown (Top 5)</summary>",
                    "",
                    "| Component | Runtime |",
                    "|-----------|---------|",
                ]
            )

            for name, runtime in component_list[:5]:
                lines.append(f"| {name} | {self._format_duration(runtime)} |")

            lines.extend(["", "</details>"])

        lines.extend(["", "[View Full Report in Job Summary]"])

        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate detailed full report."""
        heartbeat = self._load_json(self.results_dir / "system_heartbeat.json")
        hierarchical: List[Dict[str, Any]] = self._load_json(self.results_dir / "hierarchical_report.json")  # type: ignore
        components = self._load_component_results()
        history = self._load_history(limit=10)

        # If no heartbeat but we have components, we can still generate a partial report
        if not heartbeat and not components:
            return "No metrics data available. Tests may have failed before generating results."

        sections = []

        sections.append(self._section_header())

        # Only include sections that require heartbeat if we have it
        if heartbeat:
            sections.append(self._section_insights(heartbeat, history, hierarchical))
            sections.append(self._section_executive_summary(heartbeat, history))
            sections.append(self._section_performance_trends(history))
        else:
            sections.append(
                "## ⚠️ Partial Report\n\nTests failed or did not complete. Showing available component data only."
            )

        # Component performance can work without heartbeat
        if components:
            sections.append(self._section_component_performance(components))

            # Visual insights need both components and hierarchical
            if hierarchical:
                sections.append(
                    self._section_resource_treemaps(components, hierarchical or [])
                )
            sections.append(self._section_metric_charts(components, history))
            sections.append(self._section_component_details(components))

        if hierarchical:
            sections.append(self._section_dependency_tree(hierarchical))

        return "\n\n".join(sections)

    def _section_insights(
        self,
        heartbeat: Dict[str, Any],
        history: List[Dict[str, Any]],
        hierarchical: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Generate actionable insights section using sleuth's InsightGenerator."""
        try:
            # Convert heartbeat dict to SystemHeartbeat
            system_heartbeat = SystemHeartbeat(
                timestamp=heartbeat.get("timestamp", 0),
                metrics=heartbeat.get("metrics", {}),
                component_count=heartbeat.get("component_count", 0),
                system_metrics=heartbeat.get("system_metrics", {}),
            )

            generator = InsightGenerator(
                heartbeat=system_heartbeat,
                history=history,
                hierarchical_report=hierarchical or [],
            )
            insights = generator.analyze()

            if not insights:
                return "## ✅ All Systems Healthy\n\nNo issues detected. All metrics within normal thresholds."

            lines = ["## 🚨 Action Required\n"]

            for i, insight in enumerate(insights, 1):
                # to_markdown() already includes ### header, so just add number prefix
                md = insight.to_markdown()
                # Replace the first ### with numbered version
                md = md.replace("### ", f"### {i}. ", 1)
                lines.append(md)
                lines.append("\n---\n")

            return "\n".join(lines)

        except Exception as e:
            return f"## ⚠️ Insights Generation Error\n\nCould not generate insights: {e}"

    def _generate_insights_summary(
        self,
        heartbeat: Dict[str, Any],
        history: List[Dict[str, Any]],
        hierarchical: Any,
    ) -> str:
        """Generate a brief insights summary for PR comments."""
        try:
            system_heartbeat = SystemHeartbeat(
                timestamp=heartbeat.get("timestamp", 0),
                metrics=heartbeat.get("metrics", {}),
                component_count=heartbeat.get("component_count", 0),
                system_metrics=heartbeat.get("system_metrics", {}),
            )

            generator = InsightGenerator(
                heartbeat=system_heartbeat,
                history=history,
                hierarchical_report=hierarchical or [],
            )
            insights = generator.analyze()

            if not insights:
                return "## ✅ All Checks Passed\n\nNo issues detected."

            # Count by severity
            critical = sum(1 for i in insights if i.severity.value == "critical")
            high = sum(1 for i in insights if i.severity.value == "high")
            medium = sum(1 for i in insights if i.severity.value == "medium")

            if critical > 0:
                icon = "🔴"
                status = f"{critical} critical issue(s)"
            elif high > 0:
                icon = "🟠"
                status = f"{high} high priority issue(s)"
            elif medium > 0:
                icon = "🟡"
                status = f"{medium} issue(s) to review"
            else:
                icon = "🔵"
                status = f"{len(insights)} minor issue(s)"

            lines = [f"## {icon} {status}", ""]

            # Show top 3 insights briefly
            for insight in insights[:3]:
                lines.append(f"- **{insight.title}**: {insight.description[:80]}...")

            if len(insights) > 3:
                lines.append(
                    f"\n*See full report for {len(insights) - 3} more issues.*"
                )

            return "\n".join(lines)

        except Exception:
            return ""

    def _section_header(self) -> str:
        """Generate report header."""
        return "\n".join(
            [
                "# TTTAT Pipeline Metrics Report",
                "",
                f"**Generated:** {self.timestamp}",
                f"**Commit:** {self.git_hash}",
                f"**Branch:** {self._get_git_branch()}",
                "",
                "---",
            ]
        )

    def _section_executive_summary(
        self, heartbeat: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> str:
        """Generate executive summary section."""
        system_metrics = heartbeat.get("system_metrics", {})
        component_metrics = heartbeat.get("metrics", {})
        component_count = heartbeat.get("component_count", 0)

        lines = [
            "## Executive Summary",
            "",
            f"**Total Components:** {component_count}",
            "",
        ]

        # Show system metrics prominently
        if system_metrics:
            lines.extend(
                [
                    "### 🎯 System-Level Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            for metric_name, value in sorted(system_metrics.items()):
                formatted = self._format_system_metric(metric_name, value)
                lines.append(f"| {metric_name} | {formatted} |")

            lines.extend(["", ""])

        # Show component metrics in collapsible section
        if component_metrics:
            lines.extend(
                [
                    "<details>",
                    "<summary>📦 Component-Level Metrics</summary>",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            for metric_name, value in sorted(component_metrics.items()):
                if "runtime" in metric_name.lower() or "ms" in metric_name.lower():
                    formatted = self._format_duration(value)
                elif value <= 1.0 and value >= 0.0:
                    formatted = self._format_percentage(value)
                else:
                    formatted = f"{value:.2f}"
                lines.append(f"| {metric_name} | {formatted} |")

            lines.extend(["", "</details>", "", ""])

        # Comparison with previous run (use system metrics)
        if history and system_metrics:
            prev = history[0]
            prev_system = prev.get("system_metrics", {})

            if prev_system:
                lines.extend(
                    [
                        f"### 📊 Changes Since Previous Run ({prev['commit']})",
                        "",
                        "| Metric | Previous | Current | Change |",
                        "|--------|----------|---------|--------|",
                    ]
                )

                for metric_name, current_value in sorted(system_metrics.items()):
                    prev_value = prev_system.get(metric_name)
                    if prev_value is not None:
                        arrow = self._compare_values(current_value, prev_value)
                        change_pct = (
                            ((current_value - prev_value) / prev_value * 100)
                            if prev_value != 0
                            else 0
                        )

                        prev_fmt = self._format_system_metric(metric_name, prev_value)
                        curr_fmt = self._format_system_metric(
                            metric_name, current_value
                        )

                        lines.append(
                            f"| {metric_name} | {prev_fmt} | {curr_fmt} | {arrow} {change_pct:+.1f}% |"
                        )

        return "\n".join(lines)

    def _section_performance_trends(self, history: List[Dict[str, Any]]) -> str:
        """Generate performance trends section with sparklines."""
        if len(history) < 2:
            return "\n".join(
                [
                    "## Performance Trends",
                    "",
                    "*Insufficient historical data (need at least 2 runs)*",
                ]
            )

        lines = [
            "## Performance Trends (Last 10 Commits)",
            "",
        ]

        # Collect system metric values across history (prefer system over component)
        metric_names: Set[str] = set()
        has_system_metrics = any(entry.get("system_metrics") for entry in history)

        if has_system_metrics:
            # Use system metrics
            for entry in history:
                metric_names.update(entry.get("system_metrics", {}).keys())

            lines.extend(
                [
                    "| System Metric | Trend | Current |",
                    "|---------------|-------|---------|",
                ]
            )

            for metric_name in sorted(metric_names):
                values = [
                    entry.get("system_metrics", {}).get(metric_name)
                    for entry in reversed(history)  # Oldest to newest for sparkline
                ]
                values = [v for v in values if v is not None]

                if values:
                    sparkline = self._create_sparkline(values)
                    current = values[-1]
                    current_fmt = self._format_system_metric(metric_name, current)
                    lines.append(f"| {metric_name} | {sparkline} | {current_fmt} |")

        else:
            # Fall back to component metrics
            for entry in history:
                metric_names.update(entry.get("metrics", {}).keys())

            lines.extend(
                [
                    "| Metric | Trend | Current |",
                    "|--------|-------|---------|",
                ]
            )

            for metric_name in sorted(metric_names):
                values = [
                    entry.get("metrics", {}).get(metric_name)
                    for entry in reversed(history)  # Oldest to newest for sparkline
                ]
                values = [v for v in values if v is not None]

                if values:
                    sparkline = self._create_sparkline(values)
                    current = values[-1]

                    if "runtime" in metric_name.lower() or "ms" in metric_name.lower():
                        current_fmt = self._format_duration(current)
                    elif current <= 1.0 and current >= 0.0:
                        current_fmt = self._format_percentage(current)
                    else:
                        current_fmt = f"{current:.2f}"

                    lines.append(f"| {metric_name} | {sparkline} | {current_fmt} |")

        return "\n".join(lines)

    def _section_component_performance(
        self, components: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate component performance table."""
        if not components:
            return "\n".join(
                [
                    "## Component Performance",
                    "",
                    "*No component data available*",
                ]
            )

        lines = [
            "## Component Performance",
            "",
            "| Component | Runtime | Metrics |",
            "|-----------|---------|---------|",
        ]

        # Sort by runtime descending
        component_list = []
        for name, data in components.items():
            metrics = data.get("metrics", {})
            runtime = metrics.get("runtime_ms", 0)
            component_list.append((name, runtime, metrics))

        component_list.sort(key=lambda x: x[1], reverse=True)

        for name, runtime, metrics in component_list:
            runtime_fmt = self._format_duration(runtime)

            # Format other metrics
            other_metrics = []
            for metric_name, value in sorted(metrics.items()):
                if metric_name == "runtime_ms":
                    continue
                if value <= 1.0 and value >= 0.0:
                    other_metrics.append(
                        f"{metric_name}: {self._format_percentage(value)}"
                    )
                else:
                    other_metrics.append(f"{metric_name}: {value:.2f}")

            metrics_str = ", ".join(other_metrics) if other_metrics else "N/A"
            lines.append(f"| {name} | {runtime_fmt} | {metrics_str} |")

        return "\n".join(lines)

    def _section_resource_treemaps(
        self, components: Dict[str, Dict[str, Any]], hierarchical: List[Dict[str, Any]]
    ) -> str:
        """Generate treemap visualizations for resource distribution with hierarchy."""
        if not components:
            return ""

        lines = [
            "## Resource Distribution",
            "",
            "Visual breakdown of resource usage across components.",
            "",
        ]

        # Build hierarchy map from the actual graph structure
        hierarchy_map: Dict[str, List[str]] = {}
        component_parents: Dict[str, Optional[str]] = {}

        for item in hierarchical:
            comp_name = item["component"]
            parent = item.get("parent")
            children = item.get("children", [])

            component_parents[comp_name] = parent

            # Use the children list from hierarchical report (AST graph structure)
            if children:
                hierarchy_map[comp_name] = children

        # Find root components (no parent)
        roots = [name for name, parent in component_parents.items() if parent is None]

        # If no hierarchy data, fall back to flat list
        if not hierarchical:
            roots = list(components.keys())

        def build_tree(
            comp_name: str,
            metric_key: str,
            formatter_func,
            indent: int = 1,
            visited: Optional[Set[str]] = None,
        ) -> List[str]:
            """Recursively build treemap tree structure with cycle detection."""
            if visited is None:
                visited = set()

            # Prevent infinite recursion from circular references
            if comp_name in visited:
                return []

            visited = visited | {comp_name}

            result = []
            data = components.get(comp_name, {})
            metrics = data.get("metrics", {})
            value = metrics.get(metric_key, 0)

            if value <= 0:
                return result

            # Format label and weight using provided formatter
            label, weight = formatter_func(comp_name, value)

            children = hierarchy_map.get(comp_name, [])

            if children:
                # Parent node - just label (no value per Mermaid spec)
                result.append(f"{'    ' * indent}\"{label}\"")
                # Recursively add children with increased indent
                for child in children:
                    result.extend(
                        build_tree(
                            child, metric_key, formatter_func, indent + 1, visited
                        )
                    )
            else:
                # Leaf node - label with value (required per Mermaid spec)
                result.append(f"{'    ' * indent}\"{label}\": {weight}")

            return result

        # Define formatters for each metric type
        def runtime_formatter(name: str, value: float) -> tuple:
            value_sec = value / 1000
            return (f"{name} ({value_sec:.2f}s)", int(value))

        def memory_formatter(name: str, value: float) -> tuple:
            return (f"{name} ({value:.1f}MB)", int(value))

        def cpu_formatter(name: str, value: float) -> tuple:
            return (f"{name} ({value:.1f}%)", int(value * 10))  # Scale for visibility

        def cost_formatter(name: str, value: float) -> tuple:
            return (f"{name} (${value:.4f})", int(value * 10000))

        def tokens_formatter(name: str, value: float) -> tuple:
            return (f"{name} ({int(value)} tokens)", int(value))

        # Helper to generate treemap for a given metric
        def generate_treemap(metric_key: str, title: str, unit: str, formatter_func):
            """Generate a treemap for a specific metric."""
            # Find roots that have this metric
            metric_components = [
                name
                for name in roots
                if components.get(name, {}).get("metrics", {}).get(metric_key, 0) > 0
            ]

            if not metric_components:
                return []

            # For memory_mb, filter out components with duplicate values
            # (they likely all measured the same process peak)
            if metric_key == "memory_mb":
                seen_values: Dict[float, str] = {}
                filtered_components = []
                for comp in metric_components:
                    value = (
                        components.get(comp, {}).get("metrics", {}).get(metric_key, 0)
                    )
                    rounded_value = round(value, 1)  # Round to 0.1 MB precision

                    if rounded_value not in seen_values:
                        seen_values[rounded_value] = comp
                        filtered_components.append(comp)
                    else:
                        # Keep the one with more children (more interesting hierarchy)
                        existing = seen_values[rounded_value]
                        if len(hierarchy_map.get(comp, [])) > len(
                            hierarchy_map.get(existing, [])
                        ):
                            # Replace with component that has more children
                            filtered_components.remove(existing)
                            filtered_components.append(comp)
                            seen_values[rounded_value] = comp

                metric_components = filtered_components

            section = [
                f"### {title}",
                "",
                "```mermaid",
                "treemap-beta",
                f'"{unit}"',
            ]

            # Sort by metric value descending
            for root in sorted(
                metric_components,
                key=lambda x: components.get(x, {})
                .get("metrics", {})
                .get(metric_key, 0),
                reverse=True,
            ):
                section.extend(build_tree(root, metric_key, formatter_func))

            section.extend(["```", ""])
            return section

        # Generate treemaps for all metrics
        lines.extend(
            generate_treemap(
                "runtime_ms", "Runtime Distribution", "Runtime (ms)", runtime_formatter
            )
        )
        lines.extend(
            generate_treemap(
                "memory_mb", "Memory Distribution", "Memory (MB)", memory_formatter
            )
        )
        lines.extend(
            generate_treemap(
                "cpu_percent", "CPU Distribution", "CPU (%)", cpu_formatter
            )
        )
        lines.extend(
            generate_treemap(
                "token_cost", "Cost Distribution", "Cost (USD)", cost_formatter
            )
        )
        lines.extend(
            generate_treemap(
                "total_tokens", "Token Usage Distribution", "Tokens", tokens_formatter
            )
        )

        return "\n".join(lines)

    def _section_metric_charts(
        self, components: Dict[str, Dict[str, Any]], history: List[Dict[str, Any]]
    ) -> str:
        """Generate XY charts for metric trends."""
        if len(history) < 2:
            return ""

        lines = [
            "## Metric Trends",
            "",
            "Historical trends across recent commits.",
            "",
        ]

        # Get system metrics from history
        has_system_metrics = any(entry.get("system_metrics") for entry in history)

        if has_system_metrics:
            # Accuracy trend
            accuracy_values = []
            for entry in reversed(history):  # Oldest to newest
                sys_metrics = entry.get("system_metrics", {})
                if "accuracy" in sys_metrics:
                    accuracy_values.append(sys_metrics["accuracy"])

            if len(accuracy_values) >= 2:
                lines.extend(
                    [
                        "### Accuracy Over Time",
                        "",
                        "```mermaid",
                        "xychart-beta",
                        '  title "System Accuracy Trend"',
                        "  x-axis ["
                        + ", ".join(f"{i+1}" for i in range(len(accuracy_values)))
                        + "]",
                        '  y-axis "Accuracy" 0 --> 1',
                        "  line ["
                        + ", ".join(f"{v:.3f}" for v in accuracy_values)
                        + "]",
                        "```",
                        "",
                    ]
                )

            # Runtime trend
            runtime_values = []
            for entry in reversed(history):
                sys_metrics = entry.get("system_metrics", {})
                if "runtime_ms" in sys_metrics:
                    runtime_values.append(
                        sys_metrics["runtime_ms"] / 1000
                    )  # Convert to seconds

            if len(runtime_values) >= 2:
                lines.extend(
                    [
                        "### Runtime Over Time",
                        "",
                        "```mermaid",
                        "xychart-beta",
                        '  title "System Runtime Trend (seconds)"',
                        "  x-axis ["
                        + ", ".join(f"{i+1}" for i in range(len(runtime_values)))
                        + "]",
                        '  y-axis "Runtime (s)" 0 --> '
                        + f"{max(runtime_values) * 1.1:.0f}",
                        "  line ["
                        + ", ".join(f"{v:.2f}" for v in runtime_values)
                        + "]",
                        "```",
                        "",
                    ]
                )

            # Cost trend
            cost_values = []
            for entry in reversed(history):
                sys_metrics = entry.get("system_metrics", {})
                if "cost_usd" in sys_metrics:
                    cost_values.append(sys_metrics["cost_usd"])

            if len(cost_values) >= 2:
                lines.extend(
                    [
                        "### Cost Over Time",
                        "",
                        "```mermaid",
                        "xychart-beta",
                        '  title "System Cost Trend (USD)"',
                        "  x-axis ["
                        + ", ".join(f"{i+1}" for i in range(len(cost_values)))
                        + "]",
                        '  y-axis "Cost ($)" 0 --> ' + f"{max(cost_values) * 1.1:.4f}",
                        "  line [" + ", ".join(f"{v:.4f}" for v in cost_values) + "]",
                        "```",
                        "",
                    ]
                )

        return "\n".join(lines)

    def _section_dependency_tree(self, hierarchical: List[Dict[str, Any]]) -> str:
        """Generate dependency tree visualization."""
        if not hierarchical:
            return "\n".join(
                [
                    "## Dependency Tree",
                    "",
                    "*No hierarchy data available*",
                ]
            )

        # Build component lookup - use dict to deduplicate and get the most recent parent info
        components_by_name: Dict[str, Dict[str, Any]] = {}
        for comp in hierarchical:
            name = comp["component"]
            # Keep the version with a parent if it exists, otherwise keep root
            if name not in components_by_name or comp.get("parent"):
                components_by_name[name] = comp

        lines = [
            "## Dependency Tree",
            "",
            "```",
        ]

        # Find root components (those with no parent)
        roots = [
            name for name, comp in components_by_name.items() if not comp.get("parent")
        ]

        # Recursively build tree
        def render_tree(
            component_name: str,
            indent: int = 0,
            is_last: bool = True,
            parent_chain: Optional[set] = None,
        ):
            if parent_chain is None:
                parent_chain = set()

            # Prevent infinite loops
            if component_name in parent_chain:
                return

            parent_chain = parent_chain | {component_name}

            data = components_by_name.get(component_name, {})
            metrics = data.get("metrics", {})
            runtime = metrics.get("runtime_ms", 0)

            prefix = "└── " if is_last else "├── "
            if indent == 0:
                prefix = ""
            else:
                prefix = "    " * (indent - 1) + prefix

            # Format metrics for display
            metric_strs = []
            if runtime:
                metric_strs.append(self._format_duration(runtime))
            for metric_name, value in sorted(metrics.items()):
                if metric_name != "runtime_ms" and value is not None:
                    if value <= 1.0 and value >= 0.0:
                        metric_strs.append(self._format_percentage(value))
                    else:
                        metric_strs.append(f"{value:.2f}")

            metrics_display = f" ({', '.join(metric_strs)})" if metric_strs else ""
            lines.append(f"{prefix}{component_name}{metrics_display}")

            # Recurse for children (these are string names, not full objects)
            children = data.get("children", [])
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                render_tree(child, indent + 1, is_last_child, parent_chain)

        for root in sorted(roots):
            render_tree(root)

        lines.append("```")

        return "\n".join(lines)

    def _section_component_details(self, components: Dict[str, Dict[str, Any]]) -> str:
        """Generate detailed component breakdown."""
        if not components:
            return ""

        lines = [
            "## Component Details",
            "",
        ]

        for name in sorted(components.keys()):
            data = components[name]
            metrics = data.get("metrics", {})

            lines.extend(
                [
                    f"### {name}",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            for metric_name, value in sorted(metrics.items()):
                if "runtime" in metric_name.lower() or "ms" in metric_name.lower():
                    formatted = self._format_duration(value)
                elif isinstance(value, float) and value <= 1.0 and value >= 0.0:
                    formatted = self._format_percentage(value)
                elif isinstance(value, bool):
                    formatted = "Yes" if value else "No"
                else:
                    formatted = (
                        f"{value:.2f}" if isinstance(value, float) else str(value)
                    )
                lines.append(f"| {metric_name} | {formatted} |")

            lines.append("")

        return "\n".join(lines)
