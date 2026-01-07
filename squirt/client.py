"""
Squirt Metrics Client

Session-level client for managing metrics collection and report generation.
Supports both in-memory and file-based persistence.

Usage:
    from squirt import MetricsClient

    # In-memory only
    client = MetricsClient()

    # File-based persistence
    client = MetricsClient(results_dir="./tests/results", persist=True)

    # Add results as tests run
    client.add_result(result)

    # Generate all reports at end
    reports = client.generate_reports()
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.types import MetricResult
from .reporting.aggregation import (
    ComponentReport,
    SystemHeartbeat,
    aggregate_results,
    generate_heartbeat,
)


def _get_git_info() -> dict[str, str]:
    """Get current git commit and branch."""
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception:
        pass
    return info


class MetricsClient:
    """
    Client for managing metrics collection and report generation.

    Collects MetricResult objects during a test session and generates
    various reports at the end.

    Supports two modes:
    - In-memory: Results stored in memory only
    - File-based: Results persisted to disk as *_results.json files with all executions (default when results_dir provided)
    """

    def __init__(
        self,
        results_dir: str | Path | None = None,
        history_dir: str | Path | None = None,
        persist: bool | None = None,
    ):
        """
        Initialize the metrics client.

        Args:
            results_dir: Directory for output reports
            history_dir: Directory for historical reports
            persist: If True, save results to files. Defaults to True if results_dir is provided.
        """
        from . import get_config

        config = get_config()

        self.results_dir = Path(
            results_dir or config.get("results_dir", "tests/results")
        )
        self.history_dir = Path(
            history_dir or config.get("history_dir", "tests/history")
        )
        # Default persist=True if results_dir was explicitly provided
        self.persist = persist if persist is not None else (results_dir is not None)
        self.results: list[MetricResult] = []
        self._session_id = str(uuid.uuid4())
        self.start_time = time.time()

        # Always create directories (for backward compatibility)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def record_result(self, result: MetricResult) -> None:
        """
        Record a metric result (alias for add_result with file persistence).

        Args:
            result: MetricResult from a tracked component
        """
        self.results.append(result)

        if self.persist:
            self._save_result(result)

    def add_result(self, result: MetricResult) -> None:
        """
        Add a metric result.

        Args:
            result: MetricResult from a tracked component
        """
        self.record_result(result)

    def add_results(self, results: list[MetricResult]) -> None:
        """
        Add multiple metric results.

        Args:
            results: List of MetricResult objects
        """
        for result in results:
            self.record_result(result)

    def _save_result(self, result: MetricResult) -> None:
        """Save a result to the file system, appending metric values to lists."""
        result_file = self.results_dir / f"{result.component}_results.json"

        # Read existing data if file exists
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                # If file is corrupted, start fresh
                data = None
        else:
            data = None

        # Initialize structure if needed
        if data is None:
            data = {
                "component": result.component,
                "test_case_ids": [],
                "metrics": {},
                "aggregation_types": result.aggregation_types,
                "timestamps": [],
                "execution_ids": [],
                "system_metric_map": result.system_metric_map,
            }

        # Append test case ID
        data["test_case_ids"].append(result.test_case_id)
        data["timestamps"].append(result.timestamp)
        data["execution_ids"].append(result.execution_id)

        # Append each metric value to its list
        for metric_name, metric_value in result.metrics.items():
            if metric_name not in data["metrics"]:
                data["metrics"][metric_name] = []
            data["metrics"][metric_name].append(metric_value)

        # Save updated data
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_results(
        self, component: str | None = None
    ) -> dict[str, list[MetricResult]]:
        """
        Get results, optionally filtered by component.

        Args:
            component: Optional component name to filter by

        Returns:
            Dict mapping component name to list of results
        """
        if component:
            filtered = [r for r in self.results if r.component == component]
            return {component: filtered}

        # Group by component
        by_component: dict[str, list[MetricResult]] = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        return by_component

    def clear(self) -> None:
        """Clear all collected results."""
        self.results.clear()

    def generate_heartbeat(self) -> SystemHeartbeat:
        """
        Generate a system heartbeat from collected results.

        Returns:
            SystemHeartbeat with aggregated metrics
        """
        return generate_heartbeat(self.results)

    def generate_hierarchical_report(self) -> list[dict[str, Any]]:
        """
        Generate a hierarchical component report.

        Returns:
            List of component reports with parent-child relationships
        """
        # Group by component
        by_component: dict[str, list[MetricResult]] = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)

        # Build hierarchical report
        reports = []
        for component, results in by_component.items():
            # Aggregate metrics for this component
            metrics = aggregate_results(results)

            # Get aggregation types from first result
            agg_types = results[0].aggregation_types if results else {}

            report = ComponentReport(
                component=component,
                metrics=metrics,
                aggregation_types=agg_types,
                parent=None,  # TODO: Track parent-child relationships
                children=[],
                timestamp=max(r.timestamp for r in results),
            )
            reports.append(report.to_dict())

        return reports

    def generate_reports(self) -> dict[str, str]:
        """
        Generate all reports and save to disk.

        Returns:
            Dictionary mapping report type to file path
        """
        if not self.results:
            return {}

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        reports_generated = {}
        git_info = _get_git_info()

        # 1. Generate system heartbeat
        heartbeat = self.generate_heartbeat()
        heartbeat_path = self.results_dir / "system_heartbeat.json"
        heartbeat.save(heartbeat_path)
        reports_generated["heartbeat"] = str(heartbeat_path)

        # Save versioned copy to history
        versioned_heartbeat = (
            self.history_dir / f"system_heartbeat.{git_info['commit']}.json"
        )
        heartbeat.save(versioned_heartbeat)

        # 2. Generate hierarchical report
        hierarchical = self.generate_hierarchical_report()
        hierarchical_path = self.results_dir / "hierarchical_report.json"
        with open(hierarchical_path, "w") as f:
            json.dump(hierarchical, f, indent=2)
        reports_generated["hierarchical"] = str(hierarchical_path)

        # Save versioned copy
        versioned_hierarchical = (
            self.history_dir / f"hierarchical_report.{git_info['commit']}.json"
        )
        with open(versioned_hierarchical, "w") as f:
            json.dump(hierarchical, f, indent=2)

        # 3. Append to metrics history (JSONL)
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "commit": git_info["commit"],
            "branch": git_info["branch"],
            "metrics": heartbeat.metrics,
            "component_count": heartbeat.component_count,
        }

        history_path = self.history_dir / "metrics_history.jsonl"
        with open(history_path, "a") as f:
            f.write(json.dumps(history_entry) + "\n")
        reports_generated["history"] = str(history_path)

        return reports_generated

    def generate_insights(self) -> str:
        """
        Generate insight report from current metrics.

        Returns:
            Markdown-formatted insight report
        """
        from .reporting.insights import generate_insight_report

        heartbeat = self.generate_heartbeat()
        hierarchical = self.generate_hierarchical_report()

        # Load history for comparison
        history = []
        history_path = self.history_dir / "metrics_history.jsonl"
        if history_path.exists():
            with open(history_path) as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))

        return generate_insight_report(
            heartbeat=heartbeat,
            history=history,
            hierarchical_report=hierarchical,
        )


__all__ = ["MetricsClient"]
