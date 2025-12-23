"""
Sleuth CLI

Command-line interface for generating reports and analyzing metrics.

Usage:
    # Generate reports from test results
    sleuth report generate

    # View metric trends
    sleuth report trends --metric accuracy --last 30

    # Compare against baseline
    sleuth report compare --baseline main --current HEAD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _get_git_hash() -> str:
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


def _get_git_branch() -> str:
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


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate reports from existing metric results."""
    from .client import MetricsClient
    from .core.types import MetricResult

    results_dir = Path(args.results_dir)
    history_dir = Path(args.history_dir)

    # Load existing results
    results_file = results_dir / "raw_results.json"
    if not results_file.exists():
        print(f"No results found at {results_file}")
        print("Run tests with --collect-metrics first")
        return 1

    with open(results_file) as f:
        raw_results = json.load(f)

    # Convert to MetricResult objects
    results = []
    for r in raw_results:
        results.append(
            MetricResult(
                component=r["component"],
                test_case_id=r.get("test_case_id", ""),
                metrics=r["metrics"],
                aggregation_types=r.get("aggregation_types", {}),
                inputs=r.get("inputs", {}),
                output=r.get("output"),
                timestamp=r.get("timestamp", 0),
                execution_id=r.get("execution_id", ""),
                system_metric_map=r.get("system_metric_map", {}),
            )
        )

    # Generate reports
    client = MetricsClient(results_dir=results_dir, history_dir=history_dir)
    client.add_results(results)
    reports = client.generate_reports()

    print("📊 Reports generated:")
    for report_type, path in reports.items():
        print(f"   {report_type}: {path}")

    return 0


def cmd_trends(args: argparse.Namespace) -> int:
    """Show metric trends over time."""
    history_dir = Path(args.history_dir)
    history_file = history_dir / "metrics_history.jsonl"

    if not history_file.exists():
        print(f"No history found at {history_file}")
        return 1

    # Load history
    history: List[Dict[str, Any]] = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))

    if not history:
        print("No historical data found")
        return 1

    # Get last N entries
    entries = history[-args.last :]

    metric_name = args.metric.lower()
    print(f"\n📈 Trend for '{metric_name}' (last {len(entries)} runs):\n")

    for entry in entries:
        timestamp = entry.get("timestamp", "unknown")[:19]
        commit = entry.get("commit", "?")[:7]

        # Look in both metrics and system_metrics
        value = entry.get("metrics", {}).get(metric_name)
        if value is None:
            value = entry.get("system_metrics", {}).get(metric_name)

        if value is not None:
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            print(f"  {timestamp} ({commit}): {value_str}")
        else:
            print(f"  {timestamp} ({commit}): N/A")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare metrics between two points."""
    history_dir = Path(args.history_dir)
    history_file = history_dir / "metrics_history.jsonl"

    if not history_file.exists():
        print(f"No history found at {history_file}")
        return 1

    # Load history
    history: List[Dict[str, Any]] = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))

    if len(history) < 2:
        print("Need at least 2 history entries to compare")
        return 1

    # Find baseline and current
    def find_entry(identifier: str) -> Optional[Dict[str, Any]]:
        for entry in reversed(history):
            if identifier in (entry.get("commit", ""), entry.get("branch", "")):
                return entry
        return None

    baseline = find_entry(args.baseline) or history[-2]
    current = find_entry(args.current) or history[-1]

    print(f"\n📊 Comparing metrics:\n")
    print(
        f"  Baseline: {baseline.get('commit', '?')[:7]} ({baseline.get('timestamp', '')[:19]})"
    )
    print(
        f"  Current:  {current.get('commit', '?')[:7]} ({current.get('timestamp', '')[:19]})"
    )
    print()

    # Compare metrics
    all_metrics = set(baseline.get("metrics", {}).keys()) | set(
        current.get("metrics", {}).keys()
    )

    changes = []
    for metric in sorted(all_metrics):
        base_val = baseline.get("metrics", {}).get(metric)
        curr_val = current.get("metrics", {}).get(metric)

        if base_val is None or curr_val is None:
            continue

        if isinstance(base_val, (int, float)) and isinstance(curr_val, (int, float)):
            if base_val != 0:
                pct_change = ((curr_val - base_val) / abs(base_val)) * 100
            else:
                pct_change = 0 if curr_val == 0 else 100

            if abs(pct_change) > 0.1:  # Only show meaningful changes
                direction = "↑" if pct_change > 0 else "↓"
                changes.append((metric, base_val, curr_val, pct_change, direction))

    if changes:
        print("  Metric Changes:")
        for metric, base_val, curr_val, pct, direction in changes:
            if isinstance(base_val, float):
                print(
                    f"    {metric}: {base_val:.4f} → {curr_val:.4f} ({direction} {abs(pct):.1f}%)"
                )
            else:
                print(
                    f"    {metric}: {base_val} → {curr_val} ({direction} {abs(pct):.1f}%)"
                )
    else:
        print("  No significant metric changes detected")

    return 0


def cmd_check_regression(args: argparse.Namespace) -> int:
    """Check for significant accuracy regression compared to previous commit."""
    results_dir = Path(args.results_dir)
    history_dir = Path(args.history_dir)

    # Load current heartbeat
    heartbeat_file = results_dir / "system_heartbeat.json"
    if not heartbeat_file.exists():
        print("❌ No current metrics found")
        return 1

    with open(heartbeat_file) as f:
        current = json.load(f)

    # Load history
    history_file = history_dir / "metrics_history.jsonl"
    if not history_file.exists():
        print("⚠️  No history found - cannot check regression (first run?)")
        return 0  # Don't fail on first run

    history: List[Dict[str, Any]] = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))

    if not history:
        print("⚠️  No historical data - cannot check regression")
        return 0

    # Get previous commit's metrics
    previous = history[-1]
    current_metrics = current.get("system_metrics", current.get("metrics", {}))
    previous_metrics = previous.get("system_metrics", previous.get("metrics", {}))

    # Check accuracy
    current_accuracy = current_metrics.get("accuracy")
    previous_accuracy = previous_metrics.get("accuracy")

    if current_accuracy is None:
        print("⚠️  No accuracy metric in current results")
        return 0

    if previous_accuracy is None:
        print("⚠️  No accuracy metric in previous results")
        return 0

    # Calculate change
    accuracy_drop = previous_accuracy - current_accuracy
    pct_drop = (accuracy_drop / previous_accuracy) * 100 if previous_accuracy > 0 else 0

    threshold = args.threshold
    print(f"\\n📊 Accuracy Regression Check")
    print(
        f"   Previous: {previous_accuracy:.2%} (commit {previous.get('commit', 'unknown')[:7]})"
    )
    print(f"   Current:  {current_accuracy:.2%}")
    print(f"   Change:   {accuracy_drop:+.2%} ({pct_drop:+.1f}%)")
    print(f"   Threshold: {threshold}%\\n")

    if pct_drop > threshold:
        print(f"❌ REGRESSION DETECTED: Accuracy dropped by {pct_drop:.1f}%")
        print(f"   This exceeds the {threshold}% threshold.")
        return 1
    else:
        print(f"✅ No significant regression detected")
        return 0


def cmd_insights(args: argparse.Namespace) -> int:
    """Generate insight report."""
    results_dir = Path(args.results_dir)
    history_dir = Path(args.history_dir)

    # Load current heartbeat
    heartbeat_file = results_dir / "system_heartbeat.json"
    if not heartbeat_file.exists():
        print(f"No heartbeat found at {heartbeat_file}")
        return 1

    with open(heartbeat_file) as f:
        current = json.load(f)

    # Load history
    history: List[Dict[str, Any]] = []
    history_file = history_dir / "metrics_history.jsonl"
    if history_file.exists():
        with open(history_file) as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))

    # Load hierarchical report
    hierarchical: List[Dict[str, Any]] = []
    hierarchical_file = results_dir / "hierarchical_report.json"
    if hierarchical_file.exists():
        with open(hierarchical_file) as f:
            hierarchical = json.load(f)

    # Generate insights
    from .reporting.insights import generate_insight_report

    report = generate_insight_report(current, history, hierarchical)
    print(report)

    return 0


def cmd_full_report(args: argparse.Namespace) -> int:
    """Generate full markdown report."""
    from .reporting import MetricsReporter

    results_dir = Path(args.results_dir)
    history_dir = Path(args.history_dir)

    reporter = MetricsReporter(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    report = reporter.generate_full_report()

    # Save history if requested
    if args.save_history:
        reporter.save_historical_snapshot()
        print(f"📊 Saved snapshot for commit {reporter.git_hash}", file=sys.stderr)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✅ Report written to: {output_path}", file=sys.stderr)
    else:
        print(report)

    return 0


def cmd_pr_comment(args: argparse.Namespace) -> int:
    """Generate PR comment format report."""
    from .reporting import MetricsReporter

    results_dir = Path(args.results_dir)
    history_dir = Path(args.history_dir)

    reporter = MetricsReporter(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    report = reporter.generate_pr_comment()

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✅ PR comment written to: {output_path}", file=sys.stderr)
    else:
        print(report)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sleuth",
        description="Sleuth - Metrics collection and analysis",
    )
    parser.add_argument(
        "--results-dir",
        default="tests/results",
        help="Directory for metric results",
    )
    parser.add_argument(
        "--history-dir",
        default="tests/history",
        help="Directory for historical reports",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # report subcommand
    report_parser = subparsers.add_parser("report", help="Report commands")
    report_subparsers = report_parser.add_subparsers(dest="report_command")

    # report generate
    gen_parser = report_subparsers.add_parser("generate", help="Generate reports")

    # report trends
    trends_parser = report_subparsers.add_parser("trends", help="Show metric trends")
    trends_parser.add_argument(
        "--metric", "-m", required=True, help="Metric name to track"
    )
    trends_parser.add_argument(
        "--last", "-n", type=int, default=10, help="Number of runs to show"
    )

    # report compare
    compare_parser = report_subparsers.add_parser(
        "compare", help="Compare metrics between runs"
    )
    compare_parser.add_argument(
        "--baseline", "-b", default="", help="Baseline commit/branch"
    )
    compare_parser.add_argument(
        "--current", "-c", default="", help="Current commit/branch"
    )

    # report insights
    insights_parser = report_subparsers.add_parser(
        "insights", help="Generate insight report"
    )

    # report full
    full_parser = report_subparsers.add_parser(
        "full", help="Generate full markdown report"
    )
    full_parser.add_argument(
        "--output", "-o", help="Output file path (default: stdout)"
    )
    full_parser.add_argument(
        "--save-history", action="store_true", help="Save snapshot to history"
    )

    # report pr
    pr_parser = report_subparsers.add_parser(
        "pr", help="Generate PR comment format report"
    )
    pr_parser.add_argument("--output", "-o", help="Output file path (default: stdout)")

    # report check-regression
    regression_parser = report_subparsers.add_parser(
        "check-regression", help="Check for accuracy regression vs previous commit"
    )
    regression_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=5.0,
        help="Percentage drop threshold to fail (default: 5.0%%)",
    )

    args = parser.parse_args(argv)

    if args.command == "report":
        if args.report_command == "generate":
            return cmd_generate(args)
        elif args.report_command == "trends":
            return cmd_trends(args)
        elif args.report_command == "compare":
            return cmd_compare(args)
        elif args.report_command == "insights":
            return cmd_insights(args)
        elif args.report_command == "full":
            return cmd_full_report(args)
        elif args.report_command == "pr":
            return cmd_pr_comment(args)
        elif args.report_command == "check-regression":
            return cmd_check_regression(args)
        else:
            report_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
