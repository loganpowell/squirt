"""
Integration test for two-tier metrics system.

Tests that system metrics are properly generated from real component results.
"""

import json
import sys
from pathlib import Path

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent

from squirt import (
    generate_heartbeat_from_graph,
    aggregate_by_system_metrics,
    DependencyGraphBuilder,
    MetricsClient,
    configure_metrics,
)


def test_system_metrics_integration(tmp_path):
    """
    Test that system metrics are properly generated from component results.

    This validates the full two-tier metrics flow:
    1. Component results saved with system_metric_map
    2. Heartbeat generation includes system_metrics
    3. System metrics are properly aggregated
    """
    # Create sample component results with system metric mappings
    component_results = {
        "extract_json": {
            "component": "extract_json",
            "test_case_id": "test_001",
            "metrics": {
                "field_accuracy": 0.85,
                "runtime_ms": 1500,
                "token_cost": 0.002,
                "total_tokens": 1200,
            },
            "aggregation_types": {
                "field_accuracy": "average",
                "runtime_ms": "sum",
                "token_cost": "sum",
                "total_tokens": "sum",
            },
            "system_metric_map": {
                "field_accuracy": "accuracy",
                "runtime_ms": "runtime_ms",
                "token_cost": "cost_usd",
                "total_tokens": "total_tokens",
            },
            "timestamp": 1234567890.0,
            "execution_id": "exec-001",
        },
        "chunk_pdf": {
            "component": "chunk_pdf",
            "test_case_id": "test_001",
            "metrics": {
                "chunk_count": 50,
                "runtime_ms": 500,
                "error_free": 1.0,
            },
            "aggregation_types": {
                "chunk_count": "sum",
                "runtime_ms": "sum",
                "error_free": "average",
            },
            "system_metric_map": {
                "runtime_ms": "runtime_ms",
                "error_free": "error_rate",
            },
            "timestamp": 1234567890.0,
            "execution_id": "exec-002",
        },
        "enrich_fields": {
            "component": "enrich_fields",
            "test_case_id": "test_001",
            "metrics": {
                "completeness": 0.92,
                "runtime_ms": 800,
                "token_cost": 0.003,
                "total_tokens": 1800,
            },
            "aggregation_types": {
                "completeness": "average",
                "runtime_ms": "sum",
                "token_cost": "sum",
                "total_tokens": "sum",
            },
            "system_metric_map": {
                "completeness": "accuracy",
                "runtime_ms": "runtime_ms",
                "token_cost": "cost_usd",
                "total_tokens": "total_tokens",
            },
            "timestamp": 1234567890.0,
            "execution_id": "exec-003",
        },
    }

    # Save component results to temp directory
    for component_name, result in component_results.items():
        result_file = tmp_path / f"{component_name}_latest.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    # Test system metric aggregation directly
    system_metrics = aggregate_by_system_metrics(component_results)

    # Check accuracy (average of field_accuracy:0.85 and completeness:0.92)
    assert "accuracy" in system_metrics
    expected_accuracy = (0.85 + 0.92) / 2
    assert (
        abs(system_metrics["accuracy"] - expected_accuracy) < 0.01
    ), f"Expected accuracy {expected_accuracy}, got {system_metrics['accuracy']}"

    # Check runtime_ms (sum of all runtime_ms: 1500 + 500 + 800)
    assert "runtime_ms" in system_metrics
    expected_runtime = 1500 + 500 + 800
    assert (
        system_metrics["runtime_ms"] == expected_runtime
    ), f"Expected runtime {expected_runtime}, got {system_metrics['runtime_ms']}"

    # Check cost_usd (sum of token_cost: 0.002 + 0.003)
    assert "cost_usd" in system_metrics
    expected_cost = 0.002 + 0.003
    assert (
        abs(system_metrics["cost_usd"] - expected_cost) < 0.0001
    ), f"Expected cost {expected_cost}, got {system_metrics['cost_usd']}"

    # Check total_tokens (sum: 1200 + 1800)
    assert "total_tokens" in system_metrics
    expected_tokens = 1200 + 1800
    assert (
        system_metrics["total_tokens"] == expected_tokens
    ), f"Expected tokens {expected_tokens}, got {system_metrics['total_tokens']}"

    # Check error_rate (inverted from error_free: 1.0 → 0.0)
    assert "error_rate" in system_metrics
    assert (
        system_metrics["error_rate"] == 0.0
    ), f"Expected error_rate 0.0, got {system_metrics['error_rate']}"

    print("\n✅ System metrics integration test passed!")
    print(f"\nSystem Metrics Generated:")
    for metric_name, value in sorted(system_metrics.items()):
        print(f"  {metric_name}: {value}")


def test_system_metrics_with_inversion(tmp_path):
    """Test that error metrics are properly inverted to error_rate."""

    component_results = {
        "component1": {
            "component": "component1",
            "test_case_id": "test_001",
            "metrics": {
                "error_free": 0.9,  # 90% error-free
            },
            "aggregation_types": {
                "error_free": "average",
            },
            "system_metric_map": {
                "error_free": "error_rate",
            },
            "timestamp": 1234567890.0,
            "execution_id": "exec-001",
        },
        "component2": {
            "component": "component2",
            "test_case_id": "test_001",
            "metrics": {
                "structure_valid": 0.95,  # 95% valid
            },
            "aggregation_types": {
                "structure_valid": "average",
            },
            "system_metric_map": {
                "structure_valid": "error_rate",
            },
            "timestamp": 1234567890.0,
            "execution_id": "exec-002",
        },
    }

    # Save results
    for component_name, result in component_results.items():
        result_file = tmp_path / f"{component_name}_latest.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    # Test system metric aggregation directly
    system_metrics = aggregate_by_system_metrics(component_results)

    # error_free: 0.9 → error_rate: 0.1 (inverted)
    # structure_valid: 0.95 → error_rate: 0.05 (inverted)
    # Average: (0.1 + 0.05) / 2 = 0.075
    assert "error_rate" in system_metrics
    expected_error_rate = 0.075
    assert (
        abs(system_metrics["error_rate"] - expected_error_rate) < 0.01
    ), f"Expected error_rate {expected_error_rate}, got {system_metrics['error_rate']}"

    print("\n✅ Metric inversion test passed!")
    print(f"  Component1 error_free: 0.9 → error_rate: 0.1")
    print(f"  Component2 structure_valid: 0.95 → error_rate: 0.05")
    print(f"  Aggregated error_rate: {system_metrics['error_rate']:.3f}")


def test_component_only_metrics_not_in_system(tmp_path):
    """Test that component-specific metrics don't appear in system metrics."""

    component_results = {
        "chunk_pdf": {
            "component": "chunk_pdf",
            "test_case_id": "test_001",
            "metrics": {
                "chunk_count": 100,  # Component-specific, no system mapping
                "header_count": 50,  # Component-specific, no system mapping
                "runtime_ms": 500,
            },
            "aggregation_types": {
                "chunk_count": "sum",
                "header_count": "sum",
                "runtime_ms": "sum",
            },
            "system_metric_map": {
                "runtime_ms": "runtime_ms",
                # chunk_count and header_count have no system mapping
            },
            "timestamp": 1234567890.0,
            "execution_id": "exec-001",
        },
    }

    # Save results
    for component_name, result in component_results.items():
        result_file = tmp_path / f"{component_name}_latest.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    # Test system metric aggregation
    system_metrics = aggregate_by_system_metrics(component_results)

    # Component-specific metrics should NOT be in system_metrics
    # (chunk_count and header_count have no system_metric mapping)
    assert "chunk_count" not in system_metrics
    assert "header_count" not in system_metrics

    # runtime_ms SHOULD be in system_metrics
    assert "runtime_ms" in system_metrics
    assert system_metrics["runtime_ms"] == 500

    print("\n✅ Component-only metrics test passed!")
    print(f"  System metrics: {list(system_metrics.keys())}")


if __name__ == "__main__":
    """Run tests directly with pytest."""
    import tempfile

    print("=" * 70)
    print("Two-Tier Metrics Integration Tests")
    print("=" * 70)

    try:
        # Use separate temp directories for each test
        with tempfile.TemporaryDirectory() as tmpdir1:
            test_system_metrics_integration(Path(tmpdir1))

        print()
        with tempfile.TemporaryDirectory() as tmpdir2:
            test_system_metrics_with_inversion(Path(tmpdir2))

        print()
        with tempfile.TemporaryDirectory() as tmpdir3:
            test_component_only_metrics_not_in_system(Path(tmpdir3))

        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


# =============================================================================
# Metrics Reporting Tests
# =============================================================================


class TestMetricsReporting:
    """Test metrics reporting and heartbeat generation."""

    def test_heartbeat_generation(self, tmp_path):
        """Test that heartbeat can be generated from results."""
        from squirt import (
            generate_heartbeat_from_graph,
            DependencyGraphBuilder,
            MetricsClient,
        )

        # Build a simple graph for testing
        builder = DependencyGraphBuilder()
        # Create an empty graph (no files to scan)
        graph = builder.build_graph(tmp_path, exclude_patterns=["*"])

        # Create sample result files in temp dir
        sample_result = {
            "component": "full_pipeline",
            "metrics": {"runtime_ms": 1000, "accuracy": 0.9},
            "aggregation_types": {"runtime_ms": "sum", "accuracy": "average"},
        }
        result_file = tmp_path / "full_pipeline_latest.json"
        with open(result_file, "w") as f:
            json.dump(sample_result, f)

        # Use temp directory for test isolation
        temp_client = MetricsClient(results_dir=str(tmp_path))

        # Generate heartbeat using graph-based API
        heartbeat = generate_heartbeat_from_graph(
            graph=graph, results_dir=str(tmp_path)
        )

        assert "timestamp" in heartbeat
        assert "metrics" in heartbeat

    def test_hierarchical_report_format(self, tmp_path):
        """Test the hierarchical report format."""
        from squirt import save_hierarchical_reports

        # Create sample node metrics
        node_metrics = {
            "full_pipeline": {
                "metrics": {"runtime_ms": 3000, "accuracy": 0.9},
                "aggregation_types": {"runtime_ms": "sum", "accuracy": "average"},
                "parent": None,
                "children": ["extract_nested_text", "extract_json"],
            },
            "extract_nested_text": {
                "metrics": {"runtime_ms": 1000, "accuracy": 0.95},
                "aggregation_types": {"runtime_ms": "sum", "accuracy": "average"},
                "parent": "full_pipeline",
                "children": [],
            },
        }

        # Save report to temp directory (not actual results dir)
        save_hierarchical_reports(node_metrics, tmp_path)

        # Verify file was created
        report_file = tmp_path / "hierarchical_report.json"
        assert report_file.exists()

        # Verify content
        with open(report_file) as f:
            reports = json.load(f)

        assert len(reports) == 2
        assert any(r["component"] == "full_pipeline" for r in reports)
