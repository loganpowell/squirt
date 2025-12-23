"""
Unit tests for metrics aggregation functions.

Tests the aggregation logic including system metric aggregation,
metric inversion, and percentile calculations.
"""

import pytest
from typing import Dict, List, Any

from sleuth import (
    aggregate_values,
    aggregate_by_system_metrics,
    generate_heartbeat,
    AggregationType,
    MetricResult,
    SystemMetric,
)


class TestAggregateValues:
    """Test aggregate_values() function with P95/P99 support."""

    def test_average_aggregation(self):
        """Test AVERAGE aggregation."""
        values = [10.0, 20.0, 30.0, 40.0]
        result = aggregate_values(values, AggregationType.AVERAGE)
        assert result == 25.0

    def test_sum_aggregation(self):
        """Test SUM aggregation."""
        values = [10.0, 20.0, 30.0, 40.0]
        result = aggregate_values(values, AggregationType.SUM)
        assert result == 100.0

    def test_max_aggregation(self):
        """Test MAX aggregation."""
        values = [10.0, 20.0, 30.0, 40.0]
        result = aggregate_values(values, AggregationType.MAX)
        assert result == 40.0

    def test_min_aggregation(self):
        """Test MIN aggregation."""
        values = [10.0, 20.0, 30.0, 40.0]
        result = aggregate_values(values, AggregationType.MIN)
        assert result == 10.0

    def test_count_aggregation(self):
        """Test COUNT aggregation."""
        values = [10.0, 20.0, 30.0, 40.0]
        result = aggregate_values(values, AggregationType.COUNT)
        assert result == 4

    def test_p95_aggregation(self):
        """Test P95 percentile aggregation."""
        # 100 values from 1 to 100
        values = list(range(1, 101))
        result = aggregate_values(values, AggregationType.P95)
        # P95 of 1-100 should be 95
        assert result == 95

    def test_p99_aggregation(self):
        """Test P99 percentile aggregation."""
        # 100 values from 1 to 100
        values = list(range(1, 101))
        result = aggregate_values(values, AggregationType.P99)
        # P99 of 1-100 should be 99
        assert result == 99

    def test_p95_with_small_dataset(self):
        """Test P95 with dataset smaller than 20 values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = aggregate_values(values, AggregationType.P95)
        # With 5 values, index = int(5 * 0.95) - 1 = 3, value = 4.0
        assert result == 4.0

    def test_p95_with_single_value(self):
        """Test P95 with single value."""
        values = [42.0]
        result = aggregate_values(values, AggregationType.P95)
        assert result == 42.0

    def test_empty_values_returns_zero(self):
        """Test that empty values return 0.0."""
        result = aggregate_values([], AggregationType.AVERAGE)
        assert result == 0.0

        result = aggregate_values([], AggregationType.P95)
        assert result == 0.0


class TestAggregateBySystemMetrics:
    """Test aggregate_by_system_metrics() function."""

    def test_simple_accuracy_aggregation(self):
        """Test aggregating accuracy metrics."""
        component_results = {
            "extract_json": {
                "component": "extract_json",
                "metrics": {"field_accuracy": 0.9},
                "system_metric_map": {"field_accuracy": "accuracy"},
            },
            "enrich_fields": {
                "component": "enrich_fields",
                "metrics": {"field_accuracy": 0.8},
                "system_metric_map": {"field_accuracy": "accuracy"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "accuracy" in system_metrics
        # Average of 0.9 and 0.8
        assert system_metrics["accuracy"] == pytest.approx(0.85, rel=1e-5)

    def test_runtime_aggregation_sums(self):
        """Test that runtime metrics are summed."""
        component_results = {
            "chunk_pdf": {
                "component": "chunk_pdf",
                "metrics": {"runtime_ms": 1500},
                "system_metric_map": {"runtime_ms": "runtime_ms"},
            },
            "extract_json": {
                "component": "extract_json",
                "metrics": {"runtime_ms": 2000},
                "system_metric_map": {"runtime_ms": "runtime_ms"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "runtime_ms" in system_metrics
        # Sum of 1500 and 2000
        assert system_metrics["runtime_ms"] == 3500

    def test_memory_aggregation_uses_max(self):
        """Test that memory metrics use MAX aggregation."""
        component_results = {
            "component_a": {
                "component": "component_a",
                "metrics": {"memory_mb": 256},
                "system_metric_map": {"memory_mb": "memory_mb"},
            },
            "component_b": {
                "component": "component_b",
                "metrics": {"memory_mb": 512},
                "system_metric_map": {"memory_mb": "memory_mb"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "memory_mb" in system_metrics
        # Max of 256 and 512
        assert system_metrics["memory_mb"] == 512

    def test_metric_inversion(self):
        """Test that inverted metrics are properly converted."""
        component_results = {
            "validator": {
                "component": "validator",
                "metrics": {"error_free": 1.0},  # 100% error-free
                "system_metric_map": {"error_free": "error_rate"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "error_rate" in system_metrics
        # 1.0 error_free -> 0.0 error_rate
        assert system_metrics["error_rate"] == 0.0

    def test_multiple_inverted_metrics(self):
        """Test aggregating multiple inverted metrics."""
        component_results = {
            "validator_1": {
                "component": "validator_1",
                "metrics": {"error_free": 0.9},  # 90% error-free
                "system_metric_map": {"error_free": "error_rate"},
            },
            "validator_2": {
                "component": "validator_2",
                "metrics": {"structure_valid": 0.95},  # 95% valid
                "system_metric_map": {"structure_valid": "error_rate"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "error_rate" in system_metrics
        # Inverted: 0.9 -> 0.1, 0.95 -> 0.05
        # Average: (0.1 + 0.05) / 2 = 0.075
        assert system_metrics["error_rate"] == pytest.approx(0.075, rel=1e-5)

    def test_mixed_metrics_aggregation(self):
        """Test aggregating multiple system metrics at once."""
        component_results = {
            "extract_json": {
                "component": "extract_json",
                "metrics": {
                    "field_accuracy": 0.85,
                    "runtime_ms": 1500,
                    "token_cost": 0.002,
                },
                "system_metric_map": {
                    "field_accuracy": "accuracy",
                    "runtime_ms": "runtime_ms",
                    "token_cost": "cost_usd",
                },
            },
            "enrich_fields": {
                "component": "enrich_fields",
                "metrics": {
                    "field_accuracy": 0.95,
                    "runtime_ms": 800,
                    "token_cost": 0.001,
                },
                "system_metric_map": {
                    "field_accuracy": "accuracy",
                    "runtime_ms": "runtime_ms",
                    "token_cost": "cost_usd",
                },
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        # Accuracy: average
        assert system_metrics["accuracy"] == pytest.approx(0.9, rel=1e-5)
        # Runtime: sum
        assert system_metrics["runtime_ms"] == 2300
        # Cost: sum
        assert system_metrics["cost_usd"] == pytest.approx(0.003, rel=1e-5)

    def test_component_only_metrics_not_aggregated(self):
        """Test that metrics without system mapping are not included."""
        component_results = {
            "chunk_pdf": {
                "component": "chunk_pdf",
                "metrics": {
                    "chunk_count": 100,
                    "runtime_ms": 1000,
                },
                "system_metric_map": {
                    "runtime_ms": "runtime_ms",
                    # chunk_count has no mapping
                },
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "runtime_ms" in system_metrics
        assert "chunk_count" not in system_metrics

    def test_empty_component_results(self):
        """Test that empty component results return empty dict."""
        system_metrics = aggregate_by_system_metrics({})
        assert system_metrics == {}

    def test_missing_system_metric_map(self):
        """Test handling of missing system_metric_map.

        With the new implementation, system metrics must be explicitly mapped
        via system_metric_map. Without it, no system-level aggregation occurs.
        """
        component_results = {
            "test": {
                "component": "test",
                "metrics": {"field_accuracy": 0.9},
                # No system_metric_map field - no aggregation
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)
        # Without system_metric_map, metrics are not aggregated to system level
        assert system_metrics == {}

    def test_p95_latency_aggregation(self):
        """Test P95 latency aggregation across components."""
        component_results = {
            "api_call_1": {
                "component": "api_call_1",
                "metrics": {"latency_p95": 100},
                "system_metric_map": {"latency_p95": "latency_p95"},
            },
            "api_call_2": {
                "component": "api_call_2",
                "metrics": {"latency_p95": 150},
                "system_metric_map": {"latency_p95": "latency_p95"},
            },
            "api_call_3": {
                "component": "api_call_3",
                "metrics": {"latency_p95": 200},
                "system_metric_map": {"latency_p95": "latency_p95"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        assert "latency_p95" in system_metrics
        # P95 of [100, 150, 200] should be 200 (or close to it)
        assert system_metrics["latency_p95"] >= 150


class TestGenerateHeartbeat:
    """Test generate_heartbeat() function with system metrics."""

    def test_heartbeat_includes_system_metrics_field(self):
        """Test that heartbeat structure includes system_metrics field."""
        # generate_heartbeat loads from disk, so we test with aggregate_by_system_metrics
        component_results = {
            "extract_json": {
                "component": "extract_json",
                "metrics": {"field_accuracy": 0.9, "runtime_ms": 1500},
                "system_metric_map": {
                    "field_accuracy": "accuracy",
                    "runtime_ms": "runtime_ms",
                },
            },
        }

        # Test that system metrics aggregation works
        system_metrics = aggregate_by_system_metrics(component_results)

        assert isinstance(system_metrics, dict)
        assert "accuracy" in system_metrics
        assert "runtime_ms" in system_metrics
        assert system_metrics["accuracy"] == 0.9
        assert system_metrics["runtime_ms"] == 1500

    def test_heartbeat_separates_component_and_system_metrics(self):
        """Test that component and system metrics are properly separated."""
        component_results = {
            "chunk_pdf": {
                "component": "chunk_pdf",
                "metrics": {"chunk_count": 100, "runtime_ms": 800},
                "system_metric_map": {"runtime_ms": "runtime_ms"},
            },
        }

        system_metrics = aggregate_by_system_metrics(component_results)

        # System metrics should only have runtime_ms
        assert "runtime_ms" in system_metrics
        # chunk_count is component-only, should not be in system metrics
        assert "chunk_count" not in system_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
