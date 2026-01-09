"""
Quick test script for two-tier metrics implementation.

Tests Phase 1 components:
- categories.py: System metric mapping
- core.py: P95/P99 aggregation, system_metric_map
- aggregation.py: System metric aggregation
"""

from squirt import (
    AggregationType,
    SystemMetric,
    aggregate_by_system_metrics,
    aggregate_values,
    get_aggregation_type,
    should_invert,
)


def test_categories():
    """Test category mappings."""
    print("Testing categories.py...")

    # Test aggregation types
    assert get_aggregation_type(SystemMetric.ACCURACY) == AggregationType.AVERAGE
    assert get_aggregation_type(SystemMetric.RUNTIME_MS) == AggregationType.SUM
    assert get_aggregation_type(SystemMetric.MEMORY_MB) == AggregationType.MAX
    assert get_aggregation_type(SystemMetric.LATENCY_P95) == AggregationType.P95

    # Test inversion
    assert should_invert("error_free") is True
    assert should_invert("field_accuracy") is False

    print("✅ Categories tests passed")


def test_percentile_aggregation():
    """Test P95/P99 percentile aggregation."""
    print("\nTesting percentile aggregation...")

    values = list(range(1, 101))  # 1-100

    p95 = aggregate_values(values, AggregationType.P95)
    assert p95 == 95, f"Expected 95, got {p95}"

    p99 = aggregate_values(values, AggregationType.P99)
    assert p99 == 99, f"Expected 99, got {p99}"

    print("✅ Percentile aggregation tests passed")


def test_system_metric_aggregation():
    """Test aggregate_by_system_metrics function."""
    print("\nTesting system metric aggregation...")

    component_results = {
        "component1": {
            "metrics": {
                "field_accuracy": 0.85,
                "runtime_ms": 1000,
                "error_free": 1.0,
            },
            "system_metric_map": {
                "field_accuracy": "accuracy",
                "runtime_ms": "runtime_ms",
                "error_free": "error_rate",
            },
        },
        "component2": {
            "metrics": {
                "completeness": 0.90,
                "runtime_ms": 500,
                "structure_valid": 1.0,
            },
            "system_metric_map": {
                "completeness": "accuracy",
                "runtime_ms": "runtime_ms",
                "structure_valid": "error_rate",
            },
        },
    }

    result = aggregate_by_system_metrics(component_results)

    # Accuracy: average of 0.85 and 0.90 = 0.875
    assert (
        abs(result["accuracy"] - 0.875) < 0.01
    ), f"Expected 0.875, got {result['accuracy']}"

    # Runtime: sum of 1000 and 500 = 1500
    assert result["runtime_ms"] == 1500, f"Expected 1500, got {result['runtime_ms']}"

    # Error rate: inverted average of 1.0 and 1.0 = 0.0
    assert result["error_rate"] == 0.0, f"Expected 0.0, got {result['error_rate']}"

    print("✅ System metric aggregation tests passed")


def test_inversion():
    """Test metric inversion during aggregation."""
    print("\nTesting metric inversion...")

    component_results = {
        "component1": {
            "metrics": {
                "error_free": 0.9,  # 90% error-free
            },
            "system_metric_map": {
                "error_free": "error_rate",
            },
        },
    }

    result = aggregate_by_system_metrics(component_results)

    # error_free: 0.9 → error_rate: 0.1 (10% error rate)
    assert (
        abs(result["error_rate"] - 0.1) < 0.01
    ), f"Expected 0.1, got {result['error_rate']}"

    print("✅ Metric inversion tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Two-Tier Metrics System - Phase 1 Tests")
    print("=" * 60)

    try:
        test_categories()
        test_percentile_aggregation()
        test_system_metric_aggregation()
        test_inversion()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 1 implementation is working correctly.")
        print("Ready to proceed to Phase 2 (Reporting updates).")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
