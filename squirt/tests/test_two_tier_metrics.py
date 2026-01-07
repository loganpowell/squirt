"""
Quick test script for two-tier metrics implementation.

Tests Phase 1 components:
- categories.py: System metric mapping
- core.py: P95/P99 aggregation, system_metric_map
- aggregation.py: System metric aggregation
- transforms.py: New transform functions
"""

from squirt import (
    AggregationType,
    SystemMetric,
    aggregate_by_system_metrics,
    aggregate_values,
    get_aggregation_type,
    should_invert,
)
from squirt.transforms import (
    cpu_usage_transform,
    latency_p95_transform,
    memory_usage_transform,
    throughput_transform,
    token_cost_transform,
    total_tokens_transform,
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


def test_token_cost_transform():
    """Test LLM cost calculation."""
    print("\nTesting token_cost_transform...")

    # OpenAI format
    output = {
        "model": "gpt-4",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        },
    }

    cost = token_cost_transform({}, output)
    # GPT-4: $30 per 1M input, $60 per 1M output
    # (1000 / 1M) * 30 + (500 / 1M) * 60 = 0.03 + 0.03 = 0.06
    expected = 0.06
    assert abs(cost - expected) < 0.001, f"Expected {expected}, got {cost}"

    # Anthropic format
    output_anthropic = {
        "model": "claude-3-sonnet",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
        },
    }

    cost = token_cost_transform({}, output_anthropic)
    # Claude-3-Sonnet: $3 per 1M input, $15 per 1M output
    # (1000 / 1M) * 3 + (500 / 1M) * 15 = 0.003 + 0.0075 = 0.0105
    expected = 0.0105
    assert abs(cost - expected) < 0.0001, f"Expected {expected}, got {cost}"

    print("✅ Token cost transform tests passed")


def test_total_tokens_transform():
    """Test token count extraction."""
    print("\nTesting total_tokens_transform...")

    # OpenAI format with total_tokens
    output = {
        "usage": {
            "total_tokens": 1500,
        }
    }

    tokens = total_tokens_transform({}, output)
    assert tokens == 1500, f"Expected 1500, got {tokens}"

    # Anthropic format (sum of input + output)
    output_anthropic = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
        }
    }

    tokens = total_tokens_transform({}, output_anthropic)
    assert tokens == 1500, f"Expected 1500, got {tokens}"

    print("✅ Total tokens transform tests passed")


def test_resource_transforms():
    """Test resource monitoring transforms."""
    print("\nTesting resource transforms...")

    output = {
        "metadata": {
            "peak_memory_mb": 512.5,
            "avg_cpu_percent": 45.2,
            "item_count": 100,
            "runtime_ms": 2000,
            "latency_p95_ms": 150.3,
        }
    }

    # Memory
    memory = memory_usage_transform({}, output)
    assert memory == 512.5, f"Expected 512.5, got {memory}"

    # CPU
    cpu = cpu_usage_transform({}, output)
    assert cpu == 45.2, f"Expected 45.2, got {cpu}"

    # Throughput (100 items / 2000ms = 50 items/sec)
    throughput = throughput_transform({}, output)
    assert throughput == 50.0, f"Expected 50.0, got {throughput}"

    # Latency P95
    latency = latency_p95_transform({}, output)
    assert latency == 150.3, f"Expected 150.3, got {latency}"

    print("✅ Resource transform tests passed")


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
        test_token_cost_transform()
        test_total_tokens_transform()
        test_resource_transforms()

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
