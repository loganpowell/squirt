"""
Tests for metrics filtering by namespace.

Tests the filtering functionality that allows skipping or including
specific metric namespaces at runtime.
"""

import os

import pytest

from squirt import m, track
from squirt.filters import (
    apply_runtime_filters,
    configure_namespace_filters,
    get_namespace_filters,
    only_namespaces,
    skip_namespaces,
    when_env,
)


@pytest.fixture
def sample_metrics():
    """Create sample metrics from different namespaces."""
    return [
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.memory_mb.from_output("metadata.memory_mb"),
        m.expected_match.compute(lambda i, o: 0.9),
    ]


@pytest.fixture
def mixed_namespace_metrics():
    """Create metrics from builtin and contrib namespaces."""
    from squirt.contrib.llm import llm

    return [
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.expected_match.compute(lambda i, o: 0.9),
        llm.cost.from_output("usage.cost"),
        llm.total_tokens.from_output("usage.total_tokens"),
    ]


class TestSkipNamespaces:
    """Test skip_namespaces() function."""

    def test_skip_single_namespace(self, mixed_namespace_metrics):
        """Test skipping a single namespace."""
        from squirt.contrib.llm import llm

        filtered = skip_namespaces([llm], mixed_namespace_metrics)

        # Should keep only builtin metrics
        assert len(filtered) == 2
        assert all(
            metric.name in ["runtime_ms", "expected_match"] for metric in filtered
        )

    def test_skip_multiple_namespaces(self):
        """Test skipping multiple namespaces."""
        from squirt.contrib.data import data
        from squirt.contrib.llm import llm

        metrics = [
            m.runtime_ms.from_output("metadata.runtime_ms"),
            llm.cost.from_output("usage.cost"),
            data.field_count.from_output("field_count"),
        ]

        filtered = skip_namespaces([llm, data], metrics)

        # Should keep only builtin metric
        assert len(filtered) == 1
        assert filtered[0].name == "runtime_ms"

    def test_skip_returns_all_when_no_match(self, sample_metrics):
        """Test that skip_namespaces returns all metrics when namespace doesn't match."""
        from squirt.contrib.llm import llm

        # Skip llm namespace, but no llm metrics present
        filtered = skip_namespaces([llm], sample_metrics)

        assert len(filtered) == len(sample_metrics)

    def test_skip_empty_list(self, sample_metrics):
        """Test skipping with empty namespace list."""
        filtered = skip_namespaces([], sample_metrics)

        assert len(filtered) == len(sample_metrics)


class TestOnlyNamespaces:
    """Test only_namespaces() function."""

    def test_only_single_namespace(self, mixed_namespace_metrics):
        """Test including only a single namespace."""
        filtered = only_namespaces([m], mixed_namespace_metrics)

        # Should keep only builtin metrics
        assert len(filtered) == 2
        assert all(
            metric.name in ["runtime_ms", "expected_match"] for metric in filtered
        )

    def test_only_multiple_namespaces(self):
        """Test including multiple namespaces."""
        from squirt.contrib.data import data
        from squirt.contrib.llm import llm

        metrics = [
            m.runtime_ms.from_output("metadata.runtime_ms"),
            llm.cost.from_output("usage.cost"),
            data.field_count.from_output("field_count"),
        ]

        filtered = only_namespaces([m, llm], metrics)

        # Should keep builtin and llm metrics, exclude data
        assert len(filtered) == 2
        metric_names = [metric.name for metric in filtered]
        assert "runtime_ms" in metric_names
        assert "cost_usd" in metric_names

    def test_only_returns_empty_when_no_match(self, sample_metrics):
        """Test that only_namespaces returns empty when namespace doesn't match."""
        from squirt.contrib.llm import llm

        # Only llm namespace, but no llm metrics present
        filtered = only_namespaces([llm], sample_metrics)

        assert len(filtered) == 0

    def test_only_empty_list(self, sample_metrics):
        """Test including with empty namespace list."""
        filtered = only_namespaces([], sample_metrics)

        assert len(filtered) == 0


class TestWhenEnv:
    """Test when_env() function."""

    def test_when_env_true(self, sample_metrics):
        """Test when_env returns metrics when env var matches."""
        os.environ["TEST_FLAG"] = "true"

        result = when_env("TEST_FLAG", metrics=sample_metrics)

        assert len(result) == len(sample_metrics)

        # Cleanup
        del os.environ["TEST_FLAG"]

    def test_when_env_false(self, sample_metrics):
        """Test when_env returns empty when env var doesn't match."""
        os.environ["TEST_FLAG"] = "false"

        result = when_env("TEST_FLAG", metrics=sample_metrics)

        assert len(result) == 0

        # Cleanup
        del os.environ["TEST_FLAG"]

    def test_when_env_not_set(self, sample_metrics):
        """Test when_env returns empty when env var not set."""
        result = when_env("NONEXISTENT_VAR", metrics=sample_metrics)

        assert len(result) == 0

    def test_when_env_custom_value(self, sample_metrics):
        """Test when_env with custom value."""
        os.environ["TEST_MODE"] = "production"

        result = when_env("TEST_MODE", value="production", metrics=sample_metrics)

        assert len(result) == len(sample_metrics)

        # Cleanup
        del os.environ["TEST_MODE"]

    def test_when_env_case_insensitive(self, sample_metrics):
        """Test when_env is case insensitive."""
        os.environ["TEST_FLAG"] = "TRUE"

        result = when_env("TEST_FLAG", value="true", metrics=sample_metrics)

        assert len(result) == len(sample_metrics)

        # Cleanup
        del os.environ["TEST_FLAG"]


class TestRuntimeFiltering:
    """Test runtime filtering configuration."""

    def test_configure_namespace_filters_skip(self):
        """Test configuring skip filters."""
        configure_namespace_filters(skip=["llm", "vector"])

        filters = get_namespace_filters()

        assert filters["skip"] == ["llm", "vector"]
        assert filters["only"] is None

        # Cleanup
        configure_namespace_filters(skip=None, only=None)

    def test_configure_namespace_filters_only(self):
        """Test configuring only filters."""
        configure_namespace_filters(only=["m", "data"])

        filters = get_namespace_filters()

        assert filters["only"] == ["m", "data"]
        assert filters["skip"] is None

        # Cleanup
        configure_namespace_filters(skip=None, only=None)

    def test_apply_runtime_filters_no_config(self, mixed_namespace_metrics):
        """Test apply_runtime_filters with no configuration returns all metrics."""
        configure_namespace_filters(skip=None, only=None)

        filtered = apply_runtime_filters(mixed_namespace_metrics)

        assert len(filtered) == len(mixed_namespace_metrics)

    def test_apply_runtime_filters_skip(self, mixed_namespace_metrics):
        """Test apply_runtime_filters with skip configuration."""
        configure_namespace_filters(skip=["llm"])

        filtered = apply_runtime_filters(mixed_namespace_metrics)

        # Should exclude llm metrics
        assert len(filtered) == 2
        assert all(
            metric.name in ["runtime_ms", "expected_match"] for metric in filtered
        )

        # Cleanup
        configure_namespace_filters(skip=None, only=None)

    def test_apply_runtime_filters_only(self, mixed_namespace_metrics):
        """Test apply_runtime_filters with only configuration."""
        configure_namespace_filters(only=["m"])

        filtered = apply_runtime_filters(mixed_namespace_metrics)

        # Should include only builtin metrics
        assert len(filtered) == 2
        assert all(
            metric.name in ["runtime_ms", "expected_match"] for metric in filtered
        )

        # Cleanup
        configure_namespace_filters(skip=None, only=None)

    def test_apply_runtime_filters_skip_multiple(self, mixed_namespace_metrics):
        """Test apply_runtime_filters skipping multiple namespaces."""
        configure_namespace_filters(skip=["llm", "vector"])

        filtered = apply_runtime_filters(mixed_namespace_metrics)

        # Should exclude llm metrics
        assert len(filtered) == 2
        assert all(
            metric.name in ["runtime_ms", "expected_match"] for metric in filtered
        )

        # Cleanup
        configure_namespace_filters(skip=None, only=None)


class TestFilteringIntegration:
    """Test filtering integration with @track decorator."""

    def test_track_applies_runtime_filters(self, tmp_path):
        """Test that @track decorator applies runtime filters."""
        from squirt import configure_metrics, get_metrics_client
        from squirt.contrib.llm import llm

        # Configure to skip llm namespace
        configure_namespace_filters(skip=["llm"])

        # Configure metrics client
        configure_metrics(results_dir=str(tmp_path), persist=False)

        @track(
            metrics=[
                m.runtime_ms.from_output("metadata.runtime_ms"),
                llm.cost.from_output("usage.cost"),
            ]
        )
        def test_function(text: str) -> dict:
            return {
                "result": text.upper(),
                "usage": {"cost": 0.01},
                "metadata": {"runtime_ms": 100},
            }

        test_function("test")

        # Get recorded metrics
        client = get_metrics_client()
        results = client.get_results("test_function")

        # Should have recorded runtime_ms but not cost
        assert len(results["test_function"]) == 1
        metrics = results["test_function"][0].metrics
        assert "runtime_ms" in metrics
        assert "cost_usd" not in metrics  # LLM metric should be filtered out

        # Cleanup
        configure_namespace_filters(skip=None, only=None)

    def test_track_with_only_filter(self, tmp_path):
        """Test @track with only namespace filter."""
        from squirt import configure_metrics, get_metrics_client
        from squirt.contrib.llm import llm

        # Configure to only include builtin namespace
        configure_namespace_filters(only=["m"])

        # Configure metrics client
        configure_metrics(results_dir=str(tmp_path), persist=False)

        @track(
            metrics=[
                m.expected_match.compute(lambda i, o: 0.95),
                llm.cost.from_output("usage.cost"),
            ]
        )
        def test_function(text: str) -> dict:
            return {
                "result": text.upper(),
                "usage": {"cost": 0.01},
            }

        test_function("test")

        # Get recorded metrics
        client = get_metrics_client()
        results = client.get_results("test_function")

        # Should have recorded expected_match but not cost
        assert len(results["test_function"]) == 1
        metrics = results["test_function"][0].metrics
        assert "expected_match" in metrics
        assert "cost_usd" not in metrics

        # Cleanup
        configure_namespace_filters(skip=None, only=None)
