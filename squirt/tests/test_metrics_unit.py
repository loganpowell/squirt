"""
Unit tests for the AST-Driven Metrics System.

Tests cover:
1. MetricsClient - recording and retrieving metric results
2. @track decorator - wrapping functions, measuring runtime, applying transforms
3. aggregate_values - different aggregation types (SUM, AVERAGE, MAX, MIN, COUNT, FAILURE)
4. DependencyGraphBuilder - identifying decorated functions and call relationships
5. generate_heartbeat and aggregate_metrics_from_graph - hierarchical aggregation
"""

import json
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from squirt import (
    AggregationType,
    Metric,
    MetricResult,
    MetricsClient,
    aggregate_metrics_from_graph,
    aggregate_values,
    configure_metrics,
    generate_heartbeat_from_graph,
    get_metrics_client,
    get_test_context,
    set_test_context,
    m,
    track,
)
from squirt.analysis import (
    DecoratedFunctionVisitor,
    DependencyGraph,
    DependencyGraphBuilder,
    FunctionCallVisitor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fresh_metrics_client(temp_results_dir):
    """Create a fresh MetricsClient for each test."""
    return MetricsClient(results_dir=str(temp_results_dir))


@pytest.fixture
def sample_metric_result():
    """Create a sample MetricResult for testing."""
    return MetricResult(
        component="test_component",
        test_case_id="test_case_1",
        metrics={"accuracy": 0.95, "runtime_ms": 150},
        aggregation_types={"accuracy": "average", "runtime_ms": "sum"},
        inputs={"input_text": "sample input"},
        output={"result": "sample output"},
        timestamp=time.time(),
    )


# =============================================================================
# 1. MetricsClient Tests
# =============================================================================


class TestMetricsClient:
    """Tests for MetricsClient recording and retrieval functionality."""

    def test_client_creates_results_directory(self, temp_results_dir):
        """MetricsClient should create the results directory if it doesn't exist."""
        new_dir = temp_results_dir / "new_results"
        assert not new_dir.exists()

        client = MetricsClient(results_dir=str(new_dir))

        assert new_dir.exists()
        assert client.results_dir == new_dir

    def test_record_result_stores_in_memory(
        self, fresh_metrics_client, sample_metric_result
    ):
        """Recording a result should store it in memory."""
        fresh_metrics_client.record_result(sample_metric_result)

        results = fresh_metrics_client.get_results()
        assert "test_component" in results
        assert len(results["test_component"]) == 1
        assert results["test_component"][0] == sample_metric_result

    def test_record_result_saves_to_file(
        self, fresh_metrics_client, sample_metric_result
    ):
        """Recording a result should save it to a JSON file with accumulated metrics."""
        fresh_metrics_client.record_result(sample_metric_result)

        result_file = fresh_metrics_client.results_dir / "test_component_results.json"
        assert result_file.exists()

        with open(result_file) as f:
            saved_data = json.load(f)

        assert saved_data["component"] == "test_component"
        # Metrics are now stored as lists
        assert saved_data["metrics"]["accuracy"] == [0.95]
        assert saved_data["metrics"]["runtime_ms"] == [150]

    def test_record_multiple_results_same_component(self, fresh_metrics_client):
        """Multiple results for the same component should be stored as a list."""
        result1 = MetricResult(
            component="test_component",
            test_case_id="case_1",
            metrics={"accuracy": 0.9},
            aggregation_types={"accuracy": "average"},
            inputs={},
            output={},
            timestamp=time.time(),
        )
        result2 = MetricResult(
            component="test_component",
            test_case_id="case_2",
            metrics={"accuracy": 0.95},
            aggregation_types={"accuracy": "average"},
            inputs={},
            output={},
            timestamp=time.time(),
        )

        fresh_metrics_client.record_result(result1)
        fresh_metrics_client.record_result(result2)

        results = fresh_metrics_client.get_results("test_component")
        assert len(results["test_component"]) == 2

    def test_get_results_filters_by_component(self, fresh_metrics_client):
        """get_results should filter by component name when provided."""
        result1 = MetricResult(
            component="component_a",
            test_case_id="case_1",
            metrics={"accuracy": 0.9},
            aggregation_types={},
            inputs={},
            output={},
            timestamp=time.time(),
        )
        result2 = MetricResult(
            component="component_b",
            test_case_id="case_1",
            metrics={"accuracy": 0.95},
            aggregation_types={},
            inputs={},
            output={},
            timestamp=time.time(),
        )

        fresh_metrics_client.record_result(result1)
        fresh_metrics_client.record_result(result2)

        results_a = fresh_metrics_client.get_results("component_a")
        assert "component_a" in results_a
        assert "component_b" not in results_a

    def test_get_results_returns_empty_for_unknown_component(
        self, fresh_metrics_client
    ):
        """get_results should return empty list for unknown component."""
        results = fresh_metrics_client.get_results("unknown_component")
        assert results == {"unknown_component": []}

    def test_clear_removes_all_results(
        self, fresh_metrics_client, sample_metric_result
    ):
        """clear() should remove all stored results from memory."""
        fresh_metrics_client.record_result(sample_metric_result)
        assert len(fresh_metrics_client.get_results()) > 0

        fresh_metrics_client.clear()

        assert fresh_metrics_client.get_results() == {}


# =============================================================================
# 2. @track Decorator Tests
# =============================================================================


class TestTrackDecorator:
    """Tests for the @track decorator functionality."""

    def test_decorator_wraps_function_preserves_name(self, temp_results_dir):
        """Decorator should preserve the original function name."""
        configure_metrics(results_dir=str(temp_results_dir))

        @track(
            metrics=[],
        )
        def my_test_function():
            return {"result": "test"}

        assert my_test_function.__name__ == "my_test_function"

    def test_decorator_measures_runtime(self, temp_results_dir):
        """Decorator should measure runtime internally (not modify return value)."""
        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="runtime_test")

        @track(
            metrics=[
                m.runtime_ms.from_output(
                    "_internal_runtime"
                ),  # Placeholder - won't work
            ],
        )
        def slow_function():
            time.sleep(0.05)  # 50ms sleep
            return {"result": "done"}

        result = slow_function()

        # Squirt v2: decorator auto-injects resource metrics into output.metadata
        # The original result is preserved, with metadata added
        assert result["result"] == "done"
        assert "metadata" in result
        assert "runtime_ms" in result["metadata"]
        # Type check: ensure runtime_ms is numeric before comparison
        runtime_ms = result["metadata"]["runtime_ms"]  # type: ignore[index]
        assert isinstance(runtime_ms, (int, float))
        assert float(runtime_ms) >= 50  # At least 50ms from sleep

        # Verify the function was executed (result returned correctly)
        assert "result" in result

    def test_decorator_applies_metric_transforms(self, temp_results_dir):
        """Decorator should apply all configured metric transforms."""
        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="transform_test")

        def custom_transform(inputs: Dict, output: Any) -> float:
            return 0.87

        @track(
            metrics=[
                m.custom("accuracy").compute(custom_transform),
            ],
        )
        def test_function():
            return {"result": "test"}

        test_function()

        # Check metric was recorded
        client = get_metrics_client()
        results = client.get_results("test_function")
        assert len(results["test_function"]) == 1
        assert results["test_function"][0].metrics["accuracy"] == 0.87

    def test_decorator_uses_expects_for_input_mapping(self, temp_results_dir):
        """Decorator should map function args to inputs via expects.

        Note: expects is now a simple string key. The first positional arg
        is mapped to this key.
        """
        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="mapping_test")

        captured_inputs = {}

        def capture_transform(inputs: Dict, output: Any) -> float:
            captured_inputs.update(inputs)
            return 1.0

        @track(
            expects="source_field",
            metrics=[
                m.custom("accuracy").compute(capture_transform),
            ],
        )
        def test_function(mapped_param: str = "default"):
            return {"result": "test"}

        # Call with a positional arg - this gets mapped to expects
        test_function("expected_value")

        # The first arg should be mapped to the expects key
        assert captured_inputs.get("source_field") == "expected_value"

    def test_decorator_records_aggregation_types(self, temp_results_dir):
        """Decorator should record the aggregation type for each metric."""
        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="agg_type_test")

        @track(
            metrics=[
                m.custom("accuracy").compute(lambda i, o: 0.9),
                m.runtime_ms.compute(lambda i, o: 100),
            ],
        )
        def test_function():
            return {"result": "test"}

        test_function()

        client = get_metrics_client()
        results = client.get_results("test_function")
        agg_types = results["test_function"][0].aggregation_types

        assert agg_types["accuracy"] == "average"
        assert agg_types["runtime_ms"] == "sum"

    def test_decorator_handles_transform_exceptions(self, temp_results_dir):
        """Decorator should handle exceptions in transforms gracefully."""
        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="exception_test")

        def failing_transform(inputs: Dict, output: Any) -> float:
            raise ValueError("Transform failed!")

        @track(
            metrics=[
                Metric("failing", failing_transform, AggregationType.AVERAGE),
                Metric("passing", lambda i, o: 1.0, AggregationType.AVERAGE),
            ],
        )
        def test_function():
            return {"result": "test"}

        # Function should not raise
        result = test_function()
        assert result is not None

        # Failing metric should be 0 (sleuth v2 default), passing should be recorded
        client = get_metrics_client()
        results = client.get_results("test_function")
        assert results["test_function"][0].metrics["failing"] == 0
        assert results["test_function"][0].metrics["passing"] == 1.0

    def test_decorator_stores_config_on_function(self, temp_results_dir):
        """Decorator should store configuration on the function for introspection.

        Note: The decorator stores _expects (as string), _outputs_schema,
        and _metrics directly on the function.
        """
        configure_metrics(results_dir=str(temp_results_dir))

        @track(
            expects="field",
            metrics=[
                m.custom("accuracy").compute(lambda i, o: 1.0),
            ],
        )
        def test_function():
            return {}

        # Squirt stores these attributes directly
        assert hasattr(test_function, "_expects")
        assert hasattr(test_function, "_metrics")
        assert test_function._expects == "field"
        assert len(test_function._metrics) == 1


# =============================================================================
# 3. aggregate_values Tests
# =============================================================================


class TestAggregateValues:
    """Tests for the aggregate_values function."""

    def test_average_aggregation(self):
        """AVERAGE should compute arithmetic mean."""
        values = [0.8, 0.9, 1.0]
        result = aggregate_values(values, AggregationType.AVERAGE)
        assert result == pytest.approx(0.9, rel=1e-6)

    def test_sum_aggregation(self):
        """SUM should compute total of all values."""
        values = [100, 200, 300]
        result = aggregate_values(values, AggregationType.SUM)
        assert result == 600

    def test_max_aggregation(self):
        """MAX should return the maximum value."""
        values = [10, 50, 30, 20]
        result = aggregate_values(values, AggregationType.MAX)
        assert result == 50

    def test_min_aggregation(self):
        """MIN should return the minimum value."""
        values = [10, 50, 30, 20]
        result = aggregate_values(values, AggregationType.MIN)
        assert result == 10

    def test_count_aggregation(self):
        """COUNT should return the number of values."""
        values = [1, 2, 3, 4, 5]
        result = aggregate_values(values, AggregationType.COUNT)
        assert result == 5

    def test_failure_aggregation_counts_zeros(self):
        """FAILURE should count values that are 0 or False."""
        values = [1, 0, 1, 0, 0, 1]  # 3 zeros = 3 failures
        result = aggregate_values(values, AggregationType.FAILURE)
        assert result == 3

    def test_failure_aggregation_counts_false_booleans(self):
        """FAILURE should count False values."""
        values = [True, False, True, False]  # 2 False = 2 failures
        result = aggregate_values(values, AggregationType.FAILURE)
        assert result == 2

    def test_aggregation_handles_empty_list(self):
        """Aggregation should return 0 for empty lists."""
        assert aggregate_values([], AggregationType.AVERAGE) == 0
        assert aggregate_values([], AggregationType.SUM) == 0
        assert aggregate_values([], AggregationType.MAX) == 0
        assert aggregate_values([], AggregationType.MIN) == 0
        assert aggregate_values([], AggregationType.COUNT) == 0

    def test_aggregation_filters_none_values(self):
        """Aggregation should filter out None values."""
        values = [10, None, 20, None, 30]
        result = aggregate_values(values, AggregationType.AVERAGE)
        assert result == pytest.approx(20.0, rel=1e-6)

    def test_aggregation_converts_booleans_to_int(self):
        """Booleans should be converted to int (True=1, False=0)."""
        values = [True, True, False]  # 1, 1, 0
        result = aggregate_values(values, AggregationType.SUM)
        assert result == 2

    def test_aggregation_accepts_string_type(self):
        """Aggregation should accept string aggregation type."""
        values = [10, 20, 30]
        result = aggregate_values(values, "average")
        assert result == pytest.approx(20.0, rel=1e-6)

    def test_aggregation_with_single_value(self):
        """Aggregation should work with single value."""
        assert aggregate_values([42], AggregationType.AVERAGE) == 42
        assert aggregate_values([42], AggregationType.SUM) == 42
        assert aggregate_values([42], AggregationType.MAX) == 42
        assert aggregate_values([42], AggregationType.MIN) == 42


# =============================================================================
# 4. DependencyGraphBuilder Tests
# =============================================================================


class TestDependencyGraphBuilder:
    """Tests for AST-based dependency graph building."""

    def test_identifies_decorated_functions(self):
        """Builder should identify functions with @track decorator."""
        source_code = textwrap.dedent(
            """
            from squirt import track, m

            @track(
                metrics=[m.accuracy.compute(lambda i, o: 1.0)]
            )
            def my_decorated_function(param: str) -> dict:
                return {"result": param}

            def not_decorated():
                return "plain function"
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_code)
            f.flush()

            builder = DependencyGraphBuilder()
            functions = builder.analyze_file(Path(f.name))

        assert "my_decorated_function" in functions
        assert "not_decorated" not in functions

    def test_extracts_function_parameters(self):
        """Builder should extract function parameter names."""
        source_code = textwrap.dedent(
            """
            from squirt import track

            @track(metrics=[])
            def func_with_params(arg1: str, arg2: int, arg3) -> dict:
                return {}
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_code)
            f.flush()

            builder = DependencyGraphBuilder()
            functions = builder.analyze_file(Path(f.name))

        assert functions["func_with_params"]["params"] == ["arg1", "arg2", "arg3"]

    def test_extracts_return_type_annotation(self):
        """Builder should extract return type annotation."""
        source_code = textwrap.dedent(
            """
            from squirt import track

            @track(metrics=[])
            def func_with_return() -> dict:
                return {}

            @track(metrics=[])
            def func_no_return():
                return {}
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_code)
            f.flush()

            builder = DependencyGraphBuilder()
            functions = builder.analyze_file(Path(f.name))

        assert functions["func_with_return"]["return_type"] == "dict"
        assert functions["func_no_return"]["return_type"] == "Any"

    def test_identifies_function_calls(self):
        """Builder should identify functions called within decorated functions."""
        source_code = textwrap.dedent(
            """
            from squirt import track

            @track(metrics=[])
            def parent_function() -> dict:
                result1 = child_function_a()
                result2 = child_function_b()
                return {"a": result1, "b": result2}

            @track(metrics=[])
            def child_function_a() -> dict:
                return {"child": "a"}

            @track(metrics=[])
            def child_function_b() -> dict:
                return {"child": "b"}
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_code)
            f.flush()

            builder = DependencyGraphBuilder()
            functions = builder.analyze_file(Path(f.name))

        calls = functions["parent_function"]["calls"]
        assert "child_function_a" in calls
        assert "child_function_b" in calls

    def test_builds_dependency_graph(self):
        """Builder should construct a directed graph of dependencies."""
        source_code = textwrap.dedent(
            """
            from squirt import track

            @track(metrics=[])
            def root_function() -> dict:
                return child_function()

            @track(metrics=[])
            def child_function() -> dict:
                return leaf_function()

            @track(metrics=[])
            def leaf_function() -> dict:
                return {}
        """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory to avoid temp path patterns
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            source_file = src_dir / "pipeline.py"
            source_file.write_text(source_code)

            builder = DependencyGraphBuilder()
            # Pass empty exclude_patterns to avoid default exclusions
            graph = builder.build_graph(src_dir, exclude_patterns=[])

        # Check nodes exist (DependencyGraph supports `in` operator)
        assert "root_function" in graph
        assert "child_function" in graph
        assert "leaf_function" in graph

        # Check edges (call relationships)
        assert graph.has_edge("root_function", "child_function")
        assert graph.has_edge("child_function", "leaf_function")

    def test_finds_root_nodes(self):
        """Builder should identify root nodes (not called by others)."""
        source_code = textwrap.dedent(
            """
            from squirt import track

            @track(metrics=[])
            def root_a() -> dict:
                return child()

            @track(metrics=[])
            def root_b() -> dict:
                return child()

            @track(metrics=[])
            def child() -> dict:
                return {}
        """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory to avoid temp path patterns
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            source_file = src_dir / "pipeline.py"
            source_file.write_text(source_code)

            builder = DependencyGraphBuilder()
            graph = builder.build_graph(src_dir, exclude_patterns=[])
            roots = builder.get_roots(graph)

        assert "root_a" in roots
        assert "root_b" in roots
        assert "child" not in roots

    def test_excludes_patterns(self):
        """Builder should exclude files matching exclusion patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in different locations
            main_file = Path(tmpdir) / "main.py"
            test_file = Path(tmpdir) / "test_main.py"

            source_code = textwrap.dedent(
                """
                from squirt import track

                @track(metrics=[])
                def some_function() -> dict:
                    return {}
            """
            )

            main_file.write_text(source_code)
            test_file.write_text(source_code.replace("some_function", "test_function"))

            builder = DependencyGraphBuilder()
            graph = builder.build_graph(Path(tmpdir), exclude_patterns=["test"])

        assert "some_function" in graph
        assert "test_function" not in graph

    def test_handles_async_functions(self):
        """Builder should identify async decorated functions."""
        source_code = textwrap.dedent(
            """
            from squirt import track

            @track(metrics=[])
            async def async_function() -> dict:
                return {}
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_code)
            f.flush()

            builder = DependencyGraphBuilder()
            functions = builder.analyze_file(Path(f.name))

        assert "async_function" in functions
        assert functions["async_function"].get("is_async") is True


# =============================================================================
# 5. generate_heartbeat and aggregate_metrics_from_graph Tests
# =============================================================================


class TestAggregateMetricsFromGraph:
    """Tests for hierarchical metric aggregation."""

    def test_aggregates_leaf_node_metrics(self, temp_results_dir):
        """Should return metrics directly for leaf nodes."""

        # Create a simple graph with one node
        graph = DependencyGraph()
        graph.add_node("leaf_component")

        # Create result file with list format
        result = {
            "component": "leaf_component",
            "metrics": {"accuracy": [0.95], "runtime_ms": [100]},
            "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
            "test_case_ids": ["case_1"],
            "timestamps": [1000],
            "execution_ids": ["exec_1"],
        }
        result_file = temp_results_dir / "leaf_component_results.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

        client = MetricsClient(results_dir=str(temp_results_dir))
        heartbeat = aggregate_metrics_from_graph(
            graph=graph, results_dir=str(temp_results_dir), save_reports=False
        )

        assert heartbeat["accuracy"] == 0.95
        assert heartbeat["runtime_ms"] == 100

    def test_aggregates_parent_from_children(self, temp_results_dir):
        """Parent should aggregate metrics from children."""

        # Create graph: parent -> child_a, child_b
        graph = DependencyGraph()
        graph.add_node("parent")
        graph.add_node("child_a")
        graph.add_node("child_b")
        graph.add_edge("parent", "child_a")
        graph.add_edge("parent", "child_b")

        # Create result files
        results = [
            {
                "component": "parent",
                "metrics": {},
                "aggregation_types": {},
            },
            {
                "component": "child_a",
                "metrics": {"accuracy": [0.8], "runtime_ms": [100]},
                "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
            },
            {
                "component": "child_b",
                "metrics": {"accuracy": [0.9], "runtime_ms": [200]},
                "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
            },
        ]

        for result in results:
            result_file = temp_results_dir / f"{result['component']}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f)

        client = MetricsClient(results_dir=str(temp_results_dir))
        heartbeat = aggregate_metrics_from_graph(
            graph=graph, results_dir=str(temp_results_dir), save_reports=False
        )

        # accuracy should be average: (0.8 + 0.9) / 2 = 0.85
        assert heartbeat["accuracy"] == pytest.approx(0.85, rel=1e-6)
        # runtime_ms should be sum: 100 + 200 = 300
        assert heartbeat["runtime_ms"] == 300

    def test_aggregates_deep_hierarchy(self, temp_results_dir):
        """Should aggregate metrics through multiple levels."""

        # Create graph: root -> mid -> leaf
        graph = DependencyGraph()
        for name in ["root", "mid", "leaf"]:
            graph.add_node(name)
        graph.add_edge("root", "mid")
        graph.add_edge("mid", "leaf")

        results = [
            {
                "component": "root",
                "metrics": {},
                "aggregation_types": {},
            },
            {
                "component": "mid",
                "metrics": {},
                "aggregation_types": {},
            },
            {
                "component": "leaf",
                "metrics": {"accuracy": [0.95], "runtime_ms": [150]},
                "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
            },
        ]

        for result in results:
            result_file = temp_results_dir / f"{result['component']}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f)

        client = MetricsClient(results_dir=str(temp_results_dir))
        heartbeat = aggregate_metrics_from_graph(
            graph=graph, results_dir=str(temp_results_dir), save_reports=False
        )

        # Metrics should propagate up from leaf
        assert heartbeat["accuracy"] == 0.95
        assert heartbeat["runtime_ms"] == 150

    def test_aggregates_multiple_roots(self, temp_results_dir):
        """Should handle multiple root nodes."""

        # Create graph with two independent roots
        graph = DependencyGraph()
        for name in ["root_a", "root_b"]:
            graph.add_node(name)
        # No edges - both are roots

        results = [
            {
                "component": "root_a",
                "metrics": {"accuracy": [0.8], "runtime_ms": [100]},
                "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
            },
            {
                "component": "root_b",
                "metrics": {"accuracy": [0.9], "runtime_ms": [200]},
                "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
            },
        ]

        for result in results:
            result_file = temp_results_dir / f"{result['component']}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f)

        client = MetricsClient(results_dir=str(temp_results_dir))
        heartbeat = aggregate_metrics_from_graph(
            graph=graph, results_dir=str(temp_results_dir), save_reports=False
        )

        # Should aggregate across roots
        assert heartbeat["accuracy"] == pytest.approx(0.85, rel=1e-6)
        assert heartbeat["runtime_ms"] == 300

    def test_filters_by_include_components(self, temp_results_dir):
        """Should only include specified components."""

        graph = DependencyGraph()
        for name in ["comp_a", "comp_b", "comp_c"]:
            graph.add_node(name)

        results = [
            {
                "component": "comp_a",
                "metrics": {"accuracy": [0.9]},
                "aggregation_types": {"accuracy": "average"},
            },
            {
                "component": "comp_b",
                "metrics": {"accuracy": [0.8]},
                "aggregation_types": {"accuracy": "average"},
            },
            {
                "component": "comp_c",
                "metrics": {"accuracy": [0.7]},
                "aggregation_types": {"accuracy": "average"},
            },
        ]

        for result in results:
            result_file = temp_results_dir / f"{result['component']}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f)

        heartbeat = aggregate_metrics_from_graph(
            graph=graph,
            results_dir=str(temp_results_dir),
            save_reports=False,
            include_components={"comp_a", "comp_b"},
        )

        # Only comp_a and comp_b should be included
        assert heartbeat["accuracy"] == pytest.approx(0.85, rel=1e-6)

    def test_generates_hierarchical_report(self, temp_results_dir):
        """Should generate hierarchical report file when requested."""

        graph = DependencyGraph()
        for name in ["parent", "child"]:
            graph.add_node(name)
        graph.add_edge("parent", "child")

        results = [
            {
                "component": "parent",
                "metrics": {},
                "aggregation_types": {},
            },
            {
                "component": "child",
                "metrics": {"accuracy": [0.9]},
                "aggregation_types": {"accuracy": "average"},
            },
        ]

        for result in results:
            result_file = temp_results_dir / f"{result['component']}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f)

        aggregate_metrics_from_graph(
            graph=graph, results_dir=str(temp_results_dir), save_reports=True
        )

        report_file = temp_results_dir / "hierarchical_report.json"
        assert report_file.exists()

        with open(report_file) as f:
            reports = json.load(f)

        # Should have entries for both components
        component_names = [r["component"] for r in reports]
        assert "parent" in component_names
        assert "child" in component_names


class TestGenerateHeartbeat:
    """Tests for the generate_heartbeat function."""

    def test_generates_heartbeat_with_graph(self, temp_results_dir):
        """Should generate heartbeat using dependency graph."""

        graph = DependencyGraph()
        graph.add_node("component")

        result = {
            "component": "component",
            "metrics": {"accuracy": [0.9], "runtime_ms": [100]},
            "aggregation_types": {"accuracy": "average", "runtime_ms": "sum"},
        }
        result_file = temp_results_dir / "component_results.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

        client = MetricsClient(results_dir=str(temp_results_dir))
        heartbeat = generate_heartbeat_from_graph(
            graph=graph, results_dir=str(temp_results_dir)
        )

        assert "timestamp" in heartbeat
        assert "metrics" in heartbeat
        # Metrics now have suffixes based on aggregation type
        assert heartbeat["metrics"]["accuracy.avg"] == 0.9
        assert heartbeat["metrics"]["runtime_ms.sum"] == 100

    def test_saves_heartbeat_file(self, temp_results_dir):
        """Should save heartbeat to JSON file."""

        graph = DependencyGraph()
        graph.add_node("component")

        result = {
            "component": "component",
            "metrics": {"accuracy": [0.9]},
            "aggregation_types": {"accuracy": "average"},
        }
        result_file = temp_results_dir / "component_results.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

        client = MetricsClient(results_dir=str(temp_results_dir))
        generate_heartbeat_from_graph(graph=graph, results_dir=str(temp_results_dir))

        heartbeat_file = temp_results_dir / "system_heartbeat.json"
        assert heartbeat_file.exists()

        with open(heartbeat_file) as f:
            saved_heartbeat = json.load(f)

        # Metrics now have suffixes based on aggregation type
        assert saved_heartbeat["metrics"]["accuracy.avg"] == 0.9


# =============================================================================
# Integration Tests - Full Flow
# =============================================================================


class TestFullMetricsFlow:
    """Integration tests for the complete metrics flow."""

    def test_end_to_end_single_component(self, temp_results_dir):
        """Test complete flow: decorate -> execute -> aggregate."""

        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="e2e_test")

        # Define and execute instrumented function
        @track(
            metrics=[
                m.custom("accuracy").compute(lambda i, o: 0.92),
                m.runtime_ms.from_output("metadata.runtime_ms"),
            ],
        )
        def test_component():
            return {"result": "success"}

        test_component()

        # Build graph and aggregate
        graph = DependencyGraph()
        graph.add_node("test_component")

        client = get_metrics_client()
        heartbeat = generate_heartbeat_from_graph(
            graph=graph, results_dir=str(temp_results_dir)
        )

        # Metrics now have suffixes based on aggregation type
        assert heartbeat["metrics"]["accuracy.avg"] == 0.92
        assert "runtime_ms.sum" in heartbeat["metrics"]

    def test_end_to_end_parent_child(self, temp_results_dir):
        """Test complete flow with parent-child relationship."""

        configure_metrics(results_dir=str(temp_results_dir))
        set_test_context(test_case_id="e2e_hierarchy")

        @track(
            metrics=[
                m.custom("accuracy").compute(lambda i, o: 0.8),
            ],
            record_when_child=True,  # Important: record even when called as child
        )
        def child_component():
            return {"child_result": True}

        @track(
            metrics=[
                m.custom("accuracy").compute(lambda i, o: 0.9),
            ],
        )
        def parent_component():
            child_result = child_component()
            return {"parent_result": True, "child": child_result}

        # Execute
        parent_component()

        # Build graph
        graph = DependencyGraph()
        graph.add_node("parent_component")
        graph.add_node("child_component")
        graph.add_edge("parent_component", "child_component")

        client = get_metrics_client()
        heartbeat = generate_heartbeat_from_graph(
            graph=graph, results_dir=str(temp_results_dir)
        )

        # Parent accuracy should aggregate with child
        # (0.9 + 0.8) / 2 = 0.85
        # Metrics now have suffixes based on aggregation type
        assert heartbeat["metrics"]["accuracy.avg"] == pytest.approx(0.85, rel=1e-6)
