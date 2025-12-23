"""
Tests for the Sleuth metrics library.
"""

import pytest
from sleuth import m, track, AggregationType, SystemMetric
from sleuth.metrics import MetricBuilder
from sleuth.plugins import MetricNamespace
from sleuth.core.decorator import get_results, clear_results
from sleuth.contrib.data import data
from sleuth.contrib.vector import vector
from sleuth.contrib.chunk import chunk
from sleuth.contrib.llm import llm


class TestBuiltinMetrics:
    """Test built-in metrics (m.*)."""

    def test_runtime_ms(self):
        """Test runtime_ms metric."""
        builder = m.runtime_ms
        assert isinstance(builder, MetricBuilder)

        metric = builder.from_output("metadata.runtime_ms")
        assert metric.name == "runtime_ms"
        assert metric.agg == AggregationType.SUM

    def test_memory_mb(self):
        """Test memory_mb metric."""
        metric = m.memory_mb.from_output("metadata.memory_mb")
        assert metric.name == "memory_mb"
        assert metric.agg == AggregationType.MAX

    def test_error_free(self):
        """Test error_free metric."""
        metric = m.error_free.from_output("metadata.error_free")
        assert metric.name == "error_free"
        assert metric.agg == AggregationType.AVERAGE

    def test_structure_valid(self):
        """Test structure_valid metric."""
        metric = m.structure_valid.from_output("valid")
        assert metric.name == "structure_valid"
        assert metric.agg == AggregationType.AVERAGE

    def test_expected_match(self):
        """Test expected_match metric."""
        metric = m.expected_match.compare_to_expected("expected", "actual")
        assert metric.name == "expected_match"
        assert metric.agg == AggregationType.AVERAGE


class TestMetricBuilder:
    """Test MetricBuilder fluent API."""

    def test_from_output_string_path(self):
        """Test from_output with string path."""
        metric = m.runtime_ms.from_output("metadata.runtime_ms")

        # Test the transform
        output = {"metadata": {"runtime_ms": 123.45}}
        result = metric.transform({}, output)
        assert result == 123.45

    def test_from_output_lambda(self):
        """Test from_output with lambda."""
        metric = m.runtime_ms.from_output(lambda o: o["time"] * 1000)

        output = {"time": 1.5}
        result = metric.transform({}, output)
        assert result == 1500

    def test_compute(self):
        """Test compute with custom function."""

        def custom_transform(inputs, output):
            return output["score"] * inputs.get("weight", 1.0)

        metric = m.expected_match.compute(custom_transform)

        result = metric.transform({"weight": 2.0}, {"score": 0.5})
        assert result == 1.0

    def test_compare_to_expected(self):
        """Test compare_to_expected."""
        metric = m.expected_match.compare_to_expected("expected", "actual")

        # Exact match
        result = metric.transform({"expected": "hello"}, {"actual": "hello"})
        assert result == 1.0

        # No match
        result = metric.transform({"expected": "hello"}, {"actual": "goodbye"})
        assert result == 0.0


class TestTrackDecorator:
    """Test @track decorator."""

    def setup_method(self):
        """Clear results before each test."""
        clear_results()

    def test_basic_tracking(self):
        """Test basic metric tracking."""

        @track(
            metrics=[
                m.runtime_ms.from_output("metadata.runtime_ms"),
            ],
        )
        def my_component(text: str) -> dict:
            return {"result": text.upper(), "metadata": {"runtime_ms": 100}}

        result = my_component("hello")
        assert result["result"] == "HELLO"

        results = get_results()
        assert len(results) == 1
        assert results[0].component == "my_component"
        assert results[0].metrics["runtime_ms"] == 100

    def test_multiple_metrics(self):
        """Test tracking multiple metrics."""

        @track(
            metrics=[
                m.runtime_ms.from_output("metadata.runtime_ms"),
                m.memory_mb.from_output("metadata.memory_mb"),
                m.error_free.from_output("metadata.error_free"),
            ],
        )
        def my_component(text: str) -> dict:
            return {
                "result": text,
                "metadata": {"runtime_ms": 50, "memory_mb": 256, "error_free": 1.0},
            }

        my_component("test")

        results = get_results()
        assert len(results) == 1
        assert results[0].metrics["runtime_ms"] == 50
        assert results[0].metrics["memory_mb"] == 256
        assert results[0].metrics["error_free"] == 1.0

    def test_expects_contract(self):
        """Test expects contract."""

        @track(
            expects="text",
            metrics=[m.runtime_ms.from_output("runtime")],
        )
        def my_component(text: str) -> dict:
            return {"result": text.upper(), "runtime": 25}

        my_component("hello")

        results = get_results()
        assert len(results) == 1


class TestContribPlugins:
    """Test contrib plugins."""

    def test_data_metrics(self):
        """Test data plugin metrics."""
        assert isinstance(data.field_count, MetricBuilder)
        assert isinstance(data.nesting_depth, MetricBuilder)
        assert isinstance(data.structure_valid, MetricBuilder)

        metric = data.field_accuracy.from_output("accuracy")
        assert metric.name == "field_accuracy"

    def test_vector_metrics(self):
        """Test vector plugin metrics."""
        assert isinstance(vector.top_similarity, MetricBuilder)
        assert isinstance(vector.hit_rate, MetricBuilder)
        assert isinstance(vector.embedding_dimension, MetricBuilder)

        metric = vector.top_similarity.from_output("similarity")
        assert metric.name == "top_similarity"

    def test_chunk_metrics(self):
        """Test chunk plugin metrics."""
        assert isinstance(chunk.count, MetricBuilder)
        assert isinstance(chunk.avg_size, MetricBuilder)
        assert isinstance(chunk.max_size, MetricBuilder)

        metric = chunk.count.from_output("chunks")
        assert metric.name == "chunk_count"

    def test_llm_metrics(self):
        """Test LLM plugin metrics."""
        assert isinstance(llm.total_tokens, MetricBuilder)
        assert isinstance(llm.cost, MetricBuilder)
        assert isinstance(llm.completeness, MetricBuilder)

        metric = llm.total_tokens.from_output("usage.tokens")
        assert metric.name == "total_tokens"


class TestCustomPlugin:
    """Test creating custom plugins."""

    def test_custom_namespace(self):
        """Test creating a custom metric namespace."""

        class MyMetrics(MetricNamespace):
            @property
            def custom_score(self) -> MetricBuilder:
                return self._define(
                    name="custom_score",
                    aggregation=AggregationType.AVERAGE,
                    system_metric=SystemMetric.ACCURACY,
                    description="My custom score",
                )

            @property
            def custom_count(self) -> MetricBuilder:
                return self._define(
                    name="custom_count",
                    aggregation=AggregationType.SUM,
                    system_metric=None,
                )

        my = MyMetrics()

        score_metric = my.custom_score.from_output("score")
        assert score_metric.name == "custom_score"
        assert score_metric.agg == AggregationType.AVERAGE

        count_metric = my.custom_count.from_output("count")
        assert count_metric.name == "custom_count"
        assert count_metric.agg == AggregationType.SUM

    def test_custom_plugin_with_track(self):
        """Test using custom plugin with @track."""
        clear_results()

        class GameMetrics(MetricNamespace):
            @property
            def player_score(self) -> MetricBuilder:
                return self._define(
                    name="player_score",
                    aggregation=AggregationType.MAX,
                    system_metric=None,
                )

        game = GameMetrics()

        @track(
            metrics=[
                m.runtime_ms.from_output("time"),
                game.player_score.from_output("score"),
            ]
        )
        def play_game() -> dict:
            return {"score": 9999, "time": 1234}

        play_game()

        results = get_results()
        assert len(results) == 1
        assert results[0].metrics["player_score"] == 9999
        assert results[0].metrics["runtime_ms"] == 1234


class TestReporting:
    """Test reporting and aggregation features."""

    def setup_method(self):
        """Clear results before each test."""
        clear_results()

    def test_generate_heartbeat(self):
        """Test heartbeat generation."""
        from sleuth.reporting import generate_heartbeat

        @track(
            metrics=[
                m.runtime_ms.from_output("runtime"),
                m.memory_mb.from_output("memory"),
            ]
        )
        def component() -> dict:
            return {"runtime": 100, "memory": 256}

        component()

        results = get_results()
        heartbeat = generate_heartbeat(results)

        assert heartbeat.component_count == 1
        # Metrics now have suffixes based on aggregation type
        assert heartbeat.metrics["runtime_ms.sum"] == 100
        assert heartbeat.metrics["memory_mb.max"] == 256

    def test_aggregate_results(self):
        """Test metric aggregation."""
        from sleuth.reporting import aggregate_results

        @track(
            metrics=[
                m.runtime_ms.from_output("runtime"),
                # Use custom metric instead of error_free (which now has assertion mode)
                m.custom("quality").compute(lambda i, o: o.get("quality", 0.0)),
            ]
        )
        def component(val: int) -> dict:
            return {"runtime": val, "quality": 1.0 if val < 200 else 0.0}

        component(100)
        component(200)
        component(300)

        results = get_results()
        aggregated = aggregate_results(results)

        # runtime_ms uses SUM
        assert aggregated["runtime_ms"] == 600

        # quality uses AVERAGE (default)
        assert abs(aggregated["quality"] - 0.333) < 0.01

    def test_heartbeat_to_json(self):
        """Test heartbeat JSON serialization."""
        from sleuth.reporting import generate_heartbeat
        import json

        @track(metrics=[m.runtime_ms.from_output("runtime")])
        def component() -> dict:
            return {"runtime": 50}

        component()

        heartbeat = generate_heartbeat(get_results())
        json_str = heartbeat.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "timestamp" in data
        assert "metrics" in data
        assert "component_count" in data


class TestInsights:
    """Test insight generation."""

    def setup_method(self):
        """Clear results before each test."""
        clear_results()

    def test_healthy_system_no_insights(self):
        """Test that healthy system generates no critical insights."""
        from sleuth.reporting import generate_heartbeat, InsightGenerator, Severity

        @track(
            metrics=[
                m.runtime_ms.from_output("runtime"),
                m.error_free.from_output("ok"),
                m.expected_match.from_output("accuracy"),
            ]
        )
        def healthy_component() -> dict:
            return {"runtime": 100, "ok": 1.0, "accuracy": 0.95}

        healthy_component()

        heartbeat = generate_heartbeat(get_results())
        generator = InsightGenerator(heartbeat)
        insights = generator.analyze()

        # No critical or high severity insights
        critical_high = [
            i for i in insights if i.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        assert len(critical_high) == 0

    def test_low_accuracy_generates_insight(self):
        """Test that low accuracy generates critical insight."""
        from sleuth.reporting import generate_heartbeat, InsightGenerator, Severity

        @track(
            metrics=[
                m.expected_match.from_output("accuracy"),
                m.error_free.from_output("ok"),
            ]
        )
        def bad_component() -> dict:
            return {"accuracy": 0.3, "ok": 1.0}

        bad_component()

        heartbeat = generate_heartbeat(get_results())
        generator = InsightGenerator(heartbeat)
        insights = generator.analyze()

        # Should have critical accuracy insight
        critical = [i for i in insights if i.severity == Severity.CRITICAL]
        assert len(critical) >= 1
        assert any("Accuracy" in i.title for i in critical)

    def test_high_error_rate_generates_insight(self):
        """Test that high error rate generates critical insight."""
        from sleuth.reporting import generate_heartbeat, InsightGenerator, Severity

        @track(
            metrics=[
                m.error_free.from_output("ok"),
                m.expected_match.from_output("accuracy"),
            ]
        )
        def failing_component() -> dict:
            return {"ok": 0.2, "accuracy": 0.9}

        failing_component()

        heartbeat = generate_heartbeat(get_results())
        generator = InsightGenerator(heartbeat)
        insights = generator.analyze()

        # Should have error rate insight
        critical = [i for i in insights if i.severity == Severity.CRITICAL]
        assert len(critical) >= 1
        assert any("Error" in i.title for i in critical)

    def test_generate_insight_report(self):
        """Test markdown report generation."""
        from sleuth.reporting import generate_heartbeat, generate_insight_report

        @track(
            metrics=[
                m.runtime_ms.from_output("runtime"),
                m.expected_match.from_output("accuracy"),
                m.error_free.from_output("ok"),
            ]
        )
        def component() -> dict:
            return {"runtime": 100, "accuracy": 0.3, "ok": 0.5}

        component()

        heartbeat = generate_heartbeat(get_results())
        report = generate_insight_report(heartbeat)

        # Should be markdown
        assert "##" in report
        assert "Severity" in report or "Healthy" in report


class TestMetricsClient:
    """Test MetricsClient for report generation."""

    def setup_method(self):
        """Clear results before each test."""
        clear_results()

    def test_client_basic(self):
        """Test basic client operations."""
        from sleuth import MetricsClient
        from sleuth.core.types import MetricResult
        import time

        client = MetricsClient()

        # Add a result manually
        result = MetricResult(
            component="test_component",
            test_case_id="test_1",
            metrics={"runtime_ms": 100, "accuracy": 0.95},
            aggregation_types={"runtime_ms": "sum", "accuracy": "average"},
            inputs={"text": "hello"},
            output={"result": "world"},
            timestamp=time.time(),
        )
        client.add_result(result)

        assert len(client.get_results()) == 1
        heartbeat = client.generate_heartbeat()
        assert heartbeat.component_count == 1

    def test_client_hierarchical_report(self):
        """Test hierarchical report generation."""
        from sleuth import MetricsClient
        from sleuth.core.types import MetricResult
        import time

        client = MetricsClient()

        for i in range(3):
            result = MetricResult(
                component=f"component_{i}",
                test_case_id=f"test_{i}",
                metrics={"value": i * 10},
                aggregation_types={"value": "sum"},
                inputs={},
                output={},
                timestamp=time.time(),
            )
            client.add_result(result)

        report = client.generate_hierarchical_report()
        assert len(report) == 3


class TestExtensions:
    """Test extension system for custom aggregations and system metrics."""

    def test_register_aggregation(self):
        """Test registering custom aggregation function."""
        from sleuth import register_aggregation
        from sleuth.extensions import apply_aggregation

        def geometric_mean(values):
            import math

            if not values:
                return 0
            return math.exp(sum(math.log(v) for v in values if v > 0) / len(values))

        register_aggregation("geometric_mean", geometric_mean)

        result = apply_aggregation("geometric_mean", [2, 8])
        assert abs(result - 4.0) < 0.01  # sqrt(2 * 8) = 4

    def test_register_system_metric(self):
        """Test registering custom system metric."""
        from sleuth import register_system_metric
        from sleuth.extensions import get_system_metric

        register_system_metric("quality_score", "Quality Score")

        assert get_system_metric("quality_score") == "Quality Score"

    def test_list_extensions(self):
        """Test listing registered extensions."""
        from sleuth.extensions import list_aggregations, list_system_metrics

        aggs = list_aggregations()
        metrics = list_system_metrics()

        # Should return dicts (may have items from previous tests)
        assert isinstance(aggs, dict)
        assert isinstance(metrics, dict)


class TestCustomMetric:
    """Test m.custom() for ad-hoc metrics."""

    def test_custom_metric(self):
        """Test creating custom metrics with m.custom()."""
        custom = m.custom("my_custom_metric", AggregationType.SUM)
        metric = custom.from_output("value")

        assert metric.name == "my_custom_metric"
        assert metric.agg == AggregationType.SUM

    def test_custom_metric_with_system_metric(self):
        """Test custom metric with system metric mapping."""
        custom = m.custom(
            "quality_score",
            aggregation=AggregationType.AVERAGE,
            system_metric=SystemMetric.ACCURACY,
        )
        metric = custom.from_output("score")

        assert metric.name == "quality_score"
        assert metric.agg == AggregationType.AVERAGE
