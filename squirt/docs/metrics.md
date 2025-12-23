# Metrics Guide

Squirt provides a rich set of built-in metrics and an extensible system for creating custom metrics.

## Built-in Metrics

All built-in metrics are available through the `m` namespace:

```python
from squirt import m
```

### Performance Metrics

| Metric          | Type  | Description                    | Aggregation |
| --------------- | ----- | ------------------------------ | ----------- |
| `m.runtime_ms`  | float | Execution time in milliseconds | Sum         |
| `m.memory_mb`   | float | Memory usage in megabytes      | Max         |
| `m.cpu_percent` | float | CPU utilization percentage     | Max         |
| `m.latency_p95` | float | 95th percentile latency        | Max         |
| `m.throughput`  | float | Items processed per second     | Average     |

### Quality Metrics

| Metric             | Type        | Description                      | Aggregation |
| ------------------ | ----------- | -------------------------------- | ----------- |
| `m.accuracy`       | float (0-1) | Overall accuracy score           | Average     |
| `m.completeness`   | float (0-1) | Data completeness ratio          | Average     |
| `m.error_rate`     | float (0-1) | Error occurrence rate            | Average     |
| `m.error_free`     | bool        | Whether execution was error-free | All         |
| `m.expected_match` | bool        | Whether output matches expected  | All         |

### Cost Metrics

| Metric            | Type  | Description        | Aggregation |
| ----------------- | ----- | ------------------ | ----------- |
| `m.cost_usd`      | float | Cost in US dollars | Sum         |
| `m.total_tokens`  | int   | Total tokens used  | Sum         |
| `m.input_tokens`  | int   | Input tokens       | Sum         |
| `m.output_tokens` | int   | Output tokens      | Sum         |

### Structure Metrics

| Metric              | Type | Description                | Aggregation |
| ------------------- | ---- | -------------------------- | ----------- |
| `m.structure_valid` | bool | Whether structure is valid | All         |
| `m.node_count`      | int  | Number of nodes in output  | Average     |
| `m.nesting_depth`   | int  | Maximum nesting depth      | Max         |
| `m.line_count`      | int  | Number of lines            | Average     |

## Using Metrics

### Extracting from Output

Most metrics can extract values from the function's return value:

```python
@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),
    m.memory_mb.from_output("stats.memory"),
    m.accuracy.from_output("quality.accuracy"),
])
def my_function() -> dict:
    return {
        "result": "...",
        "metadata": {"runtime_ms": 150.5},
        "stats": {"memory": 256.0},
        "quality": {"accuracy": 0.95},
    }
```

### Computing from Output

For metrics that need to be computed from the output:

```python
@track(metrics=[
    m.node_count.compute(lambda output: count_nodes(output["tree"])),
    m.nesting_depth.compute(lambda output: get_max_depth(output["tree"])),
])
def parse_document(text: str) -> dict:
    return {"tree": build_tree(text)}
```

### Static Values

For metrics with known static values:

```python
@track(metrics=[
    m.cost_usd.static(0.001),  # Fixed cost per call
])
def api_call():
    ...
```

## Metric Categories

Metrics are organized into categories that determine how they're aggregated:

### SystemMetric (Aggregated to System Level)

```python
from squirt.categories import SystemMetric

class CustomSystemMetric(SystemMetric):
    """Metrics that aggregate to system-level reporting."""
    pass
```

System metrics include: `accuracy`, `error_rate`, `runtime_ms`, `memory_mb`, `cost_usd`

### QualityMetric

```python
from squirt.categories import QualityMetric

@track(metrics=[
    m.accuracy,      # QualityMetric
    m.completeness,  # QualityMetric
])
```

Quality metrics are averaged when aggregating across components.

### PerformanceMetric

```python
from squirt.categories import PerformanceMetric

@track(metrics=[
    m.runtime_ms,  # PerformanceMetric - summed
    m.memory_mb,   # PerformanceMetric - maxed
])
```

Performance metrics are summed (runtime) or maxed (memory, CPU).

## Creating Custom Metrics

### Simple Custom Metric

```python
from squirt.core.types import Metric, AggregationType

# Define a custom metric
custom_score = Metric(
    name="custom_score",
    aggregation=AggregationType.AVERAGE,
)

# Use it
@track(metrics=[
    custom_score.from_output("score"),
])
def my_function():
    return {"score": 0.85}
```

### Custom Metric with Validation

```python
from squirt.core.types import Metric, MetricResult, AggregationType

class ConfidenceMetric(Metric):
    """Custom metric for confidence scores."""

    def __init__(self):
        super().__init__(
            name="confidence",
            aggregation=AggregationType.AVERAGE,
        )

    def validate(self, value: float) -> MetricResult:
        if not 0 <= value <= 1:
            return MetricResult(
                name=self.name,
                value=0.0,
                valid=False,
                error="Confidence must be between 0 and 1",
            )
        return MetricResult(name=self.name, value=value, valid=True)

confidence = ConfidenceMetric()
```

### Domain-Specific Metrics (Plugins)

For domain-specific metrics, use the contrib system:

```python
# sleuth/contrib/tax.py
from squirt.core.types import Metric, AggregationType

class TaxMetrics:
    """Tax domain metrics."""

    field_accuracy = Metric(
        name="field_accuracy",
        aggregation=AggregationType.AVERAGE,
    )

    rule_coverage = Metric(
        name="rule_coverage",
        aggregation=AggregationType.AVERAGE,
    )

tax = TaxMetrics()

# Usage
from squirt.contrib.tax import tax

@track(metrics=[
    tax.field_accuracy.from_output("accuracy"),
    tax.rule_coverage.compute(calculate_coverage),
])
def extract_tax_rules():
    ...
```

## Aggregation Types

| Type      | Description       | Use Case                 |
| --------- | ----------------- | ------------------------ |
| `SUM`     | Add all values    | Runtime, tokens, cost    |
| `AVERAGE` | Calculate mean    | Accuracy, quality scores |
| `MAX`     | Take maximum      | Memory peak, CPU peak    |
| `MIN`     | Take minimum      | Minimum confidence       |
| `ALL`     | Boolean AND       | All must be true         |
| `ANY`     | Boolean OR        | Any can be true          |
| `COUNT`   | Count occurrences | Error count              |

## Best Practices

### 1. Use Appropriate Categories

```python
# ✅ Good: Use the right category
m.accuracy     # QualityMetric - averaged
m.runtime_ms   # PerformanceMetric - summed

# ❌ Bad: Wrong aggregation leads to wrong reports
```

### 2. Extract from Output When Possible

```python
# ✅ Good: Let the function return metrics
@track(metrics=[m.runtime_ms.from_output("metadata.runtime_ms")])
def my_fn():
    start = time.time()
    result = do_work()
    return {"result": result, "metadata": {"runtime_ms": (time.time() - start) * 1000}}

# ❌ Avoid: Global state or side effects
runtime = 0
def my_fn():
    global runtime
    runtime = measure()  # Side effect
```

### 3. Name Metrics Consistently

```python
# ✅ Good: snake_case, descriptive
m.accuracy
m.field_accuracy
m.extraction_runtime_ms

# ❌ Bad: Inconsistent naming
m.Accuracy
m.fieldAcc
m.time
```

### 4. Validate Custom Metrics

```python
# ✅ Good: Validate inputs
def validate(self, value):
    if value < 0:
        return MetricResult(name=self.name, value=0, valid=False, error="Negative value")
    return MetricResult(name=self.name, value=value, valid=True)
```

## Next Steps

- [Instrumentation Guide](instrumentation.md) - Learn about `@track` and expectations
- [Reporting Guide](reporting.md) - Generate reports from your metrics
- [API Reference](api.md) - Complete API documentation
