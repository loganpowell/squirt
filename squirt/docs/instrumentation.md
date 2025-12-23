# Instrumentation Guide

Learn how to instrument your code with sleuth's `@track` decorator and expectations system.

## The @track Decorator

The `@track` decorator is the primary way to instrument functions for metrics collection.

### Basic Usage

```python
from squirt import track, m

@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),
    m.accuracy.from_output("result.accuracy"),
])
def process_document(text: str) -> dict:
    result = analyze(text)
    return {
        "result": {"data": result, "accuracy": 0.95},
        "metadata": {"runtime_ms": 150.0},
    }
```

### Parameters

| Parameter   | Type           | Description                                |
| ----------- | -------------- | ------------------------------------------ |
| `metrics`   | `List[Metric]` | Metrics to collect from this function      |
| `expects`   | `Expects`      | Expected output configuration              |
| `component` | `str`          | Component name (defaults to function name) |
| `parent`    | `str`          | Parent component for hierarchy             |

## Expectations System

The expectations system allows you to compare function outputs against expected results.

### Setting Up Expectations

1. Create an expectations file (`expectations.json`):

```json
{
  "test_case_1": {
    "input": { "description": "Sample input text" },
    "expected": {
      "bullets": ["Point 1", "Point 2", "Point 3"]
    }
  },
  "test_case_2": {
    "input": { "description": "Another input" },
    "expected": {
      "bullets": ["Different point"]
    }
  }
}
```

2. Configure sleuth to use it:

```python
from squirt import configure

configure(expectations_file="./tests/data/expectations.json")
```

### Using Expects

```python
from squirt import track, m, Expects

@track(
    expects=Expects(
        input_key="description",   # Key in input to match
        output_key="bullets",      # Key in output to compare
    ),
    metrics=[
        m.expected_match.from_output("__expected_match__"),
        m.accuracy.from_output("__accuracy__"),
    ],
)
def extract_bullets(description: str) -> dict:
    bullets = generate_bullets(description)
    return {"bullets": bullets}
```

### How Expects Works

1. When the function is called, sleuth looks up the input in expectations
2. After execution, it compares the output against the expected result
3. Special keys are injected into the result:
   - `__expected_match__`: Boolean indicating exact match
   - `__accuracy__`: Similarity score (0-1)
   - `__expected__`: The expected value for reference

### Custom Comparison Functions

```python
from squirt import Expects

def custom_compare(actual, expected) -> float:
    """Custom comparison returning accuracy 0-1."""
    # Your comparison logic
    return similarity_score

@track(
    expects=Expects(
        input_key="text",
        output_key="result",
        compare_fn=custom_compare,
    ),
    metrics=[m.accuracy.from_output("__accuracy__")],
)
def my_function(text: str) -> dict:
    ...
```

## Component Hierarchy

Squirt supports hierarchical component relationships for better aggregation.

### Defining Parent-Child Relationships

```python
@track(component="full_pipeline", metrics=[...])
def run_pipeline(text: str):
    nested = extract_nested_text(text)
    json_data = extract_json(nested)
    enriched = enrich_fields(json_data)
    return enriched

@track(
    component="extract_nested_text",
    parent="full_pipeline",
    metrics=[m.runtime_ms.from_output("metadata.runtime_ms")],
)
def extract_nested_text(text: str):
    ...

@track(
    component="extract_json",
    parent="full_pipeline",
    metrics=[m.runtime_ms.from_output("metadata.runtime_ms")],
)
def extract_json(nested: str):
    ...

@track(
    component="enrich_fields",
    parent="full_pipeline",
    metrics=[m.runtime_ms.from_output("metadata.runtime_ms")],
)
def enrich_fields(json_data: dict):
    ...
```

### Hierarchy Benefits

- **Aggregation**: Child metrics roll up to parent
- **Visualization**: Dependency trees in reports
- **Analysis**: Find bottlenecks in component chains

## Test Context

Set test context to associate metrics with specific test runs.

### In pytest

```python
import pytest
from squirt import set_test_context, get_test_context

@pytest.fixture(autouse=True)
def sleuth_context(request):
    """Set sleuth test context for each test."""
    set_test_context(
        test_name=request.node.name,
        test_file=request.fspath.basename,
    )
    yield
```

### Manual Context

```python
from squirt import set_test_context

set_test_context(
    test_name="test_extraction",
    test_file="test_pipeline.py",
    extra={"environment": "staging"},
)
```

## Metrics Client

For more control, use the MetricsClient directly.

### Basic Client Usage

```python
from squirt import configure_metrics, get_metrics_client

# Configure
configure_metrics(
    results_dir="./tests/results",
    history_dir="./tests/history",
)

# Get client
client = get_metrics_client()

# Record metrics manually
client.record(
    component="my_component",
    metrics={
        "runtime_ms": 150.5,
        "accuracy": 0.95,
    },
    parent="pipeline",
)

# Save results
client.save()
```

### Client with Context Manager

```python
from squirt.client import MetricsClient

with MetricsClient(results_dir="./results") as client:
    # Run your tests
    result = my_function()

    # Record metrics
    client.record("my_function", {"runtime_ms": 100})

# Auto-saves on exit
```

## Dependency Graph Analysis

Squirt can analyze your codebase to build dependency graphs.

```python
from squirt.analysis import DependencyGraph

# Build graph from decorated functions
graph = DependencyGraph.from_codebase("./src")

# Or manually
graph = DependencyGraph()
graph.add_component("pipeline", parent=None)
graph.add_component("extractor", parent="pipeline")
graph.add_component("processor", parent="pipeline")

# Analyze
bottlenecks = graph.find_bottlenecks(threshold=0.5)
critical_path = graph.get_critical_path()
```

## Advanced: Custom Decorators

Create specialized decorators for your domain:

```python
from squirt import track, m, Expects
from functools import wraps

def track_llm_call(model: str, max_tokens: int = 1000):
    """Custom decorator for LLM calls."""
    def decorator(fn):
        @track(
            metrics=[
                m.runtime_ms.from_output("metadata.runtime_ms"),
                m.total_tokens.from_output("metadata.tokens.total"),
                m.cost_usd.compute(
                    lambda out: calculate_cost(model, out["metadata"]["tokens"])
                ),
            ],
        )
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Usage
@track_llm_call(model="gpt-4", max_tokens=2000)
def generate_summary(text: str) -> dict:
    response = call_llm(text)
    return {
        "summary": response.text,
        "metadata": {
            "runtime_ms": response.latency,
            "tokens": {"total": response.total_tokens},
        },
    }
```

## Best Practices

### 1. Return Metrics in Output

```python
# ✅ Good: Metrics are part of the return value
@track(metrics=[m.runtime_ms.from_output("meta.runtime_ms")])
def my_fn():
    start = time.time()
    result = work()
    return {
        "result": result,
        "meta": {"runtime_ms": (time.time() - start) * 1000}
    }

# ❌ Bad: Side effects or global state
```

### 2. Use Meaningful Component Names

```python
# ✅ Good
@track(component="extract_tax_rules")
@track(component="validate_json_structure")

# ❌ Bad
@track(component="fn1")
@track(component="process")
```

### 3. Set Up Proper Hierarchy

```python
# ✅ Good: Clear parent-child relationships
@track(component="pipeline")
def pipeline(): ...

@track(component="step1", parent="pipeline")
def step1(): ...

# ❌ Bad: Flat structure loses context
@track(component="step1")
@track(component="step2")
```

### 4. Use Expectations for Regression Testing

```python
# ✅ Good: Compare against known-good outputs
@track(expects=Expects(input_key="text", output_key="result"))
def my_fn(text): ...

# Run tests with different inputs from expectations.json
```

## Next Steps

- [Metrics Guide](metrics.md) - Learn about available metrics
- [Reporting Guide](reporting.md) - Generate reports from your metrics
- [GitHub Actions](github-actions.md) - Set up CI/CD integration
