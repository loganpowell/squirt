# Metric Filtering Guide

Squirt provides flexible runtime filtering to conditionally include or exclude metrics by namespace. This is essential for:

- **CI optimization** - Skip expensive LLM/API metrics to speed up tests
- **Environment control** - Enable debug metrics only in development
- **Test isolation** - Exclude external service calls in unit tests

## Quick Start

### Pytest CLI Flags (Recommended)

The easiest way to filter metrics is using pytest CLI flags:

```bash
# Skip expensive namespaces in CI
pytest tests/ --skip-metrics-namespaces=llm,vector

# Only collect core metrics
pytest tests/ --only-metrics-namespaces=m

# Skip multiple namespaces
pytest tests/ --skip-metrics-namespaces=llm,vector,data
```

Your code stays the same - filtering happens at runtime:

```python
from squirt import m, track
from squirt.contrib.llm import llm

@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),
    llm.cost.from_output("usage.cost"),  # Skipped when --skip-metrics-namespaces=llm
])
def my_component(text: str) -> dict:
    ...
```

### GitHub Actions

```yaml
- name: Run Tests
  run: |
    pytest tests/ --skip-metrics-namespaces=llm,vector
```

## Available Namespaces

| Name     | Module                  | Metrics                              |
| -------- | ----------------------- | ------------------------------------ |
| `m`      | `squirt.builtins`       | Core metrics (runtime, memory, etc.) |
| `llm`    | `squirt.contrib.llm`    | LLM tokens, cost, latency            |
| `vector` | `squirt.contrib.vector` | Similarity, embeddings, search       |
| `chunk`  | `squirt.contrib.chunk`  | Chunking counts and sizes            |
| `data`   | `squirt.contrib.data`   | Data structure metrics               |

## Filtering Methods

### 1. Runtime Filtering (Pytest CLI)

**Best for:** CI/CD pipelines, different test environments

```bash
# Skip specific namespaces
pytest --skip-metrics-namespaces=llm,vector

# Only include specific namespaces
pytest --only-metrics-namespaces=m,data
```

**How it works:**

- The pytest plugin intercepts CLI flags
- Configures filters before test collection
- All `@track` decorators automatically respect the configuration

### 2. Manual Filtering

**Best for:** Fine-grained control, conditional logic

```python
from squirt import m, track
from squirt.filters import skip_namespaces, only_namespaces
from squirt.contrib.llm import llm

# Skip specific namespaces
metrics = skip_namespaces([llm], [
    m.runtime_ms.from_output("metadata.runtime_ms"),
    llm.cost.from_output("usage.cost"),  # Excluded
])

@track(metrics=metrics)
def my_component(text: str) -> dict:
    ...
```

Or include only specific namespaces:

```python
# Only include builtin metrics
metrics = only_namespaces([m], [
    m.runtime_ms.from_output("metadata.runtime_ms"),
    llm.cost.from_output("usage.cost"),  # Excluded
])
```

### 3. Programmatic Configuration

**Best for:** Test fixtures, conftest.py setup

```python
# conftest.py
from squirt.filters import configure_namespace_filters

def pytest_configure(config):
    """Configure namespace filters before tests run."""
    if config.option.ci:
        # Skip expensive metrics in CI
        configure_namespace_filters(skip=['llm', 'vector'])
```

All `@track` decorators automatically respect this configuration.

### 4. Environment-Based Filtering

**Best for:** Optional debug metrics, feature flags

```python
from squirt import m, track
from squirt.filters import when_env
from squirt.contrib.llm import llm

@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),  # Always collected
    *when_env("COLLECT_LLM_METRICS", metrics=[
        llm.cost.from_output("usage.cost"),  # Only when env var = "true"
        llm.total_tokens.from_output("usage.tokens"),
    ]),
])
def my_component(text: str) -> dict:
    ...
```

Usage:

```bash
# Metrics not collected
pytest tests/

# Metrics collected
COLLECT_LLM_METRICS=true pytest tests/
```

Custom values:

```python
*when_env("TEST_MODE", value="debug", metrics=[...])
```

## Filter API Reference

### `skip_namespaces(namespaces, metrics)`

Filter out metrics from specified namespaces.

**Parameters:**

- `namespaces: list[MetricNamespace]` - Namespace objects to exclude
- `metrics: list[Metric]` - Metrics to filter

**Returns:** Filtered list excluding specified namespaces

**Example:**

```python
from squirt import m
from squirt.contrib.llm import llm
from squirt.filters import skip_namespaces

filtered = skip_namespaces([llm], [
    m.runtime_ms.from_output("metadata.runtime_ms"),  # Kept
    llm.cost.from_output("usage.cost"),  # Removed
])
```

### `only_namespaces(namespaces, metrics)`

Filter to only include metrics from specified namespaces.

**Parameters:**

- `namespaces: list[MetricNamespace]` - Namespace objects to include
- `metrics: list[Metric]` - Metrics to filter

**Returns:** Filtered list including only specified namespaces

**Example:**

```python
from squirt import m
from squirt.contrib.llm import llm
from squirt.filters import only_namespaces

filtered = only_namespaces([m], [
    m.runtime_ms.from_output("metadata.runtime_ms"),  # Kept
    llm.cost.from_output("usage.cost"),  # Removed
])
```

### `when_env(var, value="true", metrics=None)`

Conditionally include metrics based on environment variable.

**Parameters:**

- `var: str` - Environment variable name
- `value: str` - Value to match (default: "true", case-insensitive)
- `metrics: list[Metric]` - Metrics to include if condition met

**Returns:** Metrics list if condition met, empty list otherwise

**Example:**

```python
from squirt.filters import when_env
from squirt.contrib.llm import llm

expensive_metrics = when_env("ENABLE_EXPENSIVE_METRICS", metrics=[
    llm.cost.from_output("usage.cost"),
    llm.total_tokens.from_output("usage.tokens"),
])
```

### `configure_namespace_filters(skip=None, only=None)`

Configure runtime namespace filters globally.

**Parameters:**

- `skip: list[str] | None` - Namespace names to skip
- `only: list[str] | None` - Namespace names to include exclusively

**Example:**

```python
from squirt.filters import configure_namespace_filters

# Skip LLM and vector metrics
configure_namespace_filters(skip=['llm', 'vector'])

# Only collect builtin metrics
configure_namespace_filters(only=['m'])
```

## Common Use Cases

### CI Optimization

Skip expensive API/LLM metrics in CI:

```yaml
# .github/workflows/test.yml
- name: Run Tests (fast)
  run: pytest tests/ --skip-metrics-namespaces=llm,vector
```

### Development vs Production

```python
# conftest.py
import os
from squirt.filters import configure_namespace_filters

def pytest_configure(config):
    if os.getenv("CI"):
        # Skip expensive metrics in CI
        configure_namespace_filters(skip=['llm', 'vector'])
    elif os.getenv("ENVIRONMENT") == "dev":
        # Collect all metrics in dev
        configure_namespace_filters(skip=None, only=None)
```

### Feature Flags

```python
from squirt import m, track
from squirt.filters import when_env
from squirt.contrib.llm import llm

@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),
    *when_env("BETA_FEATURES", metrics=[
        llm.cost.from_output("usage.cost"),
    ]),
])
def beta_component(text: str) -> dict:
    ...
```

### Test Isolation

Skip external service metrics in unit tests:

```python
# tests/unit/conftest.py
from squirt.filters import configure_namespace_filters

def pytest_configure(config):
    # Unit tests shouldn't call external services
    configure_namespace_filters(skip=['llm', 'vector'])
```

```python
# tests/integration/conftest.py
from squirt.filters import configure_namespace_filters

def pytest_configure(config):
    # Integration tests can use all metrics
    configure_namespace_filters(skip=None, only=None)
```

## Creating Filterable Plugins

To make your custom plugin filterable, inherit from `MetricNamespace`:

```python
from squirt.plugins import MetricNamespace, MetricBuilder, AggregationType

class MyMetrics(MetricNamespace):
    """My custom metrics namespace."""

    @property
    def my_metric(self) -> MetricBuilder:
        """My custom metric."""
        return self._define(
            name="my_metric",
            aggregation=AggregationType.AVERAGE,
            description="My custom metric",
        )

# Export singleton
my_metrics = MyMetrics()
```

Users can now filter your metrics:

```bash
pytest --skip-metrics-namespaces=my_metrics
```

## Best Practices

1. **Use pytest flags for environment-specific behavior** - Keep code the same across environments
2. **Skip expensive metrics in CI** - Speed up test runs by excluding LLM/API calls
3. **Use `when_env()` for optional debug metrics** - Don't clutter default runs
4. **Configure filters in `conftest.py`** - Centralize test configuration
5. **Document namespace names** - Make it easy for users to know what to skip

## Troubleshooting

### Metrics still collected after filtering

Make sure your namespace inherits from `MetricNamespace`:

```python
# ✅ Correct - filterable
class MyMetrics(MetricNamespace):
    ...

# ❌ Wrong - not filterable
class MyMetrics:
    ...
```

### Namespace name not recognized

The namespace name is derived from the module. Check available names:

```python
from squirt.filters import get_namespace_filters

# After configuring
print(get_namespace_filters())
```

Supported names: `m`, `llm`, `vector`, `chunk`, `data`, or your custom namespace class name.
