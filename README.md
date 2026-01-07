# Squirt 🔍

**Metrics collection and analysis library for component testing**

Squirt provides a unified framework for instrumenting code, collecting metrics, analyzing performance, and generating actionable reports. Built for CI/CD pipelines with first-class GitHub Actions support.

## Features

- 🎯 **Decorator-based instrumentation** - Add metrics to any function with `@track`
- 📊 **Built-in metrics** - Runtime, memory, accuracy, and more out of the box
- 🔌 **Extensible** - Create custom metrics with simple Python classes
- 📈 **Historical tracking** - Sparkline trends and commit-over-commit comparisons
- 🚨 **Actionable insights** - Automatic detection of regressions and issues
- 🤖 **GitHub Actions ready** - PR comments and job summaries

## Quick Start

### Installation

```bash
# With pip
pip install squirt

# With uv
uv pip install squirt

# From source
pip install git+https://github.com/loganpowell/squirt.git
```

### Basic Usage

```python
from squirt import m, track

@track(
    expects="text",  # Input key from test expectations
    metrics=[
        # runtime_ms, memory_mb, cpu_percent are auto-recorded!
        m.expected_match.compare_to_expected("data", "result"),
        m.structure_valid.compute(lambda i, o: validate_structure(o)),
    ],
)
def my_component(text: str) -> dict:
    result = process(text)
    return {
        "result": result,
        "metadata": {},  # runtime_ms, memory_mb, cpu_percent auto-injected here
    }
```

The `@track` decorator automatically:

- Measures and records `runtime_ms`, `memory_mb`, and `cpu_percent`
- Injects these values into `output.metadata` for downstream use
- No manual timing code needed!

## Pytest Integration

Squirt provides a zero-config pytest plugin that handles everything automatically.

### Zero-Config Setup

```python
# conftest.py
pytest_plugins = ["squirt.pytest"]
```

That's it! The plugin auto-discovers:

- `tests/results` for metrics output
- `tests/history` for historical data
- `tests/data/expectations.json` for test expectations
- `tests/instrumented` for component dependency graph

### Custom Configuration

```python
# conftest.py
from squirt.pytest import configure_squirt

configure_squirt(
    results_dir="custom/results",
    history_dir="custom/history",
    default_source="custom/expectations.json",
    auto_heartbeat=True,   # Generate heartbeat on session finish
    verbose=True,          # Print configuration info
)

pytest_plugins = ["squirt.pytest"]
```

### CLI Options

```bash
# Default - metrics enabled
pytest tests/

# Custom expectations file
pytest --squirt-expectations tests/data/custom.json tests/

# Disable metrics collection
pytest --no-metrics tests/

# Custom results directory
pytest --metrics-dir ./custom/results tests/

# Skip expensive metric namespaces (e.g., for CI)
pytest --skip-metrics-namespaces=llm,vector tests/

# Only collect specific metric namespaces
pytest --only-metrics-namespaces=m tests/
```

### Provided Fixtures

The plugin provides these fixtures automatically:

| Fixture             | Scope    | Description                   |
| ------------------- | -------- | ----------------------------- |
| `metrics_client`    | session  | Access to collected metrics   |
| `expectations_data` | session  | Loaded expectations from JSON |
| `dependency_graph`  | session  | Component dependency graph    |
| `set_context`       | function | Set test context for metrics  |

### Example Test

```python
def test_my_component(set_context, expectations_data):
    """Test with metrics collection."""
    test_case = expectations_data[0]
    set_context(test_case)

    result = my_component(test_case["text"])

    assert result["result"] is not None
    # Metrics are automatically collected!
```

## IDE Support

Squirt is designed for robust IDE autocomplete with Pylance/Pyright. However, there's an important caveat:

### ⚠️ Long Metrics Lists

**Pylance can time out on long inline lists inside decorators.** If you have more than ~5 metrics, extract them to a module-level variable:

```python
# ✅ Good - IDE autocomplete works reliably
_my_component_metrics = [
    # Note: runtime_ms, memory_mb, cpu_percent are auto-recorded!
    m.expected_match.compare_to_expected("data", "result"),
    m.structure_valid.compute(my_validation_fn),
    m.error_free.compute(error_check_fn),
]

@track(metrics=_my_component_metrics)
def my_component(text: str) -> dict:
    ...
```

```python
# ❌ Avoid - Pylance may lose autocomplete on long inline lists
@track(metrics=[
    m.expected_match.compare_to_expected("data", "result"),
    m.structure_valid.compute(my_validation_fn),
    m.error_free.compute(error_check_fn),
])
def my_component(text: str) -> dict:
    ...
```

## Filtering Metrics by Namespace

Squirt provides flexible runtime filtering to conditionally include or exclude metrics by namespace - perfect for:

- **CI optimization** - Skip expensive LLM/API metrics to speed up tests
- **Environment control** - Enable debug metrics only in development
- **Test isolation** - Exclude external service calls in unit tests

### Runtime Filtering (Recommended)

**Use pytest flags to filter at test time** - no code changes needed:

```bash
# Skip expensive LLM metrics in CI
pytest tests/ --skip-metrics-namespaces=llm,vector

# Only collect core built-in metrics
pytest tests/ --only-metrics-namespaces=m

# Skip multiple namespaces
pytest tests/ --skip-metrics-namespaces=llm,vector,data
```

Your decorators stay the same regardless of environment:

```python
from squirt import m, track
from squirt.contrib.llm import llm

# Define all metrics - filtering happens at runtime based on pytest flags
@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),
    m.expected_match.compute(accuracy_fn),
    llm.cost.from_output("usage.cost"),  # Skipped when --skip-metrics-namespaces=llm
    llm.total_tokens.from_output("usage.tokens"),  # Skipped when --skip-metrics-namespaces=llm
])
def my_component(text: str) -> dict:
    ...
```

**In GitHub Actions:**

```yaml
- name: Run Tests (skip expensive metrics)
  run: pytest tests/ --skip-metrics-namespaces=llm,vector
```

### Manual Filtering (Advanced)

For programmatic control, use the filter functions directly:

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

Or configure filters programmatically:

```python
from squirt.filters import configure_namespace_filters

# Configure once at the start of your test session
configure_namespace_filters(skip=['llm', 'vector'])

# All @track decorators automatically respect this configuration
```

### Environment-Based Filtering

Use `when_env()` to conditionally enable metrics:

```python
from squirt import m, track
from squirt.filters import when_env
from squirt.contrib.llm import llm

@track(metrics=[
    m.runtime_ms.from_output("metadata.runtime_ms"),  # Always collected
    *when_env("COLLECT_LLM_METRICS", metrics=[
        llm.cost.from_output("usage.cost"),  # Only when COLLECT_LLM_METRICS=true
        llm.total_tokens.from_output("usage.tokens"),
    ]),
])
def my_component(text: str) -> dict:
    ...
```

### Available Namespace Names

| Name     | Module                  | Metrics                              |
| -------- | ----------------------- | ------------------------------------ |
| `m`      | `squirt.builtins`       | Core metrics (runtime, memory, etc.) |
| `llm`    | `squirt.contrib.llm`    | LLM tokens, cost, latency            |
| `vector` | `squirt.contrib.vector` | Similarity, embeddings, search       |
| `chunk`  | `squirt.contrib.chunk`  | Chunking counts and sizes            |
| `data`   | `squirt.contrib.data`   | Data structure metrics               |

````

### Creating Custom Plugins

Create reusable metric namespaces using `MetricNamespace`:

```python
from squirt.plugins import MetricNamespace, MetricBuilder, AggregationType, SystemMetric

class TaxMetrics(MetricNamespace):
    """Custom metrics for tax domain."""

    @property
    def field_accuracy(self) -> MetricBuilder:
        """Accuracy of field extraction."""
        return self._define(
            name="field_accuracy",
            system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE
            description="Per-field accuracy score",
        )

    @property
    def validation_passed(self) -> MetricBuilder:
        """Whether validation passed."""
        return self._define(
            name="validation_passed",
            system_metric=SystemMetric.ERROR_RATE,
            inverted=True,  # 1.0 = passed, 0.0 = failed
            description="Tax validation status",
        )

    @property
    def field_count(self) -> MetricBuilder:
        """Number of extracted fields."""
        return self._define(
            name="field_count",
            aggregation=AggregationType.AVERAGE,  # Non-system metric
            description="Average field count per document",
        )

# Create singleton instance
tax = TaxMetrics()

# Usage with full IDE support
@track(metrics=[
    tax.field_accuracy.from_output("accuracy"),
    tax.field_count.from_output("count"),
])
def extract_tax_fields(document: str) -> dict:
    ...
```

Key benefits of using `MetricNamespace`:

1. **Namespace filtering support** - Users can skip your entire plugin with `--skip-metrics-namespaces=tax`
2. **Full IDE autocomplete** - Properties provide robust type hints
3. **Consistent patterns** - `_define()` handles aggregation and system metrics automatically

### Configuration

```python
from squirt import configure

configure(
    results_dir="./tests/results",
    history_dir="./tests/history",
    expectations_file="./tests/data/expectations.json",
)
```

### CLI Usage

```bash
# Generate full report
squirt report full --output report.md --save-history

# Generate PR comment
squirt report pr --output pr-comment.md

# View trends
squirt report trends --metric accuracy --last 10

# Generate insights
squirt report insights

# Check for regressions
squirt report check-regression
```

## Documentation

| Document                                          | Description                                   |
| ------------------------------------------------- | --------------------------------------------- |
| [Metrics Guide](squirt/docs/metrics.md)           | Built-in metrics and creating custom metrics  |
| [Instrumentation](squirt/docs/instrumentation.md) | Using `@track` decorator and expectations     |
| [Filtering Guide](squirt/docs/filtering.md)       | Filtering metrics by namespace                |
| [Reporting](squirt/docs/reporting.md)             | Generating reports, PR comments, and insights |
| [GitHub Actions](squirt/docs/github-actions.md)   | CI/CD integration and workflows               |
| [API Reference](squirt/docs/api.md)               | Complete API documentation                    |

## Architecture

```
squirt/
├── core/           # Core types, decorators, and base classes
├── categories/     # Metric categories (quality, performance, cost)
├── builtins.py     # Pre-configured metrics (m.runtime_ms, m.accuracy, etc.)
├── reporting/      # Aggregation, insights, and report generation
├── analysis/       # Code analysis and dependency graphs
├── contrib/        # Domain-specific metric plugins
├── cli.py          # Command-line interface
└── tests/          # Library tests
```

## Metric Categories

Squirt organizes metrics into categories for proper aggregation:

| Category        | Metrics                                  | Aggregation |
| --------------- | ---------------------------------------- | ----------- |
| **Performance** | `runtime_ms`, `memory_mb`, `cpu_percent` | Sum/Max     |
| **Quality**     | `accuracy`, `completeness`, `error_rate` | Average     |
| **Cost**        | `cost_usd`, `total_tokens`               | Sum         |
| **Structure**   | `node_count`, `nesting_depth`            | Average     |

## Example Output

### PR Comment

```markdown
# Metrics Summary - abc1234

**Branch:** feature/new-extraction
**Timestamp:** 2025-12-21T10:30:00

## 🔴 1 critical issue(s)

- **Critical Accuracy Drop**: System accuracy is at 45%, below 50%...

## 📊 System Metrics

| Metric     | Current | Δ   |
| ---------- | ------- | --- |
| accuracy   | 45.0%   | ↓   |
| runtime_ms | 2.5s    | →   |
| memory_mb  | 512 MB  | ↑   |
```

### Full Report Features

- Executive summary with system-level metrics
- Performance sparkline trends (▁▂▃▄▅▆▇█)
- Mermaid treemap visualizations
- Dependency tree with metrics
- Actionable insights with suggested fixes

## Testing

```bash
# Run squirt tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=squirt --cov-report=html
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

### Quick Development Setup

```bash
# Clone repository
git clone https://github.com/loganpowell/squirt.git
cd squirt

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Releasing

Maintainers can release new versions using the automated script:

```bash
./release.sh patch  # 0.1.0 → 0.1.1
./release.sh minor  # 0.1.1 → 0.2.0
./release.sh major  # 0.2.0 → 1.0.0
```

See [RELEASE.md](RELEASE.md) for detailed release documentation.

## License

MIT

---

_Squirt - Because metrics should be invisible until they matter._
````
