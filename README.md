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
    llm.completeness.evaluate(completeness_eval),
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
    llm.completeness.evaluate(completeness_eval),
])
def my_component(text: str) -> dict:
    ...
```

### Creating Custom Plugins

Use `MetricBuilder` with explicit type annotations for full IDE support:

```python
from squirt.plugins import MetricBuilder, AggregationType, SystemMetric

class MyDomainMetrics:
    """Custom metrics for my domain."""

    # System metric - auto-derives AVERAGE aggregation
    field_accuracy: MetricBuilder = MetricBuilder(
        "field_accuracy",
        system_metric=SystemMetric.ACCURACY,
        description="Per-field accuracy score",
    )

    # System metric with inversion
    validation_passed: MetricBuilder = MetricBuilder(
        "validation_passed",
        system_metric=SystemMetric.ERROR_RATE,
        inverted=True,
        description="Whether validation passed",
    )

    # Non-system metric - must specify aggregation
    field_count: MetricBuilder = MetricBuilder(
        "field_count",
        aggregation=AggregationType.AVERAGE,
        description="Average field count per document",
    )

# Create singleton instance
my_domain = MyDomainMetrics()

# Usage: my_domain.field_accuracy.from_output("score")
```

Key points for plugin authors:

1. Use plain classes (no inheritance needed)
2. Add explicit `MetricBuilder` type annotations to all attributes
3. Use `MetricBuilder()` directly (preferred) or the `custom()` factory
4. Export a singleton instance

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

| Document                                   | Description                                   |
| ------------------------------------------ | --------------------------------------------- |
| [Metrics Guide](docs/metrics.md)           | Built-in metrics and creating custom metrics  |
| [Instrumentation](docs/instrumentation.md) | Using `@track` decorator and expectations     |
| [Reporting](docs/reporting.md)             | Generating reports, PR comments, and insights |
| [GitHub Actions](docs/github-actions.md)   | CI/CD integration and workflows               |
| [API Reference](docs/api.md)               | Complete API documentation                    |

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
