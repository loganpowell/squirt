# API Reference

Complete API documentation for the Sleuth library.

## Core Module

### sleuth

Main entry point for the library.

```python
from sleuth import (
    # Configuration
    configure,
    get_config,
    configure_metrics,
    get_metrics_client,

    # Instrumentation
    track,
    Expects,
    set_test_context,
    get_test_context,

    # Metrics
    m,  # Built-in metrics namespace

    # Types
    Metric,
    MetricResult,
    AggregationType,
)
```

---

## Configuration

### configure()

```python
def configure(
    results_dir: Optional[Union[str, Path]] = None,
    history_dir: Optional[Union[str, Path]] = None,
    expectations_file: Optional[Union[str, Path]] = None,
) -> None
```

Configure global sleuth settings.

**Parameters:**

- `results_dir`: Directory to store metric results
- `history_dir`: Directory to store historical reports
- `expectations_file`: Path to expectations.json file

**Example:**

```python
from sleuth import configure

configure(
    results_dir="./tests/results",
    history_dir="./tests/history",
    expectations_file="./tests/data/expectations.json",
)
```

---

### get_config()

```python
def get_config() -> dict
```

Get current sleuth configuration.

**Returns:** Dictionary with current configuration

---

### configure_metrics()

```python
def configure_metrics(
    results_dir: str = "tests/results",
    history_dir: Optional[str] = None,
) -> "MetricsClient"
```

Configure and return a MetricsClient instance.

**Parameters:**

- `results_dir`: Directory for results
- `history_dir`: Directory for history

**Returns:** Configured MetricsClient

---

### get_metrics_client()

```python
def get_metrics_client() -> Optional["MetricsClient"]
```

Get the global MetricsClient instance.

**Returns:** MetricsClient or None if not configured

---

## Instrumentation

### @track

```python
def track(
    metrics: List[Metric] = None,
    expects: Expects = None,
    component: str = None,
    parent: str = None,
)
```

Decorator to instrument a function for metrics collection.

**Parameters:**

- `metrics`: List of metrics to collect
- `expects`: Expected output configuration
- `component`: Component name (defaults to function name)
- `parent`: Parent component for hierarchy

**Example:**

```python
@track(
    component="my_extractor",
    parent="pipeline",
    expects=Expects(input_key="text", output_key="result"),
    metrics=[
        m.runtime_ms.from_output("meta.runtime_ms"),
        m.accuracy.from_output("meta.accuracy"),
    ],
)
def my_extractor(text: str) -> dict:
    return {"result": "...", "meta": {"runtime_ms": 100, "accuracy": 0.95}}
```

---

### Expects

```python
@dataclass
class Expects:
    input_key: str
    output_key: str
    compare_fn: Optional[Callable[[Any, Any], float]] = None
```

Configuration for expected output comparison.

**Attributes:**

- `input_key`: Key in input to match against expectations
- `output_key`: Key in output to compare
- `compare_fn`: Optional custom comparison function returning 0-1

**Injected Keys:**
When `Expects` is configured, these keys are added to the output:

- `__expected_match__`: Boolean indicating exact match
- `__accuracy__`: Similarity score (0-1)
- `__expected__`: The expected value

---

### set_test_context()

```python
def set_test_context(
    test_name: str = None,
    test_file: str = None,
    **extra,
) -> None
```

Set the current test context for metric association.

---

### get_test_context()

```python
def get_test_context() -> dict
```

Get the current test context.

---

## Metrics

### m (Metrics Namespace)

Built-in metrics available through the `m` namespace:

```python
from sleuth import m

# Performance
m.runtime_ms      # Execution time in milliseconds
m.memory_mb       # Memory usage in megabytes
m.cpu_percent     # CPU utilization percentage
m.latency_p95     # 95th percentile latency
m.throughput      # Items processed per second

# Quality
m.accuracy        # Overall accuracy score (0-1)
m.completeness    # Data completeness ratio (0-1)
m.error_rate      # Error occurrence rate (0-1)
m.error_free      # Boolean: no errors occurred
m.expected_match  # Boolean: output matches expected

# Cost
m.cost_usd        # Cost in US dollars
m.total_tokens    # Total tokens used
m.input_tokens    # Input tokens
m.output_tokens   # Output tokens

# Structure
m.structure_valid # Boolean: structure is valid
m.node_count      # Number of nodes
m.nesting_depth   # Maximum nesting depth
m.line_count      # Number of lines
```

---

### Metric

```python
@dataclass
class Metric:
    name: str
    aggregation: AggregationType = AggregationType.AVERAGE

    def from_output(self, path: str) -> "BoundMetric"
    def compute(self, fn: Callable[[Any], float]) -> "BoundMetric"
    def static(self, value: float) -> "BoundMetric"
```

Base metric class.

**Methods:**

#### from_output()

```python
def from_output(self, path: str) -> "BoundMetric"
```

Extract metric value from output at given path.

```python
m.runtime_ms.from_output("metadata.runtime_ms")
# Extracts output["metadata"]["runtime_ms"]
```

#### compute()

```python
def compute(self, fn: Callable[[Any], float]) -> "BoundMetric"
```

Compute metric value from output using function.

```python
m.node_count.compute(lambda out: count_nodes(out["tree"]))
```

#### static()

```python
def static(self, value: float) -> "BoundMetric"
```

Use a static value for the metric.

```python
m.cost_usd.static(0.001)
```

---

### MetricResult

```python
@dataclass
class MetricResult:
    name: str
    value: float
    valid: bool = True
    error: Optional[str] = None
```

Result of a metric extraction.

---

### AggregationType

```python
class AggregationType(Enum):
    SUM = "sum"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    ALL = "all"      # Boolean AND
    ANY = "any"      # Boolean OR
    COUNT = "count"
```

How to aggregate multiple values.

---

## Client

### MetricsClient

```python
class MetricsClient:
    def __init__(
        self,
        results_dir: Union[str, Path] = "tests/results",
        history_dir: Optional[Union[str, Path]] = None,
    )

    def record(
        self,
        component: str,
        metrics: Dict[str, float],
        parent: Optional[str] = None,
    ) -> None

    def save(self) -> None

    def get_component_results(self, component: str) -> Dict[str, Any]

    def get_all_results(self) -> Dict[str, Dict[str, Any]]
```

Client for recording and saving metrics.

**Methods:**

#### record()

Record metrics for a component.

```python
client.record(
    component="my_extractor",
    metrics={"runtime_ms": 150.5, "accuracy": 0.95},
    parent="pipeline",
)
```

#### save()

Save all recorded metrics to disk.

#### get_component_results()

Get results for a specific component.

#### get_all_results()

Get all recorded results.

---

## Reporting

### sleuth.reporting

```python
from sleuth.reporting import (
    # Aggregation
    aggregate_values,
    aggregate_results,
    generate_heartbeat,
    aggregate_metrics_from_graph,
    generate_heartbeat_from_graph,
    save_hierarchical_reports,
    aggregate_by_system_metrics,
    find_bottlenecks,
    find_underperforming_components,

    # Data classes
    ComponentReport,
    SystemHeartbeat,

    # Insights
    Severity,
    Insight,
    InsightGenerator,
    generate_insight_report,

    # Reporting
    MetricsReporter,
)
```

---

### MetricsReporter

```python
class MetricsReporter:
    def __init__(
        self,
        results_dir: Path,
        history_dir: Path,
        git_hash: Optional[str] = None,
    )

    def generate_full_report(self) -> str
    def generate_pr_comment(self) -> str
    def save_historical_snapshot(self) -> None
```

Generate markdown reports from metrics.

**Methods:**

#### generate_full_report()

Generate detailed markdown report.

#### generate_pr_comment()

Generate concise PR comment format.

#### save_historical_snapshot()

Save current results to history.

---

### SystemHeartbeat

```python
@dataclass
class SystemHeartbeat:
    timestamp: float
    metrics: Dict[str, float]      # Component-level aggregated
    system_metrics: Dict[str, float]  # System-level aggregated
    component_count: int

    def save(self, path: Union[str, Path]) -> None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SystemHeartbeat"
```

System-level metrics snapshot.

---

### ComponentReport

```python
@dataclass
class ComponentReport:
    component: str
    parent: Optional[str]
    children: List[str]
    metrics: Dict[str, float]
```

Component-level metrics report.

---

### InsightGenerator

```python
class InsightGenerator:
    def __init__(
        self,
        heartbeat: SystemHeartbeat,
        history: List[Dict[str, Any]] = None,
        hierarchical_report: List[Dict[str, Any]] = None,
    )

    def analyze(self) -> List[Insight]
```

Generate actionable insights from metrics.

---

### Insight

```python
@dataclass
class Insight:
    severity: Severity
    title: str
    description: str
    likely_cause: str
    suggested_actions: List[str]

    def to_markdown(self) -> str
```

An actionable insight.

---

### Severity

```python
class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
```

Insight severity levels.

---

### Aggregation Functions

#### aggregate_values()

```python
def aggregate_values(
    values: List[float],
    aggregation: AggregationType,
) -> float
```

Aggregate a list of values.

#### aggregate_results()

```python
def aggregate_results(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, float]
```

Aggregate metrics across components.

#### generate_heartbeat()

```python
def generate_heartbeat(
    results: Dict[str, Dict[str, Any]],
) -> SystemHeartbeat
```

Generate heartbeat from results.

#### find_bottlenecks()

```python
def find_bottlenecks(
    results: Dict[str, Dict[str, float]],
    threshold: float = 0.5,
) -> List[str]
```

Find components using more than threshold of total resources.

---

## Analysis

### sleuth.analysis

```python
from sleuth.analysis import (
    DependencyGraph,
    analyze_codebase,
)
```

---

### DependencyGraph

```python
class DependencyGraph:
    def __init__(self)

    def add_component(
        self,
        name: str,
        parent: Optional[str] = None,
    ) -> None

    def get_children(self, name: str) -> List[str]
    def get_parent(self, name: str) -> Optional[str]
    def get_all_components(self) -> List[str]
    def get_roots(self) -> List[str]

    @classmethod
    def from_codebase(cls, path: str) -> "DependencyGraph"
```

Component dependency graph.

---

## CLI

### sleuth

```bash
sleuth [OPTIONS] COMMAND [ARGS]

Options:
  --results-dir PATH    Directory for metric results
  --history-dir PATH    Directory for historical reports
  --help                Show help message

Commands:
  report                Report commands
```

### sleuth report

```bash
sleuth report SUBCOMMAND [OPTIONS]

Subcommands:
  full      Generate full markdown report
  pr        Generate PR comment format report
  trends    Show metric trends
  compare   Compare metrics between runs
  insights  Generate insight report
  generate  Generate basic report
```

### sleuth report full

```bash
sleuth report full [OPTIONS]

Options:
  --output, -o FILE     Output file path
  --save-history        Save snapshot to history
```

### sleuth report pr

```bash
sleuth report pr [OPTIONS]

Options:
  --output, -o FILE     Output file path
```

### sleuth report trends

```bash
sleuth report trends [OPTIONS]

Options:
  --metric, -m NAME     Metric name (required)
  --last, -n COUNT      Number of runs (default: 10)
```

---

## Contrib

### sleuth.contrib.tax

Domain-specific metrics for tax processing.

```python
from sleuth.contrib.tax import tax

tax.field_accuracy    # Tax field extraction accuracy
tax.rule_coverage     # Rule coverage percentage
```

---

## Constants

### Thresholds

```python
# Default insight thresholds
ACCURACY_CRITICAL = 0.5      # Below 50% triggers critical
ACCURACY_WARNING = 0.8       # Below 80% triggers warning
MEMORY_SPIKE_THRESHOLD = 0.5 # 50% increase triggers warning
RUNTIME_REGRESSION = 0.2     # 20% increase triggers warning
COST_INCREASE = 0.25         # 25% increase triggers warning
```

---

## Exceptions

### SleuthError

```python
class SleuthError(Exception):
    """Base exception for sleuth errors."""
    pass

class MetricExtractionError(SleuthError):
    """Failed to extract metric from output."""
    pass

class ConfigurationError(SleuthError):
    """Invalid configuration."""
    pass
```

---

## Type Hints

```python
from sleuth.core.types import (
    Metric,
    MetricResult,
    AggregationType,
    BoundMetric,
    MetricExtractor,
)

from sleuth.reporting import (
    SystemHeartbeat,
    ComponentReport,
    Insight,
    Severity,
)
```
