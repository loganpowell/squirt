# Creating Custom Metrics and Transforms

This guide shows you how to create custom metrics for your domain-specific needs using two patterns: the **Builder Pattern** (quick and simple) and the **Namespace Pattern** (organized and reusable).

## Table of Contents

1. [Quick Start: Builder Pattern](#quick-start-builder-pattern)
2. [Organized Approach: Namespace Pattern](#organized-approach-namespace-pattern)
3. [Understanding Transforms](#understanding-transforms)
4. [Common Patterns](#common-patterns)
5. [Best Practices](#best-practices)
6. [Examples](#examples)

---

## Quick Start: Builder Pattern

The Builder Pattern is perfect for ad-hoc metrics or when you're just getting started.

### Method 1: Extract from Output Path

Extract values directly from your function's return value using dot notation:

```python
from squirt import m, track

@track(metrics=[
    m.custom("extraction_time").from_output("metadata.duration_ms"),
    m.custom("word_count").from_output("stats.words"),
])
def process_document(text: str) -> dict:
    return {
        "content": "processed...",
        "metadata": {"duration_ms": 1500},
        "stats": {"words": 342}
    }
```

**Use when:**

- Your function already returns structured data with the metric values
- You need simple value extraction with no computation

### Method 2: Compute from Lambda

Use a lambda or function to compute the metric from the output:

```python
@track(metrics=[
    m.custom("complexity_score").compute(
        lambda inputs, output: len(output.get("rules", [])) * 2.5
    ),
    m.custom("avg_field_length").compute(
        lambda i, o: sum(len(str(v)) for v in o.values()) / len(o)
    ),
])
def extract_rules(text: str) -> dict:
    return {"rules": ["rule1", "rule2", "rule3"], "source": text}
```

**Use when:**

- You need to compute a value from the output
- The calculation is simple and inline

### Method 3: Compute from Custom Transform

For complex logic, create a dedicated transform function:

```python
def calculate_accuracy(inputs: dict, output: dict) -> float:
    """Calculate accuracy by comparing output to expected."""
    expected = inputs.get("expected_data", [])
    actual = output.get("results", [])

    if not expected:
        return 0.0

    matches = sum(1 for e, a in zip(expected, actual) if e == a)
    return matches / len(expected)

@track(metrics=[
    m.custom("rule_accuracy").compute(calculate_accuracy),
])
def extract_rules(text: str, expected_data: list) -> dict:
    return {"results": ["extracted", "rules"]}
```

**Use when:**

- The calculation is complex or reused across components
- You want testable, documented metric logic
- The transform needs multiple helper functions

### Method 4: Compare to Expected

Compare output to expected ground truth:

```python
@track(metrics=[
    m.expected_match.compare_to_expected("expected_output", "result"),
    m.custom("field_match").compare_to_expected("expected_fields", "fields"),
])
def process_data(text: str, expected_output: dict, expected_fields: list) -> dict:
    return {
        "result": {"processed": "data"},
        "fields": ["field1", "field2"]
    }
```

**Use when:**

- You have ground truth expectations in your test data
- You want automatic similarity scoring

### Aggregation and System Metrics

Control how metrics are combined across test cases and components:

```python
from squirt import m, AggregationType, SystemMetric

@track(metrics=[
    # Explicit aggregation type
    m.custom("total_tokens", aggregation=AggregationType.SUM),
    m.custom("peak_memory", aggregation=AggregationType.MAX),
    m.custom("avg_accuracy", aggregation=AggregationType.AVERAGE),
    m.custom("p95_latency", aggregation=AggregationType.P95),

    # System metric (auto-derives aggregation)
    m.custom("accuracy", system_metric=SystemMetric.ACCURACY),  # → AVERAGE
    m.custom("runtime_ms", system_metric=SystemMetric.RUNTIME_MS),  # → SUM
    m.custom("error_rate", system_metric=SystemMetric.ERROR_RATE),  # → AVERAGE
])
def my_component(): ...
```

**Aggregation Types:**

- `SUM` - Add values across tests (tokens, runtime, counts)
- `AVERAGE` - Mean across tests (accuracy, scores, rates)
- `MAX` - Maximum value across tests (peak memory, worst latency)
- `P95` - 95th percentile across tests (latency, response times)

**System Metrics** (auto-map to standard categories):

- `ACCURACY`, `PRECISION`, `RECALL`, `F1_SCORE` → AVERAGE
- `RUNTIME_MS`, `TOTAL_TOKENS` → SUM
- `MEMORY_MB`, `CPU_PERCENT` → MAX
- `ERROR_RATE` → AVERAGE

**Use when:**

- Need specific aggregation behavior across test runs
- Want metrics grouped by standard categories in reports
- Building reusable metrics that should aggregate consistently

---

## Organized Approach: Namespace Pattern

The Namespace Pattern is ideal for reusable, domain-specific metrics that you'll use across many components.

### Creating a Custom Namespace

```python
# myproject/metrics.py
from squirt import MetricNamespace, MetricBuilder, SystemMetric, AggregationType

class TaxMetrics(MetricNamespace):
    """Domain-specific metrics for tax processing."""

    @property
    def rule_accuracy(self) -> MetricBuilder:
        """Accuracy of extracted tax rules."""
        return self._define(
            "rule_accuracy",
            system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE aggregation
            description="Tax rule extraction accuracy"
        )

    @property
    def rule_count(self) -> MetricBuilder:
        """Number of rules extracted."""
        return self._define(
            "rule_count",
            aggregation=AggregationType.SUM,  # Explicit aggregation
            description="Total rules extracted"
        )

    @property
    def rule_complexity(self) -> MetricBuilder:
        """Maximum complexity score across rules."""
        return self._define(
            "rule_complexity",
            aggregation=AggregationType.MAX,
            description="Maximum rule complexity"
        )

# Create singleton instance
tax = TaxMetrics()
```

### Using Your Custom Namespace

```python
from myproject.metrics import tax

@track(metrics=[
    tax.rule_accuracy.compute(calculate_rule_accuracy),
    tax.rule_count.from_output("rules.count"),
    tax.rule_complexity.from_output("metadata.max_complexity"),
])
def extract_tax_rules(text: str) -> dict:
    return {
        "rules": {"count": 5, "items": [...]},
        "metadata": {"max_complexity": 8.5}
    }
```

### Real-World Example: Echo Namespace

Here's how the built-in `echo` namespace is implemented:

```python
# squirt/contrib/echo/metrics.py
class EchoMetrics(MetricNamespace):
    """Debug metrics for I/O logging."""

    def save(
        self,
        input_field: str,
        expected_field: str,
        actual_field: str,
    ) -> Metric:
        """Save input/expected/actual comparison to markdown file."""

        def save_transform(inputs: dict, output: dict) -> float:
            # Extract values using field paths
            input_val = _extract_field(inputs, input_field)
            expected_val = _extract_field(inputs, expected_field)
            actual_val = _extract_field(output, actual_field)

            # Format and save to file
            markdown = format_comparison(input_val, expected_val, actual_val)
            save_to_file(markdown, f"{component_name}_echo.md")

            # Return similarity score
            return calculate_similarity(expected_val, actual_val)

        return self._define(
            f"echo_save_{input_field}",
            aggregation=AggregationType.AVERAGE,
            description=f"Save {input_field} → {expected_field} comparison"
        ).compute(save_transform)

# Usage
from squirt.contrib.echo import echo

@track(metrics=[
    echo.save(
        input_field="description",
        expected_field="expected_rule",
        actual_field="taxAssistRule"
    )
])
def extract_rule(description: str, expected_rule: dict) -> dict:
    return {"taxAssistRule": {...}}
```

### Real-World Example: Tokens Namespace

Here's how the `tokens` namespace provides configurable token/cost tracking:

```python
# squirt/contrib/tokens/metrics.py
class TokensMetrics(MetricNamespace):
    """Token and cost tracking metrics."""

    def count(
        self,
        input_path: str,
        output_path: str,
        system_prompt: str = "",
    ) -> Metric:
        """Estimate token count from specific fields."""

        def estimate_tokens(inputs: dict, output: dict) -> int:
            input_text = _extract_field(inputs, input_path)
            output_text = _extract_field(output, output_path)

            input_tokens = len(input_text) / 3.5
            output_tokens = len(output_text) / 3.5
            system_tokens = len(system_prompt) / 3.5

            return int(input_tokens + system_tokens + output_tokens)

        return self._define(
            "total_tokens",
            system_metric=SystemMetric.TOTAL_TOKENS,  # SUM aggregation
            description=f"Tokens from {input_path} + {output_path}"
        ).compute(estimate_tokens)

# Usage
from squirt.contrib.tokens import tokens

@track(metrics=[
    tokens.count(
        input_path="query",
        output_path="response",
        system_prompt="You are a helpful assistant."
    ),
    tokens.cost(
        input_cost_per_1m=5.0,
        output_cost_per_1m=15.0,
        input_path="query",
        output_path="response"
    )
])
def call_llm(query: str) -> dict:
    return {"response": "..."}
```

---

## Understanding Transforms

Transforms are the functions that extract or compute metric values. They have a consistent signature:

```python
def my_transform(inputs: dict[str, Any], output: Any) -> float | int | bool:
    """
    Extract or compute a metric value.

    Args:
        inputs: Mapped values from expectations.json or function parameters
        output: The function's return value

    Returns:
        The metric value (float, int, or bool)
    """
    # Your logic here
    return value
```

### Transform Anatomy

```python
def field_accuracy_transform(inputs: dict, output: dict) -> float:
    """
    Calculate field extraction accuracy.

    Compares extracted fields to expected fields from test expectations.
    """
    # 1. Extract data from inputs (test expectations)
    expected_fields = inputs.get("expected_fields", {})

    # 2. Extract data from output (function result)
    actual_fields = output.get("extracted_fields", {})

    # 3. Handle edge cases
    if not expected_fields:
        return 0.0  # Return 0, not None

    # 4. Compute the metric
    matches = sum(
        1 for key in expected_fields
        if actual_fields.get(key) == expected_fields[key]
    )

    # 5. Return the value
    return matches / len(expected_fields)
```

### Transform Best Practices

1. **Always return a value** - Never return `None`, use `0`, `0.0`, or `False` for missing data
2. **Handle edge cases** - Check for empty dicts, None values, missing keys
3. **Use type hints** - Makes the transform self-documenting
4. **Add docstrings** - Explain what the transform measures
5. **Keep them pure** - No side effects (except for special cases like `echo.save()`)
6. **Use helpers** - Extract common logic into helper functions

---

## Common Patterns

### Pattern 1: Nested Field Extraction

```python
def extract_nested_field(data: dict, path: str) -> Any:
    """Extract field using dot notation (e.g., 'data.user.name')."""
    value = data
    for key in path.split("."):
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value

def nested_field_transform(inputs: dict, output: dict) -> str:
    """Extract deeply nested field."""
    return extract_nested_field(output, "result.data.taxRule.condition") or ""
```

### Pattern 2: List/Array Metrics

```python
def item_count_transform(inputs: dict, output: dict) -> int:
    """Count items in a list."""
    items = output.get("items", [])
    return len(items) if isinstance(items, list) else 0

def avg_item_length_transform(inputs: dict, output: dict) -> float:
    """Average length of items."""
    items = output.get("items", [])
    if not items:
        return 0.0
    return sum(len(str(item)) for item in items) / len(items)
```

### Pattern 3: Boolean Validation

```python
def has_required_structure_transform(inputs: dict, output: dict) -> bool:
    """Check if output has required structure."""
    if not isinstance(output, dict):
        return False

    required_keys = ["id", "name", "data"]
    return all(key in output for key in required_keys)

def valid_format_transform(inputs: dict, output: dict) -> bool:
    """Check if format is valid."""
    data = output.get("data")
    if not data:
        return False

    # Custom validation logic
    return isinstance(data, dict) and "version" in data
```

### Pattern 4: Similarity/Comparison

```python
def compute_similarity(expected: Any, actual: Any) -> float:
    """Compute similarity between expected and actual values."""
    if expected == actual:
        return 1.0

    # String similarity
    if isinstance(expected, str) and isinstance(actual, str):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, expected, actual).ratio()

    # List similarity
    if isinstance(expected, list) and isinstance(actual, list):
        matches = sum(1 for e, a in zip(expected, actual) if e == a)
        return matches / max(len(expected), len(actual))

    return 0.0

def similarity_transform(inputs: dict, output: dict) -> float:
    """Compare output to expected value."""
    expected = inputs.get("expected_data")
    actual = output.get("result")
    return compute_similarity(expected, actual)
```

### Pattern 5: Factory Functions

For configurable transforms, use factory functions:

```python
def create_threshold_checker(
    field_path: str,
    threshold: float,
    above: bool = True
) -> Callable[[dict, dict], bool]:
    """
    Factory to create threshold-checking transforms.

    Args:
        field_path: Dot-notation path to field
        threshold: Threshold value
        above: True to check >= threshold, False for <= threshold

    Returns:
        Transform function
    """
    def transform(inputs: dict, output: dict) -> bool:
        value = extract_nested_field(output, field_path)
        if value is None:
            return False

        if above:
            return float(value) >= threshold
        else:
            return float(value) <= threshold

    return transform

# Usage
@track(metrics=[
    m.custom("high_confidence").compute(
        create_threshold_checker("confidence", 0.8, above=True)
    ),
    m.custom("low_latency").compute(
        create_threshold_checker("latency_ms", 100, above=False)
    ),
])
def my_component(): ...
```

---

## Best Practices

### 1. Choose the Right Pattern

**Use Builder Pattern when:**

- Prototyping or exploring
- One-off metrics for specific components
- Simple extraction or computation
- Learning the system

**Use Namespace Pattern when:**

- Building reusable domain metrics
- Multiple components share metrics
- Need organized metric definitions
- Building a metrics library

### 2. Naming Conventions

```python
# Good metric names
m.custom("rule_extraction_accuracy")  # Clear and specific
m.custom("avg_processing_time_ms")    # Includes units
m.custom("has_valid_structure")       # Boolean prefix

# Avoid
m.custom("accuracy")                  # Too generic
m.custom("time")                      # No units
m.custom("check")                     # Unclear purpose
```

### 3. System Metric Mapping

Map to system metrics for automatic aggregation and reporting:

```python
# Maps to SystemMetric.ACCURACY (AVERAGE aggregation)
@property
def extraction_accuracy(self) -> MetricBuilder:
    return self._define(
        "extraction_accuracy",
        system_metric=SystemMetric.ACCURACY,  # AVERAGE
    )

# Maps to SystemMetric.RUNTIME_MS (SUM aggregation)
@property
def processing_time(self) -> MetricBuilder:
    return self._define(
        "processing_time_ms",
        system_metric=SystemMetric.RUNTIME_MS,  # SUM
    )

# No system mapping (custom aggregation)
@property
def rule_count(self) -> MetricBuilder:
    return self._define(
        "rule_count",
        aggregation=AggregationType.SUM,  # Explicit
    )
```

### 4. Error Handling

```python
def robust_transform(inputs: dict, output: dict) -> float:
    """Robust transform with proper error handling."""
    try:
        # Extract with defaults
        value = output.get("result", {}).get("score", 0.0)

        # Validate type
        if not isinstance(value, (int, float)):
            return 0.0

        # Clamp to valid range
        return max(0.0, min(1.0, float(value)))

    except Exception as e:
        # Log but don't crash
        print(f"Transform error: {e}")
        return 0.0
```

### 5. Testing Transforms

```python
# test_transforms.py
def test_field_accuracy_transform():
    """Test field accuracy calculation."""
    inputs = {
        "expected_fields": {"name": "John", "age": 30}
    }
    output = {
        "extracted_fields": {"name": "John", "age": 25}
    }

    accuracy = field_accuracy_transform(inputs, output)
    assert accuracy == 0.5  # 1 of 2 matches

def test_field_accuracy_empty():
    """Test with empty expectations."""
    accuracy = field_accuracy_transform({}, {})
    assert accuracy == 0.0  # Not None!
```

---

## Examples

### Example 1: Tax Rule Extraction Metrics

```python
# project/metrics.py
from squirt import MetricNamespace, MetricBuilder, SystemMetric, AggregationType

class TaxRuleMetrics(MetricNamespace):
    """Metrics for tax rule extraction pipeline."""

    @property
    def rule_structure_valid(self) -> MetricBuilder:
        """Whether extracted rule has valid structure."""
        return self._define(
            "rule_structure_valid",
            system_metric=SystemMetric.ERROR_RATE,
            inverted=True,  # Valid structure means no error
            description="Tax rule structure validity"
        )

    @property
    def field_accuracy(self) -> MetricBuilder:
        """Field-level extraction accuracy."""
        return self._define(
            "field_accuracy",
            system_metric=SystemMetric.ACCURACY,
            description="Field extraction accuracy"
        )

    @property
    def condition_depth(self) -> MetricBuilder:
        """Maximum nesting depth of conditions."""
        return self._define(
            "condition_depth",
            aggregation=AggregationType.MAX,
            description="Max condition nesting depth"
        )

    @property
    def conclusion_count(self) -> MetricBuilder:
        """Number of conclusions in rule."""
        return self._define(
            "conclusion_count",
            aggregation=AggregationType.AVERAGE,
            description="Avg conclusions per rule"
        )

tax = TaxRuleMetrics()

# project/transforms.py
def validate_rule_structure(inputs: dict, output: dict) -> bool:
    """Validate tax rule structure."""
    rule = output.get("taxAssistRule", {})

    required_fields = ["taxAssistCondition", "taxAssistConclusions"]
    if not all(field in rule for field in required_fields):
        return False

    # Validate condition is not empty
    condition = rule.get("taxAssistCondition", {})
    if not condition or not condition.get("operator"):
        return False

    # Validate at least one conclusion
    conclusions = rule.get("taxAssistConclusions", [])
    return isinstance(conclusions, list) and len(conclusions) > 0

def calculate_field_accuracy(inputs: dict, output: dict) -> float:
    """Calculate field extraction accuracy."""
    expected = inputs.get("expected_rule", {})
    actual = output.get("taxAssistRule", {})

    if not expected:
        return 0.0

    matches = sum(
        1 for key in expected
        if actual.get(key) == expected[key]
    )
    return matches / len(expected)

# Usage
from project.metrics import tax
from project.transforms import validate_rule_structure, calculate_field_accuracy

@track(metrics=[
    tax.rule_structure_valid.compute(validate_rule_structure),
    tax.field_accuracy.compute(calculate_field_accuracy),
    tax.condition_depth.from_output("taxAssistRule.condition.depth"),
    tax.conclusion_count.from_output("taxAssistRule.conclusions.count"),
])
def extract_tax_rule(description: str, expected_rule: dict) -> dict:
    return {"taxAssistRule": {...}}
```

### Example 2: Vector Search Metrics

```python
class SearchMetrics(MetricNamespace):
    """Metrics for vector search quality."""

    @property
    def hit_rate(self) -> MetricBuilder:
        """Percentage of queries that found results."""
        return self._define(
            "hit_rate",
            system_metric=SystemMetric.ACCURACY,
            description="Search hit rate"
        )

    @property
    def mrr(self) -> MetricBuilder:
        """Mean Reciprocal Rank of correct results."""
        return self._define(
            "mrr",
            system_metric=SystemMetric.ACCURACY,
            description="Mean Reciprocal Rank"
        )

    @property
    def top_similarity(self) -> MetricBuilder:
        """Similarity score of top result."""
        return self._define(
            "top_similarity",
            system_metric=SystemMetric.ACCURACY,
            description="Top result similarity"
        )

search = SearchMetrics()

# Transforms
def calculate_mrr(inputs: dict, output: dict) -> float:
    """Calculate Mean Reciprocal Rank."""
    results = output.get("results", [])
    expected_id = inputs.get("expected_id")

    if not results or not expected_id:
        return 0.0

    for rank, result in enumerate(results, 1):
        if result.get("id") == expected_id:
            return 1.0 / rank

    return 0.0

@track(metrics=[
    search.hit_rate.compute(lambda i, o: len(o.get("results", [])) > 0),
    search.mrr.compute(calculate_mrr),
    search.top_similarity.from_output("results.0.similarity"),
])
def vector_search(query: str, expected_id: str) -> dict:
    return {"results": [...]}
```

---

## Quick Reference

### Builder Pattern Cheat Sheet

```python
# Extract from path
m.custom("name").from_output("path.to.field")

# Compute with lambda
m.custom("name").compute(lambda i, o: calculation)

# Compute with function
m.custom("name").compute(my_transform_function)

# Compare to expected
m.expected_match.compare_to_expected("expected_key", "output_key")

# System metric (auto-aggregation)
m.custom("accuracy", system_metric=SystemMetric.ACCURACY)

# Custom aggregation
m.custom("count", aggregation=AggregationType.SUM)
```

### Namespace Pattern Cheat Sheet

```python
class MyMetrics(MetricNamespace):
    @property
    def my_metric(self) -> MetricBuilder:
        return self._define(
            "metric_name",
            system_metric=SystemMetric.ACCURACY,  # Optional
            aggregation=AggregationType.AVERAGE,   # If no system_metric
            description="What this measures"
        )

    def configurable_metric(self, param: str) -> Metric:
        def transform(inputs: dict, output: dict) -> float:
            # Use param in calculation
            return value

        return self._define(
            f"metric_{param}",
            aggregation=AggregationType.AVERAGE,
        ).compute(transform)

metrics = MyMetrics()
```

### Transform Cheat Sheet

```python
def my_transform(inputs: dict, output: dict) -> float:
    # 1. Extract from inputs (test expectations)
    expected = inputs.get("key", default)

    # 2. Extract from output (function result)
    actual = output.get("key", default)

    # 3. Handle edge cases
    if not actual:
        return 0.0  # Never return None!

    # 4. Compute metric
    value = calculate(expected, actual)

    # 5. Return typed value
    return float(value)  # or int, bool
```

---

## Next Steps

- See [metrics.md](metrics.md) for built-in metrics reference
- See [instrumentation.md](instrumentation.md) for `@track` decorator details
- See [reporting.md](reporting.md) for metrics reports and analysis
- See `squirt/contrib/echo` and `squirt/contrib/tokens` for real-world namespace examples
