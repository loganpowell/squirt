"""
Example: Using Assertion Metrics to Block Failing PRs

This example shows how to use m.assert_passes() to fail tests when
critical metrics indicate errors, preventing bad code from being merged.
"""

from squirt import m, track

# Example 1: Structure validation that blocks PRs
# ================================================


def validate_tax_rule_structure(inputs, output):
    """
    Validate that tax rule has all required fields.
    Returns 1.0 if valid, 0.0 if invalid.
    """
    rule = output.get("taxAssistRule", {})
    required_fields = ["name", "conditions", "conclusion"]

    has_all_fields = all(field in rule for field in required_fields)
    return 1.0 if has_all_fields else 0.0


@track(
    expects="description",
    metrics=[
        # This will FAIL THE TEST if structure is invalid
        m.custom("structure_validation").assert_passes(
            validate_tax_rule_structure,
            threshold=0.0,  # Any value <= 0.0 fails
        ),
        # Regular metrics continue to be collected
        m.runtime_ms.from_output("metadata.runtime_ms"),
    ],
)
def extract_tax_rule(text: str) -> dict:
    """Extract tax rule - test fails if structure is invalid."""
    # ... processing logic ...
    return {
        "taxAssistRule": {
            "name": "Example Rule",
            "conditions": [],
            "conclusion": {},
        },
        "metadata": {"runtime_ms": 150},
    }


# Example 2: Quality threshold that blocks PRs
# =============================================


def calculate_field_accuracy(inputs, output):
    """Calculate what % of fields were mapped correctly."""
    expected = inputs.get("expected_mappings", {})
    actual = output.get("field_mappings", {})

    if not expected:
        return 1.0

    correct = sum(1 for k, v in expected.items() if actual.get(k) == v)
    return correct / len(expected)


@track(
    expects="field_data",
    metrics=[
        # Block PR if field accuracy < 80%
        m.custom("field_accuracy").assert_passes(
            calculate_field_accuracy,
            threshold=0.8,  # Must be > 0.8 to pass
        ),
    ],
)
def enrich_fields(data: dict) -> dict:
    """Enrich fields - test fails if accuracy < 80%."""
    # ... enrichment logic ...
    return {
        "field_mappings": {
            "customer": "Person",
            "amount": "Currency",
        }
    }


# Example 3: Multiple assertions - all must pass
# ===============================================


def check_no_errors(inputs, output):
    """Returns 1.0 if no errors, 0.0 if errors present."""
    return 0.0 if output.get("errors") else 1.0


def check_has_data(inputs, output):
    """Returns 1.0 if has data, 0.0 if empty."""
    return 1.0 if output.get("data") else 0.0


def check_latency_acceptable(inputs, output):
    """Returns 1.0 if latency < 1000ms, 0.0 otherwise."""
    latency = output.get("metadata", {}).get("runtime_ms", 0)
    return 1.0 if latency < 1000 else 0.0


@track(
    expects="query",
    metrics=[
        # All three must pass - any failure blocks the PR
        m.custom("no_errors").assert_passes(check_no_errors),
        m.custom("has_data").assert_passes(check_has_data),
        m.custom("latency_ok").assert_passes(check_latency_acceptable),
    ],
)
def search_database(query: str) -> dict:
    """Search database - multiple quality gates."""
    # ... search logic ...
    return {
        "data": ["result1", "result2"],
        "errors": None,
        "metadata": {"runtime_ms": 250},
    }


# Example 4: Using with existing metrics
# =======================================


@track(
    expects="description",
    metrics=[
        # Assertion: fail if structure invalid
        m.structure_valid.assert_passes(
            lambda i, o: 1.0 if validate_structure(o) else 0.0,
        ),
        # Regular metrics: collected but don't fail test
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.memory_mb.from_output("metadata.memory_mb"),
        m.expected_match.compare_to_expected("result", "expected_result"),
    ],
)
def process_input(text: str) -> dict:
    """Process input with both assertions and regular metrics."""
    return {
        "result": process(text),
        "metadata": {"runtime_ms": 100, "memory_mb": 50},
    }


def validate_structure(output: dict) -> bool:
    """Validate output structure."""
    return "result" in output and isinstance(output["result"], dict)


def process(text: str) -> dict:
    """Dummy processing function."""
    return {"processed": text}


# How it works in CI/CD
# =====================
"""
When you run pytest in CI:

1. Tests with assertion metrics will FAIL if:
   - Metric value <= threshold
   - Transform raises exception
   
2. This blocks the PR from merging

3. Regular metrics continue to be collected for monitoring,
   but don't block the PR

4. You get detailed error messages:
   
   AssertionError: Assertion metric(s) failed in extract_tax_rule:
     structure_validation=0.0 (threshold: >0.0)
     
5. Metrics are still recorded even when assertions fail,
   so you can analyze what went wrong
"""


# Usage in pytest
# ===============
"""
# tests/test_pipeline.py

from squirt import configure_expectations, set_test_context

def test_extract_tax_rule():
    # Setup
    configure_expectations(path="tests/data/expectations.json")
    set_test_context(
        test_case_id="tc_001",
        expectations={
            "description": "A person under 65 gets full pension",
            "expected_result": {...}
        }
    )
    
    # This will fail the test if structure_validation <= 0.0
    result = extract_tax_rule("A person under 65 gets full pension")
    
    # If we reach here, all assertions passed!
    assert result["taxAssistRule"]["name"]
"""
