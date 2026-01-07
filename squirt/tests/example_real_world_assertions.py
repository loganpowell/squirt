"""
Real-world example: Adding assertion metrics to pipeline components

This shows how to add assert_passes() to existing instrumented components
to block PRs when quality falls below thresholds.
"""

from squirt import m, track

# BEFORE: Component with regular metrics only
# ============================================


@track(
    expects="description",
    metrics=[
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.expected_match.compare_to_expected("bullets", "bullets"),
        m.structure_valid.compute(lambda i, o: 1.0 if o.get("output") else 0.0),
    ],
)
def extract_text_v1(text: str) -> dict:
    """This collects metrics but doesn't fail tests on errors."""
    return {"output": process(text), "metadata": {"runtime_ms": 100}}


# AFTER: Component with assertion metrics
# ========================================


def validate_structure(inputs, output):
    """Ensure output has required structure."""
    output_data = output.get("output")
    if not output_data:
        return 0.0
    if not isinstance(output_data, list):
        return 0.0
    if len(output_data) < 3:  # Must have at least 3 bullet points
        return 0.0
    return 1.0


def validate_quality(inputs, output):
    """Ensure output quality meets threshold."""
    expected = inputs.get("bullets", "")
    actual = output.get("output", "")
    # Simple similarity check
    similarity = calculate_similarity(expected, actual)
    return similarity


@track(
    expects="description",
    metrics=[
        # Regular metrics - collected but don't block
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.expected_match.compare_to_expected("bullets", "bullets"),
        # ASSERTION METRICS - BLOCK PR IF FAIL
        # ------------------------------------
        # Block if structure invalid
        m.structure_valid.assert_passes(
            validate_structure,
            threshold=0.0,
        ),
        # Block if quality < 70%
        m.custom("quality_score").assert_passes(
            validate_quality,
            threshold=0.7,
        ),
        # Block if processing took too long
        m.custom("latency_check").assert_passes(
            lambda i, o: (
                1.0 if o.get("metadata", {}).get("runtime_ms", 0) < 5000 else 0.0
            ),
            threshold=0.0,
        ),
    ],
)
def extract_text_v2(text: str) -> dict:
    """
    This component has quality gates that block PRs.

    Test will FAIL if:
    - Output structure is invalid
    - Quality score < 70%
    - Runtime >= 5000ms
    """
    return {"output": process(text), "metadata": {"runtime_ms": 100}}


# CI/CD Integration Example
# ==========================


def test_extract_text_blocks_bad_quality():
    """
    This test will fail in CI if quality drops below 70%.

    When the test fails, you'll see:

        AssertionError: Assertion metric(s) failed in extract_text_v2:
          quality_score=0.65 (threshold: >0.7)

    This prevents the PR from merging until quality is fixed.
    """
    from squirt import set_test_context

    set_test_context(
        test_case_id="tc_001",
        expectations={
            "description": "A person under 65 gets full pension",
            "bullets": "• Condition: Age < 65\\n• Action: Full pension\\n• Amount: 100%",
        },
    )

    # This will raise AssertionError if quality < 70%
    result = extract_text_v2("A person under 65 gets full pension")

    # If we reach here, all quality gates passed!
    assert result["output"]


# Gradual Rollout Strategy
# =========================


def test_with_warning_only():
    """
    Start by collecting metrics without blocking, then gradually add assertions.

    Phase 1: Just collect metrics
    Phase 2: Add assertions with low thresholds (catch obvious failures)
    Phase 3: Increase thresholds as quality improves
    """

    # Phase 1: No assertions - just monitoring
    @track(
        expects="data",
        metrics=[
            m.custom("accuracy").compute(accuracy_fn),
        ],
    )
    def phase1_component(data):
        return process(data)

    # Phase 2: Add assertion with low threshold (catch 0% accuracy)
    @track(
        expects="data",
        metrics=[
            m.custom("accuracy").assert_passes(accuracy_fn, threshold=0.0),
        ],
    )
    def phase2_component(data):
        return process(data)

    # Phase 3: Raise threshold as quality improves
    @track(
        expects="data",
        metrics=[
            m.custom("accuracy").assert_passes(accuracy_fn, threshold=0.8),
        ],
    )
    def phase3_component(data):
        return process(data)


# Helper functions
def process(text):
    """Dummy processing."""
    return ["bullet1", "bullet2", "bullet3"]


def calculate_similarity(expected, actual):
    """Dummy similarity calculation."""
    return 0.85


def accuracy_fn(inputs, output):
    """Dummy accuracy calculation."""
    return 0.9
