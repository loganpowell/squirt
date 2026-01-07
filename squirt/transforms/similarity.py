"""
Similarity Transform Functions

Compare outputs to expected values with similarity scoring.
"""

from collections.abc import Callable
from typing import Any


def compute_similarity(expected: Any, actual: Any, depth: int = 0) -> float:
    """
    Compute similarity between expected and actual values.

    Handles multiple data types:
    - Strings: Token overlap ratio
    - Dicts: Recursive field comparison
    - Lists: Length + element matching
    - Numbers: Percentage difference

    Returns:
        Score from 0.0 (no match) to 1.0 (perfect match)
    """
    if expected == actual:
        return 1.0

    if expected is None or actual is None:
        return 0.0 if expected != actual else 1.0

    # String comparison
    if isinstance(expected, str) and isinstance(actual, str):
        expected_norm = expected.lower().strip()
        actual_norm = actual.lower().strip()
        if expected_norm == actual_norm:
            return 1.0
        # Simple token overlap
        expected_tokens = set(expected_norm.split())
        actual_tokens = set(actual_norm.split())
        if not expected_tokens:
            return 0.0
        overlap = len(expected_tokens & actual_tokens)
        return overlap / len(expected_tokens)

    # Dict comparison
    if isinstance(expected, dict) and isinstance(actual, dict):
        if not expected:
            return 1.0 if not actual else 0.5

        scores = []
        for key in expected:
            if key in actual:
                scores.append(compute_similarity(expected[key], actual[key], depth + 1))
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    # List comparison
    if isinstance(expected, list) and isinstance(actual, list):
        if not expected:
            return 1.0 if not actual else 0.5

        # Simple length-based + element matching
        len_score = min(len(actual), len(expected)) / max(len(actual), len(expected))

        if depth < 2:  # Limit recursion depth
            element_scores = []
            for i, exp_item in enumerate(expected):
                if i < len(actual):
                    element_scores.append(
                        compute_similarity(exp_item, actual[i], depth + 1)
                    )
                else:
                    element_scores.append(0.0)

            if element_scores:
                return (len_score + sum(element_scores) / len(element_scores)) / 2

        return len_score

    # Numeric comparison
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == 0:
            return 1.0 if actual == 0 else 0.0
        diff_pct = abs(expected - actual) / abs(expected)
        return max(0.0, 1.0 - diff_pct)

    return 0.0


def expected_match_transform(inputs: dict[str, Any], output: Any) -> float:
    """
    Compare output to expected data from inputs.

    Expects inputs["expected"] to contain the expected value.

    Returns:
        Similarity score 0.0 to 1.0
    """
    expected = inputs.get("expected")
    if expected is None:
        return 1.0  # No expected data, assume pass

    return compute_similarity(expected, output)


def create_expected_match_transform(
    expected_key: str = "expected",
    output_key: str | None = None,
    exact: bool = False,
) -> Callable[[dict[str, Any], Any], float]:
    """
    Factory to create an expected match transform.

    Args:
        expected_key: Key in inputs containing expected data
        output_key: Optional key to extract from output for comparison
        exact: If True, requires exact match (returns 0 or 1)

    Returns:
        Transform returning similarity score (0.0 to 1.0)

    Example:
        match_bullets = create_expected_match_transform(
            expected_key="expected_bullets",
            output_key="bullets"
        )

        @track(metrics=[
            Metric("accuracy", match_bullets, AggregationType.AVERAGE)
        ])
        def extract_bullets(text: str) -> dict:
            ...
    """

    def transform(inputs: dict[str, Any], output: Any) -> float:
        expected = inputs.get(expected_key)
        if expected is None:
            return 1.0  # No expected data, assume pass

        # Extract specific key from output if specified
        actual = output
        if output_key and isinstance(output, dict):
            actual = output.get(output_key, output)

        if exact:
            return 1.0 if expected == actual else 0.0

        # Compute similarity score
        return compute_similarity(expected, actual)

    return transform


__all__ = [
    "compute_similarity",
    "expected_match_transform",
    "create_expected_match_transform",
]
