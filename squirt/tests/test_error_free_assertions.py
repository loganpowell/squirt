"""
Test that error_free automatically fails tests when errors occur.
"""

import pytest
from sleuth import m, track, configure_metrics, set_test_context
from sleuth.core.decorator import clear_results


def test_error_free_fails_automatically_on_error(tmp_path):
    """error_free should automatically fail test when it returns 0."""
    configure_metrics(results_dir=str(tmp_path))
    set_test_context(test_case_id="error_test")
    clear_results()

    @track(
        metrics=[
            # Just use error_free.compute() - no need for assert_passes!
            m.error_free.compute(lambda i, o: 0.0),  # Simulates an error
        ]
    )
    def component_with_error():
        return {"result": "bad"}

    # Should automatically raise AssertionError
    with pytest.raises(AssertionError, match="error_free.*0.0"):
        component_with_error()


def test_error_free_passes_when_no_errors(tmp_path):
    """error_free should pass when it returns 1.0."""
    configure_metrics(results_dir=str(tmp_path))
    set_test_context(test_case_id="no_error_test")
    clear_results()

    @track(
        metrics=[
            m.error_free.compute(lambda i, o: 1.0),  # No errors
        ]
    )
    def component_without_error():
        return {"result": "good"}

    # Should not raise
    result = component_without_error()
    assert result["result"] == "good"


def test_error_free_from_output_also_works(tmp_path):
    """error_free.from_output() should also have assertion behavior."""
    configure_metrics(results_dir=str(tmp_path))
    set_test_context(test_case_id="from_output_test")
    clear_results()

    @track(
        metrics=[
            m.error_free.from_output("metadata.error_free"),
        ]
    )
    def component_with_error_in_metadata():
        return {"result": "ok", "metadata": {"error_free": 0.0}}

    # Should fail because error_free in output is 0.0
    with pytest.raises(AssertionError, match="error_free"):
        component_with_error_in_metadata()


def test_regular_metrics_not_affected(tmp_path):
    """Other metrics should still work normally without assertions."""
    configure_metrics(results_dir=str(tmp_path))
    set_test_context(test_case_id="regular_test")
    clear_results()

    @track(
        metrics=[
            m.runtime_ms.from_output("metadata.runtime_ms"),
            m.custom("quality").compute(lambda i, o: 0.0),  # Low quality, but OK
        ]
    )
    def component_with_low_quality():
        return {"result": "acceptable", "metadata": {"runtime_ms": 100}}

    # Should not raise - only error_free has automatic assertion behavior
    result = component_with_low_quality()
    assert result["result"] == "acceptable"
