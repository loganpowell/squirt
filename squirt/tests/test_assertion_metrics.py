"""
Tests for assertion metrics that fail tests when metrics indicate errors.
"""

import pytest
from sleuth import m, track, configure_metrics, set_test_context
from sleuth.core.decorator import clear_results


class TestAssertionMetrics:
    """Test assertion metrics that block failing PRs."""

    def test_assert_passes_succeeds_when_above_threshold(self, tmp_path):
        """Assertion metric should pass when value > threshold."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="pass_test")
        clear_results()

        @track(
            metrics=[
                m.assert_passes.assert_passes(
                    lambda i, o: 1.0,  # Returns 1.0 (> 0.0 threshold)
                    threshold=0.0,
                ),
            ]
        )
        def passing_component():
            return {"result": "success"}

        # Should not raise
        result = passing_component()
        assert result["result"] == "success"

    def test_assert_passes_fails_when_at_threshold(self, tmp_path):
        """Assertion metric should fail when value == threshold."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="fail_test")
        clear_results()

        @track(
            metrics=[
                m.assert_passes.assert_passes(
                    lambda i, o: 0.0,  # Returns 0.0 (== 0.0 threshold)
                    threshold=0.0,
                ),
            ]
        )
        def failing_component():
            return {"result": "failure"}

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="Assertion metric.*failed"):
            failing_component()

    def test_assert_passes_fails_when_below_threshold(self, tmp_path):
        """Assertion metric should fail when value < threshold."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="fail_test")
        clear_results()

        @track(
            metrics=[
                m.assert_passes.assert_passes(
                    lambda i, o: 0.5,  # Returns 0.5 (< 0.8 threshold)
                    threshold=0.8,
                ),
            ]
        )
        def low_quality_component():
            return {"result": "poor"}

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="Assertion metric.*failed"):
            low_quality_component()

    def test_assert_passes_with_custom_metric(self, tmp_path):
        """Test assertion with custom validation logic."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="custom_test")
        clear_results()

        def validate_structure(inputs, output):
            """Returns 1.0 if valid, 0.0 if invalid."""
            return 1.0 if output.get("required_field") else 0.0

        @track(
            metrics=[
                m.custom("structure_validation").assert_passes(
                    validate_structure,
                    threshold=0.0,
                ),
            ]
        )
        def validated_component():
            return {"required_field": None}  # Invalid!

        # Should raise because required_field is None
        with pytest.raises(AssertionError, match="structure_validation"):
            validated_component()

    def test_multiple_assertions_show_all_failures(self, tmp_path):
        """When multiple assertions fail, should show all failures."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="multi_fail_test")
        clear_results()

        @track(
            metrics=[
                m.custom("check_a").assert_passes(lambda i, o: 0.0, threshold=0.0),
                m.custom("check_b").assert_passes(lambda i, o: 0.0, threshold=0.0),
                m.custom("check_c").assert_passes(lambda i, o: 0.0, threshold=0.0),
            ]
        )
        def multi_fail_component():
            return {"result": "bad"}

        # Should show all three failures
        with pytest.raises(AssertionError) as exc_info:
            multi_fail_component()

        error_msg = str(exc_info.value)
        assert "check_a" in error_msg
        assert "check_b" in error_msg
        assert "check_c" in error_msg

    def test_assertion_failure_still_records_metrics(self, tmp_path):
        """Metrics should be recorded even when assertion fails."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="record_test")
        clear_results()

        @track(
            metrics=[
                m.runtime_ms.from_output("metadata.runtime_ms"),
                m.assert_passes.assert_passes(lambda i, o: 0.0, threshold=0.0),
            ]
        )
        def component_that_fails():
            return {"result": "fail", "metadata": {"runtime_ms": 100}}

        # Execute and expect failure
        with pytest.raises(AssertionError):
            component_that_fails()

        # But metrics should still be recorded
        from sleuth.core.decorator import get_results

        results = get_results()
        assert len(results) == 1
        assert results[0].metrics["runtime_ms"] == 100
        assert results[0].metrics["assert_passes"] == 0.0

    def test_non_assertion_metrics_dont_fail_tests(self, tmp_path):
        """Regular metrics should not fail tests even when they return 0."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="non_assert_test")
        clear_results()

        @track(
            metrics=[
                # Regular metric - should not fail test
                m.custom("quality").compute(lambda i, o: 0.0),
            ]
        )
        def low_quality_but_ok():
            return {"result": "acceptable"}

        # Should not raise even though quality=0.0
        result = low_quality_but_ok()
        assert result["result"] == "acceptable"

    def test_assertion_with_exception_in_transform(self, tmp_path):
        """When assertion transform raises exception, test should fail."""
        configure_metrics(results_dir=str(tmp_path))
        set_test_context(test_case_id="exception_test")
        clear_results()

        def buggy_validator(inputs, output):
            raise ValueError("Validation crashed!")

        @track(
            metrics=[
                m.custom("buggy_check").assert_passes(
                    buggy_validator,
                    threshold=0.0,
                ),
            ]
        )
        def component_with_buggy_validator():
            return {"result": "ok"}

        # Should raise AssertionError showing the exception
        with pytest.raises(AssertionError, match="buggy_check.*Validation crashed"):
            component_with_buggy_validator()
