"""
Integration test: Verify assertion metrics work in real instrumented components.

This tests that error_free and other assertion metrics properly fail tests
when they detect errors in actual pipeline components.
"""


import pytest

from squirt import m, set_test_context, track
from squirt.core.decorator import clear_results, get_results


class TestAssertionMetricsIntegration:
    """Integration tests for assertion metrics in real components."""

    def test_error_free_assertion_prevents_bad_pr(self, tmp_path):
        """
        Demonstrate how error_free prevents bad PRs from merging.

        When error_free detects an error (returns 0.0), it should:
        1. Record the metrics (for debugging)
        2. Raise AssertionError (fail the test)
        3. Block the PR in CI
        """

        def component_with_error(text: str) -> dict:
            """Simulates a component that has an error."""
            # Simulate a validation error
            return {
                "output": None,  # Missing required output
                "metadata": {"runtime_ms": 100},
            }

        # Add error_free metric which will automatically fail on error
        tracked_component = track(
            expects="description",
            metrics=[
                m.error_free.compute(
                    lambda i, o: 0.0 if o.get("output") is None else 1.0
                ),
            ],
        )(component_with_error)

        set_test_context(test_case_id="bad_pr_test", expectations={})
        clear_results()

        # This should raise AssertionError because error_free returns 0.0
        with pytest.raises(AssertionError) as exc_info:
            tracked_component("test input")

        # Verify the error message is helpful
        error_msg = str(exc_info.value)
        assert "error_free" in error_msg
        assert "0.0" in error_msg

        # Verify metrics were still recorded (for post-mortem analysis)
        results = get_results()
        assert len(results) == 1
        assert results[0].metrics["error_free"] == 0.0

    def test_multiple_assertion_metrics_all_checked(self, tmp_path):
        """
        Test that multiple assertion metrics are all checked.

        If any assertion metric fails, the test should fail with
        all failures listed in the error message.
        """

        @track(
            expects="text",
            metrics=[
                # Three different checks - all will fail
                m.custom("check_a").assert_passes(
                    lambda i, o: 0.0, threshold=0.0  # Fails
                ),
                m.custom("check_b").assert_passes(
                    lambda i, o: 0.5, threshold=0.8  # Fails (0.5 <= 0.8)
                ),
                m.custom("check_c").assert_passes(
                    lambda i, o: 1.0, threshold=0.0  # Passes
                ),
            ],
        )
        def multi_check_component(text: str) -> dict:
            return {"result": "test"}

        set_test_context(test_case_id="multi_check", expectations={})
        clear_results()

        # Should fail with both check_a and check_b in error message
        with pytest.raises(AssertionError) as exc_info:
            multi_check_component("test")

        error_msg = str(exc_info.value)
        assert "check_a" in error_msg
        assert "check_b" in error_msg
        # check_c passed, so shouldn't be listed as a failure
        assert "check_c=1.0" not in error_msg

    def test_regular_metrics_dont_block_when_low(self, tmp_path):
        """
        Verify that regular (non-assertion) metrics don't fail tests.

        This is important: we want to collect quality metrics without
        blocking PRs unless we explicitly use assertion mode.
        """

        @track(
            expects="text",
            metrics=[
                # Regular metric - just monitors, doesn't block
                m.custom("quality_score").compute(lambda i, o: 0.1),
                # Runtime is also non-blocking
                m.runtime_ms.from_output("metadata.runtime_ms"),
            ],
        )
        def low_quality_but_acceptable(text: str) -> dict:
            return {"result": "ok", "metadata": {"runtime_ms": 1000}}

        set_test_context(test_case_id="low_quality", expectations={})
        clear_results()

        # Should NOT raise even though quality_score is 0.1
        result = low_quality_but_acceptable("test")
        assert result["result"] == "ok"

        # But metrics should still be collected
        results = get_results()
        assert len(results) == 1
        assert results[0].metrics["quality_score"] == 0.1
        assert results[0].metrics["runtime_ms"] == 1000

    def test_assertion_and_regular_metrics_together(self, tmp_path):
        """
        Test that assertion and regular metrics work together correctly.

        Assertion metrics should fail tests while regular metrics just collect data.
        """

        @track(
            expects="text",
            metrics=[
                # Regular metrics - just collect
                m.runtime_ms.from_output("metadata.runtime_ms"),
                m.custom("quality").compute(lambda i, o: 0.5),
                # Assertion metric - blocks on failure
                m.error_free.compute(lambda i, o: 1.0),  # Passes
            ],
        )
        def mixed_metrics_component(text: str) -> dict:
            return {"result": "ok", "metadata": {"runtime_ms": 100}}

        set_test_context(test_case_id="mixed_test", expectations={})
        clear_results()

        # Should pass because error_free returns 1.0
        result = mixed_metrics_component("test")
        assert result["result"] == "ok"

        # All metrics should be collected
        results = get_results()
        assert len(results) == 1
        assert results[0].metrics["runtime_ms"] == 100
        assert results[0].metrics["quality"] == 0.5
        assert results[0].metrics["error_free"] == 1.0
