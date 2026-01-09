"""
Echo Metrics - Input/Output Logging and Comparison

Provides debugging utilities to capture and display:
- Input from expectations
- Expected output from expectations  
- Actual output from decorated functions

Useful for understanding data flow and debugging test iterations.
"""

from typing import Optional

from squirt.plugins.base import MetricNamespace
from squirt.core.types import AggregationType, Metric


class EchoMetrics(MetricNamespace):
    """
    Echo namespace for capturing and displaying inputs/outputs.

    Provides two simple methods:
    - save: Persist input, expected, and actual to markdown file
    - print: Log input, expected, and actual to console (survives pytest suppression)

    Both methods capture:
    1. Input from expectations.json
    2. Expected output from expectations.json
    3. Actual output from the decorated function

    Examples:
        from squirt.categories.echo import echo

        # Print all values to console during test runs
        @track(metrics=[
            echo.print(
                input_field="description",
                expected_field="bullets",
                actual_field="bullets"
            )
        ])
        def my_component(text: str) -> dict:
            ...

        # Save all values to markdown file for later analysis
        @track(metrics=[
            echo.save(
                input_field="description",
                expected_field="bullets",
                actual_field="bullets"
            )
        ])
        def my_component(text: str) -> dict:
            ...
    """

    def save(
        self,
        input_field: str,
        expected_field: str,
        actual_field: Optional[str] = None,
    ) -> Metric:
        """
        Save input, expected, and actual values to a markdown file.
        Automatically names file as {component_name}_echo.md in results directory.

        Args:
            input_field: Path into expectations.json for input (e.g., "description")
            expected_field: Path into expectations.json for expected output (e.g., "bullets")
            actual_field: Path into function output for actual value (default: root)
        """
        import json
        from pathlib import Path

        def get_nested_value(obj: dict, path: str):
            """Get value from nested dict using dot notation."""
            if not path:
                return obj
            keys = path.split(".")
            value = obj
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            return value

        def format_value(value, label: str) -> str:
            """Format a value as markdown with proper code blocks."""
            if isinstance(value, dict) or isinstance(value, list):
                # JSON payload - format as ```json
                return f"### {label}\n\n```json\n{json.dumps(value, indent=2, default=str)}\n```\n"
            elif isinstance(value, str):
                # Plain text - replace escaped newlines with actual newlines
                clean_text = value.replace("\\n", "\n")
                return f"### {label}\n\n```\n{clean_text}\n```\n"
            else:
                # Other types - convert to string
                return f"### {label}\n\n```\n{str(value)}\n```\n"

        def save_transform(inputs: dict, output: dict) -> float:
            """Save input, expected, and actual to markdown file."""
            from squirt.pytest import _squirt_config
            from squirt.core.decorator import _component_stack

            input_val = get_nested_value(inputs, input_field)
            expected_val = get_nested_value(inputs, expected_field)
            actual_val = (
                get_nested_value(output, actual_field) if actual_field else output
            )

            # Format as markdown
            markdown = "\n" + "=" * 80 + "\n"
            markdown += "## Echo Comparison\n\n"
            markdown += format_value(input_val, f"📥 INPUT ({input_field})")
            markdown += "\n"
            markdown += format_value(expected_val, f"✅ EXPECTED ({expected_field})")
            markdown += "\n"
            markdown += format_value(
                actual_val, f"🎯 ACTUAL ({actual_field or 'output'})"
            )
            markdown += "\n" + "=" * 80 + "\n\n"

            # Get results directory from squirt config, default to current directory
            results_dir_str = _squirt_config.get("results_dir", ".")
            results_dir = Path(results_dir_str) if results_dir_str else Path(".")

            # Auto-generate filepath from component name
            stack = _component_stack.get()
            if stack:
                component_name = stack[-1]  # Current component is last in stack
                filepath = f"{component_name}_echo.md"
            else:
                filepath = "echo_data.md"  # Fallback if no component context

            full_path = results_dir / filepath

            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "a") as f:
                f.write(markdown)

            return 1.0

        return self._define(
            f"save_{input_field.replace('.', '_')}_{expected_field.replace('.', '_')}",
            AggregationType.COUNT,
            description=f"Save {input_field} → {expected_field} comparison to {{component}}_echo.md",
        ).compute(save_transform)

    def print(
        self, input_field: str, expected_field: str, actual_field: Optional[str] = None
    ) -> Metric:
        """
        Print input, expected, and actual values to console.
        Uses stderr with flush to bypass pytest output capture.

        Args:
            input_field: Path into expectations.json for input (e.g., "description")
            expected_field: Path into expectations.json for expected output (e.g., "bullets")
            actual_field: Path into function output for actual value (default: root)
        """
        import sys

        def get_nested_value(obj: dict, path: str):
            """Get value from nested dict using dot notation."""
            if not path:
                return obj
            keys = path.split(".")
            value = obj
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            return value

        def format_value(value) -> str:
            """Format a value for pretty printing."""
            import json

            if isinstance(value, dict) or isinstance(value, list):
                # JSON payload - format with json.dumps
                return f"```json\n{json.dumps(value, indent=2, default=str)}\n```"
            elif isinstance(value, str):
                # Plain text - replace escaped newlines with actual newlines
                clean_text = value.replace("\\n", "\n")
                return f"```\n{clean_text}\n```"
            else:
                # Other types - convert to string
                return f"```\n{str(value)}\n```"

        def print_transform(inputs: dict, output: dict) -> float:
            """Print input, expected, and actual comparison."""
            import json

            input_val = get_nested_value(inputs, input_field)
            expected_val = get_nested_value(inputs, expected_field)
            actual_val = (
                get_nested_value(output, actual_field) if actual_field else output
            )

            # Use stderr with flush to bypass pytest capture
            print(f"\n{'='*80}", file=sys.stderr, flush=True)
            print("ECHO COMPARISON", file=sys.stderr, flush=True)
            print("=" * 80, file=sys.stderr, flush=True)

            print(f"\n📥 INPUT ({input_field}):", file=sys.stderr, flush=True)
            print("-" * 80, file=sys.stderr, flush=True)
            print(format_value(input_val), file=sys.stderr, flush=True)

            print(f"\n✅ EXPECTED ({expected_field}):", file=sys.stderr, flush=True)
            print("-" * 80, file=sys.stderr, flush=True)
            print(format_value(expected_val), file=sys.stderr, flush=True)

            print(
                f"\n🎯 ACTUAL ({actual_field or 'output'}):",
                file=sys.stderr,
                flush=True,
            )
            print("-" * 80, file=sys.stderr, flush=True)
            print(format_value(actual_val), file=sys.stderr, flush=True)

            print("=" * 80 + "\n", file=sys.stderr, flush=True)
            return 1.0

        return self._define(
            f"print_{input_field.replace('.', '_')}_{expected_field.replace('.', '_')}",
            AggregationType.COUNT,
            description=f"Print {input_field} → {expected_field} comparison",
        ).compute(print_transform)


# ============================================================================
# Singleton Instance
# ============================================================================

echo = EchoMetrics()
"""
Echo metrics namespace for input/output logging and storage.

Use this for debugging and understanding data flow:
- print: Log input, expected, and actual to console (bypasses pytest capture)
- save: Persist input, expected, and actual to markdown file for analysis

Examples:
    from squirt.categories.echo import echo

    # Print to console during test runs
    @track(metrics=[
        echo.print(
            input_field="description",
            expected_field="bullets",
            actual_field="bullets"
        )
    ])
    def my_component(text: str) -> dict:
        ...
    
    # Save to file for later analysis (auto-named as {component}_echo.md)
    @track(metrics=[
        echo.save(
            input_field="description",
            expected_field="bullets",
            actual_field="bullets"
        )
    ])
    def my_component(text: str) -> dict:
        ...
"""

__all__ = ["EchoMetrics", "echo"]
