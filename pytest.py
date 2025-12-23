"""
Sleuth Pytest Plugin

Zero-config metrics collection for pytest. Just add to conftest.py:

    pytest_plugins = ["sleuth.pytest"]

Or with custom paths:

    from sleuth.pytest import configure_sleuth

    configure_sleuth(
        results_dir="tests/results",
        default_source="tests/data/expectations.json",
    )

The plugin automatically:
- Configures sleuth directories
- Loads default source from standard locations (overridden by @track(source=...))
- Tracks component execution
- Generates heartbeat reports on session finish
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pytest

from .core.types import MetricResult

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


# =============================================================================
# Module-level Configuration
# =============================================================================

_sleuth_config: Dict[str, Any] = {
    "results_dir": None,
    "history_dir": None,
    "default_source": None,
    "instrumented_dir": None,
    "auto_heartbeat": True,
    "verbose": True,
}

# Dependency graph built at session start for parent-child detection
_dependency_graph: Optional[Any] = None


def get_dependency_graph() -> Optional[Any]:
    """Get the dependency graph built at session start."""
    return _dependency_graph


def configure_sleuth(
    results_dir: Optional[Union[str, Path]] = None,
    history_dir: Optional[Union[str, Path]] = None,
    default_source: Optional[Union[str, Path]] = None,
    instrumented_dir: Optional[Union[str, Path]] = None,
    auto_heartbeat: bool = True,
    verbose: bool = True,
) -> None:
    """
    Configure sleuth for your test suite. Call this in conftest.py.

    All paths are relative to the tests directory by default.

    Args:
        results_dir: Where to save metric results (default: tests/results)
        history_dir: Where to save historical reports (default: tests/history)
        default_source: Default source file for test data (default: tests/data/expectations.json)
                       Components can override this with their own source parameter in @track()
        instrumented_dir: Path to instrumented components (default: tests/instrumented)
        auto_heartbeat: Generate heartbeat report on session finish (default: True)
        verbose: Print configuration info (default: True)

    Example:
        # conftest.py
        from sleuth.pytest import configure_sleuth

        configure_sleuth(
            default_source="tests/data/expectations.json",
        )

        pytest_plugins = ["sleuth.pytest"]
    """
    global _sleuth_config

    if results_dir:
        _sleuth_config["results_dir"] = str(results_dir)
    if history_dir:
        _sleuth_config["history_dir"] = str(history_dir)
    if default_source:
        _sleuth_config["default_source"] = str(default_source)
    if instrumented_dir:
        _sleuth_config["instrumented_dir"] = str(instrumented_dir)

    _sleuth_config["auto_heartbeat"] = auto_heartbeat
    _sleuth_config["verbose"] = verbose


# =============================================================================
# Session State
# =============================================================================

_session_results: List[MetricResult] = []
_session_start_time: float = 0.0
_executed_components: set = set()


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_addoption(parser: "Parser") -> None:
    """Add sleuth command-line options."""
    group = parser.getgroup("sleuth", "Sleuth metrics collection")
    group.addoption(
        "--collect-metrics",
        action="store_true",
        default=True,
        help="Enable metrics collection (default: True)",
    )
    group.addoption(
        "--no-metrics",
        action="store_true",
        default=False,
        help="Disable metrics collection",
    )
    group.addoption(
        "--metrics-dir",
        action="store",
        default=None,
        help="Directory for metrics output",
    )
    group.addoption(
        "--sleuth-expectations",
        action="store",
        default=None,
        help="Path to expectations.json file",
    )


def pytest_configure(config: "Config") -> None:
    """Configure sleuth based on pytest options and module config."""
    # Skip if metrics disabled
    if config.getoption("--no-metrics", default=False):
        return

    # Register markers
    config.addinivalue_line(
        "markers", "component(name): mark test as testing a specific component"
    )
    config.addinivalue_line(
        "markers",
        "instrumented: mark test as auto-generated from instrumented component",
    )

    # Filter common warnings
    config.addinivalue_line(
        "filterwarnings",
        "ignore:builtin type.*has no __module__ attribute:DeprecationWarning",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Initialize sleuth at session start."""
    global _session_start_time
    _session_start_time = time.time()

    # Skip if metrics disabled
    if session.config.getoption("--no-metrics", default=False):
        return

    from . import configure, configure_expectations, configure_metrics

    # Resolve paths - try to find tests directory
    tests_dir = _find_tests_dir(session.config.rootdir)  # type: ignore[attr-defined]

    # Get results dir from CLI > module config > default
    results_dir = (
        session.config.getoption("--metrics-dir")
        or _sleuth_config.get("results_dir")
        or (tests_dir / "results" if tests_dir else None)
    )

    history_dir = _sleuth_config.get("history_dir") or (
        tests_dir / "history" if tests_dir else None
    )

    # Clean up old test result files to prevent duplicate entries
    if results_dir and Path(results_dir).exists():
        _cleanup_old_results(Path(results_dir))

    # Configure sleuth core AND initialize MetricsClient
    configure(
        results_dir=str(results_dir) if results_dir else None,
        history_dir=str(history_dir) if history_dir else None,
    )

    # Initialize MetricsClient for persistence
    if results_dir:
        configure_metrics(
            results_dir=str(results_dir),
            history_dir=str(history_dir) if history_dir else None,
            persist=True,
        )

    # Load expectations
    expectations_path = _resolve_expectations_path(session.config, tests_dir)

    if expectations_path and expectations_path.exists():
        configure_expectations(path=expectations_path)
        if _sleuth_config.get("verbose", True):
            print(f"\n📊 Sleuth configured")
            print(f"   Results: {results_dir}")
            print(f"   Expectations: {expectations_path}")
    elif _sleuth_config.get("verbose", True):
        print(f"\n📊 Sleuth configured (no expectations file)")
        print(f"   Results: {results_dir}")


def pytest_collect_file(parent, file_path):
    """
    Auto-generate test items for instrumented components.

    When pytest discovers the instrumented/ directory, this hook
    creates virtual test items for each component + expectation combination.
    """
    # Only process files in the instrumented directory
    if "instrumented" not in str(file_path):
        return None

    if file_path.suffix != ".py":
        return None

    if file_path.name.startswith("_") or file_path.name == "__init__.py":
        return None

    # Create a custom test file collector
    return InstrumentedFile.from_parent(parent, path=file_path)


class InstrumentedFile(pytest.File):
    """Custom collector for instrumented component files."""

    def collect(self):
        """Generate test items for all components in this file."""
        from . import get_expectations

        # Import the module and discover components
        components = {}
        try:
            import importlib.util
            import sys

            module_name = f"tests.instrumented.{self.path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, self.path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Get components from registry or scan for @track decorated functions
                if hasattr(module, "INSTRUMENTED_COMPONENTS"):
                    components = module.INSTRUMENTED_COMPONENTS
                else:
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        # Check for @track decorator markers
                        is_tracked = (
                            hasattr(attr, "__track_metadata__")
                            or hasattr(attr, "_expects")
                            or hasattr(attr, "_metrics")
                        )
                        if callable(attr) and is_tracked:
                            components[attr_name] = attr
        except Exception as e:
            # Skip files that fail to import
            return []

        if not components:
            return []

        # Create test items for each component
        # Each component may have its own source file or use global expectations
        for component_name, component_func in components.items():
            # Check if component has custom source parameter
            custom_source = getattr(component_func, "_source", None)

            if custom_source:
                # Load expectations from component's source file
                import json
                from pathlib import Path

                source_path = Path(custom_source)
                if not source_path.is_absolute():
                    # Make relative to tests directory
                    source_path = Path.cwd() / source_path

                if source_path.exists() and source_path.suffix == ".json":
                    try:
                        with open(source_path) as f:
                            source_data = json.load(f)

                        # Handle array vs object
                        if isinstance(source_data, list):
                            # Array: each item is a test case
                            component_expectations = source_data
                        elif isinstance(source_data, dict):
                            # Object: single test case
                            component_expectations = [source_data]
                        else:
                            component_expectations = []
                    except Exception as e:
                        # Failed to load source, skip this component
                        component_expectations = []
                else:
                    # Source file doesn't exist or isn't JSON, skip
                    component_expectations = []
            else:
                # Use global expectations
                component_expectations = get_expectations()

            if not component_expectations:
                # No expectations - create one test without expectations
                yield InstrumentedItem.from_parent(
                    self,
                    name=component_name,
                    component_name=component_name,
                    component_func=component_func,
                    test_case=None,
                )
                continue

            # Create test items for each expectation
            for i, expectation in enumerate(component_expectations):
                # Use 'id' field if present, otherwise generate from index
                test_case_id = expectation.get("id", f"case_{i}")
                test_name = f"{component_name}[{test_case_id}]"
                yield InstrumentedItem.from_parent(
                    self,
                    name=test_name,
                    component_name=component_name,
                    component_func=component_func,
                    test_case=expectation,
                    test_case_id=test_case_id,
                )


class InstrumentedItem(pytest.Item):
    """A test item for an instrumented component."""

    def __init__(
        self, name, parent, component_name, component_func, test_case, test_case_id=None
    ):
        super().__init__(name, parent)
        self.component_name = component_name
        self.component_func = component_func
        self.test_case = test_case
        self.test_case_id = test_case_id or "unknown"

        # Add marker
        self.add_marker(pytest.mark.instrumented)
        self.add_marker(pytest.mark.component(component_name))

    def runtest(self):
        """Run the instrumented component with the test case."""
        from . import set_test_context
        import inspect

        if self.test_case:
            # Set up expectations context
            set_test_context(
                test_case_id=self.test_case_id, expectations=self.test_case
            )

            # Get function signature to understand parameters
            sig = inspect.signature(self.component_func)
            params = list(sig.parameters.keys())

            # Determine expects key
            expects_key = None
            if hasattr(self.component_func, "__track_metadata__"):
                metadata = self.component_func.__track_metadata__
                expects = metadata.get("expects")
                if expects:
                    expects_key = expects.input_key
            elif hasattr(self.component_func, "_expects"):
                # New @track decorator stores expects directly
                expects_key = self.component_func._expects

            # Build kwargs for function call - transparently pass test_case data
            kwargs = {}

            # If expects_key is specified, it must exist in test_case
            if expects_key:
                if expects_key in self.test_case:
                    value = self.test_case[expects_key]
                    # Map to first parameter if expects_key matches
                    if expects_key in params:
                        kwargs[expects_key] = value
                else:
                    raise ValueError(
                        f"Component expects '{expects_key}' but test case doesn't provide it. "
                        f"Available keys: {list(self.test_case.keys())}"
                    )

            # Pass any other matching keys from test case to function parameters
            for param in params:
                if param not in kwargs and param in self.test_case:
                    kwargs[param] = self.test_case[param]

            # Call the instrumented component with kwargs
            result = self.component_func(**kwargs)
        else:
            # No test case - just run the component with dummy data
            result = self.component_func("test input")

        # Basic validation
        if result is None:
            raise AssertionError(f"{self.component_name} returned None")

    def repr_failure(self, excinfo):
        """Custom failure representation."""
        return f"{self.component_name} failed: {excinfo.value}"

    def reportinfo(self):
        """Report info for this test item."""
        return self.path, None, f"{self.component_name}"


def pytest_runtest_makereport(item, call):
    """Track which components ran in this session."""
    global _executed_components
    if call.when == "call":
        marker = item.get_closest_marker("component")
        if marker and marker.args:
            _executed_components.add(marker.args[0])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Generate heartbeat after tests complete."""
    global _executed_components

    # Skip if metrics disabled or auto_heartbeat off
    if session.config.getoption("--no-metrics", default=False):
        return
    if not _sleuth_config.get("auto_heartbeat", True):
        return

    from . import DependencyGraphBuilder, generate_heartbeat_from_graph

    tests_dir = _find_tests_dir(session.config.rootdir)  # type: ignore[attr-defined]
    if not tests_dir:
        return

    results_dir_str = _sleuth_config.get("results_dir")
    results_dir = Path(results_dir_str) if results_dir_str else tests_dir / "results"

    instrumented_dir_str = _sleuth_config.get("instrumented_dir")
    instrumented_dir = (
        Path(instrumented_dir_str)
        if instrumented_dir_str
        else tests_dir / "instrumented"
    )

    if not instrumented_dir.exists():
        return

    try:
        # Build dependency graph
        builder = DependencyGraphBuilder()
        graph = builder.build_graph(instrumented_dir, exclude_patterns=["__pycache__"])

        # Check if root components ran
        root_components = set(graph.get_roots())
        roots_executed = _executed_components & root_components

        if not roots_executed:
            if _sleuth_config.get("verbose", True):
                print(f"\n📊 No root components ran - skipping heartbeat")
            return

        # Get components with results
        components_with_results = set()
        for result_file in results_dir.glob("*_results.json"):
            with open(result_file) as f:
                result = json.load(f)
                components_with_results.add(result["component"])

        if components_with_results:
            # Use aggregate_metrics_from_graph to save hierarchical reports
            from . import aggregate_metrics_from_graph

            aggregate_metrics_from_graph(
                graph=graph,
                results_dir=results_dir,
                include_components=components_with_results,
                save_reports=True,  # This saves hierarchical_report.json
            )

            # Then generate heartbeat for display
            heartbeat = generate_heartbeat_from_graph(
                graph=graph,
                results_dir=results_dir,
                include_components=components_with_results,
            )

            if _sleuth_config.get("verbose", True):
                print("\n📊 System Heartbeat:")

                # Print system-level metrics first (high-level health)
                system_metrics = heartbeat.get("system_metrics", {})
                if system_metrics:
                    print("   System Metrics:")
                    for name, value in sorted(system_metrics.items()):
                        if isinstance(value, float):
                            print(f"     {name}: {value:.4f}")
                        else:
                            print(f"     {name}: {value}")

                # Print component-level metrics (detailed)
                component_metrics = heartbeat.get("metrics", {})
                if component_metrics:
                    print("   Component Metrics:")
                    for name, value in sorted(component_metrics.items()):
                        if isinstance(value, float):
                            print(f"     {name}: {value:.4f}")
                        else:
                            print(f"     {name}: {value}")

    except Exception as e:
        if _sleuth_config.get("verbose", True):
            print(f"\n⚠️  Heartbeat generation error: {e}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def metrics_client():
    """Get the sleuth MetricsClient for the session."""
    from . import get_metrics_client

    return get_metrics_client()


@pytest.fixture(scope="session")
def expectations_data() -> List[Dict[str, Any]]:
    """Load expectations data from the configured file."""
    from . import get_expectations

    return get_expectations()


@pytest.fixture(scope="session")
def dependency_graph(request):
    """Build the dependency graph from instrumented components."""
    from . import DependencyGraphBuilder

    tests_dir = _find_tests_dir(request.config.rootdir)  # type: ignore[attr-defined]
    if not tests_dir:
        pytest.skip("Could not find tests directory")

    instrumented_dir_str = _sleuth_config.get("instrumented_dir")
    instrumented_dir = (
        Path(instrumented_dir_str)
        if instrumented_dir_str
        else tests_dir / "instrumented"
    )

    if not instrumented_dir.exists():
        pytest.skip(f"Instrumented directory not found: {instrumented_dir}")

    builder = DependencyGraphBuilder()
    return builder.build_graph(instrumented_dir, exclude_patterns=["__pycache__"])


@pytest.fixture
def set_context():
    """
    Fixture to set the test context with expectations.

    Usage:
        def test_something(set_context, test_case):
            set_context(test_case)
            result = my_component(test_case["input"])
    """
    from . import set_test_context

    def _set_context(expectations: Dict[str, Any], test_case_id: Optional[str] = None):
        final_id = test_case_id or expectations.get("id", "unknown")
        set_test_context(test_case_id=final_id, expectations=expectations)

    return _set_context


# =============================================================================
# Helpers
# =============================================================================


def _find_tests_dir(rootdir) -> Optional[Path]:
    """Find the tests directory relative to rootdir."""
    # Convert to pathlib.Path if needed (pytest uses py.path.local)
    rootdir = Path(str(rootdir))

    # Common patterns
    for pattern in ["tests", "test", "backend/tests"]:
        candidate = rootdir / pattern
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _cleanup_old_results(results_dir: Path) -> None:
    """Clean up old test result files before starting new test run."""
    try:
        # Remove component result files (*_results.json)
        for result_file in results_dir.glob("*_results.json"):
            result_file.unlink()

        # Remove aggregate report files
        for report_file in ["hierarchical_report.json", "system_heartbeat.json"]:
            report_path = results_dir / report_file
            if report_path.exists():
                report_path.unlink()
    except Exception as e:
        # Don't fail the test run if cleanup fails
        if _sleuth_config.get("verbose", True):
            print(f"\n⚠️  Warning: Failed to clean up old results: {e}")


def _resolve_expectations_path(
    config: "Config", tests_dir: Optional[Path]
) -> Optional[Path]:
    """Resolve default source file path from various sources."""
    # 1. CLI option (legacy support)
    cli_path = config.getoption("--sleuth-expectations", default=None)
    if cli_path:
        return Path(cli_path)

    # 2. Module config (default_source)
    if _sleuth_config.get("default_source"):
        return Path(_sleuth_config["default_source"])

    # 3. Default locations
    if tests_dir:
        for pattern in [
            "data/expectations.json",
            "expectations.json",
            "fixtures/expectations.json",
        ]:
            candidate = tests_dir / pattern
            if candidate.exists():
                return candidate

    return None


# =============================================================================
# Public API
# =============================================================================


def add_result(result: MetricResult) -> None:
    """Add a metric result to the current session."""
    global _session_results
    _session_results.append(result)


def get_session_results() -> List[MetricResult]:
    """Get all results from the current session."""
    return _session_results.copy()


def clear_session_results() -> None:
    """Clear session results (for testing)."""
    global _session_results
    _session_results.clear()


__all__ = [
    "configure_sleuth",
    "add_result",
    "get_session_results",
    "clear_session_results",
]
