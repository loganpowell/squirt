"""
Squirt Dependency Graph Builder

Analyzes Python source files to build a dependency graph of decorated functions.
This enables automatic discovery of component relationships without manual configuration.

The graph is represented as a simple dict structure that can be easily serialized
and optionally converted to NetworkX for advanced visualization.

Usage:
    from squirt.analysis import DependencyGraphBuilder, analyze_codebase

    # Quick analysis
    graph, builder = analyze_codebase("./src")
    roots = graph.get_roots()

    # Detailed analysis
    builder = DependencyGraphBuilder()
    graph = builder.build_graph(Path("./src"))

    for root in graph.get_roots():
        print(f"Pipeline: {root}")
        for child in graph.get_children(root):
            print(f"  -> {child}")

    # Optional: Convert to NetworkX for visualization
    nx_graph = graph.to_networkx()
"""

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor that extracts function calls from a function body."""

    def __init__(self):
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a function call and record its name."""
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        self.generic_visit(node)


class DecoratedFunctionVisitor(ast.NodeVisitor):
    """AST visitor that finds functions decorated with @track."""

    def __init__(self):
        self.functions: dict[str, dict[str, Any]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition and check for our decorator."""
        if self._has_track_decorator(node):
            self.functions[node.name] = {
                "params": [arg.arg for arg in node.args.args],
                "return_type": self._get_return_annotation(node),
                "calls": self._extract_function_calls(node),
                "decorators": self._get_decorator_names(node),
                "lineno": node.lineno,
            }
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition and check for our decorator."""
        if self._has_track_decorator(node):
            self.functions[node.name] = {
                "params": [arg.arg for arg in node.args.args],
                "return_type": self._get_return_annotation(node),
                "calls": self._extract_function_calls(node),
                "decorators": self._get_decorator_names(node),
                "lineno": node.lineno,
                "is_async": True,
            }
        self.generic_visit(node)

    def _has_track_decorator(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        """Check if function has @track or related decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if decorator.func.id in (
                        "track",
                        "track_component",
                        "track_component_async",
                    ):
                        return True
            elif isinstance(decorator, ast.Name):
                if decorator.id in (
                    "track",
                    "track_component",
                    "track_component_async",
                ):
                    return True
        return False

    def _get_decorator_names(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Extract names of all decorators on a function."""
        names = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    names.append(decorator.func.id)
            elif isinstance(decorator, ast.Name):
                names.append(decorator.id)
        return names

    def _extract_function_calls(
        self, func_node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Extract names of functions called within this function."""
        visitor = FunctionCallVisitor()
        visitor.visit(func_node)
        return list(visitor.calls)

    def _get_return_annotation(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """Extract return type annotation if present."""
        if node.returns:
            return ast.unparse(node.returns)
        return "Any"


@dataclass
class DependencyGraph:
    """
    A directed graph of component dependencies.

    This is a simple, serializable graph structure with methods for
    traversal and optional conversion to NetworkX for visualization.
    """

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def __contains__(self, node_name: str) -> bool:
        return node_name in self.nodes

    def __iter__(self) -> Iterator[str]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(self, name: str, **attrs: Any) -> None:
        self.nodes[name] = attrs

    def add_edge(self, source: str, target: str) -> None:
        if (source, target) not in self.edges:
            self.edges.append((source, target))

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self.edges

    def get_roots(self) -> list[str]:
        """Find root nodes (nodes with no incoming edges)."""
        called_funcs = {dst for _, dst in self.edges}
        return [n for n in self.nodes if n not in called_funcs]

    def get_children(self, node_name: str) -> list[str]:
        """Get nodes that this node calls (outgoing edges)."""
        return [dst for src, dst in self.edges if src == node_name and dst != node_name]

    def get_parents(self, node_name: str) -> list[str]:
        """Get nodes that call this node (incoming edges)."""
        return [src for src, dst in self.edges if dst == node_name]

    def in_degree(self, node_name: str) -> int:
        return len(self.get_parents(node_name))

    def out_degree(self, node_name: str) -> int:
        return len(self.get_children(node_name))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dict."""
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DependencyGraph":
        """Create a DependencyGraph from a dict representation."""
        graph = cls()
        graph.nodes = data.get("nodes", {})
        graph.edges = [tuple(e) for e in data.get("edges", [])]  # type: ignore
        return graph

    def to_networkx(self) -> Any:
        """Convert to a NetworkX DiGraph for advanced visualization."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for to_networkx(). Install it with: pip install networkx"
            )
        nx_graph = nx.DiGraph()
        for name, attrs in self.nodes.items():
            nx_graph.add_node(name, **attrs)
        for src, dst in self.edges:
            nx_graph.add_edge(src, dst)
        return nx_graph

    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram representation."""
        lines = ["flowchart TD"]
        for name in self.nodes:
            lines.append(f"    {name}[{name}]")
        for src, dst in self.edges:
            lines.append(f"    {src} --> {dst}")
        return "\n".join(lines)


class DependencyGraphBuilder:
    """Analyzes decorated functions in a codebase and builds a dependency graph."""

    def __init__(self):
        self.functions: dict[str, dict[str, Any]] = {}
        self.file_map: dict[str, str] = {}

    def analyze_file(self, file_path: Path) -> dict[str, dict[str, Any]]:
        """Analyze a single Python file for decorated functions."""
        try:
            source = file_path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return {}

        visitor = DecoratedFunctionVisitor()
        visitor.visit(tree)

        for func_name in visitor.functions:
            self.file_map[func_name] = str(file_path)

        return visitor.functions

    def build_graph(
        self, source_dir: Path, exclude_patterns: list[str] | None = None
    ) -> DependencyGraph:
        """Parse all Python files in a directory and build the dependency graph."""
        exclude_patterns = exclude_patterns or [
            "test",
            "migration",
            "__pycache__",
            ".venv",
            "node_modules",
        ]

        for py_file in source_dir.rglob("*.py"):
            path_str = str(py_file)
            if any(pattern in path_str for pattern in exclude_patterns):
                continue
            file_functions = self.analyze_file(py_file)
            self.functions.update(file_functions)

        return self._build_graph()

    def _build_graph(self) -> DependencyGraph:
        """Build a DependencyGraph from discovered functions."""
        graph = DependencyGraph()
        decorated_names = set(self.functions.keys())

        for func_name, func_info in self.functions.items():
            graph.add_node(func_name, **func_info)
            for called_func in func_info["calls"]:
                if called_func in decorated_names:
                    graph.add_edge(func_name, called_func)

        return graph

    def get_roots(self, graph: DependencyGraph | None = None) -> list[str]:
        """Find root nodes. Prefer using graph.get_roots() directly."""
        if graph is None:
            graph = self._build_graph()
        return graph.get_roots()

    def get_children(
        self, func_name: str, graph: DependencyGraph | None = None
    ) -> list[str]:
        """Get functions called by the given function."""
        if graph is None:
            graph = self._build_graph()
        return graph.get_children(func_name)

    def get_parents(
        self, func_name: str, graph: DependencyGraph | None = None
    ) -> list[str]:
        """Get functions that call the given function."""
        if graph is None:
            graph = self._build_graph()
        return graph.get_parents(func_name)


def visualize_graph(graph: DependencyGraph, output_path: str | None = None) -> str:
    """Create a text visualization of the dependency graph."""
    lines = ["Component Dependency Graph", "=" * 40, ""]
    roots = graph.get_roots()

    def _print_tree(node: str, indent: int = 0, visited: set[str] | None = None):
        if visited is None:
            visited = set()
        prefix = "  " * indent
        if node in visited:
            lines.append(f"{prefix}└── {node} (circular ref)")
            return
        visited.add(node)
        lines.append(f"{prefix}├── {node}")
        for child in graph.get_children(node):
            _print_tree(child, indent + 1, visited.copy())

    if not roots:
        lines.append("No root components found.")
    else:
        for root in roots:
            lines.append(f"Root: {root}")
            for child in graph.get_children(root):
                _print_tree(child, 1)
            lines.append("")

    output = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(output)
    return output


def analyze_codebase(
    source_dir: str, exclude_patterns: list[str] | None = None
) -> tuple[DependencyGraph, DependencyGraphBuilder]:
    """Analyze a codebase and return the dependency graph."""
    builder = DependencyGraphBuilder()
    graph = builder.build_graph(Path(source_dir), exclude_patterns)
    return graph, builder


__all__ = [
    "FunctionCallVisitor",
    "DecoratedFunctionVisitor",
    "DependencyGraph",
    "DependencyGraphBuilder",
    "visualize_graph",
    "analyze_codebase",
]
