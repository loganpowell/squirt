"""
Squirt Analysis Module

AST-driven analysis for discovering component relationships and building
dependency graphs from instrumented code.

Usage:
    from squirt.analysis import (
        DependencyGraphBuilder,
        analyze_codebase,
        visualize_graph,
    )

    # Analyze a codebase
    graph, builder = analyze_codebase("./src")

    # Find root components
    roots = builder.get_roots(graph)

    # Visualize the graph
    print(visualize_graph(graph))
"""

from .graph_builder import (
    DecoratedFunctionVisitor,
    DependencyGraph,
    DependencyGraphBuilder,
    FunctionCallVisitor,
    analyze_codebase,
    visualize_graph,
)

__all__ = [
    "FunctionCallVisitor",
    "DecoratedFunctionVisitor",
    "DependencyGraph",
    "DependencyGraphBuilder",
    "visualize_graph",
    "analyze_codebase",
]
