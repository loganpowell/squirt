"""
Vector/Embedding Metrics for Squirt

Metrics for embeddings, vector search, and similarity operations.

Usage:
    from squirt.contrib.vector import vector

    @track(metrics=[
        vector.top_similarity.from_output("similarity"),
        vector.search_latency.from_output("metadata.search_ms"),
    ])
    def search_vectors(query: str) -> dict:
        ...
"""

from squirt.plugins import MetricBuilder, AggregationType, SystemMetric


class VectorMetrics:
    """
    Metrics for embedding and vector search operations.

    Tracks similarity scores, search quality, embedding characteristics.
    All attributes have explicit type annotations for IDE autocomplete support.
    """

    # Similarity metrics
    top_similarity: MetricBuilder = MetricBuilder(
        "top_similarity",
        AggregationType.AVERAGE,
        SystemMetric.ACCURACY,
        description="Top similarity score from search (0.0-1.0)",
    )

    avg_similarity: MetricBuilder = MetricBuilder(
        "avg_similarity",
        AggregationType.AVERAGE,
        SystemMetric.ACCURACY,
        description="Average similarity score across results",
    )

    # Search quality metrics (implementation-specific: hit_rate, accuracy, mrr)
    # Users should define these in their project metrics based on their search semantics

    # Performance metrics
    search_latency: MetricBuilder = MetricBuilder(
        "search_latency_ms",
        AggregationType.AVERAGE,
        SystemMetric.LATENCY_P95,
        description="Time to perform vector search (ms)",
    )

    embedding_latency: MetricBuilder = MetricBuilder(
        "embedding_latency_ms",
        AggregationType.AVERAGE,
        SystemMetric.LATENCY_P95,
        description="Time to generate embeddings (ms)",
    )

    # Embedding characteristics
    embedding_dimension: MetricBuilder = MetricBuilder(
        "embedding_dimension",
        AggregationType.AVERAGE,
        description="Dimension of embedding vectors",
    )

    vectors_indexed: MetricBuilder = MetricBuilder(
        "vectors_indexed",
        AggregationType.SUM,
        description="Number of vectors in index",
    )

    results_returned: MetricBuilder = MetricBuilder(
        "results_returned",
        AggregationType.AVERAGE,
        description="Number of results returned by search",
    )


# Singleton instance
vector = VectorMetrics()

__all__ = ["VectorMetrics", "vector"]
