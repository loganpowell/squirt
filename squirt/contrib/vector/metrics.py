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

from squirt.plugins import AggregationType, MetricBuilder, MetricNamespace, SystemMetric


class VectorMetrics(MetricNamespace):
    """
    Metrics for embedding and vector search operations.

    Tracks similarity scores, search quality, embedding characteristics.
    """

    @property
    def top_similarity(self) -> MetricBuilder:
        """Top similarity score from search (0.0-1.0)."""
        return self._define(
            "top_similarity",
            system_metric=SystemMetric.ACCURACY,
            description="Top similarity score from search (0.0-1.0)",
        )

    @property
    def avg_similarity(self) -> MetricBuilder:
        """Average similarity score across results."""
        return self._define(
            "avg_similarity",
            system_metric=SystemMetric.ACCURACY,
            description="Average similarity score across results",
        )

    @property
    def search_latency(self) -> MetricBuilder:
        """Time to perform vector search (ms)."""
        return self._define(
            "search_latency_ms",
            system_metric=SystemMetric.LATENCY_P95,
            description="Time to perform vector search (ms)",
        )

    @property
    def embedding_latency(self) -> MetricBuilder:
        """Time to generate embeddings (ms)."""
        return self._define(
            "embedding_latency_ms",
            system_metric=SystemMetric.LATENCY_P95,
            description="Time to generate embeddings (ms)",
        )

    @property
    def embedding_dimension(self) -> MetricBuilder:
        """Dimension of embedding vectors."""
        return self._define(
            "embedding_dimension",
            aggregation=AggregationType.AVERAGE,
            description="Dimension of embedding vectors",
        )

    @property
    def vectors_indexed(self) -> MetricBuilder:
        """Number of vectors in index."""
        return self._define(
            "vectors_indexed",
            aggregation=AggregationType.SUM,
            description="Number of vectors in index",
        )

    @property
    def results_returned(self) -> MetricBuilder:
        """Number of results returned by search."""
        return self._define(
            "results_returned",
            aggregation=AggregationType.AVERAGE,
            description="Number of results returned by search",
        )


# Singleton instance
vector = VectorMetrics()

__all__ = ["VectorMetrics", "vector"]
