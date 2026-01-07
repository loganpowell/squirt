"""
Chunking Metrics for Squirt

Metrics for text/document chunking operations - chunk counts, sizes, overlap, etc.

Usage:
    from squirt.contrib.chunk import chunk

    @track(metrics=[
        chunk.count.from_output("metadata.chunk_count"),
        chunk.avg_size.from_output("metadata.avg_chunk_size"),
    ])
    def chunk_document(text: str) -> dict:
        ...
"""

from squirt.plugins import AggregationType, MetricBuilder, MetricNamespace


class ChunkMetrics(MetricNamespace):
    """
    Metrics for text/document chunking operations.

    Tracks chunk characteristics like counts, sizes, overlap, and distribution.
    """

    @property
    def count(self) -> MetricBuilder:
        """Number of chunks produced."""
        return self._define(
            "chunk_count",
            aggregation=AggregationType.SUM,
            description="Number of chunks produced",
        )

    @property
    def avg_size(self) -> MetricBuilder:
        """Average chunk size (tokens or characters)."""
        return self._define(
            "avg_chunk_size",
            aggregation=AggregationType.AVERAGE,
            description="Average chunk size (tokens or characters)",
        )

    @property
    def total_size(self) -> MetricBuilder:
        """Total size of all chunks."""
        return self._define(
            "total_chunk_size",
            aggregation=AggregationType.SUM,
            description="Total size of all chunks",
        )

    # Quality metrics (implementation-specific)
    # Projects should define chunking quality metrics based on their needs:
    # - overlap_ratio, size_variance (uniformity)
    # - coverage, boundary_accuracy (semantic quality)
    # These depend on your chunking strategy and quality requirements


# Singleton instance
chunk = ChunkMetrics()

__all__ = ["ChunkMetrics", "chunk"]
