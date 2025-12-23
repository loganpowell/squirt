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

from squirt.plugins import MetricBuilder, AggregationType, SystemMetric


class ChunkMetrics:
    """
    Metrics for text/document chunking operations.

    Tracks chunk characteristics like counts, sizes, overlap, and distribution.
    All attributes have explicit type annotations for IDE autocomplete support.
    """

    count: MetricBuilder = MetricBuilder(
        "chunk_count",
        AggregationType.SUM,
        description="Number of chunks produced",
    )

    avg_size: MetricBuilder = MetricBuilder(
        "avg_chunk_size",
        AggregationType.AVERAGE,
        description="Average chunk size (tokens or characters)",
    )

    total_size: MetricBuilder = MetricBuilder(
        "total_chunk_size",
        AggregationType.SUM,
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
