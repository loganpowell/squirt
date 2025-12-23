"""
Chunking Metrics Plugin for Squirt

Metrics for text/document chunking operations.

Usage:
    from squirt.contrib.chunk import chunk

    @track(metrics=[
        chunk.count.from_output("metadata.chunk_count"),
        chunk.avg_size.from_output("metadata.avg_chunk_size"),
    ])
    def my_chunker(...):
        ...
"""

from .metrics import ChunkMetrics, chunk

__all__ = ["ChunkMetrics", "chunk"]
