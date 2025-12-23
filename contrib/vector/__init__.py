"""
Vector/Embedding Metrics Plugin for Sleuth

Metrics for embeddings, vector search, and similarity operations.

Usage:
    from sleuth.contrib.vector import vector

    @track(metrics=[
        vector.top_similarity.from_output("similarity"),
        vector.hit_rate.compute(my_fn),
    ])
    def search_vectors(query: str) -> dict:
        ...
"""

from .metrics import vector, VectorMetrics

__all__ = ["vector", "VectorMetrics"]
