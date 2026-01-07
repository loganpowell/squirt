"""
Vector/Embedding Metrics Plugin for Squirt

Metrics for embeddings, vector search, and similarity operations.

Usage:
    from squirt.contrib.vector import vector

    @track(metrics=[
        vector.top_similarity.from_output("similarity"),
        vector.hit_rate.compute(my_fn),
    ])
    def search_vectors(query: str) -> dict:
        ...
"""

from .metrics import VectorMetrics, vector

__all__ = ["vector", "VectorMetrics"]
