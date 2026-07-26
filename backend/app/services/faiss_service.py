"""
FAISS Vector Index Management Service.
Provides in-memory vector storage, indexing, and similarity search for precomputed embeddings.
"""

import logging
import threading
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

_faiss_service_lock = threading.Lock()


class FAISSIndexContainer:
    """
    Container wrapping a FAISS index instance alongside associated metadata.
    """

    def __init__(self, dimension: int):
        import faiss

        self.dimension = dimension
        # Using IndexFlatIP (Inner Product) for normalized unit vector cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Adds normalized float32 numpy vectors and corresponding metadata dicts."""
        self.index.add(vectors)
        self.metadata_store.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Executes similarity search against the FAISS index.
        Returns list of dicts containing 'score' and 'metadata'.
        """
        if self.index.ntotal == 0:
            return []

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_store):
                results.append({
                    "score": float(score),
                    "metadata": self.metadata_store[idx],
                })
        return results

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal


class FAISSService:
    """
    Singleton FAISS Vector Indexing & Similarity Search Service.
    """

    _instance: Optional["FAISSService"] = None
    _indexes: Dict[str, FAISSIndexContainer] = {}

    def __new__(cls) -> "FAISSService":
        if cls._instance is None:
            with _faiss_service_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._indexes = {}
        return cls._instance

    def create_index(self, index_id: str, dimension: int) -> None:
        """
        Creates a new in-memory FAISS index container for the given index_id.

        Args:
            index_id: Unique session or job index identifier.
            dimension: Vector dimension size (e.g. 384).
        """
        if not index_id or not index_id.strip():
            raise ValueError("index_id must be a non-empty string.")

        if dimension <= 0:
            raise ValueError("dimension must be a positive integer.")

        clean_id = index_id.strip()
        if clean_id in self._indexes:
            logger.info(f"FAISS index '{clean_id}' already exists. Overwriting with empty index.")

        self._indexes[clean_id] = FAISSIndexContainer(dimension=dimension)
        logger.info(f"Created new FAISS index '{clean_id}' (dimension={dimension}).")

    def has_index(self, index_id: str) -> bool:
        """Checks if a FAISS index container exists for index_id."""
        return bool(index_id and index_id.strip() in self._indexes)

    def get_index_size(self, index_id: str) -> int:
        """Returns the total number of indexed vectors for index_id."""
        if not self.has_index(index_id):
            return 0
        return self._indexes[index_id.strip()].total_vectors

    def add_vectors(
        self,
        index_id: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Adds precomputed embedding vectors and metadata items to a target FAISS index.

        Args:
            index_id: Target FAISS index identifier.
            vectors: Precomputed floating point vector embeddings.
            metadata: Associated metadata dictionaries for each vector.
        """
        if not self.has_index(index_id):
            raise KeyError(f"FAISS index '{index_id}' does not exist. Call create_index first.")

        if not vectors:
            logger.warning(f"No vectors provided to add_vectors for index '{index_id}'. Skipping.")
            return

        if len(vectors) != len(metadata):
            raise ValueError(
                f"Vector count ({len(vectors)}) must match metadata count ({len(metadata)})."
            )

        container = self._indexes[index_id.strip()]

        np_vectors = np.array(vectors, dtype=np.float32)

        if np_vectors.ndim != 2 or np_vectors.shape[1] != container.dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected shape (N, {container.dimension}), "
                f"got {np_vectors.shape}."
            )

        try:
            container.add(np_vectors, metadata)
            logger.info(
                f"Added {len(vectors)} vectors to FAISS index '{index_id}'. "
                f"Total vectors: {container.total_vectors}"
            )
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS index '{index_id}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to add vectors to index '{index_id}': {str(e)}") from e

    def search(
        self,
        index_id: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs similarity search against target FAISS index using precomputed query vector.

        Args:
            index_id: Target FAISS index identifier.
            query_vector: Precomputed query embedding vector.
            top_k: Number of nearest results to retrieve.

        Returns:
            List[Dict[str, Any]]: Search results with 'score' and 'metadata'.
        """
        if not self.has_index(index_id):
            raise KeyError(f"FAISS index '{index_id}' does not exist.")

        if not query_vector:
            raise ValueError("query_vector must be a non-empty list of floats.")

        if top_k < 1:
            raise ValueError("top_k must be a positive integer greater than 0.")

        container = self._indexes[index_id.strip()]

        np_query = np.array([query_vector], dtype=np.float32)
        if np_query.shape[1] != container.dimension:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {container.dimension}, "
                f"got {np_query.shape[1]}."
            )

        try:
            results = container.search(np_query, top_k=top_k)
            logger.info(f"Retrieved {len(results)} search results from FAISS index '{index_id}'.")
            return results
        except Exception as e:
            logger.error(f"Failed search on FAISS index '{index_id}': {e}", exc_info=True)
            raise RuntimeError(f"Search failed on FAISS index '{index_id}': {str(e)}") from e

    def search_similar(self, index_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds query text and executes similarity search against target FAISS index.
        Returns list of metadata dicts flattened with similarity score.
        """
        if not self.has_index(index_id) or not query:
            return []
        try:
            from app.services.embedding_service import embedding_service
            query_vector = embedding_service.embed_text(query)
            raw_results = self.search(index_id=index_id, query_vector=query_vector, top_k=k)
            flattened: List[Dict[str, Any]] = []
            for item in raw_results:
                meta = dict(item.get("metadata", {}))
                meta["score"] = item.get("score", 0.0)
                flattened.append(meta)
            return flattened
        except Exception as e:
            logger.warning(f"search_similar failed for index '{index_id}': {e}")
            return []

    def delete_index(self, index_id: str) -> bool:
        """
        Deletes a FAISS index container and frees associated memory.
        """
        if not self.has_index(index_id):
            return False

        clean_id = index_id.strip()
        del self._indexes[clean_id]
        logger.info(f"Deleted FAISS index '{clean_id}'.")
        return True


# Singleton instance export
faiss_service = FAISSService()


def get_faiss_service() -> FAISSService:
    """
    Returns the singleton FAISSService instance.
    """
    return faiss_service
