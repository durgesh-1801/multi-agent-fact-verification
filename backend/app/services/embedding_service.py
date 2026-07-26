"""
Embedding Service using SentenceTransformers for local vector embedding generation.
Generates normalized vector embeddings for FAISS indexing and semantic similarity RAG queries.
"""

from abc import ABC, abstractmethod
import os
import logging
import threading
from typing import Any, List, Optional
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_service_instance_lock = threading.Lock()


class BaseEmbeddingService(ABC):
    """
    Abstract Base Class for text embedding services.
    Enables swapping out embedding backends (e.g. SentenceTransformers, OpenAI embeddings).
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a normalized floating-point vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of strings into a list of normalized floating-point vectors.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Returns the vector dimension size of the embedding model.
        """
        pass


class EmbeddingService(BaseEmbeddingService):
    """
    Singleton SentenceTransformers embedding model client with thread-safe lazy loading.
    """

    _instance: Optional["EmbeddingService"] = None
    _model: Optional[Any] = None
    _dimension: Optional[int] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            with _service_instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
                    cls._instance._dimension = None
        return cls._instance

    def _get_model(self) -> Any:
        """
        Lazily loads and caches the SentenceTransformer model instance using thread-safe locking.
        """
        if self._model is None:
            with _model_lock:
                if self._model is None:
                    # Enforce single-threaded PyTorch allocations to minimize memory footprint on Render
                    os.environ["OMP_NUM_THREADS"] = "1"
                    os.environ["MKL_NUM_THREADS"] = "1"
                    os.environ["TORCH_NUM_THREADS"] = "1"

                    from sentence_transformers import SentenceTransformer
                    import torch

                    try:
                        torch.set_num_threads(1)
                        if hasattr(torch, "set_num_interop_threads"):
                            torch.set_num_interop_threads(1)
                    except Exception:
                        pass

                    model_name = settings.EMBEDDING_MODEL_NAME
                    logger.info(f"Loading SentenceTransformer embedding model: '{model_name}'...")
                    try:
                        self._model = SentenceTransformer(model_name)
                        # Compute sample embedding to determine vector dimension
                        sample_emb = self._model.encode("test", normalize_embeddings=True)
                        self._dimension = len(sample_emb)
                        logger.info(
                            f"Successfully loaded embedding model '{model_name}' (dimension={self._dimension})."
                        )
                    except Exception as e:
                        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
                        raise RuntimeError(f"Could not load SentenceTransformer model '{model_name}': {str(e)}") from e

        return self._model

    @property
    def dimension(self) -> int:
        """
        Returns the embedding vector dimension size (e.g. 384 for all-MiniLM-L6-v2).
        """
        if self._dimension is None:
            self._get_model()
        return self._dimension or 384

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a normalized embedding vector for a single text string.

        Args:
            text: Input string to embed.

        Returns:
            List[float]: Normalized floating-point vector.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for embedding must be a non-empty string.")

        clean_text = text.strip()
        model = self._get_model()

        try:
            vector = model.encode(clean_text, normalize_embeddings=True)
            if isinstance(vector, np.ndarray):
                return vector.tolist()
            return list(vector)
        except Exception as e:
            logger.error(f"Error generating embedding for text snippet: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate text embedding: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates normalized embedding vectors for a list of text strings in batch.

        Args:
            texts: List of input text strings to embed.

        Returns:
            List[List[float]]: List of normalized vector floats.
        """
        if not texts:
            return []

        cleaned_texts: List[str] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text item at index {idx}: must be a non-empty string.")
            cleaned_texts.append(text.strip())

        model = self._get_model()
        logger.info(f"Generating batch embeddings for {len(cleaned_texts)} document chunks.")

        try:
            vectors = model.encode(cleaned_texts, normalize_embeddings=True)
            if isinstance(vectors, np.ndarray):
                return vectors.tolist()
            return [list(v) for v in vectors]
        except Exception as e:
            logger.error(f"Error generating batch embeddings for {len(cleaned_texts)} documents: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate batch document embeddings: {str(e)}") from e


# Singleton instance export
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """
    Returns the singleton EmbeddingService instance.
    """
    return embedding_service
