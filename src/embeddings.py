"""
Semantic embedding pipeline — encodes JDs and candidate profiles into
dense vectors for similarity computation, with disk caching.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.utils import EMBEDDINGS_CACHE, ensure_dir, get_logger, load_config

logger = get_logger(__name__)


class EmbeddingEngine:
    """Encode text into dense vectors using ``sentence-transformers``.

    Attributes:
        model_name: HuggingFace model identifier.
        model: Lazy-loaded SentenceTransformer instance.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialise the engine (model loaded lazily on first call).

        Args:
            model_name: Override model from config.
        """
        cfg = load_config()
        emb_cfg = cfg.get("embedding", {})
        self.model_name: str = model_name or emb_cfg.get(
            "model_name", "all-MiniLM-L6-v2"
        )
        self._batch_size: int = emb_cfg.get("batch_size", 64)
        self._normalize: bool = emb_cfg.get("normalize", True)
        self._cache_dir: Path = ensure_dir(EMBEDDINGS_CACHE)
        self._model = None  # lazy

    # ── lazy model loading ───────────────────────────────────────────
    @property
    def model(self):
        """Load the model on first access to avoid import-time overhead."""
        if self._model is None:
            logger.info("Loading embedding model: %s …", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "Model loaded — dimension=%d",
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    # ── public API ───────────────────────────────────────────────────
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text.

        Returns:
            1-D numpy vector.
        """
        vec = self.model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=np.float32)

    def embed_jd(self, jd: JobDescription) -> np.ndarray:
        """Embed a parsed job description.

        Args:
            jd: Structured JD.

        Returns:
            1-D numpy vector.
        """
        text = jd.to_embedding_text()
        return self.embed_text(text)

    def embed_profile(self, profile: CandidateProfile) -> np.ndarray:
        """Embed a single candidate profile.

        Args:
            profile: Normalised profile.

        Returns:
            1-D numpy vector.
        """
        text = profile.to_embedding_text()
        return self.embed_text(text)

    def batch_embed_profiles(
        self,
        profiles: List[CandidateProfile],
        cache_key: Optional[str] = None,
        refresh: bool = False,
    ) -> np.ndarray:
        """Batch-embed a list of profiles, with optional disk caching.

        Args:
            profiles: List of normalised profiles.
            cache_key: If given, results are cached under this name.
            refresh: If True, ignore cache and recompute.

        Returns:
            2-D numpy array of shape ``(len(profiles), dim)``.
        """
        # Check cache
        if cache_key and not refresh:
            cached = self._load_cache(cache_key)
            if cached is not None and cached.shape[0] == len(profiles):
                logger.info(
                    "Loaded cached embeddings (%s) — %d vectors.",
                    cache_key,
                    len(profiles),
                )
                return cached

        texts = [p.to_embedding_text() for p in profiles]
        logger.info(
            "Batch-embedding %d profiles (batch_size=%d) …",
            len(texts),
            self._batch_size,
        )

        vecs = self.model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=True,
        )
        result = np.asarray(vecs, dtype=np.float32)

        # Save cache
        if cache_key:
            self._save_cache(cache_key, result)
            logger.info("Cached embeddings to %s.", cache_key)

        return result

    @staticmethod
    def semantic_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors (assumed normalised).

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Similarity score in ``[-1, 1]`` (typically ``[0, 1]`` for normalised).
        """
        dot = float(np.dot(vec_a, vec_b))
        # Guard against un-normalised vectors
        norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm < 1e-9:
            return 0.0
        return dot / norm

    @staticmethod
    def batch_similarity(jd_vec: np.ndarray, profile_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between one JD vector and many profiles.

        Args:
            jd_vec: 1-D JD vector.
            profile_vecs: 2-D matrix ``(N, dim)``.

        Returns:
            1-D array of N similarity scores.
        """
        jd_norm = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)
        norms = np.linalg.norm(profile_vecs, axis=1, keepdims=True) + 1e-9
        normed = profile_vecs / norms
        return normed @ jd_norm

    # ── cache helpers ────────────────────────────────────────────────
    def _cache_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()[:12]
        return self._cache_dir / f"{safe}.pkl"

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        path = self._cache_path(key)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, data: np.ndarray) -> None:
        path = self._cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
