"""
RAGEngine - hybrid retrieval over the scheme chunks.

Metadata first, vectors second: callers pass the scheme ids / sections the
SQLite + rules stages already narrowed down to, and only those rows are
scored. Chunk embeddings are pre-computed offline (rag_embeddings.npy,
~0.4 MB) and scored with one einsum, so no vector-database process or extra
index is needed. einsum rather than `@`: right after an e5 encode PyTorch's
six worker threads are still spinning and OpenBLAS's six threads fight them -
the `@` matvec measured 15.8 ms P50 in that state (0.015 ms on an idle CPU),
einsum 0.17 ms, since it runs in the calling thread (measured 2026-09-11).

The query embedder is NOT loaded here: it is the same multilingual-e5-small
instance the legacy RAGRetriever already keeps in memory, obtained through
`embedder_provider`. Query vectors are cached (small LRU).
"""

import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, repo, embedder_provider: Callable[[], Any],
                 embeddings_path: str = C.RAG_EMBEDDINGS_PATH, faq_path: str = C.FAQ_QUESTIONS_PATH):
        self.repo = repo
        self._provider = embedder_provider
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._emb: Optional[np.ndarray] = None
        self._scheme_of_row = np.array([c["scheme_id"] for c in repo.chunks])
        self._section_of_row = np.array([(c.get("section") or "") for c in repo.chunks])
        if os.path.exists(embeddings_path):
            emb = np.load(embeddings_path)
            if emb.shape[0] == len(repo.chunks):
                self._emb = np.ascontiguousarray(emb, dtype=np.float32)
            else:
                logger.warning(f"[SI] rag_embeddings.npy has {emb.shape[0]} rows but the KB has "
                               f"{len(repo.chunks)} chunks - rebuild with build_kb; vector search disabled")
        else:
            logger.warning("[SI] rag_embeddings.npy missing - vector search disabled (structured answers still work)")
        self._faq_emb: Optional[np.ndarray] = None
        self._faq_rows: List[Dict[str, Any]] = []
        self._faq_scheme = np.array([], dtype=str)
        self._load_faq_questions(faq_path)

    def _load_faq_questions(self, path: str) -> None:
        """
        FAQ questions embedded on their own at build time (faq_questions.npz), so a
        spoken question is matched question-to-question. Against passages that also
        hold the answer, the near-verbatim "When does the Sukanya Samriddhi account
        mature?" scored only 0.878 on its own FAQ (measured) - too close to other
        FAQs to read one out safely.
        """
        if not os.path.exists(path):
            return
        data = np.load(path)
        by_id = {f["id"]: f for sid in set(data["scheme_ids"].tolist()) for f in self.repo.faqs(sid)}
        rows = [by_id.get(int(i)) for i in data["ids"]]
        if any(r is None or r["question"] != q for r, q in zip(rows, data["questions"].tolist())):
            logger.warning("[SI] faq_questions.npz does not match the KB - rebuild with build_kb; FAQ matching off")
            return
        self._faq_rows = rows
        self._faq_scheme = np.array([r["scheme_id"] for r in rows])
        self._faq_emb = np.ascontiguousarray(data["embeddings"], dtype=np.float32)

    @property
    def index_ready(self) -> bool:
        return self._emb is not None

    @property
    def faq_ready(self) -> bool:
        return self._faq_emb is not None

    def _model(self):
        try:
            return self._provider()
        except Exception as exc:
            logger.warning(f"[SI] embedder unavailable: {exc}")
            return None

    def embed_query(self, text: str) -> Optional[np.ndarray]:
        key = (text or "").strip().lower()
        if not key:
            return None
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        model = self._model()
        if model is None:
            return None
        vec = model.encode([C.QUERY_PREFIX + text.strip()], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        self._cache[key] = vec
        while len(self._cache) > C.QUERY_EMBED_CACHE_SIZE:
            self._cache.popitem(last=False)
        return vec

    def _mask(self, scheme_ids: Optional[Iterable[str]], sections: Optional[Iterable[str]]) -> np.ndarray:
        mask = np.ones(len(self._scheme_of_row), dtype=bool)
        if scheme_ids:
            mask &= np.isin(self._scheme_of_row, list(scheme_ids))
        if sections:
            mask &= np.isin(self._section_of_row, list(sections))
        return mask

    def search(self, qvec: Optional[np.ndarray], scheme_ids: Optional[Iterable[str]] = None,
               sections: Optional[Iterable[str]] = None, top_k: int = C.RAG_TOP_K,
               min_score: float = C.RAG_MIN_SCORE) -> List[Tuple[Dict[str, Any], float]]:
        if self._emb is None or qvec is None:
            return []
        rows = np.nonzero(self._mask(scheme_ids, sections))[0]
        if rows.size == 0:
            return []
        scores = np.einsum("ij,j->i", self._emb[rows], qvec)
        k = min(top_k, rows.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.repo.chunks[int(rows[i])], float(scores[i])) for i in top if scores[i] >= min_score]

    def scheme_scores(self, qvec: Optional[np.ndarray], scheme_ids: Iterable[str]) -> Dict[str, float]:
        """Best chunk similarity per scheme - a relevance signal for discovery ranking."""
        if self._emb is None or qvec is None:
            return {}
        rows = np.nonzero(self._mask(scheme_ids, None))[0]
        if rows.size == 0:
            return {}
        scores = np.einsum("ij,j->i", self._emb[rows], qvec)
        best: Dict[str, float] = {}
        for r, s in zip(rows.tolist(), scores.tolist()):
            sid = self._scheme_of_row[r]
            if s > best.get(sid, -1.0):
                best[sid] = s
        return best

    def faq_match(self, qvec: Optional[np.ndarray], scheme_ids: Optional[Iterable[str]] = None,
                  min_score: float = C.FAQ_MATCH_MIN_SCORE) -> Optional[Tuple[Dict[str, Any], float]]:
        """The FAQ whose question is closest to this one (optionally within some schemes), if >= min_score."""
        if self._faq_emb is None or qvec is None:
            return None
        rows = np.arange(len(self._faq_rows))
        if scheme_ids:
            rows = rows[np.isin(self._faq_scheme, list(scheme_ids))]
        if rows.size == 0:
            return None
        scores = np.einsum("ij,j->i", self._faq_emb[rows], qvec)
        best = int(np.argmax(scores))
        if scores[best] < min_score:
            return None
        return self._faq_rows[int(rows[best])], float(scores[best])
