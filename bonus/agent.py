"""HybridMemoryAgent — episodic memory (vector) + stable profile (feature store).

Bonus challenge for Lab 19 ("Build Your Own AI Memory").

Patterns reused from the core lab:
  * embedding           -> app.embeddings.Embedder (NB1)
  * hybrid RRF fusion   -> vector rank + keyword rank, k=60 (NB2)
  * multi-tenant safety -> every point is namespaced by user_id, recalls are
                           filtered by it (NB5/NB7 lesson: never forget the filter)
  * profile features    -> Feast online store, degrades to a deterministic
                           per-user profile when Feast is not applied yet
                           (NB4 / NB6 build_context pattern)

Deliberately zero-LLM: recall() returns an *assembled context string* that an
LLM would consume, which keeps the POC free of API keys.
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
import warnings
from pathlib import Path

# Bootstrap repo root so `python bonus/demo.py` works from any cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from qdrant_client import QdrantClient, models

from app.embeddings import Embedder

warnings.filterwarnings("ignore")  # in-memory Qdrant warns about payload indexes

_COLLECTION = "bonus_memory"
RRF_K = 60  # same constant as NB2
_FALLBACK_AFFINITIES = ["cloud", "ai_ml", "security", "database", "networking", "devops"]


class HybridMemoryAgent:
    """Personal assistant memory: what the user has read + who the user is.

    remember(text)  -> chunks, embeds, stores as episodic memory (per user)
    recall(query)   -> profile features + hybrid top-K memories -> context string
    """

    def __init__(self, user_id: str = "u_001", feature_store=None,
                 embedder=None, client: QdrantClient | None = None,
                 top_k: int = 3) -> None:
        self.user_id = user_id
        self.top_k = top_k
        self.embedder = embedder or Embedder()
        self.feature_store = feature_store  # optional Feast FeatureStore
        self.client = client or QdrantClient(":memory:")
        self._next_id = 0
        self._memory_ts = 0.0

        names = {c.name for c in self.client.get_collections().collections}
        if _COLLECTION in names:
            self.client.delete_collection(_COLLECTION)
        self.client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        # Keyword index on the tenant key: recall MUST filter by user_id
        # (OWASP LLM08 — a forgotten filter leaks one user's memories to another).
        try:
            self.client.create_payload_index(
                _COLLECTION, "user_id", field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:  # pragma: no cover - in-memory mode may no-op
            pass

    # ── write path ───────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str | None = None) -> None:
        """Add a piece of episodic memory for this user (chunked + embedded)."""
        user_id = user_id or self.user_id
        for chunk in self._chunk(text):
            vec = np.asarray(next(self.embedder.embed([chunk])), dtype=np.float32).tolist()
            self.client.upsert(
                collection_name=_COLLECTION,
                points=[models.PointStruct(
                    id=self._next_id,
                    vector=vec,
                    payload={"user_id": user_id, "text": chunk,
                             "ts": self._memory_ts},
                )],
            )
            self._next_id += 1
            self._memory_ts += 1.0  # deterministic ordering without wall-clock

    @staticmethod
    def _chunk(text: str, max_chars: int = 400) -> list[str]:
        """Chunk on semantic breaks (paragraph -> sentence), not fixed tokens.

        This is Architecture Decision #1 in ARCHITECTURE.md.
        """
        text = text.strip()
        if not text:
            return []
        chunks: list[str] = []
        for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
            if len(para) <= max_chars:
                chunks.append(para)
                continue
            buf = ""
            for sent in re.split(r"(?<=[.?!])\s+", para):
                if len(buf) + len(sent) <= max_chars:
                    buf = f"{buf} {sent}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = sent
            if buf:
                chunks.append(buf)
        return chunks

    # ── read path ────────────────────────────────────────────────────────
    def recall(self, query: str, user_id: str | None = None) -> str:
        """Assemble profile + top-K memories for `query` into a context string."""
        user_id = user_id or self.user_id
        profile = self._profile(user_id)
        memories = self._hybrid_top_k(query, user_id, self.top_k)
        return self._assemble_context(profile, memories)

    def _user_points(self, user_id: str) -> list:
        qf = models.Filter(must=[models.FieldCondition(
            key="user_id", match=models.MatchValue(value=user_id))])
        pts, _ = self.client.scroll(
            collection_name=_COLLECTION, scroll_filter=qf,
            limit=10_000, with_payload=True)
        return sorted(pts, key=lambda p: p.payload["ts"])

    def _hybrid_top_k(self, query: str, user_id: str, k: int) -> list[tuple[int, str, float]]:
        """RRF fusion of vector rank + keyword rank over THIS user's memories."""
        points = self._user_points(user_id)
        if not points:
            return []

        qv = np.asarray(next(self.embedder.embed([query])), dtype=np.float32).tolist()
        qf = models.Filter(must=[models.FieldCondition(
            key="user_id", match=models.MatchValue(value=user_id))])
        vec_hits = self.client.query_points(
            collection_name=_COLLECTION, query=qv, query_filter=qf, limit=len(points)).points
        vec_rank = {p.id: i for i, p in enumerate(vec_hits)}  # rank 0 = best

        q_toks = set(self._tokenize(query))
        kw_scores = {
            p.id: len(q_toks & set(self._tokenize(p.payload["text"]))) / max(1, len(q_toks))
            for p in points
        }
        kw_rank = {pid: i for i, pid in enumerate(
            sorted(kw_scores, key=lambda pid: -kw_scores[pid]))}

        text_of = {p.id: p.payload["text"] for p in points}
        fused = {}
        for p in points:
            r_vec = vec_rank.get(p.id, len(points)) + 1   # 1-based rank
            r_kw = kw_rank.get(p.id, len(points)) + 1
            fused[p.id] = 1.0 / (RRF_K + r_vec) + 1.0 / (RRF_K + r_kw)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(pid, text_of[pid], score) for pid, score in ranked]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Whitespace split: acceptable for a VN+EN mix at POC scale.
        # (See ARCHITECTURE.md decision on pyvi / underthesea.)
        return text.lower().split()

    def _profile(self, user_id: str) -> dict:
        """Try Feast online store first; fall back to a stable per-user hash."""
        if self.feature_store is not None:
            try:
                f = self.feature_store.get_online_features(
                    features=[
                        "user_profile_features:topic_affinity",
                        "user_profile_features:preferred_language",
                        "user_profile_features:reading_speed_wpm",
                        "query_velocity_features:queries_last_hour",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                return {
                    "affinity": (f.get("topic_affinity") or [None])[0],
                    "language": (f.get("preferred_language") or [None])[0],
                    "wpm": (f.get("reading_speed_wpm") or [None])[0],
                    "queries_last_hour": (f.get("queries_last_hour") or [None])[0],
                    "source": "feast",
                }
            except Exception:
                pass  # Feast not applied / not materialized -> fallback below

        h = hashlib.blake2b(f"bonus:{user_id}".encode("utf-8"), digest_size=8).digest()
        a = int.from_bytes(h[:4], "big") % len(_FALLBACK_AFFINITIES)
        return {
            "affinity": _FALLBACK_AFFINITIES[a],
            "language": "vi",
            "wpm": 180 + int.from_bytes(h[4:6], "big") % 120,
            "queries_last_hour": int.from_bytes(h[6:], "big") % 25,
            "source": "deterministic-fallback",
        }

    def _assemble_context(self, profile: dict, memories: list) -> str:
        lang = profile["language"] or "vi"
        wpm = profile["wpm"]
        affinity = profile["affinity"] or "unknown"
        qph = profile["queries_last_hour"]
        src = profile.get("source", "?")

        lines = [
            f"[PROFILE] user={self.user_id} (source: {src})",
            f"  - topic_affinity: {affinity}",
            f"  - preferred_language: {lang}",
            f"  - reading_speed: {wpm} wpm",
            f"  - queries_last_hour: {qph}",
        ]
        lines.append("[EPISODIC MEMORIES] (RRF hybrid, top-K):")
        if memories:
            for i, (_, text, score) in enumerate(memories, 1):
                lines.append(f"  {i}. [{score:.3f}] {text[:170]}")
        else:
            lines.append("  (chưa có memory — chạy remember() trước)")
        return "\n".join(lines)
