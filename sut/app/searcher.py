"""
Поисковый движок. Две конфигурации:
  - search_dense_only  — Конфигурация А (только dense)
  - search_hybrid_rrf  — Конфигурация Б (dense + sparse, слияние RRF в Python)

Все замеры времени делаются здесь, в SUT, чтобы в latency входила
именно логика поиска (а не сетевые задержки до клиента).
"""

import os
import time
from typing import List, Dict, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.embedder import Embedder


class Searcher:
    COLLECTION = "documents"

    def __init__(self, embedder: Embedder):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        print(f"[Searcher] Подключение к Qdrant {host}:{port}")
        self.embedder = embedder
        self.client = QdrantClient(host=host, port=port, timeout=30)

    # ---------- подготовка ----------

    def create_collection(self, recreate: bool = True):
        exists = self.client.collection_exists(self.COLLECTION)
        if exists and recreate:
            print(f"[Searcher] Удаляю существующую коллекцию {self.COLLECTION}")
            self.client.delete_collection(self.COLLECTION)
        if not exists or recreate:
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams())
                },
            )
            print(f"[Searcher] Коллекция {self.COLLECTION} создана")

    def index_documents(self, docs: List[Dict], batch_size: int = 64):
        print(f"[Searcher] Индексация {len(docs)} документов")
        texts = [d["text"] for d in docs]
        dense_vectors = self.embedder.encode_batch(texts, prefix="passage")

        points = []
        for i, doc in enumerate(docs):
            sparse_vector = self._text_to_sparse_vector(doc["text"])
            points.append(
                PointStruct(
                    id=doc["id"],
                    vector={
                        "dense": dense_vectors[i].tolist(),
                        "sparse": sparse_vector,
                    },
                    payload={"text": doc["text"], "doc_id": doc["id"]},
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.COLLECTION, points=batch, wait=True)
            print(f"  upsert {i + len(batch)}/{len(points)}")
        print("[Searcher] Индексация завершена")

    # ---------- поиск ----------

    def search_dense_only(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()
        qvec = self.embedder.encode_query(query)
        res = self.client.query_points(
            collection_name=self.COLLECTION,
            query=qvec,
            limit=top_k,
            using="dense",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return self._format(res.points), elapsed

    def search_hybrid_rrf(self, query: str, top_k: int = 5, candidates: int = 20) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()
        qvec = self.embedder.encode_query(query)
        dense_res = self.client.query_points(
            collection_name=self.COLLECTION, query=qvec, limit=candidates, using="dense"
        )
        sparse_vec = self._text_to_sparse_vector(query)
        sparse_res = self.client.query_points(
            collection_name=self.COLLECTION, query=sparse_vec, limit=candidates, using="sparse"
        )
        merged = self._rrf(dense_res.points, sparse_res.points, k=60)
        merged_sorted = sorted(merged, key=lambda x: x["score"], reverse=True)[:top_k]
        elapsed = (time.perf_counter() - t0) * 1000
        return merged_sorted, elapsed

    # ---------- утилиты ----------

    @staticmethod
    def _text_to_sparse_vector(text: str) -> SparseVector:
        """Простой sparse: TF по хэшу слов (без IDF — как в НИР 2)."""
        word_counts: Dict[int, float] = {}
        for word in text.lower().split():
            wid = abs(hash(word)) % 1_000_000
            word_counts[wid] = word_counts.get(wid, 0.0) + 1.0
        indices = list(word_counts.keys())
        values = [word_counts[i] for i in indices]
        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _rrf(dense, sparse, k: int = 60) -> List[Dict]:
        scores: Dict[int, float] = {}
        info: Dict[int, Dict] = {}
        for rank, doc in enumerate(dense, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)
            info.setdefault(doc.id, {"id": doc.id, "text": (doc.payload or {}).get("text", "")})
        for rank, doc in enumerate(sparse, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)
            info.setdefault(doc.id, {"id": doc.id, "text": (doc.payload or {}).get("text", "")})
        out = []
        for doc_id, s in scores.items():
            out.append({"id": info[doc_id]["id"], "text": info[doc_id]["text"], "score": round(s, 6)})
        return out

    @staticmethod
    def _format(points) -> List[Dict]:
        return [
            {
                "id": p.id,
                "text": (p.payload or {}).get("text", ""),
                "score": round(float(p.score), 6),
            }
            for p in points
        ]