"""
Модуль поиска. Реализует два режима:
- Конфигурация А: Dense-only (только векторный поиск)
- Конфигурация Б: Hybrid RRF (векторный + BM25 + слияние в Python)
"""
import time
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    ScoredPoint,
    SparseVector,
    SparseIndexParams,
    SparseVectorParams,
    Distance,
    VectorParams,
    PointStruct,
)
from app.embedder import Embedder


class Searcher:
    """
    Поисковый движок. Обёртка над Qdrant для выполнения поиска
    в двух режимах.
    """

    def __init__(self, embedder: Embedder, host: str = "localhost", port: int = 6333):
        self.embedder = embedder
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "documents"

    def create_collection(self):
        """Создаёт коллекцию в Qdrant."""
        if self.client.collection_exists(self.collection_name):
            print(f"[Searcher] Коллекция '{self.collection_name}' уже существует. Удаляю...")
            self.client.delete_collection(self.collection_name)

        print(f"[Searcher] Создаю коллекцию '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedder.dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams()
                )
            },
        )
        print("[Searcher] Коллекция создана.")

    def index_documents(self, documents: List[Dict]):
        """Индексирует документы в Qdrant."""
        print(f"[Searcher] Индексирую {len(documents)} документов...")
        points = []

        for i, doc in enumerate(documents):
            if (i + 1) % 100 == 0:
                print(f"  Прогресс: {i + 1}/{len(documents)} документов...")

            dense_vector = self.embedder.encode_for_indexing(doc["text"])
            sparse_vector = self._text_to_sparse_vector(doc["text"])

            point = PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload={
                    "text": doc["text"],
                    "doc_id": doc["id"],
                },
            )
            points.append(point)

            if len(points) >= 100:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        print(f"[Searcher] Индексация завершена. Загружено {len(documents)} документов.")

    def search_dense_only(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """КОНФИГУРАЦИЯ А: Только плотный (семантический) поиск."""
        start_time = time.perf_counter()

        query_vector = self.embedder.encode(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            using="dense",
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return self._format_results(results.points), elapsed_ms

    def search_hybrid_rrf(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """КОНФИГУРАЦИЯ Б: Гибридный поиск с RRF-слиянием."""
        start_time = time.perf_counter()

        query_vector = self.embedder.encode(query)

        # Dense поиск
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=20,
            using="dense",
        )

        # Sparse (BM25) поиск
        sparse_vector = self._text_to_sparse_vector(query)
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            limit=20,
            using="sparse",
        )

        # RRF-слияние
        merged = self._rrf_fusion(dense_results.points, sparse_results.points, k=60)

        final_results = sorted(merged, key=lambda x: x["score"], reverse=True)[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return final_results, elapsed_ms

    def _text_to_sparse_vector(self, text: str) -> SparseVector:
        """Создаёт sparse вектор из текста."""
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_id = abs(hash(word)) % 1000000
            word_counts[word_id] = word_counts.get(word_id, 0) + 1

        indices = list(word_counts.keys())
        values = [float(v) for v in word_counts.values()]

        return SparseVector(indices=indices, values=values)

    def _rrf_fusion(
        self,
        dense_results: List[ScoredPoint],
        sparse_results: List[ScoredPoint],
        k: int = 60,
    ) -> List[Dict]:
        """RRF-слияние."""
        scores = {}
        docs_info = {}

        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc.id
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in docs_info:
                docs_info[doc_id] = {
                    "id": doc_id,
                    "text": doc.payload.get("text", "") if doc.payload else "",
                }

        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc.id
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in docs_info:
                docs_info[doc_id] = {
                    "id": doc_id,
                    "text": doc.payload.get("text", "") if doc.payload else "",
                }

        results = []
        for doc_id, rrf_score in scores.items():
            info = docs_info[doc_id]
            results.append({
                "id": info["id"],
                "text": info["text"],
                "score": round(rrf_score, 4),
            })

        return results

    def _format_results(self, qdrant_results: List[ScoredPoint]) -> List[Dict]:
        """Форматирует ответ Qdrant."""
        formatted = []
        for doc in qdrant_results:
            formatted.append({
                "id": doc.id,
                "text": doc.payload.get("text", "") if doc.payload else "",
                "score": round(doc.score, 4),
            })
        return formatted