"""
Модуль эмбеддингов. Обёртка над sentence-transformers.

Модель intfloat/multilingual-e5-large требует префиксов:
  - 'query: '    для поисковых запросов
  - 'passage: '  для документов
Без них качество падает.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    MODEL_NAME = "intfloat/multilingual-e5-large"
    EMBEDDING_DIM = 1024

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.MODEL_NAME
        print(f"[Embedder] Загрузка модели {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Готово. Размерность: {self.dim}")

    def encode_query(self, text: str) -> list[float]:
        emb = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return emb.tolist()

    def encode_passage(self, text: str) -> list[float]:
        emb = self.model.encode(f"passage: {text}", normalize_embeddings=True)
        return emb.tolist()

    def encode_batch(self, texts: list[str], prefix: str = "passage") -> np.ndarray:
        return self.model.encode(
            [f"{prefix}: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )