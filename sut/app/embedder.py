"""
Модуль для работы с эмбеддинг-моделью.
Отвечает за загрузку модели и преобразование текста в вектор.
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Класс-обёртка над sentence-transformers.
    Использует модель multilingual-e5-large, которая поддерживает русский язык.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """
        Инициализация модели.

        Args:
            model_name: название модели с HuggingFace.
        """
        print(f"[Embedder] Загружаю модель {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Модель загружена. Размерность вектора: {self.dim}")

    def encode(self, text: str) -> list[float]:
        """
        Преобразует текст в вектор (список float).

        Для модели e5 нужно добавлять префикс 'query: ' к поисковым запросам
        и 'passage: ' к документам. Это особенность модели, которая улучшает
        качество поиска.

        Args:
            text: исходный текст.

        Returns:
            список float — векторное представление текста.
        """
        # Префикс нужен именно для multilingual-e5
        embedding = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return embedding.tolist()

    def encode_for_indexing(self, text: str) -> list[float]:
        """
        Преобразует текст документа в вектор для индексации.
        Для документов используется префикс 'passage: '.

        Args:
            text: текст документа.

        Returns:
            список float.
        """
        embedding = self.model.encode(f"passage: {text}", normalize_embeddings=True)
        return embedding.tolist()