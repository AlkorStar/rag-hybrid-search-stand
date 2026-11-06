"""
FastAPI-приложение для поиска.
Предоставляет два эндпоинта:
- GET /search?query=...&mode=dense   — Конфигурация А
- GET /search?query=...&mode=hybrid  — Конфигурация Б
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from app.embedder import Embedder
from app.searcher import Searcher

# Создаём приложение
app = FastAPI(
    title="RAG Search Experiment",
    description="Экспериментальный поисковый сервис для НИР 2",
    version="1.0.0",
)

# Загружаем модель ОДИН РАЗ при старте сервера
print("[FastAPI] Инициализация модели...")
embedder = Embedder()
searcher = Searcher(embedder)
print("[FastAPI] Сервер готов к работе.")


@app.get("/search")
def search(
    query: str = Query(..., description="Поисковый запрос"),
    mode: str = Query("dense", description="Режим поиска: dense или hybrid"),
):
    """
    Поисковый эндпоинт.

    Параметры:
    - query: поисковый запрос (обязательно).
    - mode: 'dense' для Конфигурации А, 'hybrid' для Конфигурации Б.

    Возвращает:
    - results: список найденных документов.
    - latency_ms: время выполнения в миллисекундах.
    - mode: использованный режим поиска.
    """
    if mode == "dense":
        results, latency_ms = searcher.search_dense_only(query)
    elif mode == "hybrid":
        results, latency_ms = searcher.search_hybrid_rrf(query)
    else:
        return {
            "error": f"Неизвестный режим: {mode}. Используйте 'dense' или 'hybrid'."
        }

    return {
        "mode": mode,
        "query": query,
        "latency_ms": round(latency_ms, 2),
        "results_count": len(results),
        "results": results,
    }


@app.get("/health")
def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}