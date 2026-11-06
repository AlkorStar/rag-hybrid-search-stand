"""
FastAPI-приложение SUT.

Эндпоинты:
  GET /health                    — проверка
  GET /search?query=...&mode=... — поиск (mode ∈ {dense, hybrid})
"""

import os
from fastapi import FastAPI, Query

from app.embedder import Embedder
from app.searcher import Searcher

app = FastAPI(title="RAG Search SUT", version="1.0.0")

print("[FastAPI] Инициализация модели...")
embedder = Embedder()
searcher = Searcher(embedder)
print("[FastAPI] SUT готов")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(query: str = Query(...), mode: str = Query("dense")):
    if mode == "dense":
        results, latency_ms = searcher.search_dense_only(query)
    elif mode in ("hybrid", "hybrid_rrf"):
        results, latency_ms = searcher.search_hybrid_rrf(query)
    else:
        return {"error": f"unknown mode: {mode}"}

    return {
        "mode": mode,
        "query": query,
        "latency_ms": round(latency_ms, 3),
        "results": results,
    }


@app.get("/admin/index")
def admin_index():
    """Пересоздать коллекцию и проиндексировать data/documents_500.json."""
    import json
    docs_path = os.getenv("DOCS_PATH", "/app/data/documents_500.json")
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    searcher.create_collection(recreate=True)
    searcher.index_documents(docs)
    return {"indexed": len(docs)}