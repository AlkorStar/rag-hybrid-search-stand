"""
Подготовка данных: создание коллекции + индексация.
Запуск: docker compose exec sut python -m app.data_preparer
"""

import json
import os
import sys

from app.embedder import Embedder
from app.searcher import Searcher


def main():
    docs_path = os.getenv("DOCS_PATH", "/app/data/documents_500.json")
    print(f"[prepare] Загрузка {docs_path}")
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"[prepare] Документов: {len(docs)}")

    emb = Embedder()
    searcher = Searcher(emb)
    searcher.create_collection(recreate=True)
    searcher.index_documents(docs)
    print("[prepare] Готово")


if __name__ == "__main__":
    main()