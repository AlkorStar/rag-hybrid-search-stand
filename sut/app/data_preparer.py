"""
Скрипт для подготовки данных: создание коллекции и индексация документов.
Запускается ОДИН РАЗ перед экспериментом.
"""
import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedder import Embedder
from app.searcher import Searcher


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для эксперимента")
    parser.add_argument(
        "--docs",
        type=str,
        default="data/documents_500.json",
        help="Путь к файлу с документами (по умолчанию: data/documents_500.json)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ЭКСПЕРИМЕНТА")
    print("=" * 60)

    # 1. Загружаем модель
    print("\n[1/3] Загрузка эмбеддинг-модели...")
    embedder = Embedder()

    # 2. Загружаем документы
    print(f"\n[2/3] Загрузка документов из {args.docs}...")
    with open(args.docs, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Загружено {len(documents)} документов.")

    # 3. Создаём коллекцию и индексируем
    print("\n[3/3] Индексация в Qdrant...")
    searcher = Searcher(embedder)
    searcher.create_collection()
    searcher.index_documents(documents)

    print("\n" + "=" * 60)
    print("ГОТОВО! Данные загружены в Qdrant.")
    print(f"Загружено документов: {len(documents)}")
    print("Теперь можно запускать FastAPI и эксперимент.")
    print("=" * 60)


if __name__ == "__main__":
    main()