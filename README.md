# RAG Hybrid Search Stand

Экспериментальный стенд для сравнения Dense-only и Hybrid (RRF) поиска в RAG-системе.

## Что это

Воспроизводимый стенд для исследования влияния гибридного слияния результатов
поиска (BM25 + dense через Reciprocal Rank Fusion) на latency и Recall@5.

**Гипотеза:** гибрид добавляет latency (из-за второго запроса к Qdrant)
и при этом даёт прирост Recall@5.

## Требования

- Docker + Docker Compose
- Python 3.10+
- Пакеты анализа: `pip install -r analysis/requirements.txt`

## Быстрый старт

```bash
make all