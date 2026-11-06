# Makefile
# Запуск всего эксперимента одной командой.
#   make all      — полный цикл
#   make up       — поднять контейнеры
#   make prepare  — индексировать данные
#   make smoke    — smoke-test
#   make aa-test  — A/A-тест
#   make experiment — основной прогон
#   make analyze  — статистика
#   make plot     — графики
#   make clean    — очистить результаты

SHELL := /bin/bash

.PHONY: help up down prepare smoke aa-test experiment analyze plot all clean

help:
	@echo "Цели:"
	@echo "  up          - поднять контейнеры (sut + qdrant)"
	@echo "  down        - остановить"
	@echo "  prepare     - индексировать документы"
	@echo "  smoke       - smoke-test"
	@echo "  aa-test     - A/A-тест"
	@echo "  experiment  - основной прогон"
	@echo "  analyze     - статистика"
	@echo "  plot        - графики"
	@echo "  all         - всё по порядку"

up:
	docker compose up -d --build

down:
	docker compose down

prepare:
	docker compose exec -T sut python -m app.data_preparer

smoke:
	@echo "health:"
	@curl -fsS http://localhost:8000/health && echo
	@echo "search:"
	@curl -fsS "http://localhost:8000/search?query=test&mode=dense" | head -c 200; echo
	@echo "qdrant:"
	@curl -fsS http://localhost:6333 > /dev/null && echo "OK"

aa-test:
	python3 experiment/aa_test.py

experiment:
	python3 experiment/run_experiment.py

analyze:
	python3 analysis/analyze.py

plot:
	python3 analysis/plot.py

all: up
	@echo "Ждём запуска SUT (~40 сек, первая загрузка модели)..."
	@sleep 40
	$(MAKE) prepare
	$(MAKE) smoke
	$(MAKE) aa-test
	$(MAKE) experiment
	$(MAKE) analyze
	$(MAKE) plot

clean:
	rm -f experiment/results/raw/*.json
	rm -f experiment/results/processed/*.csv experiment/results/processed/*.json
	rm -f docs/img/*.png