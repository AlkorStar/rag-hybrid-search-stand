"""
plot.py — графики для отчёта ПР4.

Строит:
  docs/img/boxplot_p95.png   — распределение p95 по прогонам
  docs/img/ecdf_latency.png  — ECDF сырых латентностей
  docs/img/stability_p95.png — p95 по повторам (проверка дрейфа)

Запуск: python analysis/plot.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
RAW = ROOT / CFG["paths"]["results_raw"]
OUT = ROOT / CFG["paths"]["results_processed"]
IMG = ROOT / CFG["paths"]["figures"]
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def main():
    df = pd.read_csv(OUT / "runs.csv")

    # 1. Boxplot p95
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="impl", y="p95_lat", color="lightsteelblue", width=0.5, fliersize=0)
    sns.stripplot(data=df, x="impl", y="p95_lat", color="black", size=7, jitter=0.15)
    plt.ylabel("p95 Latency за прогон, мс")
    plt.xlabel("Конфигурация")
    plt.title("Распределение p95 по прогонам")
    plt.tight_layout()
    plt.savefig(IMG / "boxplot_p95.png", dpi=150)
    plt.close()

    # 2. ECDF по всем запросам
    plt.figure(figsize=(8, 5))
    for impl in ["dense", "hybrid_rrf"]:
        lats = []
        for m in RAW.glob(f"{impl}_*_manifest.json"):
            r = json.loads(m.with_name(m.name.replace("_manifest", "_raw")).read_text(encoding="utf-8"))
            lats.extend(r["latencies_ms"])
        if not lats:
            continue
        lats = np.sort(np.array(lats))
        y = np.arange(1, len(lats) + 1) / len(lats)
        plt.plot(lats, y, linewidth=2.5, label=impl)
    plt.xlabel("Latency, мс (сырые HTTP-запросы)")
    plt.ylabel("ECDF")
    plt.title("ECDF латентностей")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / "ecdf_latency.png", dpi=150)
    plt.close()

    # 3. Стабильность по повторам
    plt.figure(figsize=(8, 5))
    for impl, g in df.groupby("impl"):
        g = g.sort_values("rep")
        plt.plot(g["rep"] + 1, g["p95_lat"], marker="o", linewidth=2, label=impl)
    plt.xlabel("Номер повторения")
    plt.ylabel("p95, мс")
    plt.title("p95 по повторам (проверка дрейфа)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / "stability_p95.png", dpi=150)
    plt.close()

    print(f"Графики сохранены в {IMG}")


if __name__ == "__main__":
    main()