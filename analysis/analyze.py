"""
analyze.py — превращает сырые прогоны в выводы.

Ключевой принцип: единица наблюдения = ОДИН ПРОГОН.
Поэтому n = 10 на конфигурацию, а не 500 запросов.
Иначе p-value будет искусственно занижен (псевдорепликация).

Запуск: python analysis/analyze.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
RAW = ROOT / CFG["paths"]["results_raw"]
OUT = ROOT / CFG["paths"]["results_processed"]
OUT.mkdir(parents=True, exist_ok=True)

BOOT_N = CFG["analysis"]["bootstrap_n"]
ALPHA = CFG["analysis"]["alpha"]


def bootstrap_ci(x, stat_fn=np.median, n_boot=BOOT_N, alpha=ALPHA, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    samples = np.array([stat_fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for i, idx in enumerate(order):
        prev = min(prev, (n - i) * pvals[idx])
        adj[idx] = prev
    return adj


def effect_label(delta):
    d = abs(delta)
    if d < 0.147:
        return "пренебрежимо малый"
    if d < 0.330:
        return "малый"
    if d < 0.474:
        return "средний"
    return "большой"


def main():
    # --- собираем строки по прогонам ---
    rows = []
    for manifest_path in sorted(RAW.glob("*_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_path = manifest_path.with_name(manifest_path.name.replace("_manifest", "_raw"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))

        lats = np.array(raw["latencies_ms"], dtype=float)
        recalls = np.array(raw["recalls"], dtype=float)
        if len(lats) == 0:
            continue

        rows.append(
            {
                "run_id": manifest["run_id"],
                "impl": manifest["params"],
                "rep": manifest["repetition"],
                "n": len(lats),
                "mean_lat": lats.mean(),
                "median_lat": np.median(lats),
                "p95_lat": np.quantile(lats, 0.95),
                "min_lat": lats.min(),
                "max_lat": lats.max(),
                "mean_recall": recalls.mean(),
                "errors": manifest["n_errors"],
            }
        )

    if not rows:
        print("Нет данных: сначала запусти make experiment")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)

    print("\n=== Прогоны (единица = прогон) ===")
    print(df[["impl", "rep", "p95_lat", "mean_lat", "mean_recall"]].to_string(index=False))

    # --- сводка по конфигурациям ---
    summary_rows = []
    for impl, g in df.groupby("impl"):
        p95 = g["p95_lat"].values
        lo, hi = bootstrap_ci(p95)
        summary_rows.append(
            {
                "impl": impl,
                "n_runs": int(len(g)),
                "p95_median_ms": float(np.median(p95)),
                "p95_iqr_ms": float(np.quantile(p95, 0.75) - np.quantile(p95, 0.25)),
                "p95_ci_lo_ms": lo,
                "p95_ci_hi_ms": hi,
                "p95_mean_ms": float(p95.mean()),
                "p95_std_ms": float(p95.std(ddof=1)),
                "mean_recall": float(g["mean_recall"].mean()),
                "errors_total": int(g["errors"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    print("\n=== Сводка ===")
    print(summary.to_string(index=False))

    # --- сравнение ---
    a = df.loc[df["impl"] == "dense", "p95_lat"].values
    b = df.loc[df["impl"] == "hybrid_rrf", "p95_lat"].values

    if len(a) >= 3 and len(b) >= 3:
        u, p_raw = stats.mannwhitneyu(a, b, alternative="two-sided")
        p_adj = float(holm([p_raw])[0])
        delta = cliffs_delta(a, b)
        comparison = {
            "metric": "p95_latency_ms",
            "n_dense": int(len(a)),
            "n_hybrid": int(len(b)),
            "median_dense": float(np.median(a)),
            "median_hybrid": float(np.median(b)),
            "abs_diff_ms": float(np.median(b) - np.median(a)),
            "rel_diff_pct": float((np.median(b) / np.median(a) - 1) * 100),
            "mann_whitney_U": float(u),
            "p_raw": float(p_raw),
            "p_holm": p_adj,
            "cliffs_delta": delta,
            "cliffs_effect_label": effect_label(delta),
        }
        print("\n=== Сравнение Dense vs Hybrid (единица = прогон) ===")
        for k, v in comparison.items():
            if isinstance(v, float):
                print(f"  {k:<22} = {v:.4f}")
            else:
                print(f"  {k:<22} = {v}")
    else:
        comparison = {"error": "слишком мало прогонов"}
        print("\n[WARN] Слишком мало прогонов для статистики (нужно >= 3 на группу)")

    # --- сохраняем общий JSON ---
    report = {
        "summary": summary.to_dict(orient="records"),
        "comparison": comparison,
    }
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {OUT / 'analysis.json'}")


if __name__ == "__main__":
    main()