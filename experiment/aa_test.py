"""
A/A-тест: 20 прогонов Dense, случайное разбиение 10/10, проверка,
что медианы p95 не различаются значимо.

Зачем: доказать, что стенд не даёт ложных срабатываний.

Запуск: python experiment/aa_test.py
"""

import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))


def one(base_url, query, mode, timeout):
    t0 = time.perf_counter()
    requests.get(f"{base_url}/search", params={"query": query, "mode": mode}, timeout=timeout)
    return (time.perf_counter() - t0) * 1000


def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def main():
    cfg = CFG
    seed = cfg["experiment"]["seed"]
    base_url = cfg["sut"]["base_url"]
    timeout = cfg["experiment"]["timeout_s"]
    queries = json.loads((ROOT / cfg["data"]["queries"]).read_text(encoding="utf-8"))
    N = 20

    print(f"[aa] {N} прогонов Dense")
    runs = []
    for i in range(N):
        lats = [one(base_url, q["query"], "dense", timeout) for q in queries]
        p95 = sorted(lats)[int(len(lats) * 0.95)]
        runs.append(p95)
        print(f"  run {i+1:02d}/{N}: p95 = {p95:.2f} ms")

    rng = random.Random(seed)
    rng.shuffle(runs)
    A, B = runs[: N // 2], runs[N // 2 :]

    u, p = stats.mannwhitneyu(A, B, alternative="two-sided")
    delta = cliffs_delta(A, B)

    verdict = "PASS" if p > 0.05 else "FAIL"
    print("\n=== A/A-тест ===")
    print(f"Медиана A = {statistics.median(A):.2f} ms (n={len(A)})")
    print(f"Медиана B = {statistics.median(B):.2f} ms (n={len(B)})")
    print(f"Mann-Whitney U = {u:.1f}, p = {p:.4f}")
    print(f"Cliff's δ = {delta:+.3f}")
    print(f"Вердикт: {verdict}")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "N": N,
        "group_A": A,
        "group_B": B,
        "median_A": statistics.median(A),
        "median_B": statistics.median(B),
        "mann_whitney_U": float(u),
        "p_value": float(p),
        "cliffs_delta": float(delta),
        "verdict": verdict,
    }
    out_dir = ROOT / cfg["paths"]["results_raw"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aa_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[aa] Сохранено: {out_dir / 'aa_test.json'}")


if __name__ == "__main__":
    main()