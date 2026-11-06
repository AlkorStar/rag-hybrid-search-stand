"""
run_experiment.py — оркестратор основного эксперимента.

Что делает:
  1. Читает configs/experiment.yaml
  2. Проверяет SUT через /health
  3. Делает прогрев (warmup_queries на режим)
  4. Строит план прогонов (10 повторов × 2 конфигурации), перемешивает
  5. Для каждого прогона проходит 50 запросов, сохраняет latency и recall
  6. Пишет на прогон два файла: {run_id}_raw.json и {run_id}_manifest.json

Запуск:
    python experiment/run_experiment.py
или через Makefile:
    make experiment
"""

import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiment.yaml"


def sh(*args) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return ""


def git_info() -> dict:
    return {
        "commit": sh("git", "rev-parse", "HEAD"),
        "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(sh("git", "status", "--porcelain")),
    }


def host_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }


def docker_images() -> list:
    raw = sh("docker", "compose", "images", "--format", "json")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def wait_for_sut(base_url: str, timeout_s: int = 90) -> bool:
    print(f"[run] Ждём SUT: {base_url}/health")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                print("[run] SUT доступен")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("[run] ERROR: SUT не поднялся")
    return False


def one_request(base_url: str, query: str, mode: str, timeout: int):
    t0 = time.perf_counter()
    try:
        r = requests.get(
            f"{base_url}/search", params={"query": query, "mode": mode}, timeout=timeout
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return None, [], f"HTTP {r.status_code}"
        return elapsed, [x["id"] for x in r.json().get("results", [])], None
    except Exception as e:
        return None, [], str(e)


def recall_at_k(found: list, relevant: list, k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(found[:k]) & set(relevant)) / len(relevant)


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    seed = cfg["experiment"]["seed"]
    reps = cfg["experiment"]["repetitions"]
    warmup_n = cfg["experiment"]["warmup_queries"]
    timeout = cfg["experiment"]["timeout_s"]
    top_k = cfg["data"]["top_k"]
    base_url = cfg["sut"]["base_url"]
    modes = cfg["factors"]["impl"]

    raw_dir = ROOT / cfg["paths"]["results_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads((ROOT / cfg["data"]["queries"]).read_text(encoding="utf-8"))

    # 1. Проверка SUT
    if not wait_for_sut(base_url):
        sys.exit(1)

    # 2. Прогрев
    print(f"[run] Прогрев ({warmup_n} запросов × {len(modes)} режимов)")
    warmup_texts = ["тест", "поиск", "Docker"][:warmup_n]
    for q in warmup_texts:
        for mode in modes:
            one_request(base_url, q, mode, timeout)
    print("[run] Прогрев завершён")

    # 3. План прогонов
    plan = [{"impl": m, "rep": r} for r in range(reps) for m in modes]
    if cfg["experiment"]["randomize_order"]:
        random.Random(seed).shuffle(plan)

    common_manifest = {
        "config": cfg,
        "git": git_info(),
        "host": host_info(),
        "images": docker_images(),
        "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": cfg["experiment"]["name"],
    }

    print(f"[run] Всего прогонов: {len(plan)}")

    for i, p in enumerate(plan, 1):
        run_id = f"{p['impl']}_r{p['rep']:02d}_{uuid.uuid4().hex[:6]}"
        print(f"\n[run] [{i}/{len(plan)}] {run_id}")

        latencies, recalls, errors = [], [], []
        for q in queries:
            lat, ids, err = one_request(base_url, q["query"], p["impl"], timeout)
            if err:
                errors.append({"query_id": q["id"], "error": err})
                continue
            latencies.append(lat)
            recalls.append(recall_at_k(ids, q["relevant_ids"], top_k))

        raw = {
            "run_id": run_id,
            "params": {"impl": p["impl"]},
            "repetition": p["rep"],
            "latencies_ms": latencies,
            "recalls": recalls,
            "errors": errors,
        }
        manifest = {
            **common_manifest,
            "run_id": run_id,
            "params": p["impl"],
            "repetition": p["rep"],
            "n_queries": len(queries),
            "n_successful": len(latencies),
            "n_errors": len(errors),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

        (raw_dir / f"{run_id}_raw.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (raw_dir / f"{run_id}_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if latencies:
            print(f"  n={len(latencies)}  mean={sum(latencies)/len(latencies):.2f} ms")

    print(f"\n[run] Готово. Сырые данные в {raw_dir}")


if __name__ == "__main__":
    main()