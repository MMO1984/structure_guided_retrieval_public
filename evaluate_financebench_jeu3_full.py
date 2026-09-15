#!/usr/bin/env python3
"""
JEU 3 a l'echelle complete (150 questions, 84 documents) : meme modele que
le run original, budgets de contexte reduits (Phase 1 TOC : 4000->2000
tokens, Phase 2b BM25+contagion : 3000->1500 tokens). Metriques decimales
(Jeu 1) uniquement. Cache incremental par financebench_id.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import retrieve_ours_financebench as ro
import metrics_financebench as mx
import jeu_metrics_financebench as jm

BASE_DIR = os.path.dirname(__file__)
CACHE_FILE = os.path.join(BASE_DIR, "results", "jeu3_full_cache.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "results", "jeu3_full_summary.json")

MAX_TOKENS_SECTIONS = 2000
MAX_TOKENS_BM25 = 1500
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

import threading
_lock_print = threading.Lock()


def process_one(q):
    target_file = q["doc_name"] + ".md"
    try:
        answer, context, section_sources, bloc_sources, tokens_used = ro.answer_question(
            q["question"], target_file,
            max_tokens_sections=MAX_TOKENS_SECTIONS, max_tokens_bm25=MAX_TOKENS_BM25)
    except Exception as e:
        answer, context, section_sources, bloc_sources, tokens_used = f"[ERROR: {e}]", "", [], [], 0

    gold_pages = set(ev["page_num"] for ev in (q.get("evidence_pages") or []))
    recall, precision = jm.decimal_page_metrics(section_sources, bloc_sources, gold_pages)
    acc = mx.answer_accuracy(q["question"], q["answer"], answer)

    return {
        "financebench_id": q["financebench_id"],
        "question": q["question"],
        "gold": q["answer"],
        "doc_name": q["doc_name"],
        "answer": answer,
        "context_chars": len(context),
        "gold_pages": sorted(gold_pages),
        "accuracy": acc,
        "page_recall_decimal": recall,
        "page_precision_decimal": precision,
        "tokens_used": tokens_used,
    }


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def main():
    with open(os.path.join(BASE_DIR, "all_150_questions.json"), encoding="utf-8") as f:
        all_q = json.load(f)

    cache = load_cache()
    todo = [q for q in all_q if q["financebench_id"] not in cache]
    print(f"Questions totales: {len(all_q)} | deja en cache: {len(all_q) - len(todo)} | a calculer: {len(todo)}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, q): q for q in todo}
            done = 0
            for future in as_completed(futures):
                q = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    r = {"financebench_id": q["financebench_id"], "question": q["question"],
                         "accuracy": 0.0, "page_recall_decimal": None, "page_precision_decimal": None,
                         "tokens_used": 0, "error": str(e)}
                cache[r["financebench_id"]] = r
                save_cache(cache)
                done += 1
                with _lock_print:
                    print(f"[{done}/{len(todo)}] acc={r['accuracy']:.0f} recall_dec={r['page_recall_decimal']} "
                          f"prec_dec={r['page_precision_decimal']} tokens={r['tokens_used']} -- {r['question'][:60]}", flush=True)

    results = [cache[q["financebench_id"]] for q in all_q]
    n = len(results)

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_questions": n,
        "max_tokens_sections": MAX_TOKENS_SECTIONS,
        "max_tokens_bm25": MAX_TOKENS_BM25,
        "accuracy": avg("accuracy"),
        "page_recall_decimal": avg("page_recall_decimal"),
        "page_precision_decimal": avg("page_precision_decimal"),
        "avg_tokens_used": avg("tokens_used"),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTATS JEU 3 COMPLET (150 questions) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
