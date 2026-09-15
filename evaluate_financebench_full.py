#!/usr/bin/env python3
"""
Evaluation complete (150 questions publiques, 84 documents) : notre methode
(version de base originale) sur FinanceBench, avec Page Recall/Precision
(Jeu 1 decimal + Jeu 2 page entiere) et Answer Accuracy (original + variante
Jeu 2 pages entieres). Cache incremental par financebench_id (comme pour
(comme pour nos autres pipelines de recherche) -- aucune question deja calculee n'est rejouee.
"""

import os
import re
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import retrieve_ours_financebench as ro
import metrics_financebench as mx
import jeu_metrics_financebench as jm

BASE_DIR = os.path.dirname(__file__)
CORPUS_MD_DIR = os.path.join(BASE_DIR, "corpus_md")
# Overrides optionnels (non utilises en fonctionnement normal) pour rejouer l'evaluation sur un
# sous-ensemble de questions dans un cache/summary separe, sans toucher au cache de production
# (ex. test de validation sur 15 questions avant de relancer les 150).
CACHE_FILE = os.environ.get("CACHE_FILE_OVERRIDE") or os.path.join(BASE_DIR, "results", "full_cache.json")
SUMMARY_FILE = os.environ.get("SUMMARY_FILE_OVERRIDE") or os.path.join(BASE_DIR, "results", "full_summary.json")
SAMPLE_IDS_FILE = os.environ.get("SAMPLE_IDS_FILE")

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

with open(os.path.join(BASE_DIR, "rag_index_financebench.pkl"), "rb") as f:
    RAG_INDEX = pickle.load(f)
BLOC_PAGES = RAG_INDEX["bloc_pages"]

_PAGE_POSITIONS_CACHE = {}
_lock_print = __import__("threading").Lock()


def get_page_positions(fname):
    if fname not in _PAGE_POSITIONS_CACHE:
        positions = []
        path = os.path.join(CORPUS_MD_DIR, fname)
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                m = re.match(r"<!-- page:(\d+) -->", line.strip())
                if m:
                    positions.append((i, int(m.group(1))))
        _PAGE_POSITIONS_CACHE[fname] = positions
    return _PAGE_POSITIONS_CACHE[fname]


def page_at_line(positions, line_no):
    page = positions[0][1] if positions else 0
    for ln, pg in positions:
        if ln <= line_no:
            page = pg
        else:
            break
    return page


def pages_for_section_sources(section_sources):
    pages = set()
    for fname, line_start, line_end in section_sources:
        positions = get_page_positions(fname)
        p_start = page_at_line(positions, line_start)
        p_end = page_at_line(positions, line_end)
        pages.update(range(p_start, p_end + 1))
    return pages


def pages_for_bloc_sources(bloc_sources):
    pages = set()
    for bloc_id, _unites_ids in bloc_sources:
        rng = BLOC_PAGES.get(bloc_id)
        if rng:
            pages.update(range(rng[0], rng[1] + 1))
    return pages


def process_one(q):
    target_file = q["doc_name"] + ".md"
    try:
        answer, context, section_sources, bloc_sources, tokens_used = ro.answer_question(q["question"], target_file)
    except Exception as e:
        answer, context, section_sources, bloc_sources, tokens_used = f"[ERROR: {e}]", "", [], [], 0

    retrieved_pages = pages_for_section_sources(section_sources) | pages_for_bloc_sources(bloc_sources)
    gold_pages = set(ev["page_num"] for ev in (q.get("evidence_pages") or []))

    recall_j2, precision_j2 = mx.page_recall_precision(retrieved_pages, gold_pages)
    acc_original = mx.answer_accuracy(q["question"], q["answer"], answer)
    recall_j1, precision_j1 = jm.decimal_page_metrics(section_sources, bloc_sources, gold_pages)

    full_page_context = jm.build_full_page_context(section_sources, bloc_sources, target_file)
    ro.reset_token_counter()
    try:
        answer_j2 = ro.generate_answer(q["question"], full_page_context)
    except Exception as e:
        answer_j2 = f"[ERROR: {e}]"
    tokens_wholepage = ro.get_token_counter()
    acc_j2 = mx.answer_accuracy(q["question"], q["answer"], answer_j2)

    return {
        "financebench_id": q["financebench_id"],
        "question": q["question"],
        "gold": q["answer"],
        "doc_name": q["doc_name"],
        "answer": answer,
        "context_chars": len(context),
        "retrieved_pages": sorted(retrieved_pages),
        "gold_pages": sorted(gold_pages),
        "tokens_used": tokens_used,
        "accuracy": acc_original,
        "page_recall_decimal": recall_j1,
        "page_precision_decimal": precision_j1,
        "page_recall_wholepage": recall_j2,
        "page_precision_wholepage": precision_j2,
        "answer_wholepage": answer_j2,
        "accuracy_wholepage": acc_j2,
        "full_page_context_chars": len(full_page_context),
        "tokens_used_wholepage": tokens_wholepage,
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

    if SAMPLE_IDS_FILE:
        with open(SAMPLE_IDS_FILE, encoding="utf-8") as f:
            sample_ids = set(json.load(f))
        all_q = [q for q in all_q if q["financebench_id"] in sample_ids]
        print(f"Sous-echantillon: {len(all_q)} questions ({SAMPLE_IDS_FILE})", flush=True)

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
                         "accuracy": 0.0, "accuracy_wholepage": 0.0,
                         "page_recall_decimal": None, "page_precision_decimal": None,
                         "page_recall_wholepage": None, "page_precision_wholepage": None,
                         "tokens_used": 0, "tokens_used_wholepage": 0, "error": str(e)}
                cache[r["financebench_id"]] = r
                save_cache(cache)
                done += 1
                with _lock_print:
                    print(f"[{done}/{len(todo)}] acc={r['accuracy']:.0f} acc_wp={r['accuracy_wholepage']:.0f} "
                          f"recall_dec={r['page_recall_decimal']} prec_dec={r['page_precision_decimal']} "
                          f"-- {r['question'][:60]}", flush=True)

    results = [cache[q["financebench_id"]] for q in all_q]
    n = len(results)

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_questions": n,
        "accuracy_original": avg("accuracy"),
        "accuracy_wholepage": avg("accuracy_wholepage"),
        "page_recall_decimal": avg("page_recall_decimal"),
        "page_precision_decimal": avg("page_precision_decimal"),
        "page_recall_wholepage": avg("page_recall_wholepage"),
        "page_precision_wholepage": avg("page_precision_wholepage"),
        "avg_tokens_original": avg("tokens_used"),
        "avg_tokens_wholepage": avg("tokens_used_wholepage"),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTATS COMPLETS (150 questions) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
