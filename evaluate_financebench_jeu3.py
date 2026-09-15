#!/usr/bin/env python3
"""
JEU 3 : meme modele (meme LLM, meme prompt, pas de fix de troncature) que le
run original, mais avec des budgets de contexte reduits :
- Phase 1 (sections TOC) : 4000 -> 2000 tokens
- Phase 2b (BM25 + contagion) : 3000 -> 1500 tokens
Hypothese : Precision (decimale) en hausse, Accuracy et Page Recall en
baisse -- objectif = comparer a HiREC a un niveau de precision plus proche
du leur (~21%), pas juste comparer a des niveaux de precision tres
differents. Seules les metriques decimales (Jeu 1) sont calculees ici.
"""

import os
import json

import retrieve_ours_financebench as ro
import metrics_financebench as mx
import jeu_metrics_financebench as jm

BASE_DIR = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(BASE_DIR, "results", "pilot_results_jeu3.json")

MAX_TOKENS_SECTIONS = 2000
MAX_TOKENS_BM25 = 1500


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


def main():
    with open(os.path.join(BASE_DIR, "pilot_questions.json"), encoding="utf-8") as f:
        pilot = json.load(f)

    results = []
    for i, q in enumerate(pilot, start=1):
        r = process_one(q)
        results.append(r)
        print(f"[{i}/{len(pilot)}] acc={r['accuracy']:.0f} recall_dec={r['page_recall_decimal']} "
              f"prec_dec={r['page_precision_decimal']} tokens={r['tokens_used']} -- {r['question'][:60]}", flush=True)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
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
    print("\n=== RESULTATS JEU 3 (budgets reduits) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
