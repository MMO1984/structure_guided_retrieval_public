#!/usr/bin/env python3
"""
Evaluation du pilote (18 questions, 10 documents) : notre methode (version
de base originale) sur FinanceBench, avec Page Recall/Precision (vs les
pages gold recuperees du dataset LOFin de HiREC) et Answer Accuracy
(numerique tolerant ou juge LLM esprit FAMMA pour le textuel).
"""

import os
import re
import json
import pickle

import retrieve_ours_financebench as ro
import metrics_financebench as mx
import jeu_metrics_financebench as jm

BASE_DIR = os.path.dirname(__file__)
CORPUS_MD_DIR = os.path.join(BASE_DIR, "corpus_md")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "pilot_results.json")

with open(os.path.join(BASE_DIR, "rag_index_financebench.pkl"), "rb") as f:
    RAG_INDEX = pickle.load(f)
BLOC_PAGES = RAG_INDEX["bloc_pages"]

_PAGE_POSITIONS_CACHE = {}


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

    # Jeu 2 (1 page = 1 page) : Page Recall/Precision = convention deja
    # utilisee ci-dessus (identique par construction, meme retrieval).
    recall_j2, precision_j2 = mx.page_recall_precision(retrieved_pages, gold_pages)
    acc_original = mx.answer_accuracy(q["question"], q["answer"], answer)

    # Jeu 1 (pages decimales) : recompute a partir du MEME retrieval, sans
    # rappel LLM (aucune regeneration necessaire).
    recall_j1, precision_j1 = jm.decimal_page_metrics(section_sources, bloc_sources, gold_pages)

    # Jeu 2, variante pipeline : Phase 4 rejouee avec les pages entieres
    # (union deduppliquee TOC+BM25) au lieu des extraits cures. Retrieval
    # inchange -> Page Recall/Precision Jeu 2 = ceux deja calcules ci-dessus.
    target_file = q["doc_name"] + ".md"
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
        # Modele original (extraits cures)
        "accuracy": acc_original,
        # Jeu 1 : pages decimales (meme retrieval, meme reponse, metrique recalculee)
        "page_recall_decimal": recall_j1,
        "page_precision_decimal": precision_j1,
        # Jeu 2 : 1 page = 1 page (deja calcule ci-dessus, identique au run original)
        "page_recall_wholepage": recall_j2,
        "page_precision_wholepage": precision_j2,
        # Jeu 2, variante pipeline : Phase 4 rejouee avec pages entieres
        "answer_wholepage": answer_j2,
        "accuracy_wholepage": acc_j2,
        "full_page_context_chars": len(full_page_context),
        "tokens_used_wholepage": tokens_wholepage,
    }


def main():
    with open(os.path.join(BASE_DIR, "pilot_questions.json"), encoding="utf-8") as f:
        pilot = json.load(f)

    results = []
    for i, q in enumerate(pilot, start=1):
        r = process_one(q)
        results.append(r)
        print(f"[{i}/{len(pilot)}] acc={r['accuracy']:.0f} acc_wholepage={r['accuracy_wholepage']:.0f} "
              f"recall_dec={r['page_recall_decimal']} prec_dec={r['page_precision_decimal']} "
              f"recall_wp={r['page_recall_wholepage']} prec_wp={r['page_precision_wholepage']} "
              f"tokens={r['tokens_used']}+{r['tokens_used_wholepage']} -- {r['question'][:60]}", flush=True)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
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
    print("\n=== RESULTATS PILOTE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
