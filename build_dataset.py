#!/usr/bin/env python3
"""
Reconstruit all_150_questions.json (deja fourni dans ce depot, ce script
sert a le regenerer/verifier) directement depuis le JSONL officiel de
FinanceBench (patronus-ai/financebench), qui contient nativement
evidence_page_num (0-indexe) -- aucune source supplementaire necessaire.
"""

import json
import urllib.request

URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
OUT_FILE = "all_150_questions.json"


def main():
    with urllib.request.urlopen(URL) as f:
        lines = f.read().decode("utf-8").splitlines()

    questions = []
    for line in lines:
        r = json.loads(line)
        questions.append({
            "financebench_id": r["financebench_id"],
            "company": r["company"],
            "doc_name": r["doc_name"],
            "question_type": r["question_type"],
            "question": r["question"],
            "answer": r["answer"],
            "evidence_pages": [
                {"doc_name": ev.get("evidence_doc_name") or r["doc_name"], "page_num": ev["evidence_page_num"]}
                for ev in r["evidence"]
            ],
        })

    print(f"Questions: {len(questions)}")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Ecrit: {OUT_FILE}")


if __name__ == "__main__":
    main()
