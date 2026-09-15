#!/usr/bin/env python3
"""
Telecharge les filings PDF necessaires aux 150 questions publiques de
FinanceBench, depuis le depot officiel patronus-ai/financebench (licence
CC BY-NC 4.0 -- voir leur README). Ecrit dans ./corpus_pdf/.
"""

import os
import json
import urllib.request

BASE_DIR = os.path.dirname(__file__)
CORPUS_PDF_DIR = os.path.join(BASE_DIR, "corpus_pdf")


def main():
    os.makedirs(CORPUS_PDF_DIR, exist_ok=True)
    with open(os.path.join(BASE_DIR, "all_150_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    docs = sorted(set(q["doc_name"] for q in questions))
    print(f"Documents uniques a telecharger: {len(docs)}")

    for i, doc_name in enumerate(docs, 1):
        out_path = os.path.join(CORPUS_PDF_DIR, doc_name + ".pdf")
        if os.path.exists(out_path):
            continue
        url = f"https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"
        urllib.request.urlretrieve(url, out_path)
        if i % 10 == 0:
            print(f"{i}/{len(docs)} telecharges", flush=True)

    print("Termine.")


if __name__ == "__main__":
    main()
