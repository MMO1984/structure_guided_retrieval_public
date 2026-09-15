#!/usr/bin/env python3
"""
Comble les grands "trous" entre en-tetes natifs (niveau 1/2) detectes dans le
PDF -- des blocs de texte trop longs sans sous-titre visible (ex. tableaux
financiers consolides sans en-tete en gras). Pour chaque trou > SEUIL
caracteres, on le decoupe en morceaux et on demande a Gemini un titre court
et descriptif par morceau, insere comme en-tete '###' (niveau 3, sous-section
d'un niveau 1/2 existant) directement dans le Markdown.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-3-flash-preview"

CORPUS_MD_DIR = os.path.join(os.path.dirname(__file__), "corpus_md")
GAP_THRESHOLD = 6000
CHUNK_SIZE = 9000
MAX_WORKERS = 6

HEADER_PATTERN = re.compile(r"^#{1,2} .+$", re.MULTILINE)


def call_llm(prompt, retries=3):
    last_err = None
    for model_name in [MODEL, FALLBACK_MODEL]:
        try:
            model = genai.GenerativeModel(model_name=model_name, generation_config={"temperature": 0.2})
            result = model.generate_content(prompt, request_options={"timeout": 30})
            if result.candidates and result.candidates[0].content.parts:
                return result.candidates[0].content.parts[0].text
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"LLM indisponible: {last_err}")


def propose_title(chunk_text):
    prompt = f"""This is an excerpt from a SEC financial filing (10-K/10-Q). Give a short, descriptive section title (3-8 words) summarizing what this excerpt covers. Respond with ONLY the title, no punctuation at the end, no explanation.

EXCERPT:
{chunk_text[:2000]}

TITLE:"""
    title = call_llm(prompt).strip()
    title = re.sub(r'^["\'\-\s]+|["\'\-\s]+$', "", title)
    return title[:100]


def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current)
            current = p
        else:
            current += ("\n\n" if current else "") + p
    if current.strip():
        chunks.append(current)
    return chunks


def fill_gaps_for_file(md_path):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    matches = list(HEADER_PATTERN.finditer(text))
    boundaries = [m.start() for m in matches] + [len(text)]

    # Etape 1 : collecter tous les chunks a titrer (pas d'appel LLM encore).
    to_title = []  # (position, chunk_text)
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        body_start = text.find("\n", start) + 1 if start < len(text) else start
        gap_text = text[body_start:end]
        if len(gap_text) <= GAP_THRESHOLD:
            continue
        chunks = split_into_chunks(gap_text)
        if len(chunks) <= 1:
            continue
        offset = body_start
        for chunk in chunks[1:]:  # le 1er chunk garde le header existant
            chunk_pos = text.find(chunk, offset)
            if chunk_pos != -1:
                to_title.append((chunk_pos, chunk))
            offset = chunk_pos + len(chunk) if chunk_pos != -1 else offset + len(chunk)

    if not to_title:
        return text, 0

    # Etape 2 : appels LLM en parallele.
    def safe_title(item):
        pos, chunk = item
        try:
            return pos, propose_title(chunk)
        except Exception:
            return pos, None

    insertions = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for pos, title in executor.map(safe_title, to_title):
            if title:
                insertions.append((pos, f"\n\n### {title}\n\n"))

    insertions.sort(key=lambda x: x[0], reverse=True)
    for pos, insert_text in insertions:
        text = text[:pos] + insert_text + text[pos:]

    return text, len(insertions)


def main():
    for fname in sorted(os.listdir(CORPUS_MD_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(CORPUS_MD_DIR, fname)
        new_text, n_new = fill_gaps_for_file(path)
        if n_new:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
        print(f"{fname}: {n_new} nouveaux titres de niveau 3 inseres")


if __name__ == "__main__":
    main()
