#!/usr/bin/env python3
"""Genere une TOC par document du corpus FinanceBench (meme logique que
src-gemini-proxy/preparation-toc-et-pkl/generate_toc.py, boucle sur tous
les fichiers du dossier au lieu d'une liste fixe)."""

import json
import re
import os

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus_md")
TOC_DIR = os.path.join(os.path.dirname(__file__), "corpus_toc")


def parse_markdown_headers(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    sections = []
    current_path = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}

    for i, line in enumerate(lines):
        match = header_pattern.match(line.strip())
        if match:
            level = len(match.group(1))
            raw_title = match.group(2).strip()
            title = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw_title)
            title = re.sub(r"\*([^*]+)\*", r"\1", title)
            title = re.sub(r"<[^>]+>", "", title)
            title = title.strip()

            current_path[level] = title
            for l in range(level + 1, 7):
                current_path[l] = None

            path_parts = [current_path[l] for l in range(1, level + 1) if current_path[l]]
            full_path = " > ".join(path_parts)

            sections.append({
                "id": f"L{i + 1}",
                "level": level,
                "title": title,
                "line_start": i + 1,
                "line_end": None,
                "path": full_path,
            })

    for i, section in enumerate(sections):
        next_boundary = len(lines)
        for j in range(i + 1, len(sections)):
            if sections[j]["level"] <= section["level"]:
                next_boundary = sections[j]["line_start"] - 1
                break
        section["line_end"] = next_boundary

    return sections


def generate_toc_text(sections, source_name):
    lines = [f"=== TABLE DES MATIERES: {source_name} ===\n"]
    for section in sections:
        indent = "  " * (section["level"] - 1)
        lines.append(f"{indent}[{section['id']}] {section['title']}")
    return "\n".join(lines)


def generate_toc_for_file(filename):
    filepath = os.path.join(CORPUS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    sections = parse_markdown_headers(filepath)
    toc_text = generate_toc_text(sections, filename)
    return {
        "source": filename,
        "total_lines": total_lines,
        "sections_count": len(sections),
        "sections": sections,
        "toc_text": toc_text,
    }


def main():
    os.makedirs(TOC_DIR, exist_ok=True)
    for filename in sorted(os.listdir(CORPUS_DIR)):
        if not filename.endswith(".md"):
            continue
        toc_data = generate_toc_for_file(filename)
        out_path = os.path.join(TOC_DIR, filename.replace(".md", "_toc.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(toc_data, f, ensure_ascii=False, indent=2)
        print(f"{filename}: {toc_data['sections_count']} sections, {toc_data['total_lines']} lignes")


if __name__ == "__main__":
    main()
