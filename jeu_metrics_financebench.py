#!/usr/bin/env python3
"""
Jeu 1 (pages decimales) et Jeu 2 (1 page = 1 page, variante "pages entieres
en Phase 4") -- voir discussion. Le retrieval (Phase 1 TOC + Phase 2b BM25)
n'est JAMAIS rejoue ici : on reutilise les section_sources/bloc_sources deja
produits et sauvegardes par le run original, pour garantir que les deux jeux
partagent EXACTEMENT le meme ciblage de retrieval (seule la Phase 4 differe
pour le Jeu 2).
"""

import os
import re
import pickle

BASE_DIR = os.path.dirname(__file__)
CORPUS_MD_DIR = os.path.join(BASE_DIR, "corpus_md")
PAGE_MARKER = re.compile(r"<!-- page:(\d+) -->")

with open(os.path.join(BASE_DIR, "rag_index_financebench.pkl"), "rb") as f:
    _RAG_INDEX = pickle.load(f)
BLOC_LINES = _RAG_INDEX["bloc_lines"]
BLOC_PAGES = _RAG_INDEX["bloc_pages"]
UNITE_PAGES = _RAG_INDEX["unite_pages"]
BLOCS = _RAG_INDEX["blocs"]

_PAGE_LINE_SPANS_CACHE = {}
_PAGE_FULLTEXT_CACHE = {}


def _raw_lines(fname):
    with open(os.path.join(CORPUS_MD_DIR, fname), encoding="utf-8") as f:
        return f.readlines()


def get_page_line_spans(fname):
    """{page_num: (line_start, line_end)} -- bornes de lignes de chaque page."""
    if fname not in _PAGE_LINE_SPANS_CACHE:
        lines = _raw_lines(fname)
        positions = []
        for i, line in enumerate(lines, start=1):
            m = PAGE_MARKER.match(line.strip())
            if m:
                positions.append((i, int(m.group(1))))
        spans = {}
        for i, (line_no, page_num) in enumerate(positions):
            next_line = positions[i + 1][0] - 1 if i + 1 < len(positions) else len(lines)
            spans[page_num] = (line_no, next_line)
        _PAGE_LINE_SPANS_CACHE[fname] = spans
    return _PAGE_LINE_SPANS_CACHE[fname]


def get_page_fulltext(fname, page_num):
    """Texte integral d'une page (marqueur retire), pour le Jeu 2."""
    key = (fname, page_num)
    if key not in _PAGE_FULLTEXT_CACHE:
        lines = _raw_lines(fname)
        spans = get_page_line_spans(fname)
        if page_num not in spans:
            _PAGE_FULLTEXT_CACHE[key] = ""
        else:
            l_start, l_end = spans[page_num]
            text = "".join(lines[l_start - 1:l_end])
            text = PAGE_MARKER.sub("", text).strip()
            _PAGE_FULLTEXT_CACHE[key] = text
    return _PAGE_FULLTEXT_CACHE[key]


def fractional_coverage_for_span(fname, line_start, line_end):
    """Pour un item retrouve (section ou bloc) couvrant [line_start, line_end]
    dans fname, retourne {page_num: fraction_de_la_page_couverte}."""
    spans = get_page_line_spans(fname)
    coverage = {}
    for page_num, (p_start, p_end) in spans.items():
        overlap_start = max(line_start, p_start)
        overlap_end = min(line_end, p_end)
        if overlap_start <= overlap_end:
            overlap_lines = overlap_end - overlap_start + 1
            page_total_lines = max(1, p_end - p_start + 1)
            coverage[page_num] = overlap_lines / page_total_lines
    return coverage


def aggregate_coverage(section_sources, bloc_sources):
    """Fusionne la couverture fractionnaire de toutes les sections (Phase 1)
    et unites de blocs REELLEMENT EXTRAITES par la contagion (Phase 2b) pour
    UNE question, en {page_num: fraction}, fraction plafonnee a 1.0 par page.

    Sections (Phase 1) : texte entier envoye -> overlap de lignes exact
    (fractional_coverage_for_span) reste correct tel quel.

    Blocs (Phase 2b) : PLUS l'etendue nominale du bloc entier (biais corrige
    -- un bloc peut couvrir toute une page alors que la contagion n'en
    extrait que 1-2 phrases) -- on utilise desormais la position PRECISE de
    chaque unite reellement extraite (unite_pages, calculee a la construction
    de l'index), ponderee par sa longueur relative au contenu total de la
    page qu'elle touche."""
    coverage = {}
    for fname, line_start, line_end in section_sources:
        for page_num, frac in fractional_coverage_for_span(fname, line_start, line_end).items():
            coverage[page_num] = min(1.0, coverage.get(page_num, 0.0) + frac)

    for bloc_id, unites_ids in bloc_sources:
        bloc = BLOCS[bloc_id]
        for unite_id in unites_ids:
            page_span = UNITE_PAGES.get((bloc_id, unite_id))
            if not page_span:
                continue
            unite_texte = bloc.unites[unite_id].texte
            unite_len = len(unite_texte)
            pages_touched = range(page_span[0], page_span[1] + 1)
            # Si l'unite s'etale sur plusieurs pages (rare, une phrase ne
            # traverse quasi jamais une coupure de page), on repartit sa
            # longueur a parts egales entre les pages touchees.
            len_per_page = unite_len / max(1, len(list(pages_touched)))
            for page_num in pages_touched:
                page_total_len = len(get_page_fulltext(bloc.source, page_num))
                if page_total_len == 0:
                    continue
                frac = len_per_page / page_total_len
                coverage[page_num] = min(1.0, coverage.get(page_num, 0.0) + frac)
    return coverage


def decimal_page_metrics(section_sources, bloc_sources, gold_pages):
    """JEU 1 : Page Recall/Precision ponderes par la fraction reelle de
    chaque page couverte (pas juste "touchee = 1 page entiere")."""
    if not gold_pages:
        return None, None
    coverage = aggregate_coverage(section_sources, bloc_sources)
    total_retrieved_decimal = sum(coverage.values())
    intersect_decimal = sum(coverage.get(p, 0.0) for p in gold_pages)
    recall = intersect_decimal / len(gold_pages)
    precision = intersect_decimal / total_retrieved_decimal if total_retrieved_decimal > 0 else 0.0
    return recall, precision


def touched_pages_union(section_sources, bloc_sources):
    """Union des pages touchees (convention '1 page = 1 page', pour le Jeu 2
    -- identique a la convention deja utilisee pour Page Recall/Precision
    du run original)."""
    pages = set()
    for fname, line_start, line_end in section_sources:
        spans = get_page_line_spans(fname)
        for page_num, (p_start, p_end) in spans.items():
            if line_start <= p_end and line_end >= p_start:
                pages.add(page_num)
    for bloc_id, _unites_ids in bloc_sources:
        rng = BLOC_PAGES.get(bloc_id)
        if rng:
            pages.update(range(rng[0], rng[1] + 1))
    return pages


def build_full_page_context(section_sources, bloc_sources, fname):
    """JEU 2 : contexte = texte INTEGRAL de chaque page unique touchee par le
    retrieval (union deduppliquee TOC + BM25), au lieu des extraits curés."""
    pages = sorted(touched_pages_union(section_sources, bloc_sources))
    contexte = ""
    for page_num in pages:
        text = get_page_fulltext(fname, page_num)
        if text:
            contexte += f"\n\n=== {fname} | page {page_num} ===\n{text}"
    return contexte
