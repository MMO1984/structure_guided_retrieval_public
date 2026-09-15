#!/usr/bin/env python3
"""
Construit l'index de recherche (BM25 + embeddings + groupes de contagion) sur le
corpus FinanceBench. Difference cle : chaque bloc doit connaitre sa page
PDF d'origine (pour la metrique Page Recall, comparable a HiREC) -- les
marqueurs '<!-- page:N -->' inseres par financebench_to_markdown.py sont
lus pour calculer un intervalle (page_start, page_end) par bloc, PUIS
retires du texte avant indexation BM25/embeddings (pour ne pas polluer
les representations avec du bruit "page 53").
"""

import os
import re
import pickle
import logging
import numpy as np
from datetime import datetime

from rag_common import Bloc, Unite, tokenize_for_bm25

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus_md")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "rag_index_financebench.pkl")

MIN_WORDS_PER_UNIT = 6
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BM25_K1 = 1.5
BM25_B = 0.75
SEUIL_CONTAGION = 0.55

PAGE_MARKER = re.compile(r"<!-- page:(\d+) -->")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def strip_markers_track_pages(raw_text, base_page):
    """Retire les marqueurs de page de raw_text et retourne (texte_nettoye,
    transitions) ou transitions est une liste [(offset_dans_texte_nettoye,
    page_num), ...] triee par offset -- permet de retrouver la page de
    N'IMPORTE QUELLE position dans le texte NETTOYE (celui utilise pour
    construire phrases/unites), sans avoir a remonter au fichier brut.
    Le collapse des lignes vides (>=3 '\\n' -> 2) est fait PAR SEGMENT (entre
    deux marqueurs), pas apres coup, pour que les offsets restent exacts sur
    le texte final (seul un tres rare cas limite -- pile a la jointure d'un
    marqueur retire -- peut laisser 3 '\\n' au lieu de 2, cosmetique, sans
    impact sur l'exactitude des transitions)."""
    transitions = [(0, base_page)]
    parts = []
    pos = 0
    for m in PAGE_MARKER.finditer(raw_text):
        segment = re.sub(r"\n{3,}", "\n\n", raw_text[pos:m.start()])
        parts.append(segment)
        pos = m.end()
        offset_nettoye = sum(len(p) for p in parts)
        transitions.append((offset_nettoye, int(m.group(1))))
    parts.append(re.sub(r"\n{3,}", "\n\n", raw_text[pos:]))
    # PAS de .strip() global ici (deformerait les offsets de `transitions` en
    # decalant tout le texte) -- le contenu final est deja quasi-propre car
    # raw_text est lui-meme issu d'un .strip() en amont.
    cleaned = "".join(parts)
    return cleaned, transitions


def page_at_offset(transitions, offset):
    page = transitions[0][1]
    for off, pg in transitions:
        if off <= offset:
            page = pg
        else:
            break
    return page


def parse_markdown_files(dir_path):
    blocs = []
    bloc_pages = {}
    bloc_lines = {}
    bloc_page_transitions = {}
    bloc_id = 0
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    files = sorted(f for f in os.listdir(dir_path) if f.endswith(".md"))
    logger.info(f"Fichiers a traiter: {len(files)}")

    for filename in files:
        with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as f:
            content = f.read()

        # Page en cours a chaque position de caractere -- calculee une fois par
        # fichier via les positions des marqueurs, puis interrogee par recherche
        # binaire pour chaque bloc.
        page_positions = [(m.start(), int(m.group(1))) for m in PAGE_MARKER.finditer(content)]

        def page_at(pos):
            page = page_positions[0][1] if page_positions else 0
            for p_pos, p_num in page_positions:
                if p_pos <= pos:
                    page = p_num
                else:
                    break
            return page

        headers = [(m.start(), m.group(1), m.group(2).strip())
                   for m in header_pattern.finditer(content)]
        if not headers:
            # Cas limite (ex. communiques de resultats sans aucun titre en
            # gras/majuscules) : fill_toc_gaps.py ne peut rien combler sans au
            # moins un en-tete pour ancrer les "trous" -- on traite alors le
            # document entier comme un unique bloc, pour que BM25/embeddings
            # restent utilisables plutot que de perdre le document.
            page_start = page_at(0)
            page_end = page_at(len(content))
            line_start, line_end = 1, content.count("\n") + 1
            block_content_final, transitions = strip_markers_track_pages(content, page_start)
            if block_content_final.strip():
                blocs.append(Bloc(id=bloc_id, source=filename,
                                   headers=[filename], level=1,
                                   titre=filename, contenu_complet=block_content_final, phrases=[]))
                bloc_pages[bloc_id] = (page_start, page_end)
                bloc_lines[bloc_id] = (line_start, line_end)
                bloc_page_transitions[bloc_id] = transitions
                bloc_id += 1
            continue
        hierarchy_stack = []
        for i, (pos, hashes, titre) in enumerate(headers):
            level = len(hashes)
            end_pos = headers[i + 1][0] if i < len(headers) - 1 else len(content)
            header_line_end = content.find("\n", pos)
            if header_line_end == -1:
                header_line_end = len(content)
            block_content_raw = content[header_line_end:end_pos].strip()
            while hierarchy_stack and hierarchy_stack[-1][0] >= level:
                hierarchy_stack.pop()
            hierarchy_stack.append((level, titre))
            if block_content_raw:
                page_start = page_at(header_line_end)
                page_end = page_at(end_pos)
                line_start = content.count("\n", 0, header_line_end) + 1
                line_end = content.count("\n", 0, end_pos) + 1
                block_content_final, transitions = strip_markers_track_pages(block_content_raw, page_start)
                if not block_content_final.strip():
                    continue
                blocs.append(Bloc(id=bloc_id, source=filename,
                                   headers=[t for _, t in hierarchy_stack], level=level,
                                   titre=titre, contenu_complet=block_content_final, phrases=[]))
                bloc_pages[bloc_id] = (page_start, page_end)
                bloc_lines[bloc_id] = (line_start, line_end)
                bloc_page_transitions[bloc_id] = transitions
                bloc_id += 1

    logger.info(f"TOTAL: {len(blocs)} blocs")
    return blocs, bloc_pages, bloc_lines, bloc_page_transitions


def segmenter_en_phrases_avec_positions(texte):
    """Identique a segmenter_en_phrases, mais retourne aussi la position
    (start, end) de chaque phrase DANS `texte` (le contenu du bloc deja
    nettoye des marqueurs de page) -- necessaire pour attribuer une page
    exacte a chaque UNITE plus tard (pas seulement au bloc entier), afin que
    la metrique de couverture decimale (Jeu 1) reflete ce qui est vraiment
    extrait par la contagion, pas l'etendue totale du bloc. Le remplacement
    des abreviations se fait sur des SPANS DE MEME LONGUEUR (padding) pour
    que les positions ne derivent jamais, meme legerement."""
    if not texte.strip():
        return []
    texte_work = texte
    abbreviations = ["M.", "Mme.", "Dr.", "etc.", "cf.", "ex.", "vol.", "p.", "pp.",
                      "Inc.", "Corp.", "Ltd.", "No.", "e.g.", "i.e.", "et al.", "Fig.", "Eq."]
    placeholders = {}
    for abbr in abbreviations:
        # Le "." final est remplace par un octet prive (\x00, absent du texte
        # reel) -- MEME LONGUEUR garantie par construction, donc AUCUNE derive
        # de position possible, quelle que soit la longueur de l'abreviation.
        placeholder = abbr[:-1] + "\x00" if abbr.endswith(".") else abbr
        placeholders[placeholder] = abbr
        texte_work = texte_work.replace(abbr, placeholder)
    pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+")

    parts = pattern.split(texte_work)
    seps = list(pattern.finditer(texte_work))
    spans = []
    pos = 0
    for i, part in enumerate(parts):
        start = pos
        end = start + len(part)
        spans.append((start, end))
        pos = seps[i].end() if i < len(seps) else end

    phrases = []
    for (start, end) in spans:
        raw = texte_work[start:end]
        p = raw.strip()
        # Ajuste (start,end) pour exclure le whitespace de bordure retire par strip()
        lstrip_n = len(raw) - len(raw.lstrip())
        rstrip_n = len(raw) - len(raw.rstrip())
        p_start = start + lstrip_n
        p_end = end - rstrip_n
        for placeholder, abbr in placeholders.items():
            p = p.replace(placeholder, abbr)
        p = re.sub(r"\s+", " ", p)
        if p:
            phrases.append((p, p_start, p_end))
    return phrases


def creer_unites_avec_positions(phrases_avec_positions, seuil_mots=MIN_WORDS_PER_UNIT):
    """Identique a creer_unites, mais retourne aussi {unite_id: (start,end)}
    (position DANS bloc.contenu_complet), en fusionnant les positions des
    phrases combinees dans une meme unite."""
    if not phrases_avec_positions:
        return [], {}
    unites = []
    unite_positions = {}
    unite_id = 0
    for phrase, p_start, p_end in phrases_avec_positions:
        nb_mots = len(phrase.split())
        if nb_mots >= seuil_mots:
            unites.append(Unite(id=unite_id, phrases_indices=[len(unites)], texte=phrase))
            unite_positions[unite_id] = (p_start, p_end)
            unite_id += 1
        else:
            if unites:
                last_id = unites[-1].id
                unites[-1].phrases_indices.append(len(unites))
                unites[-1].texte += " " + phrase
                s0, e0 = unite_positions[last_id]
                unite_positions[last_id] = (min(s0, p_start), max(e0, p_end))
            else:
                unites.append(Unite(id=unite_id, phrases_indices=[0], texte=phrase))
                unite_positions[unite_id] = (p_start, p_end)
                unite_id += 1
    return unites, unite_positions


def calculer_embeddings(blocs, model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    textes, mapping = [], []
    for bloc in blocs:
        for unite in bloc.unites:
            textes.append(unite.texte)
            mapping.append((bloc.id, unite.id))
    logger.info(f"Unites a encoder: {len(textes)}")
    embeddings = model.encode(textes, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    for i, (bloc_id, unite_id) in enumerate(mapping):
        blocs[bloc_id].unites[unite_id].embedding = embeddings[i].tolist()


def precalculer_groupes_contagion(blocs, seuil=SEUIL_CONTAGION):
    for bloc in blocs:
        unites = bloc.unites
        if not unites:
            bloc.contagion_groups = []
            continue
        if len(unites) == 1:
            bloc.contagion_groups = [[0]]
            continue
        groups = []
        current_group = [0]
        for i in range(len(unites) - 1):
            emb_a = np.array(unites[i].embedding)
            emb_b = np.array(unites[i + 1].embedding)
            sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
            if sim >= seuil:
                current_group.append(i + 1)
            else:
                groups.append(current_group)
                current_group = [i + 1]
        groups.append(current_group)
        bloc.contagion_groups = groups


def construire_index_bm25_par_papier(blocs):
    """Un index BM25 SEPARE par document (statistiques IDF locales), comme
    ailleurs -- FinanceBench est aussi un benchmark mono-document par
    question."""
    from rank_bm25 import BM25Okapi

    blocs_by_source = {}
    for idx, bloc in enumerate(blocs):
        blocs_by_source.setdefault(bloc.source, []).append(idx)

    bm25_by_source = {}
    for source, bloc_ids in blocs_by_source.items():
        corpus = [tokenize_for_bm25(blocs[i].get_texte_pour_bm25()) for i in bloc_ids]
        bm25_by_source[source] = {
            "bm25": BM25Okapi(corpus, k1=BM25_K1, b=BM25_B),
            "bloc_ids": bloc_ids,
        }

    logger.info(f"Index BM25 par document construits: {len(bm25_by_source)}")
    return bm25_by_source


def main():
    blocs, bloc_pages, bloc_lines, bloc_page_transitions = parse_markdown_files(CORPUS_DIR)

    unite_pages = {}  # {(bloc_id, unite_id): (page_start, page_end)}
    for bloc in blocs:
        phrases_pos = segmenter_en_phrases_avec_positions(bloc.contenu_complet)
        bloc.phrases = [p for p, _, _ in phrases_pos]
        unites, unite_positions = creer_unites_avec_positions(phrases_pos)
        bloc.unites = unites
        transitions = bloc_page_transitions[bloc.id]
        for unite_id, (start, end) in unite_positions.items():
            unite_pages[(bloc.id, unite_id)] = (page_at_offset(transitions, start),
                                                 page_at_offset(transitions, end))
    logger.info(f"Phrases totales: {sum(len(b.phrases) for b in blocs)}")
    logger.info(f"Unites totales: {sum(len(b.unites) for b in blocs)}")
    calculer_embeddings(blocs)
    precalculer_groupes_contagion(blocs)
    bm25_by_source = construire_index_bm25_par_papier(blocs)

    data = {
        "blocs": blocs, "bm25_by_source": bm25_by_source, "bloc_pages": bloc_pages, "bloc_lines": bloc_lines,
        "unite_pages": unite_pages,
        "config": {"embedding_model": EMBEDDING_MODEL, "min_words_per_unit": MIN_WORDS_PER_UNIT,
                   "bm25_k1": BM25_K1, "bm25_b": BM25_B, "created_at": datetime.now().isoformat()},
        "stats": {"nb_blocs": len(blocs), "nb_unites": sum(len(b.unites) for b in blocs),
                  "nb_phrases": sum(len(b.phrases) for b in blocs)},
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Index sauvegarde: {OUTPUT_FILE}")
    print(f"Blocs: {data['stats']['nb_blocs']}, Unites: {data['stats']['nb_unites']}")


if __name__ == "__main__":
    main()
