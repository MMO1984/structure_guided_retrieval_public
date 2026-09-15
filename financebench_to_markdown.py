#!/usr/bin/env python3
"""
Convertit un 10-K/10-Q FinanceBench (PDF) en Markdown avec en-tetes '#'/'##',
pour reutiliser generate_toc.py/prepare_embeddings.py tels quels.

Niveau 1 (#) : marqueurs standard SEC "PART I/II/III/IV" et "Item N[A-Z]." --
detectes via regex + police en gras (le texte en gras + ce pattern coincide
exactement avec les vrais en-tetes de section, verifie manuellement : les
renvois internes ("as discussed in Item 7") et la page de sommaire du filing
lui-meme ne sont PAS en gras, seuls les vrais titres de section le sont).
Niveau 2 (##) : lignes courtes (<=8 mots) tout en majuscules et en gras
(convention frequente pour les sous-sections a l'interieur d'un Item, ex.
"COMPETITION", "SEASONALITY" dans Item 1).
"""

import os
import re
import fitz

CORPUS_PDF_DIR = os.path.join(os.path.dirname(__file__), "corpus_pdf")
CORPUS_MD_DIR = os.path.join(os.path.dirname(__file__), "corpus_md")

LEVEL1_PATTERN = re.compile(r"^(PART\s+[IVX]+\.?|Item\s+\d+[A-Z]?\.?)\s*(.*)$")


def is_bold(line):
    return any("bold" in s["font"].lower() for s in line["spans"])


def line_text(line):
    return "".join(s["text"] for s in line["spans"]).strip()


def is_level2_candidate(text):
    if not text or not text.isupper():
        return False
    words = text.split()
    if not (1 <= len(words) <= 8):
        return False
    if LEVEL1_PATTERN.match(text):
        return False
    return True


# Fenetre (en lignes, de part et d'autre) et seuil de densite utilises pour
# distinguer un vrai sous-titre isole (entoure de texte de corps normal) d'un
# libelle de tableau financier (bilan, compte de resultat...) : ces libelles
# sont eux aussi courts/gras/tout en majuscules (TOTAL ASSETS, % CHANGE,
# FISCAL 2021...) et remontent donc comme candidats niveau 2 au sens de
# is_level2_candidate, mais apparaissent en RAFALES DENSES (plusieurs par
# ligne de tableau) alors qu'un vrai sous-titre de section est isole -- un
# vrai sous-titre n'a quasiment jamais 2 autres candidats a moins de 3 lignes
# de distance, un libelle de tableau si.
BURST_WINDOW = 3
BURST_MIN_COUNT = 3

# Complement du signal de rafale ci-dessus : un libelle de ligne de tableau
# financier (ex. "TOTAL NIKE, INC. REVENUES") est bien souvent ISOLE parmi
# les AUTRES candidats niveau 2 (les valeurs numeriques adjacentes, "$",
# "44,538 $", "-4 %", "(11)"..., ne sont pas elles-memes candidates niveau 2
# faute de lettres en majuscule), donc invisible au seul comptage de rafale
# ci-dessus. On le detecte plutot par la PROXIMITE immediate d'au moins une
# ligne purement numerique/monetaire -- un vrai sous-titre de section n'est
# quasiment jamais entoure de ce type de ligne.
NUMERIC_MARKER_WINDOW = 2
NUMERIC_MARKER_PATTERN = re.compile(r"^[\d\$%()\-.,\s]+$")


def is_numeric_marker(text):
    if not text or not NUMERIC_MARKER_PATTERN.match(text):
        return False
    return any(c.isdigit() for c in text) or text.strip() in ("$", "%")


def is_table_burst(raw_lvl2_flags, numeric_flags, idx):
    lo = max(0, idx - BURST_WINDOW)
    hi = min(len(raw_lvl2_flags), idx + BURST_WINDOW + 1)
    if sum(raw_lvl2_flags[lo:hi]) >= BURST_MIN_COUNT:
        return True
    lo2 = max(0, idx - NUMERIC_MARKER_WINDOW)
    hi2 = min(len(numeric_flags), idx + NUMERIC_MARKER_WINDOW + 1)
    return any(numeric_flags[lo2:hi2])


def convert_pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    out_lines = []
    for page_idx, page in enumerate(doc):
        # Marqueur de page -- 0-indexe, verifie manuellement pour matcher exactement
        # la convention page_num du dataset LOFin de HiREC (doc[page_num] en PyMuPDF,
        # aucun decalage +1/-1). Invisible pour le rendu mais retrouvable par regex
        # pour remonter n'importe quelle ligne a sa page PDF d'origine (Page Recall).
        out_lines.append(f"<!-- page:{page_idx} -->")
        d = page.get_text("dict")
        # Detecte si cette page est la page de sommaire du filing (beaucoup
        # de marqueurs Item/PART mais NON en gras) -- on saute la detection
        # d'en-tetes sur cette page (le texte reste dans le markdown en corps
        # simple, sans structure, ce qui n'est pas grave : ce n'est que le
        # sommaire imprime du document, pas du contenu utile).
        candidates = []
        nonbold_markers = 0
        for block in d["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = line_text(line)
                if not text:
                    continue
                m = LEVEL1_PATTERN.match(text)
                if m and not is_bold(line):
                    nonbold_markers += 1
                candidates.append((text, line))

        is_toc_page = nonbold_markers >= 5

        raw_lvl2 = [(not is_toc_page) and is_level2_candidate(text) and is_bold(line)
                    for text, line in candidates]
        numeric_flags = [is_numeric_marker(text) for text, line in candidates]

        for idx, (text, line) in enumerate(candidates):
            if is_toc_page:
                out_lines.append(text)
                continue
            m = LEVEL1_PATTERN.match(text)
            if m and is_bold(line):
                out_lines.append(f"\n# {text}\n")
            elif raw_lvl2[idx] and not is_table_burst(raw_lvl2, numeric_flags, idx):
                out_lines.append(f"\n## {text}\n")
            else:
                out_lines.append(text)
        out_lines.append("")  # saut de page

    return "\n".join(out_lines)


def main():
    os.makedirs(CORPUS_MD_DIR, exist_ok=True)
    for fname in sorted(os.listdir(CORPUS_PDF_DIR)):
        if not fname.endswith(".pdf"):
            continue
        doc_name = fname[:-4]
        md = convert_pdf_to_markdown(os.path.join(CORPUS_PDF_DIR, fname))
        out_path = os.path.join(CORPUS_MD_DIR, doc_name + ".md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        n_h1 = md.count("\n# ")
        n_h2 = md.count("\n## ")
        print(f"{doc_name}: {len(md)} chars, {n_h1} niveau-1, {n_h2} niveau-2")


if __name__ == "__main__":
    main()
