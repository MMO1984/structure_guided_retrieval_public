#!/usr/bin/env python3
"""
Notre methode (TOC-navigation + BM25 ancre + contagion semantique) appliquee
au corpus FinanceBench, harmonisee avec le mecanisme de retrieval de
solvency2_rag (voir analyse.md a la racine de research-financebench) :
- extraction en combinaisons de mots-cles ordonnees par priorite, chacune
  avec des synonymes optionnels ("~"), au lieu du format PRIMARY/SECONDARY/
  TERTIARY d'origine.
- recherche BM25 COMBINAISON PAR COMBINAISON (une requete independante par
  combo, avec son propre quota d'ancres selon son rang de priorite), au lieu
  d'une seule requete poolee sur tous les termes fusionnes.
- correspondance par PALIERS (tier_correspondance : 3=formulation principale
  complete/mot entier, 2=synonyme complet, 1=partielle, 0=exclue) pour choisir
  les ancres de contagion, au lieu d'une simple detection de sous-chaine.
- consigne de prompt explicite : prioriser les sections les plus imbriquees/
  specifiques de la table des matieres plutot qu'une section parente large.
- plus de repli arbitraire (1ere+derniere unite du bloc) quand aucune ancre
  n'est trouvee : un bloc sans ancre pour un combo contribue simplement zero
  candidat.
- LLM : Gemini (2.5-flash par defaut, 3-flash-preview en secours).
"""

import os
import re
import json
import pickle
import time
from dataclasses import dataclass, field

import google.generativeai as genai

from rag_common import tokenize_for_bm25

BASE_DIR = os.path.dirname(__file__)
RAG_INDEX_FILE = os.path.join(BASE_DIR, "rag_index_financebench.pkl")
TOC_DIR = os.path.join(BASE_DIR, "corpus_toc")
CORPUS_MD_DIR = os.path.join(BASE_DIR, "corpus_md")

CHARS_PER_TOKEN = 4
RAG_MAX_TOKENS_SECTIONS = 4000
RAG_MAX_TOKENS_BM25 = 3000
RAG_TOP_K_BLOCS = 20

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-3-flash-preview"

COMBO_QUOTAS = [3, 2, 1, 1, 1, 1]  # unites etendues (passages) max par combinaison, selon son rang de priorite
# Identique a solvency2_rag. Teste avec un quota plus large ([4,3,2,2,1,1]) et un budget BM25
# releve a 4000 tokens : sur les 150 questions, cette variante degradait l'accuracy (52,0% contre
# 55,3% ici) sans ameliorer le recall de facon significative -- retenu tel quel, meilleur
# compromis precision/accuracy mesure empiriquement sur ce corpus.


@dataclass
class ComboMotsCles:
    """Une combinaison de mots-cles avec ses formulations synonymes optionnelles (voir la
    syntaxe "principal" ~ "synonyme1", "synonyme2" du prompt d'extraction)."""
    principal: str
    alternates: list = field(default_factory=list)


@dataclass
class SearchQuery:
    """Combinaisons de mots-cles ordonnees par priorite decroissante (voir COMBO_QUOTAS)."""
    combos: list

    def is_empty(self):
        return not self.combos


with open(RAG_INDEX_FILE, "rb") as f:
    RAG_INDEX = pickle.load(f)

TOC_BY_FILE = {}
for fname in os.listdir(TOC_DIR):
    with open(os.path.join(TOC_DIR, fname), encoding="utf-8") as f:
        toc = json.load(f)
    TOC_BY_FILE[toc["source"]] = toc

MD_LINES_BY_FILE = {}
for fname in os.listdir(CORPUS_MD_DIR):
    with open(os.path.join(CORPUS_MD_DIR, fname), encoding="utf-8") as f:
        MD_LINES_BY_FILE[fname] = f.read().splitlines()


import threading

_token_usage = threading.local()  # par-thread : evite les races en execution parallele (ThreadPoolExecutor)


def reset_token_counter():
    _token_usage.total = 0


def get_token_counter():
    return getattr(_token_usage, "total", 0)


def call_llm(prompt, retries=3):
    """total_token_count inclut les tokens de raisonnement internes (facture
    mais pas detaille separement dans cette version du SDK) -- c'est le vrai
    cout consomme, pas une approximation caracteres/4."""
    last_err = None
    for model_name in [LLM_MODEL, FALLBACK_MODEL]:
        try:
            model = genai.GenerativeModel(model_name=model_name,
                                           generation_config={"temperature": 0.2, "top_p": 0.95, "top_k": 40})
            result = model.generate_content(prompt)
            if result.candidates and result.candidates[0].content.parts:
                if getattr(result, "usage_metadata", None):
                    _token_usage.total = getattr(_token_usage, "total", 0) + result.usage_metadata.total_token_count
                return result.candidates[0].content.parts[0].text
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue
    raise RuntimeError(f"LLM indisponible: {last_err}")


# ============================================================
# EXTRACTION MOTS-CLES + SECTIONS
# ============================================================

def extract_quoted(text):
    # Guillemets uniquement (" et «) : l'apostrophe est exclue des delimiteurs pour ne pas
    # tronquer une combinaison contenant une apostrophe interne (ex. "shareholders' equity").
    quoted = re.findall(r'["«]([^"»]{2,80})["»]', text)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]
    # Repli si le LLM n'a pas mis de guillemets : decoupage simple sur les virgules.
    parts = re.split(r"[,;]", text)
    termes = []
    for p in parts:
        clean = re.sub(r'^["\'«»\[\]\(\)]+|["\'«»\[\]\(\)]+$', "", p).strip()
        clean = re.sub(r"[.:!?]+$", "", clean).strip()
        if clean and 1 < len(clean) < 80:
            termes.append(clean)
    return termes


STOPWORDS_QUESTION = {"what", "which", "how", "why", "does", "that", "from", "with",
                       "were", "was", "are", "the", "for", "this", "have", "has"}


def parser_mots_cles(response, question):
    """Parse la reponse du LLM : une liste de combinaisons de mots-cles, ordonnee par priorite
    decroissante, chacune avec des formulations synonymes optionnelles introduites par "~"."""
    combos = []
    match = re.search(r"KEYWORDS?\s*[:\-]\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        ligne = match.group(1)
        if ";" in ligne:
            # Format attendu : "principal1" ~ "syn1a", "syn1b" ; "principal2" ; ...
            for groupe in ligne.split(";"):
                groupe = groupe.strip()
                if not groupe:
                    continue
                principal_part, _, alt_part = groupe.partition("~")
                principaux = extract_quoted(principal_part)
                if not principaux:
                    continue
                combos.append(ComboMotsCles(principal=principaux[0], alternates=extract_quoted(alt_part)))
        else:
            # Repli : pas de ";" (donc pas de synonymes) -- chaque terme entre guillemets
            # (ou separe par des virgules) est une combinaison simple.
            combos = [ComboMotsCles(principal=c) for c in extract_quoted(ligne)]

    if not combos:
        words = [w for w in re.findall(r"\b\w+\b", question.lower())
                  if len(w) > 3 and w not in STOPWORDS_QUESTION]
        combos = [ComboMotsCles(principal=w) for w in words[:4]]

    return SearchQuery(combos=combos[:len(COMBO_QUOTAS)])


def extraire_mots_cles_et_sections(question, target_file):
    toc = TOC_BY_FILE[target_file]
    prompt = f"""You are an expert assistant helping answer a question about ONE specific SEC financial filing (10-K/10-Q).

QUESTION: "{question}"

TABLE OF CONTENTS OF THIS FILING:
{toc['toc_text']}

TASK: analyze the question and the table of contents to provide:
1. KEYWORD COMBINATIONS for text search within this filing
2. SPECIFIC SECTIONS of this filing to extract, identified by [Lxx|filename]

STRICT RESPONSE FORMAT:

KEYWORDS: "combination1" ~ "synonym1a", "synonym1b" ; "combination2" ; "combination3" ~ "synonym3a"

SECTIONS: [L45|{target_file}], [L129|{target_file}]

RULES FOR KEYWORD COMBINATIONS:
- 1 to 6 combinations maximum, in quotes, on a single line, SEPARATED BY SEMICOLONS ";"
  (not commas: commas separate synonyms inside a combination)
- ORDER = DECREASING PRIORITY (token budget constraint): put the most decisive combination
  for answering the question first, least important last
- A combination can be A SINGLE WORD ("revenue") or MULTIPLE WORDS combined into an expression
  to search together ("total net sales"): the strongest match requires ALL its words to be
  present in the same passage -- only combine words that truly must appear together
- SYNONYMS: after a combination, add "~" followed by one or more synonym phrasings in quotes
  separated by commas, e.g. "total revenue" ~ "net sales", "total net revenue" -- useful when
  the wording of the question differs from the filing's exact vocabulary (line-item label,
  abbreviation, equivalent phrasing). A synonym never replaces the main phrasing, it is added
  as a safety net. A combination without "~" remains valid.
- Think about precise financial/accounting vocabulary and line-item labels as they literally
  appear in the financial statements (e.g. "total revenues", "cost of sales", "net income
  attributable to")

RULES FOR SECTIONS:
- At most 5, ordered by priority, "none" if no relevant section
- Prioritize the MOST NESTED/SPECIFIC sections in the table of contents over a broad parent
  section: a nested section is more targeted and shorter, hence cheaper in tokens -- a parent
  section also embeds the full text of all its own subsections, which can exhaust the token
  budget before other requested sections get extracted.

Now answer:"""

    response = call_llm(prompt)
    keywords = parser_mots_cles(response, question)

    section_requests = []
    m = re.search(r"SECTIONS\s*[:\-]\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if m and m.group(1).strip().lower() not in ("none", "aucune", ""):
        pairs = re.findall(r"\[L(\d+)\|([^\]]+)\]", m.group(1))
        for priority, (line_num, fname) in enumerate(pairs[:5], start=1):
            section_requests.append((f"L{line_num}", fname.strip(), priority))

    return keywords, section_requests


def normaliser_titre(titre):
    titre = re.sub(r"\*\*([^*]+)\*\*", r"\1", titre or "")
    titre = re.sub(r"\*([^*]+)\*", r"\1", titre)
    return " ".join(titre.split()).strip()


def extraire_sections_demandees(section_requests, max_tokens=RAG_MAX_TOKENS_SECTIONS):
    if not section_requests:
        return "", set(), []
    max_chars = max_tokens * CHARS_PER_TOKEN
    chars_total = 0
    contexte = ""
    titres_couverts = set()
    bloc_sources_used = []
    for section_id, fname, priority in sorted(section_requests, key=lambda x: x[2]):
        toc = TOC_BY_FILE.get(fname)
        if not toc:
            continue
        section_info = next((s for s in toc["sections"] if s["id"] == section_id), None)
        if not section_info:
            continue
        lines = MD_LINES_BY_FILE.get(fname)
        if not lines:
            continue
        text = "\n".join(lines[section_info["line_start"] - 1: section_info["line_end"]])
        if chars_total + len(text) > max_chars:
            remaining = max_chars - chars_total
            if remaining < 500:
                break
            text = text[:remaining] + "\n[...section truncated...]"
        contexte += f"\n\n=== {fname} | {section_info['title']} ===\n{text}"
        chars_total += len(text)
        titres_couverts.add(normaliser_titre(section_info["title"]))
        bloc_sources_used.append((fname, section_info["line_start"], section_info["line_end"]))
    return contexte, titres_couverts, bloc_sources_used


# ============================================================
# RECHERCHE BM25 PAR COMBINAISON + CORRESPONDANCE PAR PALIERS
# ============================================================

def recherche_bm25_combo(combo, target_file, top_k=RAG_TOP_K_BLOCS):
    """Recherche BM25 pour UNE combinaison de mots-cles, independamment des autres, sur
    l'index local (par document) du fichier cible. Les tokens de la formulation principale ET
    de ses synonymes eventuels sont combines pour cette recherche."""
    entry = RAG_INDEX["bm25_by_source"].get(target_file)
    if not entry:
        return []
    bm25_local, bloc_ids = entry["bm25"], entry["bloc_ids"]

    tokens = tokenize_for_bm25(combo.principal)
    for alt in combo.alternates:
        tokens.extend(tokenize_for_bm25(alt))
    if not tokens:
        return []

    local_scores = bm25_local.get_scores(tokens)
    results = [(bloc_ids[i], score) for i, score in enumerate(local_scores) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _mots_significatifs(phrase):
    """Tokenise une phrase et ecarte les mots de 1-2 lettres de l'exigence "tous les mots
    presents" : trop frequents/peu discriminants en correspondance sur mot entier."""
    mots = tokenize_for_bm25(phrase)
    significatifs = [m for m in mots if len(m) > 2]
    return significatifs or mots


def _phrase_entierement_presente(texte_lower, phrase):
    """Tous les mots (significatifs) de la phrase sont presents, mot entier, dans le texte."""
    mots = _mots_significatifs(phrase)
    if not mots:
        return False
    return all(re.search(r"\b" + re.escape(m) + r"\b", texte_lower) for m in mots)


def tier_correspondance(texte_unite, combo):
    """Palier de correspondance d'une unite de texte pour une combinaison, du meilleur au moins
    bon :
    3 = tous les mots de la formulation PRINCIPALE presents dans l'unite (mot entier)
    2 = pas de correspondance complete sur le principal, mais tous les mots d'un SYNONYME le sont
    1 = ni l'un ni l'autre, mais au moins un mot (principal ou synonyme) est present
    0 = rien trouve -> l'unite n'est pas retenue comme ancre
    """
    texte_lower = texte_unite.lower()

    if _phrase_entierement_presente(texte_lower, combo.principal):
        return 3

    if any(_phrase_entierement_presente(texte_lower, alt) for alt in combo.alternates):
        return 2

    tous_mots = _mots_significatifs(combo.principal)
    for alt in combo.alternates:
        tous_mots.extend(_mots_significatifs(alt))
    if any(re.search(r"\b" + re.escape(m) + r"\b", texte_lower) for m in tous_mots):
        return 1

    return 0


def extraire_texte_unites(bloc, unites_ids):
    if not unites_ids:
        return ""
    textes, previous_id = [], None
    for uid in unites_ids:
        if previous_id is not None and uid > previous_id + 1:
            textes.append("[...]")
        textes.append(bloc.unites[uid].texte)
        previous_id = uid
    return " ".join(textes)


def agreger_contexte(query, target_file, max_tokens=RAG_MAX_TOKENS_BM25, titres_couverts=None):
    """Recherche BM25 combinaison par combinaison (query.combos, deja ordonnees par priorite
    decroissante par le LLM) et agrege les passages ("unites etendues") trouves, en evitant les
    doublons avec la PARTIE A (TOC).

    Pour chaque combinaison, les unites candidates (dans ses meilleurs blocs BM25) sont d'abord
    notees par PALIER de correspondance (tier_correspondance), puis triees par palier decroissant
    (et a palier egal par score BM25 du bloc) avant de piocher jusqu'au quota de la combinaison
    (COMBO_QUOTAS) -- les correspondances les plus precises sont donc toujours retenues en
    priorite, sans pour autant exclure les partielles si rien de mieux n'est disponible. Aucun
    repli arbitraire : un bloc sans unite en tier > 0 pour un combo contribue zero candidat.
    """
    titres_couverts = titres_couverts or set()
    blocs = RAG_INDEX["blocs"]
    max_chars = max_tokens * CHARS_PER_TOKEN
    chars_total = 0
    contexte = ""
    bloc_sources_used = []
    passages_utilises = set()  # (bloc_id, unites_du_passage) deja extraits, toutes combos confondues
    budget_atteint = False

    for rang, combo in enumerate(query.combos[:len(COMBO_QUOTAS)]):
        if budget_atteint:
            break

        quota = COMBO_QUOTAS[rang]

        candidats = []  # (tier, score_bm25, bloc_id, unite_id)
        for bloc_id, score_bm25 in recherche_bm25_combo(combo, target_file):
            bloc = blocs[bloc_id]
            if normaliser_titre(bloc.titre) in titres_couverts:
                continue
            for unite in bloc.unites:
                tier = tier_correspondance(unite.texte, combo)
                if tier > 0:
                    candidats.append((tier, score_bm25, bloc_id, unite.id))

        candidats.sort(key=lambda c: (c[0], c[1]), reverse=True)

        n_trouves = 0
        for tier, score_bm25, bloc_id, ancre_id in candidats:
            if n_trouves >= quota:
                break

            bloc = blocs[bloc_id]
            groupe = next((g for g in bloc.contagion_groups if ancre_id in g), [ancre_id])
            passage_key = (bloc_id, tuple(sorted(groupe)))
            if passage_key in passages_utilises:
                continue
            passages_utilises.add(passage_key)

            texte_extrait = extraire_texte_unites(bloc, sorted(groupe))
            if not texte_extrait:
                continue

            if chars_total + len(texte_extrait) > max_chars:
                remaining = max_chars - chars_total
                if remaining > 500:
                    texte_extrait = texte_extrait[:remaining] + " [...]"
                else:
                    budget_atteint = True
                    break

            contexte += f"\n\n=== {bloc.source} | {' > '.join(bloc.headers)} ===\n{texte_extrait}"
            chars_total += len(texte_extrait)
            bloc_sources_used.append((bloc_id, sorted(groupe)))
            n_trouves += 1

        if budget_atteint:
            break

    return contexte, bloc_sources_used


def retrieve(question, target_file, max_tokens_sections=RAG_MAX_TOKENS_SECTIONS, max_tokens_bm25=RAG_MAX_TOKENS_BM25):
    query, section_requests = extraire_mots_cles_et_sections(question, target_file)
    contexte_a, titres_couverts, section_sources = extraire_sections_demandees(section_requests, max_tokens=max_tokens_sections)
    contexte_b = ""
    bloc_sources = []
    if not query.is_empty():
        contexte_b, bloc_sources = agreger_contexte(query, target_file, max_tokens=max_tokens_bm25, titres_couverts=titres_couverts)
    return contexte_a + contexte_b, section_sources, bloc_sources


def generate_answer(question, context):
    """Prompt ORIGINAL (pas aligne BookRAG) : concis, dit 'Unanswerable' si absent."""
    if not context.strip():
        return "Unanswerable: no relevant context was retrieved."

    prompt = f"""You are an expert assistant answering questions about a SEC financial filing, based only on the provided context extracted from the filing.

QUESTION: {question}

CONTEXT (extracted from the filing via table-of-contents navigation and keyword search):
{context}

RULES:
- Answer using only the information in the context above.
- If the answer is a short factual span (a number, a name, a short phrase), give it directly and concisely.
- If the question cannot be answered from the context, say "Unanswerable".
- Be precise and concise; do not add unrelated commentary.

ANSWER:"""
    return call_llm(prompt)


def answer_question(question, target_file, max_tokens_sections=RAG_MAX_TOKENS_SECTIONS, max_tokens_bm25=RAG_MAX_TOKENS_BM25):
    reset_token_counter()
    context, section_sources, bloc_sources = retrieve(question, target_file, max_tokens_sections, max_tokens_bm25)
    answer = generate_answer(question, context)
    tokens_used = get_token_counter()
    return answer, context, section_sources, bloc_sources, tokens_used
