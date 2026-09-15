#!/usr/bin/env python3
"""
Metriques comparables a HiREC (Table 6) : Page Recall/Precision + Answer
Accuracy. Page Recall/Precision utilisent les numeros de page reels
(page_num, 0-indexe PyMuPDF, recupere depuis le dataset LOFin de HiREC).
Answer Accuracy : egalite numerique tolerante (arrondi/troncature, comme
FinanceBench original et HiREC) pour les reponses chiffrees ; jugement LLM
(esprit FAMMA -- verifier la correspondance semantique avec le gold, pas
une correspondance exacte de chaine) pour les reponses textuelles. Note :
prompt de jugement simplifie, pas une reproduction caractere pour
caractere du prompt FAMMA original (Xue et al. 2024) -- a signaler
explicitement dans toute presentation des resultats.
"""

import os
import re
import time

import google.generativeai as genai

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
JUDGE_MODEL = "gemini-2.5-flash"
JUDGE_FALLBACK_MODEL = "gemini-3-flash-preview"


def call_judge(prompt, retries=3):
    last_err = None
    for model_name in [JUDGE_MODEL, JUDGE_FALLBACK_MODEL]:
        try:
            model = genai.GenerativeModel(model_name=model_name, generation_config={"temperature": 0.0})
            result = model.generate_content(prompt)
            if result.candidates and result.candidates[0].content.parts:
                return result.candidates[0].content.parts[0].text
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"Judge LLM indisponible: {last_err}")


def page_recall_precision(bloc_pages_used, gold_pages):
    """bloc_pages_used : ensemble de pages couvertes par le contexte retrieve.
    gold_pages : ensemble de pages annotees comme evidence (gold)."""
    if not gold_pages:
        return None, None
    retrieved = set(bloc_pages_used)
    gold = set(gold_pages)
    if not retrieved:
        return 0.0, 0.0
    inter = retrieved & gold
    recall = len(inter) / len(gold)
    precision = len(inter) / len(retrieved)
    return recall, precision


def parse_numeric(text):
    """Extrait un nombre d'une chaine (gere $, %, virgules milliers, mots million/billion)."""
    if text is None:
        return None
    t = str(text).strip().lower()
    multiplier = 1.0
    if "billion" in t:
        multiplier = 1e9
    elif "million" in t:
        multiplier = 1e6
    elif "thousand" in t:
        multiplier = 1e3
    t = re.sub(r"[a-z,$%]", "", t)
    m = re.search(r"-?\d+\.?\d*", t)
    if not m:
        return None
    try:
        return float(m.group()) * multiplier
    except ValueError:
        return None


def is_numeric_gold(gold):
    """Gold = valeur chiffree pure (ex. '1577.00', '$1,577 million', '30.8%'),
    PAS une phrase narrative qui contient des chiffres au milieu de mots
    (ex. 'declined from 36.8% to 34.6%...') -- sinon parse_numeric extrait
    le mauvais nombre pour la comparaison. Heuristique : peu de mots ET
    aucun verbe/mot de liaison typique d'une phrase."""
    if parse_numeric(gold) is None:
        return False
    words = str(gold).strip().split()
    if len(words) > 4:
        return False
    return True


def numeric_match(gold, predicted, rel_tol=0.02):
    gold_val = parse_numeric(gold)
    pred_val = parse_numeric(predicted)
    if gold_val is None or pred_val is None:
        return 0.0
    if gold_val == 0:
        return 1.0 if abs(pred_val) < 1e-6 else 0.0
    return 1.0 if abs(pred_val - gold_val) / abs(gold_val) <= rel_tol else 0.0


def textual_judge_match(question, gold, predicted):
    """Jugement LLM esprit FAMMA (verifie la correspondance semantique, pas
    une chaine exacte) -- prompt simplifie, pas la reproduction exacte du
    prompt FAMMA original."""
    prompt = f"""You are grading a financial QA system. Given the question, the correct answer, and the model's response, judge whether the model's response conveys the same correct answer (allow paraphrasing, different formatting, or additional correct supporting detail -- what matters is whether the core factual claim matches).

Question: {question}
Correct Answer: {gold}
Model Response: {predicted}

Respond with ONLY one word: "Correct" or "Incorrect"."""
    raw = call_judge(prompt).strip().lower()
    return 1.0 if "correct" in raw and "incorrect" not in raw else 0.0


def answer_accuracy(question, gold, predicted):
    if is_numeric_gold(gold):
        return numeric_match(gold, predicted)
    return textual_judge_match(question, gold, predicted)
