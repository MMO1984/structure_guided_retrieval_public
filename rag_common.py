"""
RAG_COMMON.PY
=============
Classes et fonctions communes partagées entre prepare_embeddings.py et rag_search.py.
"""

import re
from typing import List
from dataclasses import dataclass, field


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class Unite:
    """Une unité = une ou plusieurs phrases regroupées."""
    id: int
    phrases_indices: List[int]  # Indices des phrases originales
    texte: str
    embedding: List[float] = field(default_factory=list)
    
    def __repr__(self):
        return f"Unite({self.id}, phrases={self.phrases_indices}, len={len(self.texte)})"


@dataclass
class Bloc:
    """Un bloc = un paragraphe avec ses headers de rattachement."""
    id: int
    source: str  # Fichier source
    headers: List[str]  # Hiérarchie des headers
    level: int  # Niveau du header (#, ##, ###, ####)
    titre: str  # Titre du bloc (dernier header)
    contenu_complet: str  # Texte brut complet
    phrases: List[str]  # Liste des phrases
    unites: List[Unite] = field(default_factory=list)
    contagion_groups: List[List[int]] = field(default_factory=list)  # Groupes pré-calculés d'unités adjacentes sémantiquement liées
    
    def get_texte_pour_bm25(self) -> str:
        """Retourne le texte à indexer pour BM25 (headers + contenu)."""
        headers_text = " ".join(self.headers)
        return f"{headers_text} {self.contenu_complet}"


# ==========================================
# TOKENIZER BM25
# ==========================================

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer pour BM25.
    Défini ici pour être partagé et sérialisable.
    """
    text = text.lower()
    # Garder lettres, chiffres, caractères arabes
    tokens = re.findall(r'[\w\u0600-\u06FF]+', text)
    return tokens