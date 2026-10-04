"""BM25 retrieval over registry entries (an arrêt is small: no embeddings needed in V1)."""

import math
import re
import unicodedata
from collections import Counter

from lawhack.schema import Registry, RegistryEntry, Zone

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "le la les l un une des du de d au aux et ou a en dans par pour sur que qui quoi est sont ce cette ces se sa son ses"
    " il elle ils elles lui leur leurs ne pas plus n s y on nous vous je tu me te t qu c avec sans etre ete avait"
    " a-t-elle t-il".split()
)
SEARCHABLE = {Zone.EXPOSE, Zone.MOYENS, Zone.MOTIVATIONS, Zone.DISPOSITIF}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [_stem(w) for w in _WORD.findall(text) if w not in _STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    for suffix in ("ements", "ement", "ations", "ation", "es", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class BM25Index:
    def __init__(self, registry: Registry, k1: float = 1.5, b: float = 0.75):
        self.entries = [e for e in registry.entries if e.zone in SEARCHABLE]
        self.docs = [tokenize(e.text) for e in self.entries]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 8) -> list[tuple[RegistryEntry, float]]:
        terms = tokenize(query)
        scored = []
        for entry, doc in zip(self.entries, self.docs):
            tf = Counter(doc)
            score = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            if score > 0:
                scored.append((entry, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


def retrieve(registry: Registry, query: str, top_k: int = 8, index: BM25Index | None = None) -> list[RegistryEntry]:
    """Top BM25 hits plus the dispositif (short, and the ground truth of what the Cour decides)."""
    hits = [e for e, _ in (index or BM25Index(registry)).search(query, top_k)]
    seen = {e.id for e in hits}
    hits += [e for e in registry.entries if e.zone is Zone.DISPOSITIF and e.id not in seen]
    order = {e.id: i for i, e in enumerate(registry.entries)}
    return sorted(hits, key=lambda e: order[e.id])
