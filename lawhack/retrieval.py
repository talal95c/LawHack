"""Dependency-free BM25: plain-text index (connector) and registry index with pinned segments (Système 2)."""

import math
import re
import unicodedata
from collections import Counter

from lawhack.schema import Registry, RegistryEntry, Speaker, Zone

_STOPWORDS = set(
    "a au aux avec ce ces cet cette d dans de des du elle en est et il ils l la le les leur lui m mais me "
    "n ne ni nos notre on ou par pas pour qu que qui s sa se ses si son sur t ta te tes ton un une vos votre "
    "y à été être a-t-il ont sont était quel quelle quels quelles est-ce".split()
)


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS and len(t) > 1]


def _stem(token: str) -> str:
    """Very light French stemming: rejette / rejeté / rejetés → rejet."""
    if token.isdigit() or len(token) <= 4:
        return token
    token = re.sub(r"(.)\1", r"\1", token)
    return re.sub(r"(ent|es|e|s)$", "", token) or token


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokens(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out

    def top(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]


def coverage(query: str, text: str) -> float:
    """Share of the query's content words that appear in `text`."""
    q = set(tokens(query))
    return len(q & set(tokens(text))) / len(q) if q else 0.0


_WORD = re.compile(r"[a-z0-9]+")
_REGISTRY_STOPWORDS = set(
    "le la les l un une des du de d au aux et ou a en dans par pour sur que qui quoi est sont ce cette ces se sa son ses"
    " il elle ils elles lui leur leurs ne pas plus n s y on nous vous je tu me te t qu c avec sans etre ete avait"
    " a-t-elle t-il".split()
)
SEARCHABLE = {Zone.EXPOSE, Zone.MOYENS, Zone.MOTIVATIONS, Zone.DISPOSITIF}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [_registry_stem(w) for w in _WORD.findall(text) if w not in _REGISTRY_STOPWORDS and len(w) > 1]


def _registry_stem(word: str) -> str:
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


def pinned(registry: Registry) -> list[RegistryEntry]:
    """What the Cour itself says (its own reasons and the dispositif): short, and the ground truth of the decision."""
    return [e for e in registry.entries if e.zone is Zone.DISPOSITIF
            or (e.zone is Zone.MOTIVATIONS and e.speaker.value == Speaker.COUR_CASSATION.value)]


def retrieve(registry: Registry, query: str, top_k: int = 8, index: BM25Index | None = None) -> list[RegistryEntry]:
    """Top BM25 hits plus the pinned segments, which BM25 often misses on "why" questions."""
    hits = [e for e, _ in (index or BM25Index(registry)).search(query, top_k)]
    seen = {e.id for e in hits}
    hits += [e for e in pinned(registry) if e.id not in seen]
    order = {e.id: i for i, e in enumerate(registry.entries)}
    return sorted(hits, key=lambda e: order[e.id])
