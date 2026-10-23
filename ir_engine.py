"""
ir_engine.py - Hybrid information retrieval engine (no UI code).

Pipeline
  ingest -> passage chunking -> index (BM25 + TF-IDF/LSA [+ optional dense embeddings])
  query  -> typo correction (corpus vocabulary only) -> weighted expansion
            (domain thesaurus + LSA neighbours + pseudo-relevance feedback)
         -> BM25 and semantic scoring -> Reciprocal Rank Fusion
         -> group passages into documents -> snippets + highlighting
         -> cluster the *result set* (SVD + KMeans, k by silhouette) with term labels
"""
from __future__ import annotations

import html
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
STOP = set(ENGLISH_STOP_WORDS)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def stem(w: str) -> str:
    """Tiny suffix stripper (keeps the project dependency-free)."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ses") and len(w) > 4:
        base = w[:-2]
        return base if base.endswith(("us", "ss", "is")) else w[:-1]
    if w.endswith(("ches", "shes", "xes", "zes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def analyze(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOP and len(t) > 1]


def split_passages(text: str, target_words: int = 70) -> list[str]:
    """Paragraph-aware chunking into ~target_words passages on sentence boundaries."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        cur: list[str] = []
        n = 0
        for s in SENT_RE.split(para):
            w = len(s.split())
            if cur and n + w > target_words * 1.5:
                out.append(" ".join(cur))
                cur, n = [], 0
            cur.append(s)
            n += w
            if n >= target_words:
                out.append(" ".join(cur))
                cur, n = [], 0
        if cur:
            out.append(" ".join(cur))
    return [p for p in out if len(p.split()) >= 3]


def guess_title(name: str, text: str) -> str:
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if first and len(first) <= 90 and not first.endswith("."):
        return first
    return Path(name).stem.replace("_", " ").title()


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        import io
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return ""


def load_folder(path: str | Path) -> list[tuple[str, str]]:
    root = Path(path)
    out = []
    for p in sorted(root.rglob("*")):
        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            text = p.read_text(errors="ignore")
        elif ext == ".pdf":
            text = read_pdf_bytes(p.read_bytes())
        else:
            continue
        if text.strip():
            out.append((str(p.relative_to(root)), text))
    return out


# ----------------------------------------------------------------------------
# Optional dense embedder
# ----------------------------------------------------------------------------
class STEmbedder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=False))


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class Passage:
    doc: int
    text: str


@dataclass
class QueryPlan:
    original: str
    corrected: str
    corrections: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)   # stem -> weight
    origin: dict[str, str] = field(default_factory=dict)      # stem -> original|synonym|lsa|feedback
    embed_text: str = ""

    @property
    def suggestion(self) -> str | None:
        return self.corrected if self.corrections else None


@dataclass
class Hit:
    doc_id: int
    name: str
    title: str
    score: float              # relative to best hit (0-1)
    passage_id: int
    snippet_html: str
    matched: list[str]
    bm25_rank: int | None
    sem_rank: int | None
    more_passages: int
    cluster: int = 0


@dataclass
class SearchResult:
    plan: QueryPlan
    hits: list[Hit]
    cluster_names: list[str]
    silhouette: float
    ms: float


@dataclass
class ClusterResult:
    labels: np.ndarray
    names: list[str]
    silhouette: float


# ----------------------------------------------------------------------------
# Baseline (what the original project did) - used for evaluation only
# ----------------------------------------------------------------------------
class BaselineTfidf:
    def __init__(self, docs: list[tuple[str, str]]):
        self.names = [n for n, _ in docs]
        self.vec = TfidfVectorizer(stop_words="english")
        self.X = self.vec.fit_transform([t for _, t in docs])

    def search(self, query: str, top: int = 10) -> list[str]:
        s = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        return [self.names[i] for i in np.argsort(-s)[:top] if s[i] > 0]


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class SearchEngine:
    def __init__(self, docs: list[tuple[str, str]], thesaurus_path: str | None = None,
                 embedder=None, k1: float = 1.5, b: float = 0.75, passage_words: int = 70):
        self.docs = docs
        self.titles = [guess_title(n, t) for n, t in docs]
        self.passages: list[Passage] = []
        for di, (_, text) in enumerate(docs):
            self.passages += [Passage(di, p) for p in split_passages(text, passage_words)]
        texts = [p.text for p in self.passages]
        if not texts:
            raise ValueError("No indexable text found.")

        # --- vocabulary and counts ------------------------------------------------
        self.cv = CountVectorizer(analyzer=analyze)
        C = self.cv.fit_transform(texts).tocsr().astype(np.float64)
        self.vocab: dict[str, int] = self.cv.vocabulary_
        self.inv = {i: t for t, i in self.vocab.items()}
        self.V = C.shape[1]
        self.N = C.shape[0]

        # --- BM25 matrix (passages x terms), query-time = one sparse mat-vec -------
        df = np.diff(C.tocsc().indptr)
        idf = np.log(1 + (self.N - df + 0.5) / (df + 0.5))
        dl = np.asarray(C.sum(axis=1)).ravel()
        avg = max(dl.mean(), 1e-9)
        W = C.copy()
        rows = np.repeat(np.arange(self.N), np.diff(C.indptr))
        tf = C.data
        W.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / avg)) * idf[C.indices]
        self.bm25 = W.tocsr()

        # --- TF-IDF + LSA -----------------------------------------------------------
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.X = self.tfidf.fit_transform(C).tocsr()
        self.idf = self.tfidf.idf_
        ncomp = max(2, min(100, min(self.X.shape) - 1))
        self.svd = TruncatedSVD(n_components=ncomp, random_state=0)
        self.Z = normalize(self.svd.fit_transform(self.X))
        self.T = normalize(self.svd.components_.T)          # term vectors (V x k)

        # --- document-level aggregates (for corpus topic map) ----------------------
        D = sp.csr_matrix(([1.0] * self.N, ([p.doc for p in self.passages], range(self.N))),
                          shape=(len(docs), self.N))
        self.docZ = normalize(D @ self.Z)
        self.docX = normalize(D @ self.X)

        # --- surface vocabulary for typo correction --------------------------------
        surface = Counter(t for tx in texts for t in tokens(tx)
                          if t not in STOP and t.isalpha() and len(t) > 2)
        self.surface_df = surface
        self.typo_words = list(surface)
        self._max_df = max(surface.values()) if surface else 1
        by_stem: dict[str, Counter] = {}
        for w, c in surface.items():
            by_stem.setdefault(stem(w), Counter())[w] += c
        self.surface_of = {s: c.most_common(1)[0][0] for s, c in by_stem.items()}

        # --- thesaurus --------------------------------------------------------------
        self.syn_groups: list[list[str]] = []
        self.syn_index: dict[str, list[int]] = {}
        if thesaurus_path and Path(thesaurus_path).exists():
            for g in json.loads(Path(thesaurus_path).read_text())["groups"]:
                stems = list(dict.fromkeys(stem(t) for ph in g for t in tokens(ph)
                                           if t not in STOP))
                gid = len(self.syn_groups)
                self.syn_groups.append(stems)
                for s in stems:
                    self.syn_index.setdefault(s, []).append(gid)

        # --- optional dense embeddings --------------------------------------------
        self.embedder = embedder
        self.E = embedder.encode(texts) if embedder is not None else None

    # ------------------------------------------------------------------ typo layer
    def correct(self, tok: str) -> str | None:
        """Correct only if the token is OOV. Candidates come from the corpus vocabulary."""
        if len(tok) < 5:                      # short words are too ambiguous to auto-correct
            return None
        cands = process.extract(tok, self.typo_words,
                                scorer=DamerauLevenshtein.normalized_similarity,
                                limit=8, score_cutoff=0.74)
        # Guards against "correcting" valid words that merely are not in a small corpus:
        #  - same first letter (typos rarely hit the first character)
        #  - skip morphological variants (respond -> responder)
        cands = [c for c in cands
                 if c[0][0] == tok[0] and not c[0].startswith(tok) and not tok.startswith(c[0])]
        if not cands:
            return None
        def key(c):  # similarity first, document frequency as tie-breaker
            w, score, _ = c
            return score + 0.02 * math.log1p(self.surface_df[w]) / math.log1p(self._max_df)
        return max(cands, key=key)[0]

    # ------------------------------------------------------------- query planning
    def make_plan(self, query: str, typo: bool = True, expand: bool = True,
                  lsa_expand: bool = True, lsa_threshold: float = 0.75,
                  syn_weight: float = 0.4, max_syn_per_term: int = 5) -> QueryPlan:
        corrected_tokens, corrections = [], {}
        for t in tokens(query):
            new = t
            if typo and t not in STOP and t.isalpha() and stem(t) not in self.vocab:
                c = self.correct(t)
                if c:
                    new = c
                    corrections[t] = c
            corrected_tokens.append(new)

        plan = QueryPlan(original=query, corrected=" ".join(corrected_tokens), corrections=corrections)
        for t in corrected_tokens:
            if t in STOP or len(t) <= 1:
                continue
            plan.weights[stem(t)] = 1.0
            plan.origin[stem(t)] = "original"

        originals = list(plan.weights)
        if expand:
            for s in originals:                                   # domain thesaurus
                added = 0
                for gid in self.syn_index.get(s, []):
                    for m in self.syn_groups[gid]:
                        if m != s and m in self.vocab and m not in plan.weights and added < max_syn_per_term:
                            plan.weights[m] = syn_weight
                            plan.origin[m] = "synonym"
                            added += 1
        if lsa_expand:
            for s in originals:                                   # corpus-driven neighbours
                if s not in self.vocab:
                    continue
                sims = self.T @ self.T[self.vocab[s]]
                for j in np.argsort(-sims)[1:4]:
                    m = self.inv[int(j)]
                    if sims[j] >= lsa_threshold and m not in plan.weights:
                        plan.weights[m] = round(0.3 * float(sims[j]), 3)
                        plan.origin[m] = "lsa"
        self._set_embed_text(plan)
        return plan

    def _set_embed_text(self, plan: QueryPlan):
        extra = [self.surface_of.get(s, s) for s, o in plan.origin.items() if o != "original"]
        plan.embed_text = " ".join([plan.corrected] + extra)

    # ----------------------------------------------------------------- scoring
    def _qvec(self, weights: dict[str, float], idf: bool = False) -> np.ndarray:
        q = np.zeros(self.V)
        for s, w in weights.items():
            if s in self.vocab:
                q[self.vocab[s]] = w * (self.idf[self.vocab[s]] if idf else 1.0)
        return q

    def _bm25(self, weights) -> np.ndarray:
        return self.bm25 @ self._qvec(weights)

    def _semantic(self, plan: QueryPlan) -> np.ndarray:
        if self.E is not None:
            return self.E @ self.embedder.encode([plan.embed_text])[0]
        q = self._qvec(plan.weights, idf=True)
        if not q.any():
            return np.zeros(self.N)
        return self.Z @ normalize(self.svd.transform(q.reshape(1, -1)))[0]

    def _feedback(self, plan: QueryPlan, bm: np.ndarray, k_docs: int = 3, n_terms: int = 5):
        """Pseudo-relevance feedback (Rocchio-style) with low weights."""
        top = np.argsort(-bm)[:k_docs]
        v = np.asarray(self.X[top].sum(axis=0)).ravel()
        for s in plan.weights:
            if s in self.vocab:
                v[self.vocab[s]] = 0
        vmax = v.max()
        if vmax <= 0:
            return
        for j in np.argsort(-v)[:n_terms]:
            if v[j] > 0:
                s = self.inv[int(j)]
                plan.weights[s] = round(0.2 * float(v[j] / vmax), 3)
                plan.origin[s] = "feedback"
        self._set_embed_text(plan)

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, typo=True, expand=True, lsa_expand=True, semantic=True,
               prf=True, top_docs=15, cluster=True, rrf_k=60, w_sem=0.7, sem_min=0.2) -> SearchResult:
        t0 = time.perf_counter()
        plan = self.make_plan(query, typo=typo, expand=expand, lsa_expand=lsa_expand)
        if not any(s in self.vocab for s in plan.weights) and self.E is None:
            return SearchResult(plan, [], [], 0.0, (time.perf_counter() - t0) * 1000)

        bm = self._bm25(plan.weights)
        if prf and (bm > 0).any():
            self._feedback(plan, bm)
            bm = self._bm25(plan.weights)
        sem = self._semantic(plan) if semantic else None

        bm_rank = {int(i): r for r, i in enumerate(np.argsort(-bm)[:200]) if bm[i] > 0}
        sem_rank = ({int(i): r for r, i in enumerate(np.argsort(-sem)[:100]) if sem[i] > sem_min}
                    if sem is not None else {})
        fused = {}
        for i, r in bm_rank.items():
            fused[i] = fused.get(i, 0) + 1.0 / (rrf_k + r + 1)
        for i, r in sem_rank.items():
            fused[i] = fused.get(i, 0) + w_sem / (rrf_k + r + 1)
        if not fused:
            return SearchResult(plan, [], [], 0.0, (time.perf_counter() - t0) * 1000)

        order = sorted(fused, key=fused.get, reverse=True)
        per_doc = Counter(self.passages[i].doc for i in order)
        top_score = fused[order[0]]
        seen, hits = set(), []
        for pid in order:
            d = self.passages[pid].doc
            if d in seen:
                continue
            seen.add(d)
            ptext = self.passages[pid].text
            ptoks = set(analyze(ptext))
            hits.append(Hit(
                doc_id=d, name=self.docs[d][0], title=self.titles[d],
                score=fused[pid] / top_score, passage_id=pid,
                snippet_html=make_snippet(ptext, plan.weights, plan.origin),
                matched=sorted({self.surface_of.get(s, s) for s in plan.weights if s in ptoks}),
                bm25_rank=bm_rank.get(pid), sem_rank=sem_rank.get(pid),
                more_passages=per_doc[d] - 1))
            if len(hits) >= top_docs:
                break

        names, sil = ["All results"], 0.0
        if cluster and len(hits) >= 6:
            pids = [h.passage_id for h in hits]
            cr = self._cluster(self.Z[pids], self.X[pids], exclude=set(plan.weights))
            for h, l in zip(hits, cr.labels):
                h.cluster = int(l)
            names, sil = cr.names, cr.silhouette
        return SearchResult(plan, hits, names, sil, (time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------- clustering
    def _cluster(self, Z, X, exclude=frozenset(), kmax=6, min_sil=0.05) -> ClusterResult:
        n = Z.shape[0]
        labels, best_sil = np.zeros(n, dtype=int), 0.0
        if n >= 6:
            best = (-1.0, None)
            for k in range(2, min(kmax, n // 2) + 1):
                labs = KMeans(k, n_init=10, random_state=0).fit(Z).labels_
                if len(set(labs)) < 2:
                    continue
                sc = silhouette_score(Z, labs, metric="cosine")
                if sc > best[0]:
                    best = (sc, labs)
            if best[1] is not None and best[0] >= min_sil:
                labels, best_sil = best[1], float(best[0])
        ex = {self.vocab[s] for s in exclude if s in self.vocab}
        overall = np.asarray(X.mean(axis=0)).ravel()
        names = []
        for c in sorted(set(labels)):
            diff = np.asarray(X[labels == c].mean(axis=0)).ravel() - overall
            top = [j for j in np.argsort(-diff) if j not in ex][:3]
            names.append(" · ".join(self.surface_of.get(self.inv[int(j)], self.inv[int(j)])
                                    for j in top) or "All results")
        if len(set(labels)) == 1:
            names = ["All results"]
        return ClusterResult(labels, names, best_sil)

    def corpus_topics(self, kmax: int = 8) -> ClusterResult:
        return self._cluster(self.docZ, self.docX, kmax=kmax, min_sil=0.0)


# ----------------------------------------------------------------------------
# Snippets + highlighting
# ----------------------------------------------------------------------------
def highlight(text: str, weights: dict[str, float], origin: dict[str, str]) -> str:
    out, last = [], 0
    for m in TOKEN_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        w = m.group()
        s = stem(w.lower())
        if s in weights and w.lower() not in STOP and origin.get(s) != "feedback":
            cls = "o" if origin.get(s) == "original" else "e"
            out.append(f'<mark class="{cls}">{html.escape(w)}</mark>')
        else:
            out.append(html.escape(w))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def make_snippet(text: str, weights: dict[str, float], origin: dict[str, str],
                 max_chars: int = 320, window: int = 2) -> str:
    """Pick the sentence window with highest weighted keyword coverage + density."""
    sents = SENT_RE.split(text)
    best, best_score = (0, min(window, len(sents))), -1.0
    for i in range(max(1, len(sents) - window + 1)):
        chunk = " ".join(sents[i:i + window])
        toks = analyze(chunk)
        if not toks:
            continue
        distinct = {t for t in toks if t in weights}
        coverage = sum(weights[t] for t in distinct)
        density = sum(weights[t] for t in toks if t in weights) / math.sqrt(len(toks))
        score = coverage + 0.5 * density
        if score > best_score:
            best, best_score = (i, i + window), score
    snippet = " ".join(sents[best[0]:best[1]])
    ellipsis = ""
    if len(snippet) > max_chars:
        first = next((m.start() for m in TOKEN_RE.finditer(snippet)
                      if stem(m.group().lower()) in weights and m.group().lower() not in STOP), 0)
        start = max(0, first - max_chars // 3)
        start = snippet.rfind(" ", 0, start) + 1 if start else 0
        end = start + max_chars
        end = snippet.rfind(" ", start, end) if end < len(snippet) else len(snippet)
        snippet = ("… " if start else "") + snippet[start:end] + (" …" if end < len(snippet) else "")
    return highlight(snippet, weights, origin) + ellipsis
