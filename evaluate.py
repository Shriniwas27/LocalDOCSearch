"""
evaluate.py - compare the baseline TF-IDF engine against each added module.

Usage:  python evaluate.py [docs_dir] [eval_queries.json]

Metrics (document level): P@3, Recall@10, MRR, nDCG@10.
NOTE: replace eval_queries.json with judgments on YOUR corpus. Numbers on the bundled
sample set are illustrative only (the sample corpus, thesaurus and queries share an author).
"""
import json
import math
import sys
from collections import defaultdict

from ir_engine import BaselineTfidf, SearchEngine, load_folder

docs_dir = sys.argv[1] if len(sys.argv) > 1 else "docs"
qfile = sys.argv[2] if len(sys.argv) > 2 else "eval_queries.json"
docs = load_folder(docs_dir)
queries = json.load(open(qfile))
engine = SearchEngine(docs, thesaurus_path="thesaurus.json")
baseline = BaselineTfidf(docs)


def metrics(ranked, rel):
    ranked = ranked[:10]
    p3 = sum(d in rel for d in ranked[:3]) / 3
    rec = sum(d in rel for d in ranked) / len(rel)
    mrr = next((1 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    return p3, rec, mrr, dcg / idcg


def run(name, fn):
    per_type = defaultdict(list)
    for q in queries:
        per_type[q["type"]].append(metrics(fn(q["q"]), set(q["relevant"])))
        per_type["ALL"].append(per_type[q["type"]][-1])
    return name, {t: [sum(c) / len(c) for c in zip(*v)] for t, v in per_type.items()}


def eng(**kw):
    def f(q):
        return [h.name for h in engine.search(q, cluster=False, top_docs=10, **kw).hits]
    return f


off = dict(typo=False, expand=False, lsa_expand=False, semantic=False, prf=False)
configs = [
    run("1. Baseline TF-IDF (doc-level)", lambda q: baseline.search(q)),
    run("2. BM25 passages", eng(**off)),
    run("3. + typo correction", eng(**{**off, "typo": True})),
    run("4. + thesaurus expansion", eng(**{**off, "typo": True, "expand": True})),
    run("5. + LSA semantic (RRF)", eng(**{**off, "typo": True, "expand": True, "semantic": True})),
    run("6. Full (+LSA expand, PRF)", eng()),
]

for t in ["ALL", "typo", "synonym", "general"]:
    print(f"\n### Query type: {t}")
    print(f"{'Configuration':36s} {'P@3':>6s} {'R@10':>6s} {'MRR':>6s} {'nDCG@10':>8s}")
    for name, res in configs:
        p3, r, m, n = res[t]
        print(f"{name:36s} {p3:6.3f} {r:6.3f} {m:6.3f} {n:8.3f}")
