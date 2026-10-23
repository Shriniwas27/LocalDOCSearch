"""Streamlit front-end.  Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from ir_engine import STEmbedder, SearchEngine, load_folder, read_pdf_bytes

HERE = Path(__file__).parent
st.set_page_config(page_title="Smart Document Search", page_icon="🔎", layout="wide")
st.markdown("""
<style>
mark.o{background:#ffe066;color:#111;font-weight:600;padding:0 2px;border-radius:3px}
mark.e{background:#bfe3ff;color:#111;padding:0 2px;border-radius:3px}
.snip{line-height:1.6;margin:.2rem 0 .4rem 0}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data + index
@st.cache_resource(show_spinner="Building index…")
def get_engine(docs: tuple, thesaurus: str, use_embeddings: bool):
    emb = STEmbedder() if use_embeddings else None
    return SearchEngine(list(docs), thesaurus_path=thesaurus, embedder=emb)


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


with st.sidebar:
    st.header("Corpus")
    source = st.radio("Source", ["Folder", "Upload files"], horizontal=True)
    docs: list[tuple[str, str]] = []
    if source == "Folder":
        folder = st.text_input("Folder path", str(HERE / "docs"))
        if Path(folder).is_dir():
            docs = load_folder(folder)
        else:
            st.error("Folder not found.")
    else:
        ups = st.file_uploader("txt / md / pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)
        for u in ups or []:
            raw = u.getvalue()
            text = read_pdf_bytes(raw) if u.name.lower().endswith(".pdf") else raw.decode("utf-8", "ignore")
            if text.strip():
                docs.append((u.name, text))
    st.caption(f"{len(docs)} documents loaded")

    st.header("Retrieval options")
    typo_on = st.checkbox("Typo correction", True)
    expand_on = st.checkbox("Synonym expansion (thesaurus)", True)
    lsa_exp_on = st.checkbox("Corpus-driven expansion (LSA)", True)
    prf_on = st.checkbox("Pseudo-relevance feedback", True)
    sem_on = st.checkbox("Semantic channel (hybrid fusion)", True)
    use_emb = st.checkbox("Use neural embeddings", False, disabled=not embeddings_available(),
                          help="pip install sentence-transformers to enable") if embeddings_available() \
        else st.checkbox("Use neural embeddings (not installed)", False, disabled=True)
    top_docs = st.slider("Results", 5, 30, 12)

if not docs:
    st.info("Add documents (folder or upload) to begin.")
    st.stop()

engine = get_engine(tuple(docs), str(HERE / "thesaurus.json"), bool(use_emb))

# ------------------------------------------------------------------ search box
st.title("🔎 Smart Document Search")
q = st.text_input("Search your documents", placeholder="e.g. cybrspace, viruses, how to respond to a breach")
if not q.strip():
    st.caption("Try a typo ('firewal rules'), a synonym ('viruses'), or a broad query ('security').")
    tab_topics = st.tabs(["Corpus topics"])[0]
else:
    exact = st.session_state.get("exact_for") == q
    res = engine.search(q, typo=typo_on and not exact, expand=expand_on, lsa_expand=lsa_exp_on,
                        semantic=sem_on, prf=prf_on, top_docs=top_docs)
    plan = res.plan

    if plan.corrections and not exact:
        c1, c2 = st.columns([4, 1])
        c1.info(f"Showing results for **{plan.corrected}**  ·  searched instead of: *{q}*")
        if c2.button("Search exactly as typed"):
            st.session_state["exact_for"] = q
            st.rerun()

    tab_res, tab_q, tab_topics = st.tabs(["Results", "Query analysis", "Corpus topics"])

    with tab_res:
        if not res.hits:
            st.warning("No matches. Try fewer or different words.")
        else:
            st.caption(f"{len(res.hits)} documents · {res.ms:.0f} ms · "
                       "yellow = your words, blue = related terms")
            names = res.cluster_names
            sel = names
            if len(names) > 1:
                counts = pd.Series([h.cluster for h in res.hits]).value_counts()
                opts = [f"{n}  ({counts.get(i, 0)})" for i, n in enumerate(names)]
                chosen = st.multiselect("Filter by topic (clustered from these results)", opts, default=opts)
                sel = [names[i] for i, o in enumerate(opts) if o in chosen]
            for h in res.hits:
                if names[h.cluster] not in sel:
                    continue
                with st.container(border=True):
                    top = st.columns([6, 1])
                    top[0].markdown(f"**{h.title}**  \n`{h.name}`" +
                                    (f"  ·  🏷️ {names[h.cluster]}" if len(names) > 1 else ""))
                    top[1].markdown(f"**{h.score*100:.0f}%**")
                    st.markdown(f'<div class="snip">{h.snippet_html}</div>', unsafe_allow_html=True)
                    with st.expander("Why this result / full document"):
                        st.write("Matched terms:", ", ".join(h.matched) or "—")
                        st.write(f"BM25 rank: {h.bm25_rank if h.bm25_rank is not None else '—'} · "
                                 f"Semantic rank: {h.sem_rank if h.sem_rank is not None else '—'} · "
                                 f"+{h.more_passages} more matching passage(s)")
                        st.text(engine.docs[h.doc_id][1])

    with tab_q:
        st.write("**Corrections:**", plan.corrections or "none")
        st.write("**Final query terms** (weight 1.0 = typed; lower = expansion):")
        rows = [{"term": engine.surface_of.get(s, s), "stem": s, "weight": w, "origin": plan.origin[s],
                 "in corpus": s in engine.vocab} for s, w in plan.weights.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if res.silhouette:
            st.caption(f"Result clustering silhouette: {res.silhouette:.2f}")

with tab_topics:
    topics = engine.corpus_topics()
    st.caption(f"Automatic topic grouping of the whole corpus (silhouette {topics.silhouette:.2f}).")
    for i, name in enumerate(topics.names):
        members = [engine.docs[d][0] for d, l in enumerate(topics.labels) if l == i]
        with st.expander(f"{name}  ({len(members)})", expanded=True):
            for m in members:
                st.write("•", m)
