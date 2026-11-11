"""LexiSyncAI - Streamlit web app.

    streamlit run app.py

Tabs
  Ask                ask a question; cited answer, confidence badge, highlighted page, agent log
  Compare documents  same question to each document; conflicts between them are flagged
  Documents          detected structure, OCR, tables, unusual pages (autoencoder)
  Evaluation         all evaluation results
"""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

from src.offline import use_offline_models_if_cached

OFFLINE = use_offline_models_if_cached()          # before any Hugging Face import

import pymupdf  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.abstain import HIGH, INSUFFICIENT, LOW, MEDIUM  # noqa: E402
from src.indexer import CaseError, MAX_DOCUMENTS, add_to_case, build_index, case_dir, list_cases, load_index  # noqa: E402
from src.ingest import InvalidPDFError  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

st.set_page_config(page_title="LexiSyncAI", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

BADGES = {
    HIGH: ("#15803d", "#dcfce7", "Strong evidence, every claim verified against its source."),
    MEDIUM: ("#b45309", "#fef3c7", "Good evidence; most claims verified."),
    LOW: ("#c2410c", "#ffedd5", "Weak evidence or some claims unverified: check the cited pages."),
    INSUFFICIENT: ("#475569", "#e2e8f0", "The documents do not contain enough evidence, so the system abstains."),
}
EXAMPLES = {
    "Factual": "Which state's law governs the Tax Protection Agreement?",
    "Multi-hop": "Under the GDPR, what is the maximum fine a company can receive, and which kinds of violations "
                 "trigger the higher fine?",
    "Comparison": "Which is longer: the GDPR deadline for notifying a personal data breach to the supervisory "
                  "authority, or the time limit for the short-term right to reject under the UK Consumer Rights "
                  "Act 2015?",
    "Table": "What is the total Section 704(c) value of the Gain Limitation Properties listed in the agreement?",
    "Unanswerable": "What is the governing law of the subcontractor agreement?",
}
COMPARE_EXAMPLES = ["What is the cap on the Supplier's liability?", "Which law governs the agreement?",
                    "How long is the initial term?"]

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1380px;}
/* Header: deep navy with a thin gold rule, serif title */
.lx-hero {background: linear-gradient(135deg, #1b3a57 0%, #24527a 55%, #2f6690 100%); color: #fff;
          padding: 22px 28px 20px 28px; border-radius: 18px; margin-bottom: 16px;
          box-shadow: 0 6px 24px rgba(27, 58, 87, 0.18); border-bottom: 3px solid #c9a227;}
.lx-hero h1 {color: #fff; font-family: 'Lora', Georgia, serif; font-size: 1.9rem; font-weight: 700;
             margin: 0; padding: 0; letter-spacing: .01em;}
.lx-hero p {color: #dfe8f1; margin: 6px 0 0 0; font-size: 0.98rem;}
/* Cards (bordered containers) get a soft lift */
div[data-testid="stVerticalBlockBorderWrapper"] {background: #ffffff; box-shadow: 0 2px 12px rgba(31, 42, 55, 0.05);}
.lx-badge {display: inline-block; padding: 4px 14px; border-radius: 999px; font-weight: 600; font-size: 0.82rem;}
.lx-answer {font-size: 1.06rem; line-height: 1.75; color: #1f2a37;}
.lx-cite {display: inline-block; background: #e8eff6; color: #24527a; border-radius: 999px; padding: 0 8px;
          font-size: 0.74rem; font-weight: 700; margin: 0 2px; vertical-align: 1px;}
.lx-muted {color: #6b7280; font-size: 0.86rem;}
.lx-verdict {padding: 14px 18px; border-radius: 14px; font-weight: 600; margin: 6px 0 12px 0;}
.lx-dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}
/* Buttons: gentle lift on hover instead of a hard flat block */
.stButton button, .stFormSubmitButton button {font-weight: 600; transition: all .15s ease;}
.stButton button:hover, .stFormSubmitButton button:hover {transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(36, 82, 122, 0.18);}
/* Tabs: a little more breathing room */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: 8px 14px;}
/* Hide Streamlit's "Press Enter to submit form" hints */
div[data-testid="InputInstructions"] {display: none;}
div[data-testid="stMetricValue"] {font-size: 1.4rem; font-family: 'Lora', Georgia, serif;}
section[data-testid="stSidebar"] h2 {font-family: 'Lora', Georgia, serif;}
</style>
"""


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the models (first time only)...")
def get_pipeline(case_id: str, built_at: str) -> Pipeline:
    return Pipeline(load_index(case_id))


@st.cache_data(ttl=20, show_spinner=False)
def llm_status() -> tuple[bool, str]:
    if config.LLM_BACKEND != "ollama":
        return True, f"{config.OPENAI_MODEL} (API)"
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=1.5).json()
        names = [m["name"] for m in tags.get("models", [])]
        if config.LLM_MODEL in names:
            return True, f"{config.LLM_MODEL} ready"
        return False, f"{config.LLM_MODEL} not installed: run  ollama pull {config.LLM_MODEL}"
    except Exception:
        return False, "Ollama is not running: start the Ollama app"


@st.cache_data(show_spinner=False)
def render_page(pdf_path: str, page_no: int, boxes: tuple, zoom: float = 1.5) -> bytes | None:
    try:
        page = pymupdf.open(pdf_path)[page_no - 1]
    except Exception:
        return None
    for x0, y0, x1, y1 in boxes:
        page.draw_rect(pymupdf.Rect(x0 - 3, y0 - 3, x1 + 3, y1 + 3), color=(0.95, 0.55, 0.0),
                       fill=(1.0, 0.85, 0.1), fill_opacity=0.28, width=1.6)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def esc(text: str) -> str:
    """HTML-escape, and stop Streamlit treating $...$ in legal amounts as maths."""
    return html.escape(text).replace("$", "&#36;")


def with_cites(text: str) -> str:
    return re.sub(r"\[S(\d+)\]", r"<span class='lx-cite'>S\1</span>", esc(text))


def badge(b: str) -> str:
    fg, bg, tip = BADGES.get(b, ("#475569", "#e2e8f0", ""))
    return f"<span class='lx-badge' style='color:{fg};background:{bg};border:1px solid {fg}' title='{esc(tip)}'>{esc(b)}</span>"


def doc_paths(index) -> dict[str, str]:
    return {d["doc_id"]: d.get("path") or str(config.DATA_DIR / d["file"]) for d in index.documents}


def safe_case_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")[:40]


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------

def sidebar() -> str | None:
    sb = st.sidebar
    sb.markdown("## ⚖️ LexiSyncAI")
    ok, msg = llm_status()
    colour = "#16a34a" if ok else "#dc2626"
    sb.markdown(f"<span class='lx-dot' style='background:{colour}'></span><b>LLM</b> "
                f"<span class='lx-muted'>{esc(msg)}</span><br>"
                f"<span class='lx-dot' style='background:{'#16a34a' if OFFLINE else '#f59e0b'}'></span><b>Models</b> "
                f"<span class='lx-muted'>{'loaded from this computer' if OFFLINE else 'will download if missing'}</span>",
                unsafe_allow_html=True)
    sb.divider()

    cases = list_cases()
    if cases:
        # The picker keeps a fixed key, so a selection is never lost on rerun. A new or deleted
        # case is passed in through "pending_case" before the picker is drawn.
        pending = st.session_state.pop("pending_case", None)
        if pending is not None or st.session_state.get("case_select") not in cases:
            wanted = pending if pending in cases else ("default" if "default" in cases else cases[0])
            st.session_state.case_select = wanted
        case = sb.selectbox("📁 Case", cases, key="case_select")
        st.session_state.case = case
        index = load_index(case)
        sb.caption(f"{len(index.documents)} document(s) · {index.manifest['n_chunks']} passages")
        notes = st.session_state.pop("add_notes", None)
        if notes is not None:                     # shown once, right after documents were added
            sb.success("Documents added. The case was re-indexed with all its documents together.")
            for note in notes:
                sb.warning(note)
        if case != "default":
            add_documents_ui(case, index)
            delete_case_ui(case)
    else:
        case = None
        sb.info("No cases yet: create one below.")

    with sb.expander("➕ New case", expanded=not cases):
        with st.form("new_case", clear_on_submit=False):
            name = st.text_input("Case name", placeholder="e.g. lease_review")
            files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, help="2 to 5 legal documents")
            urls = st.text_area("Or paste PDF links", height=68, placeholder="e.g. https://site.org/act.pdf",
                                help="One link per line")
            if st.form_submit_button("Create case", type="primary", width="stretch"):
                create_case(name, files or [], [u.strip() for u in urls.splitlines() if u.strip()])
    return case


def delete_case_ui(case: str) -> None:
    sb = st.sidebar
    if st.session_state.get("confirm_delete") != case:
        if sb.button("🗑️ Delete this case", width="stretch"):
            st.session_state.confirm_delete = case
            st.rerun()
        return
    sb.warning(f"Delete case **{case}** and its uploaded copies? Your original files are not touched.")
    c1, c2 = sb.columns(2)
    if c1.button("Yes, delete", type="primary", width="stretch"):
        shutil.rmtree(case_dir(case), ignore_errors=True)
        st.session_state.pending_case = "default"
        st.session_state.confirm_delete = None
        st.session_state.history = []
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.session_state.confirm_delete = None
        st.rerun()


def add_documents_ui(case: str, index) -> None:
    """Add documents to the open case; it is re-indexed with the old and new documents together."""
    room = MAX_DOCUMENTS - len(index.documents)
    with st.sidebar.expander("📎 Add documents to this case"):
        if room <= 0:
            st.caption(f"This case already has {MAX_DOCUMENTS} documents, the maximum.")
            return
        with st.form(f"add_{case}", clear_on_submit=True):
            files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True,
                                     help=f"Up to {room} more document(s)")
            urls = st.text_area("Or paste PDF links", height=68, placeholder="e.g. https://site.org/act.pdf",
                                help="One link per line")
            submitted = st.form_submit_button("Add to case", type="primary", width="stretch")
    if submitted:
        add_documents(case, index, files or [], [u.strip() for u in urls.splitlines() if u.strip()])


def add_documents(case: str, index, files, urls: list[str]) -> None:
    sb = st.sidebar
    existing = {d["file"].lower() for d in index.documents}
    upload_dir = case_dir(case) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sources, skipped = [], []
    for f in files:
        name = Path(f.name).name
        if name.lower() in existing:            # checked before saving, so an existing copy is never overwritten
            skipped.append(f"{name} is already in this case, so it was skipped.")
            continue
        path = upload_dir / name
        path.write_bytes(f.getbuffer())
        sources.append(str(path))
    sources += urls
    if not sources:
        sb.warning(" ".join(skipped) or "Please upload a PDF or paste a link.")
        return
    bar = sb.progress(0.0, text="Re-indexing the case...")
    try:
        _, notes = add_to_case(case, sources, progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg[:70]))
    except (CaseError, InvalidPDFError) as e:
        bar.empty()
        sb.error(str(e).replace("\n", "  \n"))
        return
    bar.empty()
    st.session_state.history = []
    st.session_state.comparison = None
    st.session_state.add_notes = skipped + notes
    st.rerun()


def create_case(name: str, files, urls: list[str]) -> None:
    sb = st.sidebar
    case_id = safe_case_id(name)
    total = len(files) + len(urls)
    if not case_id:
        sb.error("Please give the case a name.")
        return
    if case_id in list_cases():
        sb.error(f"A case called '{case_id}' already exists. Choose another name.")
        return
    if total == 0:
        sb.error("Please upload at least one PDF or paste a link.")
        return
    if total > 5:
        sb.error(f"{total} documents given: a case holds at most 5.")
        return
    upload_dir = case_dir(case_id) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for f in files:
        path = upload_dir / Path(f.name).name
        path.write_bytes(f.getbuffer())
        sources.append(str(path))
    sources += urls
    bar = sb.progress(0.0, text="Starting...")
    try:
        build_index(sources, case_id, progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg[:70]))
    except InvalidPDFError as e:
        bar.empty()
        shutil.rmtree(case_dir(case_id), ignore_errors=True)
        sb.error(str(e).replace("\n", "  \n"))
        sb.warning("Please upload the files again.")
        return
    bar.empty()
    st.session_state.pending_case = case_id
    st.session_state.history = []
    st.rerun()


# ---------------------------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------------------------

def ask_tab(case_id: str, index) -> None:
    pipe = get_pipeline(case_id, index.manifest.get("built_at", ""))
    if case_id == "default":
        pick = st.pills("Not sure what to ask? Try one of these", list(EXAMPLES), selection_mode="single", key="example_pick")
        if pick and st.session_state.get("last_pick") != pick:
            st.session_state.question = EXAMPLES[pick]
            st.session_state.last_pick = pick
    with st.form("ask", border=False):
        q_col, b_col = st.columns([6, 1], vertical_alignment="bottom")
        question = q_col.text_area("Your question", key="question", height=80,
                                   placeholder="e.g. What is the liability cap?")
        go = b_col.form_submit_button("Ask", type="primary", width="stretch")
    if go and question.strip():
        with st.status("Reading the documents...", expanded=False) as status:
            status.write("Finding the relevant passages, drafting an answer and checking every claim...")
            try:
                answer = pipe.ask(question.strip())
            except LLMError as e:
                status.update(label="The language model is not reachable", state="error")
                st.error(f"{e}  \nStart the Ollama app, then ask again.")
                return
            status.update(label=f"Done in {answer.timings.get('total_s', 0):.1f} s", state="complete")
        st.session_state.setdefault("history", []).insert(0, answer)
        st.session_state.selected = 0

    history = st.session_state.get("history", [])
    if not history:
        st.info("Ask anything about these documents. Each answer shows the page it came from, and you'll be "
                "told plainly when the documents don't cover it.")
        return
    if len(history) > 1:
        labels = [a.question[:90] for a in history]
        i = st.selectbox("Recent questions", range(len(history)), format_func=lambda k: labels[k], key="hist_pick")
        answer = history[i]
    else:
        answer = history[0]
    show_answer(answer, doc_paths(index))


def show_answer(a, paths: dict[str, str]) -> None:
    left, right = st.columns([11, 9], gap="large")
    with left:
        with st.container(border=True):
            fg, bg, tip = BADGES.get(a.badge, ("#475569", "#e2e8f0", ""))
            st.markdown(f"{badge(a.badge)} <span class='lx-muted'>&nbsp;{esc(tip)}</span>", unsafe_allow_html=True)
            st.markdown(f"<p class='lx-muted' style='margin-top:10px'>Q: {esc(a.question)}</p>", unsafe_allow_html=True)
            if a.abstained:
                st.markdown(f"<div class='lx-answer'>{esc(a.answer)}</div>", unsafe_allow_html=True)
                if a.missing:
                    st.caption(f"Reason: {a.missing}")
            else:
                st.markdown(f"<div class='lx-answer'>{with_cites(a.answer)}</div>", unsafe_allow_html=True)
                if a.explanation:
                    st.markdown(f"<p class='lx-muted' style='margin-top:10px'><b>Why:</b> {esc(a.explanation)}</p>",
                                unsafe_allow_html=True)
            if a.conflict:
                st.warning(f"**Sources disagree:** {a.conflict}")
            if a.missing and not a.abstained:
                st.info(f"**Not fully covered:** {a.missing}")

        verify_t, sources_t, log_t, perf_t = st.tabs(["✅ Verification", "📚 Sources", "🧭 Agent log", "⏱️ Performance"])
        with verify_t:
            if a.claims:
                st.dataframe([{"": "✅" if c["supported"] else "❌", "Claim": c["claim"],
                               "Sources": " ".join(c["sources"]), "How it was checked": c["reason"]}
                              for c in a.claims], hide_index=True, width="stretch")
            else:
                st.caption("No claims to verify.")
        with sources_t:
            for s in a.sources:
                st.markdown(f"{'**' if s.cited else ''}{s.marker} · {esc(s.chunk.citation)}{'**' if s.cited else ''}"
                            f" <span class='lx-muted'>· relevance {s.score:.2f}{' · cited' if s.cited else ''}</span>",
                            unsafe_allow_html=True)
        with log_t:
            for step in a.retry_log:
                details = {k: v for k, v in step.items() if k not in ("step", "action", "checks")}
                st.markdown(f"**{step['action'].replace('_', ' ').capitalize()}**  "
                            f"<span class='lx-muted'>{esc(json.dumps(details, ensure_ascii=False)[:300]) if details else ''}</span>",
                            unsafe_allow_html=True)
                for ch in step.get("checks", []):
                    st.caption(f"{'✅' if ch['supported'] else '❌'} {ch['claim'][:110]} — {ch['reason']}")
            if a.sub_questions:
                st.caption("Planner sub-questions: " + " | ".join(a.sub_questions))
        with perf_t:
            t, u = a.timings, a.usage
            c = st.columns(4)
            c[0].metric("Total time", f"{t.get('total_s', 0):.1f} s")
            c[1].metric("LLM calls", u.get("llm_calls", 0))
            c[2].metric("Tokens in", f"{u.get('prompt_tokens', 0):,}")
            c[3].metric("Tokens out", f"{u.get('completion_tokens', 0):,}")
            st.caption(f"Retrieval {t.get('retrieval_s', 0):.2f} s · reranking {t.get('rerank_s', 0):.2f} s · "
                       f"LLM {t.get('llm_s', 0):.2f} s · cost 0 (local models)")

    with right:
        show_page(a, paths)


def show_page(a, paths: dict[str, str]) -> None:
    cited = [s for s in a.sources if s.cited] or a.sources
    if not cited:
        if a.searched:
            with st.container(border=True):
                st.markdown("**Documents searched**")
                for d in a.searched:
                    st.markdown(f"- {d}")
        return
    with st.container(border=True):
        options = list(range(len(cited)))
        pick = st.segmented_control("Cited source", options, format_func=lambda k: cited[k].marker,
                                    default=0, key=f"seg_{id(a)}") if len(cited) > 1 else 0
        source = cited[pick if pick is not None else 0]
        chunk = source.chunk
        st.markdown(f"**{source.marker} · {esc(chunk.citation)}**", unsafe_allow_html=True)
        st.caption(chunk.heading_path)
        pages = sorted({sp["page"] for sp in chunk.spans}) or [chunk.page_start]
        page = st.segmented_control("Page", pages, default=pages[0], key=f"pg_{id(a)}_{source.marker}") \
            if len(pages) > 1 else pages[0]
        page = page or pages[0]
        boxes = tuple(tuple(sp["bbox"]) for sp in chunk.spans if sp["page"] == page)
        png = render_page(paths.get(chunk.doc_id, ""), page, boxes)
        if png:
            st.image(png, caption=f"{chunk.doc_title}, page {page} · cited passage highlighted", width="stretch")
        else:
            st.warning("The PDF file for this page could not be found.")
        with st.expander("Cited text"):
            st.markdown(esc(chunk.text).replace("\n", "<br>"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------
# Compare tab (bonus 9)
# ---------------------------------------------------------------------------------------------

def compare_tab(case_id: str, index) -> None:
    from src.compare import compare
    docs = {d["doc_id"]: d.get("name") or d["title"] for d in index.documents}
    if len(docs) < 2:
        st.info("This case has one document. Create a case with two or more documents to compare them, e.g. the "
                "two contract versions in `samples/conflict/`.")
        return
    st.caption("Ask the same question of each document separately. Their cited answers are shown side by side "
               "and checked for conflicts (different figures, or statements that contradict each other).")
    chosen = st.multiselect("Documents", list(docs), default=list(docs)[:3], format_func=lambda d: docs[d][:60])
    if case_id == "conflict_demo":
        pick = st.pills("Try one of these", COMPARE_EXAMPLES, selection_mode="single", key="cmp_pick")
        if pick and st.session_state.get("last_cmp_pick") != pick:
            st.session_state.cmp_question = pick
            st.session_state.last_cmp_pick = pick
    with st.form("compare", border=False):
        q_col, b_col = st.columns([6, 1], vertical_alignment="bottom")
        question = q_col.text_input("Your question", key="cmp_question", placeholder="e.g. What is the liability cap?")
        go = b_col.form_submit_button("Compare", type="primary", width="stretch")
    if go and question.strip() and len(chosen) >= 2:
        with st.spinner(f"Asking {len(chosen)} documents..."):
            try:
                st.session_state.comparison = compare(get_pipeline(case_id, index.manifest.get("built_at", "")),
                                                      question.strip(), chosen)
            except LLMError as e:
                st.error(f"{e}  \nStart the Ollama app, then try again.")
                return
    elif go and len(chosen) < 2:
        st.warning("Choose at least two documents.")
    c = st.session_state.get("comparison")
    if c is None:
        return
    style = {"conflict": ("#991b1b", "#fee2e2", "⚠️ Conflict between the documents"),
             "agree": ("#166534", "#dcfce7", "✅ The documents agree"),
             "single": ("#1e40af", "#dbeafe", "ℹ️ Only one document answers this"),
             "none": ("#475569", "#e2e8f0", "No document contains the answer")}[c.status]
    st.markdown(f"<div class='lx-verdict' style='color:{style[0]};background:{style[1]}'>{style[2]}</div>",
                unsafe_allow_html=True)
    for f in c.findings:
        st.markdown(f"- {esc(f)}", unsafe_allow_html=True)
    cols = st.columns(len(c.answers))
    for col, d in zip(cols, c.answers):
        with col, st.container(border=True):
            st.markdown(f"**{esc(d.title[:70])}**", unsafe_allow_html=True)
            st.markdown(badge(d.answer.badge), unsafe_allow_html=True)
            st.markdown(f"<div class='lx-answer' style='margin-top:8px'>{with_cites(d.answer.answer)}</div>",
                        unsafe_allow_html=True)
            for s in d.answer.cited_sources:
                st.caption(f"{s.marker}: {s.chunk.citation}")


# ---------------------------------------------------------------------------------------------
# Documents tab
# ---------------------------------------------------------------------------------------------

def documents_tab(index) -> None:
    m = index.manifest
    st.caption(f"{len(m['documents'])} document(s) · {m['n_chunks']} searchable passages · indexed "
               f"{m['built_at']} in {m['build_seconds']} s")
    for r in m.get("rejected", []):
        st.error(f"**Not indexed: {r['file']}** — {r['error']}")
    paths = doc_paths(index)
    for d in m["documents"]:
        with st.container(border=True):
            st.markdown(f"#### 📄 {esc(d.get('name') or d['title'])}", unsafe_allow_html=True)
            st.caption(f"{d['file']} · {('link: ' + d['source']) if d.get('source', 'upload') != 'upload' else 'uploaded'}"
                       f" · structure: {d['mode']}")
            if d.get("legal_check", {}).get("warning"):
                st.warning(d["legal_check"]["warning"])
            c = st.columns(6)
            c[0].metric("Pages", d["pages"])
            c[1].metric("Sections", len(d["structure"]))
            c[2].metric("Tables", len(d["tables"]))
            c[3].metric("Footnotes", d["footnotes"])
            c[4].metric("Scanned", len(d["scanned_pages"]))
            c[5].metric("Passages", d["n_chunks"])
            notes = []
            if d["scanned_pages"]:
                notes.append(f"Scanned pages {d['scanned_pages']}: "
                             + (f"read with OCR {d['ocr_pages']}" if d["ocr_pages"] else "OCR not available"))
            if d["toc_pages"]:
                notes.append(f"Table-of-contents pages {d['toc_pages']} (skipped: not indexed)")
            if d["blank_pages"]:
                notes.append(f"Blank pages {d['blank_pages']}")
            if d["tables"]:
                notes.append("Tables: " + ", ".join(f"p. {t['page']} ({t['rows']}×{t['cols']})" for t in d["tables"]))
            for n in notes:
                st.markdown(f"- {n}")
            anomalies = d.get("anomalies", [])
            if anomalies:
                with st.expander(f"🔍 Unusual pages found by the autoencoder ({len(anomalies)})"):
                    st.caption("Pages the page-layout autoencoder could not reconstruct well: often covers, "
                               "tables, signature or scanned pages, where extraction needs a second look.")
                    cols = st.columns(4)
                    for k, an in enumerate(anomalies[:8]):
                        with cols[k % 4]:
                            png = render_page(paths.get(d["doc_id"], ""), an["page"], tuple(), zoom=0.45)
                            if png:
                                st.image(png, width="stretch")
                            st.caption(f"p. {an['page']} · score {an['score']}  \n{an['reason']}")
            with st.expander(f"Sections ({len(d['structure'])})"):
                st.dataframe([{"Page": s["page"], "Section": s["section"], "Title": s["title"],
                               "Within": " > ".join(s["parents"])} for s in d["structure"]],
                             hide_index=True, width="stretch")


# ---------------------------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------------------------

EVAL_SECTIONS = [
    ("Retrieval: recall@k per configuration (dense-only vs hybrid)", "retrieval_eval.md"),
    ("Answers: correctness proxy, citation precision, abstention, latency; no-retrieval and ablations",
     "answer_eval.md"),
    ("Citation precision at region level (highlighted boxes)", "region_eval.md"),
    ("Table extraction vs manual ground truth", "table_eval.md"),
    ("Autoencoder page-anomaly detector", "anomaly_eval.md"),
    ("Latency and cost", "latency_report.md"),
    ("LLM comparison: Qwen2.5 7B vs Qwen3 8B (thinking on/off)", "model_compare.md"),
]


def evaluation_tab() -> None:
    results = config.ROOT / "eval" / "results"
    st.caption("Results from the evaluation scripts in `eval/` on 23 held-out questions over the three example "
               "documents. Re-run them with the commands in the README.")
    ocr = results / "ocr_eval.json"
    if ocr.exists():
        s = json.loads(ocr.read_text(encoding="utf-8"))["summary"]
        with st.expander("OCR accuracy on scanned pages", expanded=False):
            c = st.columns(3)
            c[0].metric("Pages detected as scanned", f"{s['detected_as_scanned']}/{s['pages']}")
            c[1].metric("Character accuracy", f"{s['mean_char_accuracy']:.1%}")
            c[2].metric("Word accuracy", f"{s['mean_word_accuracy']:.1%}")
    for title, name in EVAL_SECTIONS:
        f = results / name
        if f.exists():
            with st.expander(title, expanded=name in ("retrieval_eval.md", "answer_eval.md")):
                st.markdown(f.read_text(encoding="utf-8").replace("$", r"\$"))


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    case_id = sidebar()
    st.markdown("<div class='lx-hero'><h1>⚖️ LexiSyncAI</h1><p>Your legal documents, explained in plain "
                "language, with the exact page and clause behind every answer.</p></div>",
                unsafe_allow_html=True)
    if case_id is None:
        st.info("Create a case in the sidebar: upload 2-5 public legal PDFs or paste links, then index them.")
        return
    try:
        index = load_index(case_id)
    except Exception as e:
        st.error(f"Case '{case_id}' could not be loaded: {e}")
        return
    if st.session_state.get("answer_case") != case_id:
        st.session_state.history = []
        st.session_state.comparison = None
        st.session_state.answer_case = case_id
    names = " · ".join(d.get("name") or d["title"] for d in index.documents)
    st.markdown(f"<p class='lx-muted'>📁 <b>{esc(case_id)}</b> — {esc(names[:260])}</p>", unsafe_allow_html=True)

    # The evaluation results were measured on the example case only, so the tab is shown only there.
    labels = ["💬 Ask", "📄 Documents"] + (["📊 Evaluation"] if case_id == "default" else [])
    tabs = st.tabs(labels)
    ask, docs = tabs[0], tabs[1]
    with ask:
        mode = st.segmented_control("Ask", ["Whole case", "Compare documents"], default="Whole case",
                                    key="ask_mode", label_visibility="collapsed")
        if mode == "Compare documents":
            compare_tab(case_id, index)
        else:
            ask_tab(case_id, index)
    with docs:
        documents_tab(index)
    if case_id == "default":
        with tabs[2]:
            evaluation_tab()


main()
