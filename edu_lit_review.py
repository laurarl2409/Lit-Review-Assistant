"""
Education Literature Review Assistant
=====================================
A single-file Streamlit app that searches three free academic APIs (ERIC,
OpenAlex, Semantic Scholar), synthesizes the literature with Gemini under
strict grounding rules, and renders a Consensus.app-style Deep Search Report
with BibTeX and print-ready HTML export.

Run:
    pip install streamlit requests google-genai markdown
    streamlit run edu_lit_review.py
"""

import html as html_lib
import os
import re
import time
from datetime import date

import requests
import streamlit as st

try:
    import markdown as md_lib
except ImportError:
    md_lib = None

# ==========================================================================
# Design tokens (shared between the app shell and the HTML export)
# ==========================================================================

INK = "#1B2437"       # deep ink navy — text
PAPER = "#FAF9F6"     # warm paper ground
CARD = "#FFFFFF"
LINE = "#E7E4DC"
MUTED = "#6E7480"
GREEN = "#2F6B4F"     # supporting / strong
AMBER = "#B97D2A"     # mixed / moderate
RED = "#A44444"       # contradicting / weak

VERDICT_STYLE = {
    "Strong": (GREEN, "Strong consensus"),
    "Moderate/Mixed": (AMBER, "Moderate / mixed consensus"),
    "Weak": (RED, "Weak consensus"),
}

# ==========================================================================
# API fetchers — each returns a list of normalized paper dicts, never raises
# ==========================================================================

TIMEOUT = 20
HEADERS = {"User-Agent": "EduLitReview/1.0 (mailto:researcher@example.org)"}


def fetch_eric(query):
    papers = []
    try:
        r = requests.get(
            "https://api.ies.ed.gov/eric/",
            params={"search": query, "format": "json", "rows": 15},
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        for doc in r.json().get("response", {}).get("docs", []):
            title = doc.get("title") or ""
            if not title:
                continue
            papers.append({
                "title": title.strip(),
                "authors": doc.get("author") or [],
                "year": doc.get("publicationdateyear"),
                "venue": doc.get("source") or "ERIC",
                "abstract": (doc.get("description") or "").strip(),
                "citations": None,
                "doi": None,
                "url": f"https://eric.ed.gov/?id={doc.get('id', '')}",
                "source": "ERIC",
            })
    except Exception as e:
        st.warning(f"ERIC didn't respond ({e}). Continuing with the other sources.")
    return papers


def _openalex_abstract(inv_idx):
    """Reconstruct plain text from OpenAlex's abstract_inverted_index."""
    if not inv_idx:
        return ""
    pos = {}
    for word, idxs in inv_idx.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch_openalex(query):
    papers = []
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "filter": "primary_topic.field.id:17",
                "per_page": 15,
                "mailto": "researcher@example.org",
            },
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        for w in r.json().get("results", []):
            title = w.get("display_name") or ""
            if not title:
                continue
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in (w.get("authorships") or [])
            ]
            loc = (w.get("primary_location") or {}).get("source") or {}
            doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
            papers.append({
                "title": title.strip(),
                "authors": [a for a in authors if a],
                "year": w.get("publication_year"),
                "venue": loc.get("display_name") or "OpenAlex",
                "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
                "citations": w.get("cited_by_count"),
                "doi": doi,
                "url": w.get("doi") or w.get("id", ""),
                "source": "OpenAlex",
            })
    except Exception as e:
        st.warning(f"OpenAlex didn't respond ({e}). Continuing with the other sources.")
    return papers


def fetch_semantic_scholar(query):
    papers = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query, "limit": 15,
        "fields": "title,authors,year,citationCount,abstract,externalIds,tldr",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 429:          # shared free pool — one polite retry
            time.sleep(3)
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        for p in r.json().get("data", []) or []:
            title = p.get("title") or ""
            if not title:
                continue
            tldr = (p.get("tldr") or {}).get("text") or ""
            abstract = p.get("abstract") or ""
            ext = p.get("externalIds") or {}
            papers.append({
                "title": title.strip(),
                "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                "year": p.get("year"),
                "venue": "Semantic Scholar",
                "abstract": (f"TLDR: {tldr}\n{abstract}" if tldr else abstract).strip(),
                "citations": p.get("citationCount"),
                "doi": ext.get("DOI"),
                "url": f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else "",
                "source": "Semantic Scholar",
            })
    except Exception as e:
        st.warning(f"Semantic Scholar didn't respond ({e}). Continuing with the other sources.")
    return papers


# ==========================================================================
# Screening (PRISMA), grounding context, BibTeX
# ==========================================================================

def dedupe_and_screen(all_papers, max_included=25):
    """Dedupe by DOI/normalized title (screened), then keep papers with usable
    abstracts, ranked by citation count (included)."""
    seen, screened = set(), []
    for p in all_papers:
        keys = {re.sub(r"\W+", "", p["title"].lower())}
        if p["doi"]:
            keys.add(p["doi"].lower())
        keys.discard("")                      # never dedupe on an empty key
        if not keys or not (keys & seen):
            seen |= keys
            screened.append(p)
    with_text = [p for p in screened if len(p["abstract"]) > 80]
    with_text.sort(key=lambda p: (p["citations"] or 0), reverse=True)
    return screened, with_text[:max_included]


def build_context(included):
    blocks = []
    for i, p in enumerate(included, 1):
        authors = ", ".join(p["authors"][:5]) or "Unknown authors"
        blocks.append(
            f"[{i}] {p['title']}\n"
            f"    Authors: {authors} | Year: {p['year'] or 'n.d.'} | "
            f"Venue: {p['venue']} | Citations: "
            f"{p['citations'] if p['citations'] is not None else 'n/a'} | "
            f"Source DB: {p['source']}\n"
            f"    Abstract: {p['abstract'][:1400]}"
        )
    return "\n\n".join(blocks)


def make_bibtex(included):
    entries = []
    for i, p in enumerate(included, 1):
        first = (p["authors"][0].split()[-1] if p["authors"] else "anon")
        key = re.sub(r"\W", "", f"{first}{p['year'] or ''}") + str(i)
        esc = lambda s: str(s).replace("{", "").replace("}", "").replace("\\", "")
        fields = [
            f"  title = {{{esc(p['title'])}}}",
            f"  author = {{{esc(' and '.join(p['authors']) or 'Unknown')}}}",
        ]
        if p["year"]:
            fields.append(f"  year = {{{p['year']}}}")
        if p["venue"]:
            fields.append(f"  journal = {{{esc(p['venue'])}}}")
        if p["doi"]:
            fields.append(f"  doi = {{{p['doi']}}}")
        if p["url"]:
            fields.append(f"  url = {{{p['url']}}}")
        fields.append(f"  note = {{Retrieved via {p['source']}}}")
        entries.append("@article{" + key + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries)


# ==========================================================================
# LLM synthesis (Gemini) with strict grounding
# ==========================================================================

SYSTEM_PROMPT = """You are an expert education-research synthesist producing a \
Consensus.app-style Deep Search Report.

GROUNDING RULES (non-negotiable):
- Use ONLY the numbered papers supplied in the user message. Never invent,
  recall, or cite any paper, author, statistic, or finding not present there.
- Every substantive claim must carry bracketed citations like [3] or [2,7]
  that refer to the supplied paper numbers.
- If the retrieved evidence cannot answer part of the question, say so plainly
  and flag it as a gap. Never fill gaps with outside knowledge.
- Distinguish causal evidence (RCTs, quasi-experiments) from descriptive or
  correlational evidence, based only on what the abstracts state."""

REPORT_INSTRUCTIONS = """Write a Markdown report with EXACTLY this structure.
Output raw Markdown only — no code fences, no preamble.

Line 1 (machine-readable, nothing before it):
CONSENSUS_METER: <Strong|Moderate/Mixed|Weak> | supporting=<int>% mixed=<int>% contradicting=<int>%
(Percentages must sum to 100 and reflect your paper-by-paper reading.)

Then these five sections:

## 1. Executive Summary & Consensus Meter
3-4 paragraph summary of what the included literature says about the question,
an explicit statement of the consensus category and why, and the approximate
share of papers supporting / mixed / contradicting.

## 2. Methods & PRISMA Search Flow
Describe the databases searched (ERIC, OpenAlex, Semantic Scholar), the query,
and the screening logic. Reference the counts provided: retrieved, screened
(after deduplication), included (with usable abstracts). Note evidence-hierarchy
weighting (systematic reviews/RCTs > quasi-experimental > correlational >
qualitative/descriptive).

## 3. Key Results & Synthesis
- A Markdown table "Foundational Anchor Papers" (4-6 rows): Paper [n] | Year |
  Design (as inferable from abstract) | Core finding | Citations.
- 2-4 thematic sub-syntheses with ### subheadings, each citing papers.
- A short "Timeline & Venue Breakdown" paragraph (publication-year spread,
  notable venues) using only supplied metadata.

## 4. Claim Matrix & Methodological Discussion
- A Markdown table: Claim | Evidence Strength (Strong/Moderate/Weak) |
  Reasoning | Citations. 4-6 claims.
- A paragraph on causal vs. descriptive limits of this evidence base.

## 5. Research Gaps, Emergent Questions & Conclusion
- A Markdown "Coverage Heatmap" table with rows = the main themes you found and
  columns = Causal Tests | Long-Term Outcomes | Equity | Generalization; cells =
  🟢 well covered / 🟡 partial / 🔴 gap, judged strictly from included papers.
- 3-5 emergent follow-up questions the field has not yet answered.
- A closing paragraph, including a caution that this reflects only the
  retrieved records, not the entire literature."""

GEMINI_MODEL = "gemini-3.5-flash"  # has a free tier as of mid-2026


def synthesize(api_key, question, context, counts):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    user_msg = (
        f"RESEARCH QUESTION: {question}\n\n"
        f"PRISMA COUNTS — Retrieved: {counts['retrieved']} "
        f"(ERIC {counts['eric']}, OpenAlex {counts['openalex']}, "
        f"Semantic Scholar {counts['s2']}); Screened after dedup: "
        f"{counts['screened']}; Included with usable abstracts: {counts['included']}.\n\n"
        f"INCLUDED PAPERS (your ONLY evidence base):\n\n{context}\n\n"
        f"{REPORT_INSTRUCTIONS}"
    )
    # Note: thinking tokens count toward the output cap on this model,
    # so the cap is set generously to avoid truncated reports.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, max_output_tokens=16000)
    last_err = None
    for attempt in range(3):  # free-tier RPM limits can trigger transient 429s
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=user_msg, config=config)
            text = resp.text or ""
            if not text.strip():
                raise RuntimeError(
                    "Gemini returned an empty response. This usually means the "
                    "request was rate-limited or filtered — wait a minute and "
                    "run the search again.")
            return text
        except Exception as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(15 * (attempt + 1))
            else:
                raise
    raise last_err


def clean_llm_output(text):
    """Strip code fences and stray whitespace Gemini sometimes adds."""
    t = text.strip()
    t = re.sub(r"^```(?:markdown|md)?\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def parse_meter(report):
    """Extract the CONSENSUS_METER line. Returns (meter|None, body)."""
    m = re.search(
        r"CONSENSUS_METER:\s*(Strong|Moderate/Mixed|Weak)\s*\|\s*"
        r"supporting=(\d+)%\s*mixed=(\d+)%\s*contradicting=(\d+)%", report)
    if not m:
        return None, report
    s, x, c = int(m.group(2)), int(m.group(3)), int(m.group(4))
    total = max(s + x + c, 1)                    # normalize so the bar never overflows
    meter = {
        "category": m.group(1),
        "supporting": s, "mixed": x, "contradicting": c,
        "w_sup": round(s / total * 100, 1),
        "w_mix": round(x / total * 100, 1),
        "w_con": round(c / total * 100, 1),
    }
    return meter, report[m.end():].lstrip()      # drop preamble AND meter line


# ==========================================================================
# Shared HTML fragments (verdict hero, PRISMA flow) — used in-app and in export
# ==========================================================================

def verdict_html(meter):
    if not meter:
        return ""
    color, label = VERDICT_STYLE.get(meter["category"], (MUTED, meter["category"]))
    seg = ('<span class="vseg" style="width:{w}%;background:{c}" '
           'title="{t} {p}%"></span>')
    return (
        '<div class="verdict">'
        '<div class="verdict-eyebrow">Consensus meter</div>'
        f'<div class="verdict-cat" style="color:{color}">{label}</div>'
        '<div class="vbar">'
        + seg.format(w=meter["w_sup"], c=GREEN, t="Supporting", p=meter["supporting"])
        + seg.format(w=meter["w_mix"], c=AMBER, t="Mixed", p=meter["mixed"])
        + seg.format(w=meter["w_con"], c=RED, t="Contradicting", p=meter["contradicting"])
        + "</div>"
        '<div class="vlegend">'
        f'<span><i style="background:{GREEN}"></i>Supporting {meter["supporting"]}%</span>'
        f'<span><i style="background:{AMBER}"></i>Mixed {meter["mixed"]}%</span>'
        f'<span><i style="background:{RED}"></i>Contradicting {meter["contradicting"]}%</span>'
        "</div></div>"
    )


def prisma_html(counts):
    card = ('<div class="prisma-card"><div class="prisma-n">{n}</div>'
            '<div class="prisma-l">{label}</div></div>')
    arrow = '<div class="prisma-arrow">&#8594;</div>'
    return (
        '<div class="prisma-flow">'
        + card.format(n=counts["retrieved"], label="Retrieved") + arrow
        + card.format(n=counts["screened"], label="Screened &middot; deduplicated") + arrow
        + card.format(n=counts["included"], label="Included in synthesis")
        + "</div>"
    )


def papers_html(included):
    items = []
    for i, p in enumerate(included, 1):
        title = html_lib.escape(p["title"])
        authors = html_lib.escape(", ".join(p["authors"][:4]))
        if len(p["authors"]) > 4:
            authors += " et al."
        title_el = (f'<a href="{html_lib.escape(p["url"])}" target="_blank" '
                    f'rel="noopener">{title}</a>' if p["url"] else title)
        meta = [str(p["year"]) if p["year"] else "n.d."]
        if p["venue"] and p["venue"] not in ("Semantic Scholar",):
            meta.append(html_lib.escape(str(p["venue"])[:60]))
        if p["citations"] is not None:
            meta.append(f'{p["citations"]:,} citations')
        items.append(
            '<div class="paper">'
            f'<div class="paper-n">[{i}]</div>'
            '<div class="paper-body">'
            f'<div class="paper-title">{title_el}</div>'
            f'<div class="paper-meta">{authors or "Unknown authors"}</div>'
            f'<div class="paper-meta">{" &middot; ".join(meta)}'
            f'<span class="src src-{p["source"].split()[0].lower()}">{p["source"]}</span>'
            "</div></div></div>"
        )
    return '<div class="paper-list">' + "".join(items) + "</div>"


# ==========================================================================
# HTML export (print-ready)
# ==========================================================================

def _tag_strengths(body):
    """Wrap Strong/Moderate/Weak table cells in Consensus-style badge tags."""
    for word, cls in (("Moderate/Mixed", "moderate"), ("Strong", "strong"),
                      ("Moderate", "moderate"), ("Weak", "weak")):
        body = body.replace(f"<td>{word}</td>",
                            f'<td><span class="tag tag-{cls}">{word}</span></td>')
    return body


def build_html_export(question, meter, report_md, counts):
    q = html_lib.escape(question)
    if md_lib:
        body = md_lib.markdown(report_md, extensions=["tables", "sane_lists"])
        body = _tag_strengths(body)
    else:  # graceful fallback if the markdown package is missing
        body = "<pre style='white-space:pre-wrap'>" + html_lib.escape(report_md) + "</pre>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep Search Report — {q}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
  :root {{
    --ink:{INK}; --paper:{PAPER}; --card:{CARD}; --line:{LINE}; --muted:{MUTED};
    --green:{GREEN}; --amber:{AMBER}; --red:{RED};
  }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:'Public Sans',system-ui,-apple-system,sans-serif;
         color:var(--ink); background:var(--paper); margin:0; line-height:1.65;
         font-size:15.5px; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:48px 24px; }}
  .report {{ background:var(--card); border:1px solid var(--line);
             border-radius:14px; padding:52px 56px;
             box-shadow:0 1px 3px rgba(27,36,55,.05); }}
  .eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem;
              letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
              margin-bottom:14px; }}
  h1 {{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:2rem;
        line-height:1.25; margin:0 0 8px; letter-spacing:-.01em; }}
  .subtitle {{ color:var(--muted); font-size:.88rem; margin-bottom:28px;
               padding-bottom:24px; border-bottom:1px solid var(--line); }}
  h2 {{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:1.35rem;
        margin:2.4em 0 .6em; letter-spacing:-.01em; }}
  h3 {{ font-weight:600; font-size:1.02rem; margin-top:1.7em; }}
  p {{ margin:.7em 0; }}
  a {{ color:var(--green); }}
  table {{ border-collapse:collapse; width:100%; font-size:.88rem; margin:16px 0; }}
  th, td {{ border:1px solid var(--line); padding:9px 11px; text-align:left;
            vertical-align:top; }}
  th {{ background:#F4F2EC; font-family:'IBM Plex Mono',monospace;
        font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
        color:var(--muted); font-weight:600; }}
  tr:nth-child(even) td {{ background:#FBFAF7; }}
  .tag {{ display:inline-block; font-family:'IBM Plex Mono',monospace;
          font-size:.72rem; font-weight:600; padding:2px 10px;
          border-radius:999px; color:#fff; }}
  .tag-strong {{ background:var(--green); }}
  .tag-moderate {{ background:var(--amber); }}
  .tag-weak {{ background:var(--red); }}
  .verdict {{ margin:6px 0 26px; }}
  .verdict-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem;
                      letter-spacing:.14em; text-transform:uppercase;
                      color:var(--muted); margin-bottom:4px; }}
  .verdict-cat {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
                  font-size:1.7rem; letter-spacing:-.01em; margin-bottom:12px; }}
  .vbar {{ display:flex; height:14px; border-radius:7px; overflow:hidden;
           background:var(--line); }}
  .vseg {{ display:block; height:100%; }}
  .vlegend {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px;
              font-family:'IBM Plex Mono',monospace; font-size:.76rem;
              color:var(--muted); }}
  .vlegend i {{ display:inline-block; width:9px; height:9px; border-radius:50%;
                margin-right:6px; }}
  .prisma-flow {{ display:flex; align-items:stretch; gap:12px; flex-wrap:wrap;
                  margin:0 0 8px; }}
  .prisma-card {{ border:1px solid var(--line); border-radius:12px;
                  padding:16px 24px; text-align:center; background:#FCFBF8;
                  min-width:150px; flex:1; }}
  .prisma-n {{ font-family:'IBM Plex Mono',monospace; font-size:1.7rem;
               font-weight:600; color:var(--ink); }}
  .prisma-l {{ font-size:.78rem; color:var(--muted); margin-top:2px; }}
  .prisma-arrow {{ align-self:center; font-size:1.3rem; color:#B9B4A8; }}
  @media (max-width:640px) {{
    .report {{ padding:28px 20px; }}
    .prisma-arrow {{ display:none; }}
  }}
  @media print {{
    body {{ background:#fff; font-size:12.5px; }}
    .wrap {{ max-width:100%; padding:0; }}
    .report {{ border:none; box-shadow:none; padding:0; border-radius:0; }}
    h1 {{ font-size:1.6rem; }}
    h2 {{ break-after:avoid; }}
    table, .prisma-flow, .verdict {{ break-inside:avoid; }}
    a {{ color:var(--ink); text-decoration:none; }}
  }}
</style>
</head>
<body><div class="wrap"><div class="report">
<div class="eyebrow">Deep Search Report &middot; ERIC &middot; OpenAlex &middot; Semantic Scholar</div>
<h1>{q}</h1>
<div class="subtitle">Generated {date.today().strftime("%B %d, %Y")} &middot;
{counts["included"]} papers synthesized from {counts["retrieved"]} retrieved records</div>
{verdict_html(meter)}
{prisma_html(counts)}
{body}
</div></div></body></html>"""


# ==========================================================================
# App shell CSS (injected once, styles the whole Streamlit app)
# ==========================================================================

APP_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp {{ background:{PAPER}; }}
.block-container {{ max-width:980px; padding-top:2.6rem; }}

html, body, .stApp, .stMarkdown, p, li, td {{
  font-family:'Public Sans',system-ui,-apple-system,sans-serif;
  color:{INK};
}}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {{
  font-family:'Fraunces',Georgia,serif !important;
  font-weight:600 !important; letter-spacing:-.01em;
  color:{INK} !important;
}}
.stMarkdown h2 {{ font-size:1.4rem; margin-top:1.6em; }}
.stMarkdown h3 {{ font-family:'Public Sans',sans-serif !important;
                  font-weight:600 !important; font-size:1.03rem; }}

/* Header */
.app-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  letter-spacing:.16em; text-transform:uppercase; color:{MUTED};
  margin-bottom:.4rem; }}
.app-title {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:2.1rem; line-height:1.2; letter-spacing:-.01em; margin:0; }}
.app-sub {{ color:{MUTED}; font-size:.95rem; margin:.5rem 0 0; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background:#F3F1EB; border-right:1px solid {LINE}; }}
[data-testid="stSidebar"] .stMarkdown p {{ font-size:.86rem; color:{MUTED}; }}

/* Buttons — press feedback, no sluggish transitions */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius:10px; font-weight:600;
  transition:transform 120ms cubic-bezier(0.23,1,0.32,1),
             background 150ms ease;
}}
.stButton > button:active, .stDownloadButton > button:active,
.stFormSubmitButton > button:active {{ transform:scale(0.98); }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background:{INK}; border:1px solid {INK};
}}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {{
  background:#2A3550; border-color:#2A3550;
}}
@media (prefers-reduced-motion: reduce) {{
  .stButton > button, .stDownloadButton > button,
  .stFormSubmitButton > button {{ transition:none; }}
}}

/* Report tables rendered by st.markdown */
.stMarkdown table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
.stMarkdown th, .stMarkdown td {{ border:1px solid {LINE}; padding:8px 11px;
  text-align:left; vertical-align:top; }}
.stMarkdown th {{ background:#F4F2EC; font-family:'IBM Plex Mono',monospace;
  font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
  color:{MUTED}; }}
.stMarkdown tr:nth-child(even) td {{ background:#FBFAF7; }}

/* Verdict hero */
.verdict {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px;
  padding:22px 26px; margin-bottom:14px;
  box-shadow:0 1px 3px rgba(27,36,55,.05); }}
.verdict-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  letter-spacing:.14em; text-transform:uppercase; color:{MUTED};
  margin-bottom:2px; }}
.verdict-cat {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:1.65rem; letter-spacing:-.01em; margin-bottom:12px; }}
.vbar {{ display:flex; height:14px; border-radius:7px; overflow:hidden;
  background:{LINE}; }}
.vseg {{ display:block; height:100%; }}
.vlegend {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px;
  font-family:'IBM Plex Mono',monospace; font-size:.76rem; color:{MUTED}; }}
.vlegend i {{ display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:6px; }}

/* PRISMA flow */
.prisma-flow {{ display:flex; align-items:stretch; gap:12px; flex-wrap:wrap;
  margin:0 0 6px; }}
.prisma-card {{ border:1px solid {LINE}; border-radius:12px; padding:14px 22px;
  text-align:center; background:{CARD}; min-width:140px; flex:1;
  box-shadow:0 1px 3px rgba(27,36,55,.04); }}
.prisma-n {{ font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
  font-weight:600; color:{INK}; }}
.prisma-l {{ font-size:.78rem; color:{MUTED}; margin-top:2px; }}
.prisma-arrow {{ align-self:center; font-size:1.3rem; color:#B9B4A8; }}

/* How-it-works cards (empty state) */
.how-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:6px; }}
.how-card {{ flex:1; min-width:200px; background:{CARD}; border:1px solid {LINE};
  border-radius:12px; padding:18px 20px;
  box-shadow:0 1px 3px rgba(27,36,55,.04); }}
.how-step {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem;
  letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.how-title {{ font-weight:600; margin:6px 0 4px; }}
.how-body {{ font-size:.85rem; color:{MUTED}; line-height:1.5; }}

/* Paper list */
.paper-list {{ display:flex; flex-direction:column; gap:10px; }}
.paper {{ display:flex; gap:14px; background:{CARD}; border:1px solid {LINE};
  border-radius:12px; padding:14px 18px; }}
.paper-n {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  color:{MUTED}; font-size:.85rem; min-width:34px; }}
.paper-title {{ font-weight:600; line-height:1.4; margin-bottom:2px; }}
.paper-title a {{ color:{INK}; text-decoration:none; }}
.paper-title a:hover {{ color:{GREEN}; text-decoration:underline; }}
.paper-meta {{ font-size:.82rem; color:{MUTED}; }}
.src {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem;
  font-weight:600; padding:1px 8px; border-radius:999px; margin-left:8px;
  border:1px solid {LINE}; color:{MUTED}; background:#F6F4EF; }}

.result-head {{ margin:4px 0 14px; }}
.result-q {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:1.5rem; line-height:1.3; letter-spacing:-.01em; }}
.result-meta {{ font-size:.85rem; color:{MUTED}; margin-top:4px; }}
</style>
"""

# ==========================================================================
# Streamlit UI
# ==========================================================================

st.set_page_config(page_title="Literature Review Assistant",
                   page_icon="📖", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

EXAMPLES = [
    "Does retrieval practice improve K-12 science learning?",
    "What are the effects of class size on student achievement?",
    "Does one-to-one device access improve literacy outcomes?",
]


def _set_example(text):
    st.session_state.q_input = text


# ---- Sidebar: setup only ------------------------------------------------
with st.sidebar:
    st.markdown("#### Setup")
    api_key = st.text_input(
        "Gemini API key", type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Paste a key from Google AI Studio. The free tier of "
             f"{GEMINI_MODEL} is enough — no billing needed.")
    st.markdown(
        "[Get a free key at aistudio.google.com/apikey]"
        "(https://aistudio.google.com/apikey)")
    st.divider()
    st.markdown("#### About")
    st.markdown(
        "Searches **ERIC**, **OpenAlex**, and **Semantic Scholar**, then "
        "synthesizes the results into a five-section evidence report. Every "
        "claim is cited to a retrieved paper — the model is not allowed to "
        "use outside knowledge.")

# ---- Header + question --------------------------------------------------
st.markdown(
    '<div class="app-eyebrow">ERIC &middot; OpenAlex &middot; Semantic Scholar</div>'
    '<p class="app-title">Education Literature Review Assistant</p>'
    '<p class="app-sub">Ask a research question. Get a grounded, citable '
    'evidence report with a consensus verdict, claim matrix, and gap map.</p>',
    unsafe_allow_html=True)

st.write("")
question = st.text_area(
    "Research question", key="q_input", height=100,
    placeholder="e.g., Does retrieval practice improve K-12 science learning?",
    label_visibility="collapsed")

ec1, ec2 = st.columns([3, 1])
with ec1:
    bcols = st.columns(len(EXAMPLES))
    for col, ex in zip(bcols, EXAMPLES):
        with col:
            st.button(ex.split("?")[0][:38] + "…", key=f"ex_{ex[:12]}",
                      on_click=_set_example, args=(ex,),
                      use_container_width=True, help=ex)
with ec2:
    run = st.button("Run deep search", type="primary", use_container_width=True)

# ---- Run pipeline -------------------------------------------------------
if run:
    if not question.strip():
        st.error("Type a research question first — or tap one of the examples.")
        st.stop()
    if not api_key:
        st.error("Add your free Gemini API key in the sidebar to run the "
                 "synthesis step. The link there takes you to Google AI Studio.")
        st.stop()

    q = question.strip()
    with st.status("Running deep search…", expanded=True) as status:
        st.write("Searching ERIC…")
        eric = fetch_eric(q)
        st.write(f"ERIC returned {len(eric)}. Searching OpenAlex…")
        oa = fetch_openalex(q)
        st.write(f"OpenAlex returned {len(oa)}. Searching Semantic Scholar…")
        s2 = fetch_semantic_scholar(q)
        st.write(f"Semantic Scholar returned {len(s2)}. Screening and deduplicating…")

        all_papers = eric + oa + s2
        screened, included = dedupe_and_screen(all_papers)
        counts = {"retrieved": len(all_papers), "eric": len(eric),
                  "openalex": len(oa), "s2": len(s2),
                  "screened": len(screened), "included": len(included)}

        if not included:
            status.update(label="No usable papers found", state="error")
            st.error("None of the retrieved records had usable abstracts. "
                     "Try broader wording — for example, drop grade levels or "
                     "specific program names.")
            st.stop()

        st.write(f"{counts['included']} papers included. Synthesizing with Gemini…")
        try:
            raw = clean_llm_output(
                synthesize(api_key, q, build_context(included), counts))
        except Exception as e:
            status.update(label="Synthesis failed", state="error")
            st.error(f"Synthesis failed: {e}")
            st.stop()
        status.update(label="Report ready", state="complete", expanded=False)

    meter, body = parse_meter(raw)
    st.session_state.result = {
        "question": q, "meter": meter, "body": body,
        "counts": counts, "included": included,
    }

# ---- Empty state --------------------------------------------------------
if "result" not in st.session_state:
    st.write("")
    st.markdown(
        '<div class="how-row">'
        '<div class="how-card"><div class="how-step">Search</div>'
        '<div class="how-title">Three free databases</div>'
        '<div class="how-body">Your question runs against ERIC, OpenAlex, and '
        'Semantic Scholar — up to 45 records per search.</div></div>'
        '<div class="how-card"><div class="how-step">Screen</div>'
        '<div class="how-title">PRISMA-style screening</div>'
        '<div class="how-body">Duplicates are merged and papers without usable '
        'abstracts are excluded, ranked by citation count.</div></div>'
        '<div class="how-card"><div class="how-step">Synthesize</div>'
        '<div class="how-title">Grounded report</div>'
        '<div class="how-body">A five-section evidence report where every claim '
        'cites a retrieved paper. Export to BibTeX or print-ready HTML.</div></div>'
        "</div>",
        unsafe_allow_html=True)

# ---- Results ------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state.result
    meter, counts = r["meter"], r["counts"]

    st.divider()
    st.markdown(
        '<div class="result-head">'
        f'<div class="result-q">{html_lib.escape(r["question"])}</div>'
        f'<div class="result-meta">Generated {date.today().strftime("%B %d, %Y")}'
        f' &middot; {counts["included"]} papers synthesized from '
        f'{counts["retrieved"]} retrieved records</div></div>',
        unsafe_allow_html=True)

    if meter:
        st.markdown(verdict_html(meter), unsafe_allow_html=True)
    st.markdown(prisma_html(counts), unsafe_allow_html=True)

    if "## 5" not in r["body"]:
        st.warning("This report looks shorter than expected — it may have been "
                   "truncated. Running the search again usually fixes it.")

    tab_report, tab_papers, tab_export = st.tabs(
        ["Report", f"Included papers ({counts['included']})", "Export"])

    with tab_report:
        st.markdown(r["body"])

    with tab_papers:
        st.markdown(
            "These are the only records the synthesis was allowed to cite. "
            "Bracketed numbers in the report refer to this list.")
        st.markdown(papers_html(r["included"]), unsafe_allow_html=True)

    with tab_export:
        bib = make_bibtex(r["included"])
        html_out = build_html_export(r["question"], meter, r["body"], counts)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download BibTeX for Zotero", bib,
                file_name="literature_review.bib", mime="text/plain",
                use_container_width=True)
            st.caption("Import into Zotero via File → Import to add all "
                       f"{counts['included']} included papers.")
        with d2:
            st.download_button(
                "Download HTML report", html_out,
                file_name="deep_search_report.html", mime="text/html",
                use_container_width=True)
            st.caption("Open in a browser and press Ctrl+P / Cmd+P — the "
                       "print stylesheet formats it as a clean PDF.")
