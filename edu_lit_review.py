"""
Education Literature Review Assistant
=====================================
A single-file Streamlit app that queries three free academic APIs (ERIC,
OpenAlex, Semantic Scholar), synthesizes the literature with Claude, and
renders a Consensus.app-style Deep Search Report with .bib and print-ready
HTML export.

Run:
    pip install streamlit requests anthropic markdown
    streamlit run edu_lit_review.py
"""

import re
import time
from datetime import date

import requests
import streamlit as st

try:
    import markdown as md_lib
except ImportError:
    md_lib = None

# --------------------------------------------------------------------------
# API fetchers — each returns a list of normalized paper dicts and never raises
# --------------------------------------------------------------------------

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
        st.warning(f"ERIC API unavailable ({e}) — continuing without it.")
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
        st.warning(f"OpenAlex API unavailable ({e}) — continuing without it.")
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
        if r.status_code == 429:          # rate limited — one polite retry
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
        st.warning(f"Semantic Scholar API unavailable ({e}) — continuing without it.")
    return papers


# --------------------------------------------------------------------------
# Screening (PRISMA), grounding context, BibTeX
# --------------------------------------------------------------------------

def dedupe_and_screen(all_papers, max_included=25):
    """Dedupe by DOI/title (screened), then keep only papers with usable
    abstracts, ranked by citation count (included)."""
    seen, screened = set(), []
    for p in all_papers:
        keys = {re.sub(r"\W+", "", p["title"].lower())}
        if p["doi"]:
            keys.add(p["doi"].lower())
        if not (keys & seen):
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
            f"Venue: {p['venue']} | Citations: {p['citations'] if p['citations'] is not None else 'n/a'} | "
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


# --------------------------------------------------------------------------
# LLM synthesis (Anthropic) with strict grounding
# --------------------------------------------------------------------------

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


def synthesize(api_key, question, context, counts):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    user_msg = (
        f"RESEARCH QUESTION: {question}\n\n"
        f"PRISMA COUNTS — Retrieved: {counts['retrieved']} "
        f"(ERIC {counts['eric']}, OpenAlex {counts['openalex']}, "
        f"Semantic Scholar {counts['s2']}); Screened after dedup: "
        f"{counts['screened']}; Included with usable abstracts: {counts['included']}.\n\n"
        f"INCLUDED PAPERS (your ONLY evidence base):\n\n{context}\n\n"
        f"{REPORT_INSTRUCTIONS}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_meter(report):
    m = re.search(
        r"CONSENSUS_METER:\s*(Strong|Moderate/Mixed|Weak)\s*\|\s*"
        r"supporting=(\d+)%\s*mixed=(\d+)%\s*contradicting=(\d+)%", report)
    if not m:
        return None, report
    meter = {"category": m.group(1), "supporting": int(m.group(2)),
             "mixed": int(m.group(3)), "contradicting": int(m.group(4))}
    body = report.replace(m.group(0), "", 1).lstrip()
    return meter, body


# --------------------------------------------------------------------------
# HTML export (Consensus-style, print-ready)
# --------------------------------------------------------------------------

METER_COLORS = {"Strong": "#1a7f4e", "Moderate/Mixed": "#b8860b", "Weak": "#b03a3a"}


def prisma_html(counts):
    card = ('<div class="prisma-card"><div class="prisma-n">{n}</div>'
            '<div class="prisma-l">{label}</div></div>')
    arrow = '<div class="prisma-arrow">→</div>'
    return (
        '<div class="prisma-flow">'
        + card.format(n=counts["retrieved"], label="Retrieved") + arrow
        + card.format(n=counts["screened"], label="Screened (deduplicated)") + arrow
        + card.format(n=counts["included"], label="Included in synthesis")
        + "</div>"
    )


def meter_html(meter):
    if not meter:
        return ""
    color = METER_COLORS.get(meter["category"], "#555")
    return (
        f'<div class="meter"><span class="badge" style="background:{color}">'
        f'Consensus: {meter["category"]}</span>'
        f'<div class="meter-bar">'
        f'<span style="width:{meter["supporting"]}%;background:#1a7f4e" '
        f'title="Supporting {meter["supporting"]}%"></span>'
        f'<span style="width:{meter["mixed"]}%;background:#d4a017" '
        f'title="Mixed {meter["mixed"]}%"></span>'
        f'<span style="width:{meter["contradicting"]}%;background:#b03a3a" '
        f'title="Contradicting {meter["contradicting"]}%"></span></div>'
        f'<div class="meter-legend">🟢 Supporting {meter["supporting"]}% · '
        f'🟡 Mixed {meter["mixed"]}% · 🔴 Contradicting {meter["contradicting"]}%'
        f"</div></div>"
    )


def build_html_export(question, meter, report_md, counts):
    if md_lib:
        body = md_lib.markdown(report_md, extensions=["tables"])
    else:  # graceful fallback if the markdown package is missing
        body = "<pre style='white-space:pre-wrap'>" + (
            report_md.replace("&", "&amp;").replace("<", "&lt;")) + "</pre>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep Search Report — {question}</title>
<style>
  :root {{ --ink:#1c2733; --line:#e3e8ee; --accent:#2456a6; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,-apple-system,'Helvetica Neue',sans-serif;
         color:var(--ink); background:#f6f8fa; margin:0; line-height:1.65; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:40px 24px; }}
  .report {{ background:#fff; border:1px solid var(--line); border-radius:12px;
             padding:44px 48px; box-shadow:0 1px 4px rgba(20,40,80,.06); }}
  h1 {{ font-size:1.7rem; margin:0 0 4px; }}
  h2 {{ font-size:1.25rem; border-bottom:2px solid var(--line);
        padding-bottom:6px; margin-top:2.2em; color:var(--accent); }}
  h3 {{ font-size:1.05rem; margin-top:1.6em; }}
  .subtitle {{ color:#5b6b7c; font-size:.9rem; margin-bottom:20px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; margin:14px 0; }}
  th, td {{ border:1px solid var(--line); padding:8px 10px; text-align:left;
            vertical-align:top; }}
  th {{ background:#f0f4f8; }}
  tr:nth-child(even) td {{ background:#fafbfc; }}
  .badge {{ display:inline-block; color:#fff; font-weight:600; font-size:.85rem;
            padding:4px 14px; border-radius:999px; }}
  .meter {{ margin:14px 0 24px; }}
  .meter-bar {{ display:flex; height:12px; border-radius:6px; overflow:hidden;
                margin:10px 0 6px; border:1px solid var(--line); }}
  .meter-legend {{ font-size:.85rem; color:#5b6b7c; }}
  .prisma-flow {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap;
                  margin:18px 0 6px; }}
  .prisma-card {{ border:1px solid var(--line); border-radius:10px;
                  padding:14px 22px; text-align:center; background:#fbfcfe;
                  min-width:150px; }}
  .prisma-n {{ font-size:1.6rem; font-weight:700; color:var(--accent); }}
  .prisma-l {{ font-size:.8rem; color:#5b6b7c; }}
  .prisma-arrow {{ font-size:1.4rem; color:#9aa7b5; }}
  @media (max-width:640px) {{ .report {{ padding:24px 18px; }}
    .prisma-arrow {{ display:none; }} }}
  @media print {{
    body {{ background:#fff; }}
    .wrap {{ max-width:100%; padding:0; }}
    .report {{ border:none; box-shadow:none; padding:0; }}
    h2 {{ break-after:avoid; }}
    table, .prisma-flow, .meter {{ break-inside:avoid; }}
    a {{ color:var(--ink); text-decoration:none; }}
  }}
</style>
</head>
<body><div class="wrap"><div class="report">
<h1>Deep Search Report</h1>
<div class="subtitle">Question: {question} · Generated {date.today().isoformat()}
 · Sources: ERIC, OpenAlex, Semantic Scholar</div>
{meter_html(meter)}
{prisma_html(counts)}
{body}
</div></div></body></html>"""


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Education Literature Review Assistant",
                   page_icon="📚", layout="wide")
st.title("📚 Education Literature Review Assistant")
st.caption("Free-API alternative to Consensus / Elicit — ERIC + OpenAlex + "
           "Semantic Scholar, synthesized by Claude with strict grounding.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API key", type="password")
    question = st.text_area(
        "Research question",
        placeholder="e.g., Does retrieval practice improve K-12 science learning?",
        height=110)
    run = st.button("Run Deep Search", type="primary",
                    use_container_width=True)

if run:
    if not api_key or not question.strip():
        st.error("Please provide both an Anthropic API key and a research question.")
        st.stop()

    q = question.strip()
    with st.status("Running deep search…", expanded=True) as status:
        st.write("Querying ERIC…")
        eric = fetch_eric(q)
        st.write(f"→ {len(eric)} records. Querying OpenAlex…")
        oa = fetch_openalex(q)
        st.write(f"→ {len(oa)} records. Querying Semantic Scholar…")
        s2 = fetch_semantic_scholar(q)
        st.write(f"→ {len(s2)} records. Screening & deduplicating…")

        all_papers = eric + oa + s2
        screened, included = dedupe_and_screen(all_papers)
        counts = {"retrieved": len(all_papers), "eric": len(eric),
                  "openalex": len(oa), "s2": len(s2),
                  "screened": len(screened), "included": len(included)}

        if not included:
            status.update(label="No usable papers found", state="error")
            st.error("No papers with usable abstracts were retrieved. "
                     "Try a broader or differently-worded query.")
            st.stop()

        st.write(f"{counts['included']} papers included. Synthesizing with Claude…")
        try:
            raw = synthesize(api_key, q, build_context(included), counts)
        except Exception as e:
            status.update(label="Synthesis failed", state="error")
            st.error(f"LLM synthesis failed: {e}")
            st.stop()
        status.update(label="Report ready", state="complete")

    meter, body = parse_meter(raw)
    st.session_state.result = {
        "question": q, "meter": meter, "body": body,
        "counts": counts, "included": included,
    }

if "result" in st.session_state:
    r = st.session_state.result
    meter, counts = r["meter"], r["counts"]

    # Consensus meter + PRISMA cards (in-app)
    if meter:
        st.markdown(meter_html(meter) + prisma_html(counts) +
                    """<style>
                    .badge{display:inline-block;color:#fff;font-weight:600;
                      padding:4px 14px;border-radius:999px;}
                    .meter-bar{display:flex;height:12px;border-radius:6px;
                      overflow:hidden;margin:10px 0 6px;border:1px solid #ddd;}
                    .meter-legend{font-size:.85rem;color:#5b6b7c;}
                    .prisma-flow{display:flex;align-items:center;gap:12px;
                      flex-wrap:wrap;margin:14px 0;}
                    .prisma-card{border:1px solid #e3e8ee;border-radius:10px;
                      padding:12px 20px;text-align:center;background:#fbfcfe;}
                    .prisma-n{font-size:1.5rem;font-weight:700;color:#2456a6;}
                    .prisma-l{font-size:.8rem;color:#5b6b7c;}
                    .prisma-arrow{font-size:1.3rem;color:#9aa7b5;}
                    </style>""",
                    unsafe_allow_html=True)

    st.markdown(r["body"])

    st.divider()
    bib = make_bibtex(r["included"])
    html = build_html_export(r["question"], meter, r["body"], counts)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download BibTeX (.bib) for Zotero", bib,
                           file_name="literature_review.bib",
                           mime="text/plain", use_container_width=True)
    with c2:
        st.download_button("⬇️ Download styled HTML report (print → PDF)", html,
                           file_name="deep_search_report.html",
                           mime="text/html", use_container_width=True)
    st.caption("Open the HTML file in a browser and press Ctrl+P / Cmd+P for a "
               "clean PDF. All citations are grounded in the "
               f"{counts['included']} retrieved records only.")
