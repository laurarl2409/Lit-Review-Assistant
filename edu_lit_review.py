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
# Design tokens
# ==========================================================================

INK = "#1B2437"
PAPER = "#FAF9F6"
CARD = "#FFFFFF"
LINE = "#E7E4DC"
RULE = "#EEEBE3"      # hairline table rules
MUTED = "#6E7480"
GREEN = "#2F6B4F"
AMBER = "#B97D2A"
RED = "#A44444"

# Force the light theme at Streamlit's config level so OS/browser dark mode
# can never produce unreadable widgets, and persist it for future launches.
try:
    from streamlit import config as _st_config
    _st_config.set_option("theme.base", "light")
    _st_config.set_option("theme.primaryColor", INK)
    _st_config.set_option("theme.backgroundColor", PAPER)
    _st_config.set_option("theme.secondaryBackgroundColor", "#F3F1EB")
    _st_config.set_option("theme.textColor", INK)
except Exception:
    pass
try:
    import pathlib
    _cfg = pathlib.Path(".streamlit/config.toml")
    if not _cfg.exists():
        _cfg.parent.mkdir(exist_ok=True)
        _cfg.write_text(
            '[theme]\nbase="light"\nprimaryColor="%s"\n'
            'backgroundColor="%s"\nsecondaryBackgroundColor="#F3F1EB"\n'
            'textColor="%s"\n' % (INK, PAPER, INK))
except Exception:
    pass

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


def clean_query(q):
    """Strip punctuation that some APIs reject in search strings."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", q)).strip()


def fetch_eric(query):
    papers = []
    try:
        r = requests.get(
            "https://api.ies.ed.gov/eric/",
            params={"search": clean_query(query), "format": "json", "rows": 15},
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
    if not inv_idx:
        return ""
    pos = {}
    for word, idxs in inv_idx.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch_openalex(query):
    """Education subfield is 3304 in OpenAlex's topic hierarchy (field 17 is
    Computer Science). If the filtered call fails, retry unfiltered so the
    search still returns something."""
    papers = []
    base_params = {
        "search": clean_query(query),
        "per_page": 15,
        "mailto": "researcher@example.org",
    }
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={**base_params, "filter": "primary_topic.subfield.id:3304"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        if r.status_code >= 400:          # filter rejected — retry unfiltered
            r = requests.get("https://api.openalex.org/works",
                             params=base_params, headers=HEADERS, timeout=TIMEOUT)
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


def fetch_semantic_scholar(query, s2_key=""):
    """Anonymous Semantic Scholar traffic shares one heavily-limited pool, so
    retry with growing waits; a free API key (optional) lifts the limit."""
    papers = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": clean_query(query), "limit": 15,
        "fields": "title,authors,year,citationCount,abstract,externalIds,tldr",
    }
    headers = dict(HEADERS)
    if s2_key:
        headers["x-api-key"] = s2_key
    try:
        r = None
        for attempt in range(4):
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            if r.status_code != 429:
                break
            time.sleep(3 * (attempt + 1))     # 3s, 6s, 9s between tries
        if r is not None and r.status_code == 429:
            st.warning(
                "Semantic Scholar is rate-limiting anonymous requests right "
                "now. Continuing without it — a free API key from "
                "semanticscholar.org/product/api (added in the sidebar) "
                "avoids this.")
            return papers
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
# Screening (PRISMA), grounding context, BibTeX, references
# ==========================================================================

def dedupe_and_screen(all_papers, max_included=25, prior_keys=None):
    seen, screened = set(prior_keys or ()), []
    for p in all_papers:
        keys = {re.sub(r"\W+", "", p["title"].lower())}
        if p["doi"]:
            keys.add(p["doi"].lower())
        keys.discard("")
        if not keys or not (keys & seen):
            seen |= keys
            screened.append(p)
    with_text = [p for p in screened if len(p["abstract"]) > 80]
    with_text.sort(key=lambda p: (p["citations"] or 0), reverse=True)
    return screened, with_text[:max_included]


def build_context(included, start=1):
    blocks = []
    for i, p in enumerate(included, start):
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


def make_references_md(included):
    """Deterministic reference list built from retrieved metadata only —
    the LLM never writes this section, so it can't hallucinate it."""
    lines = ["## References", ""]
    for i, p in enumerate(included, 1):
        authors = ", ".join(p["authors"][:6]) or "Unknown authors"
        if len(p["authors"]) > 6:
            authors += ", et al."
        year = p["year"] or "n.d."
        venue = p["venue"] if p["venue"] not in ("Semantic Scholar",) else ""
        ref = f"**[{i}]** {authors} ({year}). {p['title']}."
        if venue:
            ref += f" *{venue}*."
        if p["url"]:
            label = f"doi.org/{p['doi']}" if p["doi"] else "link"
            ref += f" [{label}]({p['url']})"
        lines.append(ref)
        lines.append("")
    return "\n".join(lines)


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

COMMON_RULES = """Write a Markdown report with EXACTLY the structure below.
Output raw Markdown only — no code fences, no preamble, no extra sections.
FORMATTING RULES (identical every time):
- Table column headers must match the spec exactly, word for word.
- Evidence-strength values must be EXACTLY one word: Strong, Moderate, or Weak.
- Coverage-cell values must be EXACTLY one word: Covered, Partial, or Gap.
- Do not use bold or italics inside table cells.
- Do not add a References section — it is generated separately."""

CONSENSUS_INSTRUCTIONS = COMMON_RULES + """

Line 1 (machine-readable, nothing before it):
CONSENSUS_METER: <Strong|Moderate/Mixed|Weak> | supporting=<int>% mixed=<int>% contradicting=<int>%
(Percentages must sum to 100 and reflect your paper-by-paper reading.)

Line 2:
REPORT_TITLE: <one declarative sentence answering the question, max 16 words>

## 1. Introduction
2-3 paragraphs: what the included literature says about the question, where the
evidence is uneven, and an explicit statement of the consensus category with the
approximate share of papers supporting / mixed / contradicting.

## 2. Methods
One paragraph describing the databases searched (ERIC, OpenAlex, Semantic
Scholar), the query, and the screening logic, referencing the counts provided.
Note evidence-hierarchy weighting (systematic reviews/RCTs > quasi-experimental
> correlational > qualitative/descriptive).

## 3. Results
### Key Papers
One short paragraph naming the 3-4 anchor papers and why they anchor the corpus.
Then a Markdown table (4-6 rows) with EXACTLY these columns:
Paper | Year | Design | Core finding
The Paper column uses bracketed numbers like [3].
### <Thematic subsection title> (2-4 of these, each 1-2 paragraphs citing papers)
### Timeline and Venues
One paragraph on publication-year spread and notable venues, from metadata only.

## 4. Discussion
2-3 paragraphs on what the corpus supports best and its causal vs. descriptive
limits. Then a Markdown table (4-6 rows) with EXACTLY these columns:
Claim | Evidence Strength | Reasoning | Papers

## 5. Conclusion
1-2 closing paragraphs, including a caution that this reflects only the
retrieved records, not the entire literature.
### Research Gaps
One paragraph, then a Markdown coverage table with EXACTLY these columns:
Theme | Causal Tests | Long-Term Outcomes | Equity | Generalization
### Open Research Questions
A Markdown table (3-5 rows) with EXACTLY these columns: Question | Why It Matters"""

LANDSCAPE_INSTRUCTIONS = COMMON_RULES + """

This is an enumeration question (what/which/who), so map the answer space
rather than measuring a single consensus.

Line 1 (machine-readable, nothing before it):
FINDINGS_SUMMARY: <Finding name> = <Strong|Moderate|Weak>; <Finding name> = <Strong|Moderate|Weak>; ...
(3-7 findings covering the distinct answers in the corpus; names of at most
6 words; strength reflects the evidence behind each finding.)

Line 2:
REPORT_TITLE: <one declarative sentence summarizing the main findings, max 16 words>

## 1. Introduction
2-3 paragraphs: an overview of the answer space the included literature covers,
which findings rest on the strongest evidence, and where coverage is thin.

## 2. Methods
One paragraph describing the databases searched (ERIC, OpenAlex, Semantic
Scholar), the query, and the screening logic, referencing the counts provided.
Note evidence-hierarchy weighting (systematic reviews/RCTs > quasi-experimental
> correlational > qualitative/descriptive).

## 3. Findings
For EACH finding, in the same order as FINDINGS_SUMMARY, write:
### <Finding name> :: <Strong|Moderate|Weak>
followed by 1-2 paragraphs describing the finding, citing papers, and noting
the study designs behind it.

## 4. Discussion
2-3 paragraphs on the overall shape of the evidence and its causal vs.
descriptive limits. Then a Markdown table (one row per finding) with EXACTLY
these columns: Finding | Evidence Strength | Key Papers

## 5. Conclusion
1-2 closing paragraphs, including a caution that this reflects only the
retrieved records, not the entire literature.
### Research Gaps
One paragraph, then a Markdown coverage table (rows = the findings) with
EXACTLY these columns:
Finding | Causal Tests | Long-Term Outcomes | Equity | Generalization
### Open Research Questions
A Markdown table (3-5 rows) with EXACTLY these columns: Question | Why It Matters"""


def question_mode(q):
    """Enumeration questions get a findings-landscape report; everything else
    gets the consensus-meter report."""
    first = re.match(r"\s*(\w+)", q.lower())
    first = first.group(1) if first else ""
    if first in ("what", "which", "who", "where", "why") or \
            re.match(r"\s*how\s+(do|does|are|is|can|has|have)\b.*\b(vary|differ)", q.lower()):
        return "landscape"
    return "consensus"

GEMINI_MODEL = "gemini-3.5-flash"  # has a free tier as of mid-2026


def synthesize(api_key, question, context, counts, mode="consensus"):
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
        + (LANDSCAPE_INSTRUCTIONS if mode == "landscape" else CONSENSUS_INSTRUCTIONS)
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, max_output_tokens=16000,
        temperature=0.2)   # low temperature keeps formatting identical across runs
    last_err = None
    for attempt in range(3):
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
    t = text.strip()
    t = re.sub(r"^```(?:markdown|md)?\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def parse_meter(report):
    m = re.search(
        r"CONSENSUS_METER:\s*(Strong|Moderate/Mixed|Weak)\s*\|\s*"
        r"supporting=(\d+)%\s*mixed=(\d+)%\s*contradicting=(\d+)%", report)
    if not m:
        return None, report
    s, x, c = int(m.group(2)), int(m.group(3)), int(m.group(4))
    total = max(s + x + c, 1)
    meter = {
        "category": m.group(1),
        "supporting": s, "mixed": x, "contradicting": c,
        "w_sup": round(s / total * 100, 1),
        "w_mix": round(x / total * 100, 1),
        "w_con": round(c / total * 100, 1),
    }
    return meter, report[m.end():].lstrip()


def parse_findings(report):
    """Extract the FINDINGS_SUMMARY line for landscape reports."""
    m = re.search(r"FINDINGS_SUMMARY:\s*(.+)", report)
    if not m:
        return None, report
    items = []
    for part in m.group(1).split(";"):
        if "=" in part:
            name, _, s = part.rpartition("=")
            s = s.strip().rstrip(".")
            if s in ("Strong", "Moderate", "Weak") and name.strip():
                items.append((name.strip().strip("*"), s))
    body = (report[:m.start()] + report[m.end():]).lstrip()
    return (items or None), body


def parse_title(body, fallback):
    m = re.search(r"^REPORT_TITLE:\s*(.+?)\s*$", body, flags=re.MULTILINE)
    if not m:
        return fallback, body
    title = m.group(1).strip().strip("*").rstrip(".")
    return title or fallback, (body[:m.start()] + body[m.end():]).lstrip()


def extract_open_questions(body, limit=3):
    """Pull the AI-generated Open Research Questions out of the report so they
    can be offered as one-tap follow-ups."""
    m = re.search(r"### Open Research Questions(.*?)(?=\n## |\Z)", body, re.S)
    if not m:
        return []
    qs = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.startswith("|") and not set(line) <= set("|-: "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and cells[0] and cells[0].lower() not in ("question",):
                qs.append(cells[0])
    return qs[:limit]


def synthesize_followup(api_key, question, report_title, context, start_n):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    user_msg = (
        f'You previously produced a grounded literature report titled:\n'
        f'"{report_title}"\n\n'
        f"FOLLOW-UP QUESTION: {question}\n\n"
        f"NEW PAPERS (your ONLY citable sources for new claims; their numbering "
        f"continues the original report's reference list):\n\n{context}\n\n"
        f"Write a raw-Markdown addendum — no code fences, no preamble — with "
        f"EXACTLY this shape:\n"
        f"Line 1: ## Follow-up: {question}\n"
        f"Then 2-4 paragraphs answering the follow-up using ONLY the new "
        f"papers, with bracketed citations like [{start_n}]. If the answer "
        f"enumerates several findings, add one Markdown table with EXACTLY "
        f"these columns: Finding | Evidence Strength | Key Papers "
        f"(strength EXACTLY one word: Strong, Moderate, or Weak).\n"
        f"Close with one sentence connecting this to the original report's "
        f"conclusion. Do not add a References section."
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, max_output_tokens=8000,
        temperature=0.2)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=user_msg, config=config)
            text = resp.text or ""
            if not text.strip():
                raise RuntimeError("Gemini returned an empty response — wait "
                                   "a minute and try the follow-up again.")
            return text
        except Exception as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(15 * (attempt + 1))
            else:
                raise
    raise last_err


# ==========================================================================
# Report rendering — one pipeline used both in-app and in the HTML export
# ==========================================================================

def _inline_md(s):
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


def _mini_md(md):
    """Small, dependency-free Markdown converter covering exactly the shapes
    this app's reports use (headings, paragraphs, pipe tables, lists, bold,
    italics, links). Used when the optional `markdown` package is missing, so
    the report and export always render fully styled."""
    out, para, table, ul = [], [], [], []

    def flush_para():
        if para:
            out.append("<p>" + _inline_md(" ".join(para)) + "</p>")
            para.clear()

    def flush_ul():
        if ul:
            out.append("<ul>" + "".join(f"<li>{_inline_md(x)}</li>" for x in ul)
                       + "</ul>")
            ul.clear()

    def flush_table():
        if not table:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")]
                for r in table]
        h = ["<table>", "<thead><tr>"]
        h += [f"<th>{_inline_md(c)}</th>" for c in rows[0]]
        h.append("</tr></thead><tbody>")
        body = rows[1:]
        if body and all(re.fullmatch(r":?-{2,}:?", c) for c in body[0]):
            body = body[1:]
        for r in body:
            h.append("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in r)
                     + "</tr>")
        h.append("</tbody></table>")
        out.append("".join(h))
        table.clear()

    for raw in html_lib.escape(md).splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("|"):
            flush_para(); flush_ul(); table.append(line); continue
        flush_table()
        s = line.strip()
        if not s:
            flush_para(); flush_ul(); continue
        if s.startswith("### "):
            flush_para(); flush_ul(); out.append(f"<h3>{_inline_md(s[4:])}</h3>"); continue
        if s.startswith("## "):
            flush_para(); flush_ul(); out.append(f"<h2>{_inline_md(s[3:])}</h2>"); continue
        if s.startswith(("- ", "* ")):
            flush_para(); ul.append(s[2:]); continue
        flush_ul(); para.append(s)
    flush_para(); flush_ul(); flush_table()
    return "\n".join(out)


STRENGTH_FILLS = {"Strong": (8, GREEN), "Moderate": (6, AMBER), "Weak": (2, RED)}


def _meter_html(word):
    n, color = STRENGTH_FILLS[word]
    segs = "".join(
        f'<i style="background:{color}"></i>' if k < n else "<i></i>"
        for k in range(10))
    return (f'<span class="meter10">{segs}</span>'
            f'<span class="meter-word" style="color:{color}">{word}</span>')


def _strength_meter_cell(word):
    return f'<td class="strength">{_meter_html(word)}</td>'


def findings_hero_html(findings):
    if not findings:
        return ""
    rows = "".join(
        f'<div class="fh-row"><span class="fh-name">{html_lib.escape(n)}</span>'
        f'<span class="finding-chip">{_meter_html(s)}</span></div>'
        for n, s in findings)
    return ('<div class="verdict">'
            '<div class="verdict-eyebrow">Evidence at a glance</div>'
            f'<div class="fh-list">{rows}</div></div>')


def render_report_html(body_md):
    """Markdown -> Consensus-styled HTML: segmented strength meters in the
    claim matrix, tinted coverage cells in the gaps heatmap."""
    # tolerate emoji heatmap cells from older prompts
    body_md = (body_md.replace("🟢", "Covered").replace("🟡", "Partial")
               .replace("🔴", "Gap"))
    if md_lib:
        h = md_lib.markdown(body_md, extensions=["tables", "sane_lists"])
    else:
        h = _mini_md(body_md)
    for word in ("Strong", "Moderate", "Weak"):
        h = h.replace(f"<td>{word}</td>", _strength_meter_cell(word))
    h = h.replace('<td>Moderate/Mixed</td>', _strength_meter_cell("Moderate"))
    for word, cls in (("Covered", "cov"), ("Partial", "par"), ("Gap", "gap")):
        h = h.replace(f"<td>{word}</td>", f'<td class="hm hm-{cls}">{word}</td>')
    h = re.sub(
        r"<h3>(.*?)\s*::\s*(Strong|Moderate|Weak)</h3>",
        lambda m: (f'<h3 class="finding-h"><span>{m.group(1)}</span>'
                   f'<span class="finding-chip">{_meter_html(m.group(2))}</span></h3>'),
        h)
    return h


def verdict_html(meter):
    if not meter:
        return ""
    color, label = VERDICT_STYLE.get(meter["category"], (MUTED, meter["category"]))
    seg = ('<span class="vseg" style="width:{w}%;background:{c}"></span>')
    return (
        '<div class="verdict">'
        '<div class="verdict-eyebrow">Consensus meter</div>'
        f'<div class="verdict-cat" style="color:{color}">{label}</div>'
        '<div class="vbar">'
        + seg.format(w=meter["w_sup"], c=GREEN)
        + seg.format(w=meter["w_mix"], c=AMBER)
        + seg.format(w=meter["w_con"], c=RED)
        + "</div>"
        '<div class="vlegend">'
        f'<span><i style="background:{GREEN}"></i>Supporting {meter["supporting"]}%</span>'
        f'<span><i style="background:{AMBER}"></i>Mixed {meter["mixed"]}%</span>'
        f'<span><i style="background:{RED}"></i>Contradicting {meter["contradicting"]}%</span>'
        "</div></div>"
    )


def prisma_html(counts):
    card = ('<div class="prisma-card"><div class="prisma-n">{n}</div>'
            '<div class="prisma-l">{label}</div><div class="prisma-s">{sub}</div></div>')
    arrow = '<div class="prisma-arrow">&#8594;</div>'
    return (
        '<div class="prisma-flow">'
        + card.format(n=counts["retrieved"], label="Retrieved",
                      sub=f'ERIC {counts["eric"]} &middot; OpenAlex '
                          f'{counts["openalex"]} &middot; S2 {counts["s2"]}')
        + arrow
        + card.format(n=counts["screened"], label="Screened",
                      sub="After deduplication")
        + arrow
        + card.format(n=counts["included"], label="Included",
                      sub="Usable abstracts, ranked by citations")
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
            f'<span class="src">{p["source"]}</span>'
            "</div></div></div>"
        )
    return '<div class="paper-list">' + "".join(items) + "</div>"


# --- Report CSS, scoped to .report-doc, shared by app and export ----------

REPORT_CSS = f"""
.report-doc {{ font-family:'Public Sans',system-ui,sans-serif; color:{INK};
  font-size:.95rem; line-height:1.7; }}
.report-doc h2 {{ font-family:'Public Sans',sans-serif; font-weight:700;
  font-size:1.12rem; color:{INK}; margin:2.6em 0 .7em; letter-spacing:0; }}
.report-doc h2:first-child {{ margin-top:.6em; }}
.report-doc h3 {{ font-family:'Public Sans',sans-serif; font-weight:700;
  font-size:.95rem; color:{INK}; margin:1.9em 0 .5em; }}
.report-doc p {{ margin:.75em 0; }}
.report-doc a {{ color:{GREEN}; }}
.report-doc table {{ border-collapse:collapse; width:100%; margin:18px 0 26px;
  font-size:.88rem; }}
.report-doc th {{ text-align:left; font-size:.75rem; font-weight:700;
  color:{INK}; padding:0 18px 10px 0; border-bottom:1px solid {LINE};
  vertical-align:bottom; }}
.report-doc td {{ padding:13px 18px 13px 0; border-bottom:1px solid {RULE};
  vertical-align:top; }}
.report-doc td:last-child, .report-doc th:last-child {{ padding-right:0; }}
.report-doc .strength {{ white-space:nowrap; }}
.report-doc .meter10 {{ display:inline-flex; gap:2px; vertical-align:middle; }}
.report-doc .meter10 i {{ display:inline-block; width:6px; height:15px;
  border-radius:2px; background:{LINE}; }}
.report-doc .meter-word {{ display:block; font-family:'IBM Plex Mono',monospace;
  font-size:.68rem; font-weight:600; margin-top:5px; }}
.report-doc .hm {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  font-weight:600; text-align:center; border-radius:6px; }}
.report-doc td.hm {{ padding:13px 10px; }}
.report-doc .hm-cov {{ background:#E9F1EB; color:{GREEN}; }}
.report-doc .hm-par {{ background:#F6EDDD; color:{AMBER}; }}
.report-doc .hm-gap {{ background:#F5E7E7; color:{RED}; }}
.report-doc .finding-h {{ display:flex; align-items:center;
  justify-content:space-between; gap:16px; flex-wrap:wrap; }}
.finding-chip {{ display:inline-flex; align-items:center; gap:8px;
  white-space:nowrap; }}
.finding-chip .meter-word {{ display:inline; margin-top:0; }}
.fh-list {{ display:flex; flex-direction:column; margin-top:8px; }}
.fh-row {{ display:flex; align-items:center; justify-content:space-between;
  gap:16px; padding:9px 0; border-bottom:1px solid {RULE}; }}
.fh-row:last-child {{ border-bottom:none; }}
.fh-name {{ font-weight:600; font-size:.92rem; color:{INK}; }}

.verdict {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px;
  padding:22px 26px; margin:4px 0 14px;
  box-shadow:0 1px 3px rgba(27,36,55,.05); }}
.verdict-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem;
  letter-spacing:.14em; text-transform:uppercase; color:{MUTED};
  margin-bottom:2px; }}
.verdict-cat {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:1.6rem; letter-spacing:-.01em; margin-bottom:12px; }}
.vbar {{ display:flex; height:13px; border-radius:7px; overflow:hidden;
  background:{LINE}; }}
.vseg {{ display:block; height:100%; }}
.vlegend {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px;
  font-family:'IBM Plex Mono',monospace; font-size:.74rem; color:{MUTED}; }}
.vlegend i {{ display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:6px; }}

.prisma-flow {{ display:flex; align-items:stretch; gap:12px; flex-wrap:wrap;
  margin:0 0 8px; }}
.prisma-card {{ border:1px solid {LINE}; border-radius:12px; padding:15px 22px;
  background:{CARD}; min-width:170px; flex:1;
  box-shadow:0 1px 3px rgba(27,36,55,.04); }}
.prisma-n {{ font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
  font-weight:600; color:{INK}; }}
.prisma-l {{ font-size:.82rem; font-weight:600; color:{INK}; margin-top:1px; }}
.prisma-s {{ font-size:.73rem; color:{MUTED}; margin-top:2px; }}
.prisma-arrow {{ align-self:center; font-size:1.3rem; color:#B9B4A8; }}

.paper-list {{ display:flex; flex-direction:column; gap:10px; }}
.paper {{ display:flex; gap:14px; background:{CARD}; border:1px solid {LINE};
  border-radius:12px; padding:14px 18px; }}
.paper-n {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  color:{MUTED}; font-size:.85rem; min-width:34px; }}
.paper-title {{ font-weight:600; line-height:1.4; margin-bottom:2px; color:{INK}; }}
.paper-title a {{ color:{INK}; text-decoration:none; }}
.paper-title a:hover {{ color:{GREEN}; text-decoration:underline; }}
.paper-meta {{ font-size:.82rem; color:{MUTED}; }}
.src {{ font-family:'IBM Plex Mono',monospace; font-size:.66rem;
  font-weight:600; padding:1px 8px; border-radius:999px; margin-left:8px;
  border:1px solid {LINE}; color:{MUTED}; background:#F6F4EF; }}
"""


def build_html_export(title, question, hero_html, report_html, counts):
    t, q = html_lib.escape(title), html_lib.escape(question)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
  * {{ box-sizing:border-box; }}
  body {{ font-family:'Public Sans',system-ui,sans-serif; color:{INK};
         background:{PAPER}; margin:0; }}
  .wrap {{ max-width:840px; margin:0 auto; padding:48px 24px; }}
  .sheet {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px;
            padding:56px 60px; box-shadow:0 1px 3px rgba(27,36,55,.05); }}
  .brand {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem;
            letter-spacing:.15em; text-transform:uppercase; color:{MUTED};
            margin-bottom:18px; }}
  h1.headline {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
       font-size:1.85rem; line-height:1.3; margin:0 0 10px;
       letter-spacing:-.01em; }}
  .subtitle {{ color:{MUTED}; font-size:.88rem; margin-bottom:30px;
               padding-bottom:26px; border-bottom:1px solid {LINE}; }}
  .footer {{ margin-top:44px; padding-top:20px; border-top:1px solid {LINE};
             font-size:.78rem; color:{MUTED}; font-style:italic; }}
  {REPORT_CSS}
  @media (max-width:640px) {{
    .sheet {{ padding:28px 20px; }}
    .prisma-arrow {{ display:none; }}
  }}
  @media print {{
    body {{ background:#fff; font-size:12.5px; }}
    .wrap {{ max-width:100%; padding:0; }}
    .sheet {{ border:none; box-shadow:none; padding:0; border-radius:0; }}
    h1.headline {{ font-size:1.55rem; }}
    .report-doc h2 {{ break-after:avoid; }}
    .report-doc table, .prisma-flow, .verdict {{ break-inside:avoid; }}
    a {{ color:{INK}; text-decoration:none; }}
  }}
</style>
</head>
<body><div class="wrap"><div class="sheet">
<div class="brand">Deep Search Report &middot; ERIC &middot; OpenAlex &middot; Semantic Scholar</div>
<h1 class="headline">{t}</h1>
<div class="subtitle">{q} &middot; Generated {date.today().strftime("%B %d, %Y")}
 &middot; {counts["included"]} papers synthesized from {counts["retrieved"]} retrieved records</div>
{hero_html}
{prisma_html(counts)}
<div class="report-doc">
{report_html}
</div>
<div class="footer">Generated with the Education Literature Review Assistant.
Every citation refers to the numbered References list above, built directly
from the retrieved records.</div>
</div></div></body></html>"""


# ==========================================================================
# App shell CSS — explicit light styling so dark browser themes can't break it
# ==========================================================================

APP_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp, [data-testid="stAppViewContainer"] {{ background:{PAPER}; }}
[data-testid="stHeader"] {{ background:{PAPER}; }}
.block-container {{ max-width:960px; padding-top:3.6rem; }}

html, body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
[data-testid="stText"] {{
  font-family:'Public Sans',system-ui,sans-serif; color:{INK};
}}
[data-testid="stCaptionContainer"], .stCaption {{ color:{MUTED} !important; }}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {{
  font-family:'Fraunces',Georgia,serif !important;
  font-weight:600 !important; letter-spacing:-.01em; color:{INK} !important;
}}

/* Header */
.app-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem;
  letter-spacing:.16em; text-transform:uppercase; color:{MUTED};
  margin-bottom:.4rem; }}
.app-title {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:2rem; line-height:1.2; letter-spacing:-.01em; margin:0;
  color:{INK}; }}
.app-sub {{ color:{MUTED}; font-size:.95rem; margin:.5rem 0 0; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background:#F3F1EB; border-right:1px solid {LINE}; }}
[data-testid="stSidebar"] * {{ color:{INK}; }}
[data-testid="stSidebar"] .stMarkdown p {{ font-size:.86rem; color:#565D6B; }}
[data-testid="stSidebar"] a {{ color:{GREEN} !important; }}

/* Inputs — forced light so they're readable in any browser theme */
.stTextArea textarea, .stTextInput input {{
  background:{CARD} !important; color:{INK} !important;
  border:1px solid {LINE} !important; border-radius:10px !important;
  caret-color:{INK};
}}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {{
  color:#9AA0AB !important; opacity:1;
}}
.stTextArea [data-baseweb="textarea"], .stTextInput [data-baseweb="input"],
.stTextArea [data-baseweb="base-input"], .stTextInput [data-baseweb="base-input"] {{
  background:{CARD} !important; border-color:{LINE} !important;
}}

/* Buttons — every selector variant Streamlit uses, forced light */
.stButton button, .stDownloadButton button, .stFormSubmitButton button,
button[data-testid^="stBaseButton"] {{
  background:{CARD} !important; color:{INK} !important;
  border:1px solid {LINE} !important; border-radius:10px; font-weight:600;
  transition:transform 120ms cubic-bezier(0.23,1,0.32,1),
             background 150ms ease, border-color 150ms ease;
}}
.stButton button p, .stDownloadButton button p,
button[data-testid^="stBaseButton"] p {{ color:{INK} !important; }}
.stButton button:hover, .stDownloadButton button:hover {{
  border-color:{INK} !important;
}}
.stButton button:active, .stDownloadButton button:active,
.stFormSubmitButton button:active {{ transform:scale(0.98); }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"],
button[data-testid="stBaseButton-primary"] {{
  background:{INK} !important; color:#FFFFFF !important;
  border:1px solid {INK} !important;
}}
.stButton button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {{
  background:#2A3550 !important; border-color:#2A3550 !important;
}}
.stButton button[kind="primary"] p, .stFormSubmitButton button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] p {{ color:#FFFFFF !important; }}
@media (prefers-reduced-motion: reduce) {{
  .stButton > button, .stDownloadButton > button,
  .stFormSubmitButton > button {{ transition:none; }}
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED}; font-weight:600; }}
.stTabs [aria-selected="true"] {{ color:{INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{INK}; }}

/* Status / expander */
[data-testid="stExpander"] {{ background:{CARD}; border:1px solid {LINE};
  border-radius:12px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p {{ color:{INK}; }}

/* How-it-works cards */
.how-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:6px; }}
.how-card {{ flex:1; min-width:200px; background:{CARD}; border:1px solid {LINE};
  border-radius:12px; padding:18px 20px;
  box-shadow:0 1px 3px rgba(27,36,55,.04); }}
.how-step {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem;
  letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.how-title {{ font-weight:600; margin:6px 0 4px; color:{INK}; }}
.how-body {{ font-size:.85rem; color:{MUTED}; line-height:1.5; }}

/* Result header */
.result-head {{ margin:4px 0 14px; }}
.result-title {{ font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:1.55rem; line-height:1.3; letter-spacing:-.01em; color:{INK}; }}
.result-meta {{ font-size:.85rem; color:{MUTED}; margin-top:6px; }}

{REPORT_CSS}
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
    s2_key = st.text_input(
        "Semantic Scholar API key (optional)", type="password",
        value=os.environ.get("S2_API_KEY", ""),
        help="Anonymous Semantic Scholar requests share one rate limit and "
             "often fail. A free key from semanticscholar.org/product/api "
             "makes that source reliable.")
    st.divider()
    st.markdown("#### About")
    st.markdown(
        "Searches **ERIC**, **OpenAlex**, and **Semantic Scholar**, then "
        "synthesizes the results into a five-section evidence report. Every "
        "claim is cited to a retrieved paper — the model is not allowed to "
        "use outside knowledge.")

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
        s2 = fetch_semantic_scholar(q, s2_key)
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

        mode = question_mode(q)
        st.write(f"{counts['included']} papers included. Synthesizing with Gemini "
                 f"({'findings landscape' if mode == 'landscape' else 'consensus'} report)…")
        try:
            raw = clean_llm_output(
                synthesize(api_key, q, build_context(included), counts, mode))
        except Exception as e:
            status.update(label="Synthesis failed", state="error")
            st.error(f"Synthesis failed: {e}")
            st.stop()
        status.update(label="Report ready", state="complete", expanded=False)

    meter, body = parse_meter(raw)
    findings, body = parse_findings(body)
    title, body = parse_title(body, q)
    for k in [k for k in st.session_state if str(k).startswith("sel_")]:
        del st.session_state[k]        # reset paper selections for the new result
    st.session_state.result = {
        "question": q, "title": title, "meter": meter, "findings": findings,
        "mode": mode, "body": body, "counts": counts, "included": included,
    }

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

if "result" in st.session_state:
    r = st.session_state.result
    meter, counts = r["meter"], r["counts"]
    full_body = r["body"] + "\n\n" + make_references_md(r["included"])
    report_html = render_report_html(full_body)

    st.divider()
    st.markdown(
        '<div class="result-head">'
        f'<div class="result-title">{html_lib.escape(r["title"])}</div>'
        f'<div class="result-meta">{html_lib.escape(r["question"])} &middot; '
        f'Generated {date.today().strftime("%B %d, %Y")} &middot; '
        f'{counts["included"]} papers synthesized from '
        f'{counts["retrieved"]} retrieved records</div></div>',
        unsafe_allow_html=True)

    if r.get("mode") == "landscape" and r.get("findings"):
        st.markdown(findings_hero_html(r["findings"]), unsafe_allow_html=True)
    elif meter:
        st.markdown(verdict_html(meter), unsafe_allow_html=True)
    st.markdown(prisma_html(counts), unsafe_allow_html=True)

    hero = (findings_hero_html(r["findings"])
            if r.get("mode") == "landscape" and r.get("findings")
            else verdict_html(meter))
    html_out = build_html_export(r["title"], r["question"], hero,
                                 report_html, counts)
    dl1, dl2, _sp = st.columns([1, 1, 1])
    with dl1:
        st.download_button(
            "Download report (HTML)", html_out,
            file_name="deep_search_report.html", mime="text/html",
            use_container_width=True, type="primary")
    with dl2:
        st.download_button(
            "Download bibliography (.bib)", make_bibtex(r["included"]),
            file_name="literature_review.bib", mime="text/plain",
            use_container_width=True)
    st.caption("Open the HTML report in a browser and press Ctrl+P / Cmd+P for "
               "a clean PDF. To pick which papers go to Zotero, use the "
               "Zotero export tab.")

    if "## 5" not in r["body"]:
        st.warning("This report looks shorter than expected — it may have been "
                   "truncated. Running the search again usually fixes it.")

    tab_report, tab_papers, tab_export = st.tabs(
        ["Report", f"Included papers ({counts['included']})", "Zotero export"])

    with tab_report:
        st.markdown(f'<div class="report-doc">{report_html}</div>',
                    unsafe_allow_html=True)

    with tab_papers:
        st.markdown(
            "These are the only records the synthesis was allowed to cite. "
            "Bracketed numbers in the report refer to this list.")
        st.markdown(papers_html(r["included"]), unsafe_allow_html=True)

    with tab_export:
        st.markdown("Choose which papers to include, then download the .bib "
                    "file and import it into Zotero via **File → Import**.")

        def _set_all(value):
            for k in range(len(r["included"])):
                st.session_state[f"sel_{k}"] = value

        b1, b2, _bsp = st.columns([1, 1, 4])
        with b1:
            st.button("Select all", on_click=_set_all, args=(True,),
                      use_container_width=True)
        with b2:
            st.button("Clear all", on_click=_set_all, args=(False,),
                      use_container_width=True)

        left, right = st.columns(2)
        for idx, p in enumerate(r["included"]):
            label = f"[{idx + 1}] {p['title'][:70]}" + \
                    ("…" if len(p["title"]) > 70 else "")
            with (left if idx % 2 == 0 else right):
                if f"sel_{idx}" not in st.session_state:
                    st.session_state[f"sel_{idx}"] = True
                st.checkbox(label, key=f"sel_{idx}", help=p["title"])

        selected = [p for idx, p in enumerate(r["included"])
                    if st.session_state.get(f"sel_{idx}", True)]
        st.download_button(
            f"Download {len(selected)} selected paper"
            f"{'s' if len(selected) != 1 else ''} (.bib)",
            make_bibtex(selected) if selected else "",
            file_name="literature_review.bib", mime="text/plain",
            disabled=not selected, type="primary")
        if not selected:
            st.caption("Select at least one paper to enable the download.")

    # ---- Follow-up questions: extend the report, never replace it --------
    st.divider()
    st.markdown("#### Ask a follow-up")
    st.caption("Follow-ups run a fresh search and append a new section to "
               "this report, continuing the same reference numbering.")

    fu_clicked = None
    suggestions = extract_open_questions(r["body"])
    if suggestions:
        scols = st.columns(len(suggestions))
        for col, sq in zip(scols, suggestions):
            with col:
                if st.button(sq[:64] + ("…" if len(sq) > 64 else ""),
                             key=f"fu_sugg_{abs(hash(sq)) % 10**8}",
                             help=sq, use_container_width=True):
                    fu_clicked = sq

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        fu_text = st.text_input(
            "Follow-up question", key="fu_input",
            label_visibility="collapsed",
            placeholder="e.g., Which interventions address these barriers?")
    with fc2:
        fu_run = st.button("Run follow-up", type="primary",
                           use_container_width=True)

    fu_q = fu_clicked or (fu_text.strip() if fu_run else "")
    if fu_q:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar to run follow-ups.")
        else:
            with st.status(f"Following up: {fu_q[:60]}…", expanded=True) as fst:
                st.write("Searching ERIC…")
                f_eric = fetch_eric(fu_q)
                st.write(f"ERIC returned {len(f_eric)}. Searching OpenAlex…")
                f_oa = fetch_openalex(fu_q)
                st.write(f"OpenAlex returned {len(f_oa)}. Searching Semantic Scholar…")
                f_s2 = fetch_semantic_scholar(fu_q, s2_key)
                st.write(f"Semantic Scholar returned {len(f_s2)}. Screening…")

                prior_keys = set()
                for p in r["included"]:
                    prior_keys.add(re.sub(r"\W+", "", p["title"].lower()))
                    if p["doi"]:
                        prior_keys.add(p["doi"].lower())
                f_all = f_eric + f_oa + f_s2
                f_screened, f_inc = dedupe_and_screen(
                    f_all, max_included=12, prior_keys=prior_keys)

                if not f_inc:
                    fst.update(label="No new papers found", state="error")
                    st.warning("The follow-up search found no new papers "
                               "beyond those already in this report. Try "
                               "different wording — the report is unchanged.")
                else:
                    start_n = len(r["included"]) + 1
                    st.write(f"{len(f_inc)} new papers. Synthesizing addendum…")
                    try:
                        addendum = clean_llm_output(synthesize_followup(
                            api_key, fu_q, r["title"],
                            build_context(f_inc, start=start_n), start_n))
                    except Exception as e:
                        fst.update(label="Follow-up failed", state="error")
                        st.error(f"Follow-up synthesis failed: {e} — the "
                                 "report is unchanged.")
                        addendum = None
                    if addendum:
                        if not addendum.lstrip().startswith("## Follow-up"):
                            addendum = f"## Follow-up: {fu_q}\n\n" + addendum
                        r["body"] = r["body"].rstrip() + "\n\n" + addendum
                        r["included"] = r["included"] + f_inc
                        c = r["counts"]
                        c["retrieved"] += len(f_all)
                        c["eric"] += len(f_eric)
                        c["openalex"] += len(f_oa)
                        c["s2"] += len(f_s2)
                        c["screened"] += len(f_screened)
                        c["included"] = len(r["included"])
                        fst.update(label="Follow-up added to the report",
                                   state="complete", expanded=False)
                        st.session_state.pop("fu_input", None)
                        st.rerun()
