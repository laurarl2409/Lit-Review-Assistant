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


STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "about", "into", "over", "under", "between", "among", "during",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "there", "their", "them", "they", "it", "its", "as", "any",
    "some", "such", "more", "most", "much", "many", "also", "other",
    "evidence", "research", "literature", "study", "studies", "paper",
    "papers", "article", "articles", "review", "reviews", "finding",
    "findings", "data", "show", "shows", "shown", "exist", "exists",
}


def search_terms(q, max_terms=10):
    """Turn a natural-language question into a keyword query.

    Academic APIs are built for keywords: a sentence-length `search` value
    makes OpenAlex time out (504) and makes Semantic Scholar return nothing.
    Content words only, original order, capped length.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", q)
    words, seen = [], set()
    for w in cleaned.split():
        key = w.lower().strip("-")
        if not key or len(key) < 2 or key in seen or key in STOPWORDS:
            continue
        seen.add(key)
        words.append(w)
        if len(words) >= max_terms:
            break
    return " ".join(words) if words else re.sub(r"\s+", " ", cleaned).strip()


def clean_query(q):
    """Search string used for every literature API."""
    return search_terms(q)


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
        r = None
        # Filtered first, then unfiltered; retry 5xx with backoff because
        # OpenAlex intermittently times out under load.
        for params in ({**base_params,
                        "filter": "primary_topic.subfield.id:3304"},
                       base_params):
            for attempt in range(3):
                r = requests.get("https://api.openalex.org/works",
                                 params=params, headers=HEADERS, timeout=45)
                if r.status_code < 400:
                    break
                if r.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            if r is not None and r.status_code < 400:
                break
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


def _strip_jats(s):
    """CrossRef abstracts arrive as JATS XML; reduce to plain text."""
    s = re.sub(r"<jats:title>.*?</jats:title>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


CROSSREF_TYPES = {"journal-article", "proceedings-article", "book-chapter",
                  "posted-content", "report", "monograph", "reference-entry"}


def fetch_crossref(query):
    papers = []
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": clean_query(query), "rows": 15,
                "select": ("title,author,issued,container-title,abstract,"
                           "is-referenced-by-count,DOI,URL,type"),
                "mailto": "researcher@example.org",
            },
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        for w in (r.json().get("message", {}) or {}).get("items", []) or []:
            if w.get("type") and w["type"] not in CROSSREF_TYPES:
                continue
            titles = w.get("title") or []
            title = (titles[0] if titles else "").strip()
            if not title:
                continue
            authors = []
            for a in (w.get("author") or []):
                nm = " ".join(x for x in (a.get("given"), a.get("family")) if x)
                if not nm:
                    nm = a.get("name") or ""
                if nm:
                    authors.append(nm)
            year = None
            parts = ((w.get("issued") or {}).get("date-parts") or [[]])[0]
            if parts and isinstance(parts[0], int):
                year = parts[0]
            venues = w.get("container-title") or []
            doi = w.get("DOI")
            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "venue": (venues[0] if venues else "CrossRef"),
                "abstract": _strip_jats(w.get("abstract") or ""),
                "citations": w.get("is-referenced-by-count"),
                "doi": doi,
                "url": w.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                "source": "CrossRef",
            })
    except Exception as e:
        st.warning(f"CrossRef didn't respond ({e}). Continuing with the other sources.")
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


def build_context(included, start=1, abstract_chars=1400):
    blocks = []
    for i, p in enumerate(included, start):
        authors = ", ".join(p["authors"][:5]) or "Unknown authors"
        blocks.append(
            f"[{i}] {p['title']}\n"
            f"    Authors: {authors} | Year: {p['year'] or 'n.d.'} | "
            f"Venue: {p['venue']} | Citations: "
            f"{p['citations'] if p['citations'] is not None else 'n/a'} | "
            f"Source DB: {p['source']}\n"
            f"    Abstract: {p['abstract'][:abstract_chars]}"
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

# --- Model providers ------------------------------------------------------
# Free tiers first. Every provider except Gemini and Anthropic speaks the
# OpenAI chat-completions API, so one code path covers Groq, Cerebras, and
# OpenRouter.
PROVIDERS = {
    "gemini": {
        "label": "Google Gemini 3.5 Flash", "cost": "Free",
        "kind": "gemini", "model": "gemini-3.5-flash",
        "key_label": "Gemini API key", "env": "GEMINI_API_KEY",
        "url": "https://aistudio.google.com/apikey",
        "note": "~15 requests/min, 1,500/day. Best free quality.",
        "abstract_chars": 1400, "max_papers": 25, "max_out": 16000},
    "groq": {
        "label": "Groq — Llama 3.3 70B", "cost": "Free",
        "kind": "openai", "model": "llama-3.3-70b-versatile",
        "base": "https://api.groq.com/openai/v1",
        "key_label": "Groq API key", "env": "GROQ_API_KEY",
        "url": "https://console.groq.com/keys",
        "note": "~30 requests/min, no card required. Very fast.",
        "abstract_chars": 900, "max_papers": 18, "max_out": 8000},
    "cerebras": {
        "label": "Cerebras — GPT-OSS 120B", "cost": "Free",
        "kind": "openai", "model": "gpt-oss-120b",
        "base": "https://api.cerebras.ai/v1",
        "key_label": "Cerebras API key", "env": "CEREBRAS_API_KEY",
        "url": "https://cloud.cerebras.ai",
        "note": "~1M tokens/day. Fastest throughput of the free tiers.",
        "abstract_chars": 900, "max_papers": 18, "max_out": 8000},
    "openrouter": {
        "label": "OpenRouter — free models", "cost": "Free",
        "kind": "openai", "model": "meta-llama/llama-3.3-70b-instruct:free",
        "base": "https://openrouter.ai/api/v1",
        "key_label": "OpenRouter API key", "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/keys",
        "note": "One key, dozens of free models. ~20 requests/min.",
        "abstract_chars": 900, "max_papers": 18, "max_out": 8000},
    "anthropic": {
        "label": "Claude Haiku 4.5", "cost": "Paid — about $0.03 per report",
        "kind": "anthropic", "model": "claude-haiku-4-5",
        "key_label": "Anthropic API key", "env": "ANTHROPIC_API_KEY",
        "url": "https://console.anthropic.com/settings/keys",
        "note": "$1/$5 per million tokens. Reliable when free tiers are busy.",
        "abstract_chars": 1400, "max_papers": 25, "max_out": 8000},
}
PROVIDER_ORDER = ["gemini", "groq", "cerebras", "openrouter", "anthropic"]

# Errors worth retrying or failing over on: overload, rate limit, transient 5xx.
TRANSIENT = ("429", "500", "502", "503", "504", "resource_exhausted",
             "unavailable", "overloaded", "rate limit", "rate_limit",
             "quota", "high demand", "timeout", "timed out", "capacity")


def _is_transient(e):
    s = str(e).lower()
    return any(t in s for t in TRANSIENT)


def limits_for(provider):
    p = PROVIDERS[provider]
    return {"abstract_chars": p["abstract_chars"],
            "max_papers": p["max_papers"], "max_out": p["max_out"]}


def _one_call(provider, key, system, user, max_out):
    p = PROVIDERS[provider]
    if p["kind"] == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=p["model"], contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=max_out,
                temperature=0.2))
        return resp.text or ""
    if p["kind"] == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=key, timeout=180.0)
        resp = client.messages.create(
            model=p["model"], max_tokens=max_out, temperature=0.2,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if b.type == "text")
    from openai import OpenAI                      # groq / cerebras / openrouter
    client = OpenAI(api_key=key, base_url=p["base"], timeout=180.0)
    r = client.chat.completions.create(
        model=p["model"], temperature=0.2, max_tokens=max_out,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def llm_complete(llm, system, user, max_out=None):
    """Call the primary provider; on transient failure retry, then fail over
    to the backup provider if one is configured.

    `llm` is {"provider":..., "key":..., "backup":..., "backup_key":...}
    """
    chain = [(llm.get("provider", "gemini"), (llm.get("key") or "").strip())]
    if llm.get("backup"):
        chain.append((llm["backup"], (llm.get("backup_key") or "").strip()))
    chain = [(p, k) for p, k in chain if k]      # skip providers without a key
    if not chain:
        raise RuntimeError("No API key provided. Add one in the sidebar.")

    errors = {}          # one message per provider, so both get named
    for idx, (provider, key) in enumerate(chain):
        cap = max_out or PROVIDERS[provider]["max_out"]
        cap = min(cap, PROVIDERS[provider]["max_out"])
        for attempt in range(3):
            try:
                text = _one_call(provider, key, system, user, cap)
                if not text.strip():
                    raise RuntimeError("empty response")
                if idx > 0:
                    st.info(f"{PROVIDERS[chain[0][0]]['label']} was "
                            f"unavailable — this was generated with "
                            f"{PROVIDERS[provider]['label']} instead.")
                return text
            except Exception as e:
                errors[provider] = f"{PROVIDERS[provider]['label']}: {e}"
                if not _is_transient(e):
                    break                       # bad key / bad request
                if attempt < 2:
                    time.sleep(6 * (attempt + 1))
    raise RuntimeError(" | ".join(errors.values()))


def synthesize(llm, question, context, counts, mode="consensus"):
    user_msg = (
        f"RESEARCH QUESTION: {question}\n\n"
        f"PRISMA COUNTS — Retrieved: {counts['retrieved']} "
        f"(ERIC {counts['eric']}, OpenAlex {counts['openalex']}, "
        f"Semantic Scholar {counts['s2']}, CrossRef {counts.get('crossref', 0)}); "
        f"Screened after dedup: {counts['screened']}; "
        f"Included with usable abstracts: {counts['included']}.\n\n"
        f"INCLUDED PAPERS (your ONLY evidence base):\n\n{context}\n\n"
        + (LANDSCAPE_INSTRUCTIONS if mode == "landscape"
           else CONSENSUS_INSTRUCTIONS))
    return llm_complete(llm, SYSTEM_PROMPT, user_msg)


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


def synthesize_followup(llm, question, report_title, context, start_n):
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
        f"conclusion. Do not add a References section.")
    return llm_complete(llm, SYSTEM_PROMPT, user_msg,
                        max_out=min(8000, PROVIDERS[llm["provider"]]["max_out"]))


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
        if body and all(re.fullmatch(r":?-+:?", c) for c in body[0]):
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


def stat_strip_html(included, counts):
    years = [p["year"] for p in included if p["year"]]
    span = (f"{min(years)}&ndash;{max(years)}" if len(set(years)) > 1
            else (str(years[0]) if years else "&mdash;"))
    cites = [p["citations"] for p in included if p["citations"] is not None]
    cite_s = f"{sum(cites):,}" if cites else "&mdash;"
    sources = len({p["source"] for p in included})
    stats = [(str(counts["included"]), "Papers analyzed"),
             (span, "Publication span"),
             (cite_s, "Citations tracked"),
             (str(sources), "Databases contributing")]
    cells = "".join(
        f'<div class="stat"><div class="stat-n">{n}</div>'
        f'<div class="stat-l">{l}</div></div>' for n, l in stats)
    return f'<div class="stats">{cells}</div>'


def timeline_html(included):
    years = sorted(p["year"] for p in included if isinstance(p["year"], int))
    if len(set(years)) < 2:
        return ""
    lo, hi = min(years), max(years)
    span = hi - lo
    size = 1 if span <= 14 else 2 if span <= 28 else 5
    lo -= lo % size
    buckets = list(range(lo, hi + 1, size))
    counts_by = {b: 0 for b in buckets}
    for y in years:
        counts_by[lo + ((y - lo) // size) * size] += 1
    mx = max(counts_by.values())
    label_every = 1 if len(buckets) <= 12 else 2
    bars, labels = [], []
    for i, b in enumerate(buckets):
        c = counts_by[b]
        hpx = max(int(96 * c / mx), 4) if c else 0
        cnt = f'<div class="tl-c">{c}</div>' if c else ""
        bar = (f'<div class="tl-bar" style="height:{hpx}px"></div>' if c
               else '<div class="tl-bar tl-zero"></div>')
        bars.append(f'<div class="tl-col">{cnt}{bar}</div>')
        lab = str(b) if size == 1 else f"{b}&ndash;{str(b + size - 1)[-2:]}"
        labels.append('<div class="tl-col"><div class="tl-x">'
                      + (lab if i % label_every == 0 else "&nbsp;")
                      + "</div></div>")
    return ('<div class="tl-wrap"><div class="tl">' + "".join(bars)
            + '</div><div class="tl-axis">' + "".join(labels) + "</div></div>")


def _figcap(n, text):
    return f'<div class="figcap"><b>Figure {n}</b>&ensp;{text}</div>'


def decorate_report_html(h, counts, included, fig_start=1):
    """Deterministic per-section visuals: numbered section markers, an intro
    stat strip, PRISMA inside Methods, a publication timeline in Results,
    figure captions on the claim matrix and coverage heatmap, question cards,
    and a numbered reference grid. Pure post-processing — identical output
    for identical inputs, never dependent on the LLM."""
    # numbered section headings -> markers; unnumbered h2s get the rule too
    h = re.sub(r"<h2>(\d)\.\s*(.*?)</h2>",
               lambda m: (f'<h2 class="sec"><span class="sec-n">0{m.group(1)}'
                          f'</span><span>{m.group(2)}</span></h2>'), h)
    h = re.sub(r"<h2>(?!<)(.*?)</h2>",
               r'<h2 class="sec sec-plain"><span>\1</span></h2>', h)

    fig = fig_start
    lit = bool(included) and bool(counts)   # paper-corpus visuals only apply
    # Introduction: stat strip
    if lit:
        m = re.search(r'<h2 class="sec"><span class="sec-n">01</span><span>[^<]*</span></h2>', h)
        if m:
            h = h[:m.end()] + stat_strip_html(included, counts) + h[m.end():]
    # Methods: PRISMA flow + caption
    m = (re.search(r'<h2 class="sec"><span class="sec-n">02</span><span>[^<]*</span></h2>', h)
         if lit else None)
    if m:
        block = prisma_html(counts) + _figcap(fig, "Search screening and inclusion flow")
        h = h[:m.end()] + block + h[m.end():]
        fig += 1
    # Results: timeline chart (after the Timeline heading, else before Discussion)
    tl = timeline_html(included) if lit else ""
    if tl:
        block = tl + _figcap(fig, "Included papers by publication year")
        m = re.search(r"<h3>Timeline and Venues</h3>", h)
        if m:
            h = h[:m.end()] + block + h[m.end():]
            fig += 1
        else:
            m = (re.search(r'<h2 class="sec"><span class="sec-n">04</span>', h)
                 or re.search(r'<h2 class="sec sec-plain"><span>References</span></h2>', h))
            if m:
                h = h[:m.start()] + block + h[m.start():]
                fig += 1
    # Discussion: caption under the claim/finding matrix
    m = re.search(r'(<h2 class="sec"><span class="sec-n">0[34]</span>.*?</table>)', h, re.S)
    if m:
        cap = ("Claims and the strength of evidence behind them" if lit
               else "Patterns identified in the retrieved observations")
        h = h[:m.end()] + _figcap(fig, cap) + h[m.end():]
        fig += 1
    # Conclusion: caption under the coverage heatmap
    m = re.search(r"(<h3>Research Gaps</h3>.*?</table>)", h, re.S)
    if m:
        h = h[:m.end()] + _figcap(fig, "Evidence coverage by theme") + h[m.end():]
        fig += 1
    # Open Research Questions table -> numbered cards
    def _oq_cards(mt):
        rows = re.findall(r"<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>", mt.group(0), re.S)
        cards = "".join(
            f'<div class="oq"><div class="oq-n">{i:02d}</div><div>'
            f'<div class="oq-q">{q}</div><div class="oq-why">{w}</div></div></div>'
            for i, (q, w) in enumerate(rows, 1))
        return cards or mt.group(0)
    h = re.sub(r"<table>\s*<thead>\s*<tr>\s*<th>Question</th>\s*"
               r"<th>Why It Matters</th>\s*</tr>\s*</thead>.*?</table>",
               _oq_cards, h, flags=re.S)
    # References paragraphs -> numbered grid rows
    h = re.sub(r"<p><strong>\[(\d+)\]</strong>\s*(.*?)</p>",
               r'<div class="ref"><span class="ref-n">[\1]</span>'
               r'<span class="ref-b">\2</span></div>', h, flags=re.S)
    return h


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

/* Section markers */
.report-doc h2.sec {{ display:flex; align-items:baseline; gap:14px;
  border-top:1px solid {LINE}; padding-top:20px; margin-top:2.9em; }}
.report-doc h2.sec:first-child {{ border-top:none; padding-top:0; margin-top:.6em; }}
.report-doc .sec-n {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  font-size:.78rem; letter-spacing:.08em; color:{MUTED}; }}

/* Intro stat strip */
.stats {{ display:flex; flex-wrap:wrap; border:1px solid {LINE};
  border-radius:12px; background:#FCFBF8; margin:16px 0 6px; overflow:hidden; }}
.stat {{ flex:1; min-width:130px; padding:14px 18px;
  border-left:1px solid {RULE}; }}
.stat:first-child {{ border-left:none; }}
.stat-n {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  font-size:1.25rem; color:{INK}; }}
.stat-l {{ font-size:.72rem; color:{MUTED}; margin-top:2px; }}

/* Figure captions */
.figcap {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem;
  letter-spacing:.05em; color:{MUTED}; margin:8px 0 26px; }}
.figcap b {{ color:{INK}; font-weight:600; }}

/* Publication timeline */
.tl-wrap {{ margin:16px 0 4px; }}
.tl {{ display:flex; align-items:flex-end; gap:5px; height:112px;
  border-bottom:1px solid {LINE}; padding:0 2px; }}
.tl-col {{ flex:1; display:flex; flex-direction:column; align-items:center;
  justify-content:flex-end; gap:4px; min-width:0; }}
.tl-bar {{ width:68%; max-width:30px; background:{GREEN};
  border-radius:3px 3px 0 0; }}
.tl-zero {{ height:0; }}
.tl-c {{ font-family:'IBM Plex Mono',monospace; font-size:.64rem; color:{MUTED}; }}
.tl-axis {{ display:flex; gap:5px; padding:0 2px; }}
.tl-x {{ font-family:'IBM Plex Mono',monospace; font-size:.62rem;
  color:{MUTED}; margin-top:5px; text-align:center; }}

/* Open-question cards */
.oq {{ display:flex; gap:16px; border:1px solid {LINE}; border-radius:12px;
  background:#FCFBF8; padding:15px 18px; margin:10px 0; }}
.oq-n {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  font-size:.78rem; color:{MUTED}; padding-top:2px; }}
.oq-q {{ font-weight:600; color:{INK}; }}
.oq-why {{ font-size:.85rem; color:{MUTED}; margin-top:3px; line-height:1.55; }}

/* Reference grid */
.ref {{ display:flex; gap:14px; padding:9px 0; border-bottom:1px solid {RULE};
  font-size:.85rem; line-height:1.55; }}
.ref-n {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  color:{MUTED}; min-width:36px; }}

/* Data Explorer visuals */
.chart {{ width:100%; height:auto; display:block; margin:14px 0 4px; }}
.chart .ax {{ font-family:'IBM Plex Mono',monospace; font-size:10px;
  fill:{MUTED}; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:2px 0 4px; }}
.lg {{ font-size:.78rem; color:{MUTED}; display:inline-flex;
  align-items:center; }}
.lg i {{ width:9px; height:9px; border-radius:50%; margin-right:6px;
  display:inline-block; }}
.bars {{ display:flex; flex-direction:column; gap:7px; margin:14px 0 4px; }}
.brow {{ display:flex; align-items:center; gap:12px; font-size:.85rem; }}
.blab {{ flex:0 0 176px; color:{INK}; }}
.btrack {{ flex:1; background:{RULE}; border-radius:4px; height:16px;
  overflow:hidden; }}
.bfill {{ display:block; height:100%; background:{GREEN};
  border-radius:4px 0 0 4px; }}
.bval {{ flex:0 0 92px; text-align:right;
  font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:{INK}; }}
.bmore {{ font-size:.76rem; color:{MUTED}; margin-top:8px; }}
@media (max-width:640px) {{ .blab {{ flex-basis:110px; }}
  .bval {{ flex-basis:70px; }} }}
"""


def build_html_export(title, question, hero_html, report_html, counts,
                      brand="Deep Search Report &middot; ERIC &middot; OpenAlex &middot; Semantic Scholar &middot; CrossRef", subtitle=None):
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
<div class="brand">{brand}</div>
<h1 class="headline">{t}</h1>
<div class="subtitle">{subtitle or (q + " &middot; Generated " + date.today().strftime("%B %d, %Y") + " &middot; " + str(counts["included"]) + " papers synthesized from " + str(counts["retrieved"]) + " retrieved records")}</div>
{hero_html}
<div class="report-doc">
{report_html}
</div>
<div class="footer">Generated with the Education Research Assistant.
Every figure and citation is grounded in the retrieved records shown above.</div>
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
# Data Explorer — curated catalog + fetchers for World Bank, Census, NCES
# ==========================================================================

FIPS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "District of Columbia", "12": "Florida",
    "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana",
    "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
    "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
    "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming", "72": "Puerto Rico",
}

COUNTRY_SETS = {
    "United States only": "USA",
    "US + comparison peers": "USA;GBR;CAN;AUS;DEU;FIN;JPN;KOR",
    "G7": "USA;GBR;CAN;FRA;DEU;ITA;JPN",
    "World aggregate": "WLD",
    "US vs. World & OECD": "USA;WLD;OED",
}

# Every entry maps to a verified API call pattern. The model may only choose
# from this list — it never writes indicator codes, so it cannot invent one.
DATA_CATALOG = [
    # --- World Bank (free, no key) ---------------------------------------
    {"id": "wb_edu_gdp", "src": "World Bank", "kind": "wb",
     "code": "SE.XPD.TOTL.GD.ZS", "unit": "% of GDP",
     "label": "Government expenditure on education (% of GDP)"},
    {"id": "wb_edu_govt", "src": "World Bank", "kind": "wb",
     "code": "SE.XPD.TOTL.GB.ZS", "unit": "% of gov. spending",
     "label": "Education spending (% of government expenditure)"},
    {"id": "wb_ter_enr", "src": "World Bank", "kind": "wb",
     "code": "SE.TER.ENRR", "unit": "% gross",
     "label": "Tertiary school enrollment (% gross)"},
    {"id": "wb_sec_enr", "src": "World Bank", "kind": "wb",
     "code": "SE.SEC.ENRR", "unit": "% gross",
     "label": "Secondary school enrollment (% gross)"},
    {"id": "wb_prm_enr", "src": "World Bank", "kind": "wb",
     "code": "SE.PRM.ENRR", "unit": "% gross",
     "label": "Primary school enrollment (% gross)"},
    {"id": "wb_pupil_teacher", "src": "World Bank", "kind": "wb",
     "code": "SE.PRM.ENRL.TC.ZS", "unit": "pupils per teacher",
     "label": "Pupil-teacher ratio, primary"},
    {"id": "wb_prm_compl", "src": "World Bank", "kind": "wb",
     "code": "SE.PRM.CMPT.ZS", "unit": "% of relevant age group",
     "label": "Primary completion rate"},
    {"id": "wb_literacy", "src": "World Bank", "kind": "wb",
     "code": "SE.ADT.LITR.ZS", "unit": "% of people 15+",
     "label": "Adult literacy rate"},
    {"id": "wb_youth_neet", "src": "World Bank", "kind": "wb",
     "code": "SL.UEM.NEET.ZS", "unit": "% of youth",
     "label": "Youth not in education, employment or training (NEET)"},
    {"id": "wb_gdp_pc", "src": "World Bank", "kind": "wb",
     "code": "NY.GDP.PCAP.CD", "unit": "current US$",
     "label": "GDP per capita (context indicator)"},

    # --- U.S. Census ACS 5-year (free, keyless for moderate use) ----------
    {"id": "cs_ba", "src": "U.S. Census", "kind": "census_pct",
     "num": "B15003_022E", "den": "B15003_001E", "unit": "% of adults 25+",
     "label": "Adults 25+ with a bachelor's degree"},
    {"id": "cs_grad", "src": "U.S. Census", "kind": "census_pct",
     "num": "B15003_023E", "den": "B15003_001E", "unit": "% of adults 25+",
     "label": "Adults 25+ with a master's degree"},
    {"id": "cs_hs", "src": "U.S. Census", "kind": "census_pct",
     "num": "B15003_017E", "den": "B15003_001E", "unit": "% of adults 25+",
     "label": "Adults 25+ with a high school diploma"},
    {"id": "cs_income", "src": "U.S. Census", "kind": "census_val",
     "num": "B19013_001E", "unit": "US$",
     "label": "Median household income"},
    {"id": "cs_poverty", "src": "U.S. Census", "kind": "census_pct",
     "num": "B17001_002E", "den": "B17001_001E", "unit": "% of population",
     "label": "Population below the poverty line"},
    {"id": "cs_enrolled", "src": "U.S. Census", "kind": "census_pct",
     "num": "B14001_002E", "den": "B14001_001E", "unit": "% of population 3+",
     "label": "Population enrolled in school"},

    # --- NCES via Urban Institute summary endpoints (free, no key) -------
    {"id": "nces_enroll", "src": "NCES", "kind": "nces",
     "section": "schools", "source": "ccd", "topic": "enrollment",
     "var": "enrollment", "stat": "sum", "unit": "students",
     "label": "Public school enrollment (CCD)"},
    {"id": "nces_teachers", "src": "NCES", "kind": "nces",
     "section": "schools", "source": "ccd", "topic": "directory",
     "var": "teachers_fte", "stat": "sum", "unit": "FTE teachers",
     "label": "Public school teachers, full-time equivalent (CCD)"},
    {"id": "nces_frpl", "src": "NCES", "kind": "nces",
     "section": "schools", "source": "ccd", "topic": "directory",
     "var": "free_or_reduced_price_lunch", "stat": "sum", "unit": "students",
     "label": "Students eligible for free/reduced-price lunch (CCD)"},
    {"id": "nces_dist_rev", "src": "NCES", "kind": "nces",
     "section": "school-districts", "source": "ccd", "topic": "finance",
     "var": "rev_total", "stat": "sum", "unit": "US$",
     "label": "School district total revenue (CCD finance)"},
]

CATALOG_BY_ID = {d["id"]: d for d in DATA_CATALOG}


def _num(x):
    try:
        v = float(x)
        return None if v <= -666666666 else v      # Census null sentinels
    except (TypeError, ValueError):
        return None


def fetch_worldbank(entry, countries, y0, y1):
    rows = []
    try:
        r = requests.get(
            f"https://api.worldbank.org/v2/country/{countries}/indicator/{entry['code']}",
            params={"format": "json", "per_page": 2000, "date": f"{y0}:{y1}"},
            headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            st.warning("World Bank returned no observations for that "
                       "indicator and period.")
            return rows
        for o in payload[1]:
            v = _num(o.get("value"))
            if v is None:
                continue
            rows.append({"entity": (o.get("country") or {}).get("value", "?"),
                         "year": int(o["date"]), "value": v})
    except Exception as e:
        st.warning(f"World Bank didn't respond ({e}).")
    return rows


def fetch_census(entry, year, census_key=""):
    rows, get = [], entry["num"] + (
        "," + entry["den"] if entry.get("den") else "")
    params = {"get": "NAME," + get, "for": "state:*"}
    if census_key:
        params["key"] = census_key
    try:
        r = requests.get(f"https://api.census.gov/data/{year}/acs/acs5",
                         params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        table = r.json()
        cols = table[0]
        for rec in table[1:]:
            d = dict(zip(cols, rec))
            num = _num(d.get(entry["num"]))
            if num is None:
                continue
            if entry.get("den"):
                den = _num(d.get(entry["den"]))
                if not den:
                    continue
                val = num / den * 100
            else:
                val = num
            rows.append({"entity": d.get("NAME", "?"), "year": int(year),
                         "value": val})
    except Exception as e:
        st.warning(f"Census didn't respond ({e}). ACS 5-year data for "
                   f"{year} may not be published yet — try an earlier year.")
    return rows


def fetch_nces(entry, y0, y1):
    """Urban Institute summary endpoints aggregate NCES data by state-year."""
    rows = []
    url = (f"https://educationdata.urban.org/api/v1/{entry['section']}/"
           f"{entry['source']}/{entry['topic']}/summaries")
    params = {"var": entry["var"], "stat": entry["stat"], "by": "fips",
              "year": ",".join(str(y) for y in range(y0, y1 + 1))}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=45)
        r.raise_for_status()
        payload = r.json()
        recs = payload.get("results", payload) if isinstance(payload, dict) else payload
        for o in recs or []:
            v = _num(o.get(entry["var"]))
            fips = str(o.get("fips", "")).zfill(2)
            if v is None or fips not in FIPS:
                continue
            rows.append({"entity": FIPS[fips], "year": int(o.get("year", y1)),
                         "value": v})
    except Exception as e:
        st.warning(f"NCES (Urban Institute) didn't respond ({e}).")
    return rows


def load_dataset(entry, opts):
    if entry["kind"] == "wb":
        rows = fetch_worldbank(entry, opts["countries"], opts["y0"], opts["y1"])
        note = ("World Bank Open Data, indicator "
                f"{entry['code']}, {opts['y0']}\u2013{opts['y1']}")
    elif entry["kind"].startswith("census"):
        rows = fetch_census(entry, opts["census_year"], opts.get("census_key", ""))
        note = (f"U.S. Census ACS 5-year {opts['census_year']}, "
                f"variable {entry['num']}"
                + (f" over {entry['den']}" if entry.get("den") else "")
                + ", all states")
    else:
        rows = fetch_nces(entry, opts["y0"], opts["y1"])
        note = (f"NCES {entry['source'].upper()} via the Urban Institute "
                f"Education Data Portal, {entry['var']} "
                f"({entry['stat']}) by state, {opts['y0']}\u2013{opts['y1']}")
    years = {r["year"] for r in rows}
    return {"entry": entry, "rows": rows, "note": note,
            "shape": "series" if len(years) > 1 else "cross"}


# ---- Deterministic data visuals -----------------------------------------

SERIES_COLORS = [GREEN, "#2456A6", AMBER, "#6B4FA8", RED, "#3C7D8C",
                 "#8A6D3B", "#4F6B2F"]


def _fmt(v, unit=""):
    a = abs(v)
    if a >= 1_000_000_000:
        s = f"{v/1_000_000_000:,.1f}B"
    elif a >= 1_000_000:
        s = f"{v/1_000_000:,.1f}M"
    elif a >= 10_000:
        s = f"{v:,.0f}"
    elif a >= 100:
        s = f"{v:,.1f}"
    else:
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
    return s + ("%" if unit.startswith("%") else "")


def data_stats_html(ds):
    rows, unit = ds["rows"], ds["entry"]["unit"]
    if not rows:
        return ""
    vals = [r["value"] for r in rows]
    years = sorted({r["year"] for r in rows})
    latest = [r for r in rows if r["year"] == years[-1]]
    hi = max(latest, key=lambda r: r["value"])
    lo = min(latest, key=lambda r: r["value"])
    avg = sum(r["value"] for r in latest) / len(latest)
    stats = [(str(len({r["entity"] for r in rows})), "Entities"),
             (f"{years[0]}&ndash;{years[-1]}" if len(years) > 1 else str(years[0]),
              "Period covered"),
             (_fmt(avg, unit), f"Mean, {years[-1]}"),
             (_fmt(hi["value"], unit), f"Highest: {html_lib.escape(hi['entity'])[:18]}"),
             (_fmt(lo["value"], unit), f"Lowest: {html_lib.escape(lo['entity'])[:18]}")]
    cells = "".join(f'<div class="stat"><div class="stat-n">{n}</div>'
                    f'<div class="stat-l">{l}</div></div>' for n, l in stats)
    return f'<div class="stats">{cells}</div>'


def series_chart_html(ds, max_entities=8):
    """Multi-series line chart as inline SVG — no JS, prints cleanly."""
    rows = ds["rows"]
    years = sorted({r["year"] for r in rows})
    if len(years) < 2:
        return ""
    totals = {}
    for r in rows:
        totals.setdefault(r["entity"], []).append(r["value"])
    ents = sorted(totals, key=lambda e: -max(totals[e]))[:max_entities]
    vals = [r["value"] for r in rows if r["entity"] in ents]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1
    pad = (vmax - vmin) * 0.08
    vmin, vmax = vmin - pad, vmax + pad
    W, H, L, R, T, B = 720, 260, 62, 12, 12, 34
    px = lambda y: L + (W - L - R) * (years.index(y) / max(len(years) - 1, 1))
    py = lambda v: T + (H - T - B) * (1 - (v - vmin) / (vmax - vmin))
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" '
             f'xmlns="http://www.w3.org/2000/svg" role="img">']
    for k in range(5):                                   # gridlines + y labels
        v = vmin + (vmax - vmin) * k / 4
        y = py(v)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" '
                     f'stroke="{RULE}" stroke-width="1"/>')
        parts.append(f'<text x="{L-8}" y="{y+3.5:.1f}" text-anchor="end" '
                     f'class="ax">{_fmt(v, ds["entry"]["unit"])}</text>')
    step = max(1, len(years) // 8)
    for i, yr in enumerate(years):                       # x labels
        if i % step == 0 or i == len(years) - 1:
            parts.append(f'<text x="{px(yr):.1f}" y="{H-12}" '
                         f'text-anchor="middle" class="ax">{yr}</text>')
    for i, e in enumerate(ents):                         # one path per entity
        pts = sorted((r for r in rows if r["entity"] == e),
                     key=lambda r: r["year"])
        if len(pts) < 2:
            continue
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        d = " ".join(f'{"M" if j == 0 else "L"}{px(p["year"]):.1f},'
                     f'{py(p["value"]):.1f}' for j, p in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" '
                     f'stroke-width="2.2" stroke-linejoin="round"/>')
        last = pts[-1]
        parts.append(f'<circle cx="{px(last["year"]):.1f}" '
                     f'cy="{py(last["value"]):.1f}" r="3" fill="{c}"/>')
    parts.append("</svg>")
    legend = "".join(
        f'<span class="lg"><i style="background:'
        f'{SERIES_COLORS[i % len(SERIES_COLORS)]}"></i>'
        f'{html_lib.escape(e)}</span>' for i, e in enumerate(ents))
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def bar_chart_html(ds, top_n=15):
    """Ranked horizontal bars for a single-year cross-section."""
    rows = ds["rows"]
    years = sorted({r["year"] for r in rows})
    if not rows:
        return ""
    latest = [r for r in rows if r["year"] == years[-1]]
    latest.sort(key=lambda r: -r["value"])
    shown = latest[:top_n]
    mx = max(r["value"] for r in shown) or 1
    unit = ds["entry"]["unit"]
    bars = "".join(
        f'<div class="brow"><span class="blab">{html_lib.escape(r["entity"])[:26]}</span>'
        f'<span class="btrack"><span class="bfill" style="width:'
        f'{max(r["value"] / mx * 100, 0.6):.1f}%"></span></span>'
        f'<span class="bval">{_fmt(r["value"], unit)}</span></div>'
        for r in shown)
    more = (f'<div class="bmore">Showing the top {len(shown)} of '
            f'{len(latest)} entities for {years[-1]}.</div>'
            if len(latest) > len(shown) else "")
    return f'<div class="bars">{bars}</div>{more}'


def data_table_html(ds, limit=60):
    rows = sorted(ds["rows"], key=lambda r: (-r["year"], -r["value"]))[:limit]
    unit = ds["entry"]["unit"]
    body = "".join(
        f'<tr><td>{html_lib.escape(r["entity"])}</td><td>{r["year"]}</td>'
        f'<td>{_fmt(r["value"], unit)}</td></tr>' for r in rows)
    cap = (f'<div class="bmore">Showing {len(rows)} of {len(ds["rows"])} '
           f'observations. The CSV download contains all of them.</div>'
           if len(ds["rows"]) > len(rows) else "")
    return (f'<table><thead><tr><th>Entity</th><th>Year</th>'
            f'<th>Value ({html_lib.escape(unit)})</th></tr></thead>'
            f'<tbody>{body}</tbody></table>{cap}')


def make_csv(ds):
    out = ["entity,year,value,indicator,unit,source"]
    lab = ds["entry"]["label"].replace('"', "'")
    for r in sorted(ds["rows"], key=lambda r: (r["entity"], r["year"])):
        out.append(f'"{r["entity"]}",{r["year"]},{r["value"]},'
                   f'"{lab}","{ds["entry"]["unit"]}","{ds["entry"]["src"]}"')
    return "\n".join(out)


DATA_SYSTEM_PROMPT = """You are a careful education-data analyst.

GROUNDING RULES (non-negotiable):
- Use ONLY the observations supplied in the user message. Never introduce
  outside figures, recalled statistics, or events not visible in the data.
- Every number you state must appear in, or be arithmetic on, the supplied
  observations. Round sensibly and say which year each figure refers to.
- Describe patterns, not causes. These are observational aggregates: say
  "is associated with" rather than "causes", and name confounders you cannot
  rule out with this data alone.
- If the data cannot answer part of the question, say so plainly."""

DATA_INSTRUCTIONS = """Write a raw-Markdown analysis — no code fences, no
preamble, no extra sections — with EXACTLY this structure:

Line 1:
REPORT_TITLE: <one declarative sentence stating the main pattern, max 16 words>

## 1. Overview
2-3 paragraphs: what the indicator measures, the headline pattern across
entities and years, and the overall magnitude of variation.

## 2. Data & Method
One paragraph naming the source, indicator, geography, period, and the number
of observations retrieved, plus any coverage gaps you can see in the data.

## 3. Key Patterns
2-3 short paragraphs on the most notable movements, leaders, and laggards.
Then a Markdown table with EXACTLY these columns:
Pattern | What The Data Show | Confidence
Confidence must be EXACTLY one word: Strong, Moderate, or Weak — judged by how
many observations and years support the pattern.

## 4. Caveats
One paragraph on what this data cannot establish: definitional differences,
missing years or entities, and why these aggregates cannot support causal
claims.

## 5. Takeaways
3-4 bullet points, each stating one grounded finding with its figure and year."""


def analyze_data(llm, question, ds):
    rows = sorted(ds["rows"], key=lambda r: (r["entity"], r["year"]))
    lim = limits_for(llm["provider"])
    budget = 120 if lim["max_out"] <= 4000 else 400
    if len(rows) > budget:                 # keep every entity's endpoints
        by_ent = {}
        for r in rows:
            by_ent.setdefault(r["entity"], []).append(r)
        rows, per = [], max(2, budget // max(len(by_ent), 1))
        for e, rs in by_ent.items():
            rows += rs[:1] + rs[-(per - 1):] if len(rs) > per else rs
    lines = "\n".join(f"{r['entity']} | {r['year']} | {r['value']:.4g}"
                      for r in rows)
    user_msg = (
        f"QUESTION: {question}\n\n"
        f"INDICATOR: {ds['entry']['label']} (unit: {ds['entry']['unit']})\n"
        f"SOURCE: {ds['note']}\n"
        f"TOTAL OBSERVATIONS RETRIEVED: {len(ds['rows'])}\n\n"
        f"OBSERVATIONS (entity | year | value) — your ONLY evidence:\n{lines}\n\n"
        + DATA_INSTRUCTIONS)
    return llm_complete(llm, DATA_SYSTEM_PROMPT, user_msg)


def pick_indicator(llm, question):
    """Constrained selection: the model must return one catalog id, so it can
    never invent an indicator code. Falls back to keyword matching."""
    menu = "\n".join(f"{d['id']}: {d['label']} ({d['src']})"
                     for d in DATA_CATALOG)
    try:
        out = llm_complete(
            llm,
            "You map a user's data question to exactly one dataset id from a "
            "fixed list. Reply with the id only — no punctuation, no "
            "explanation. If nothing fits well, reply with the closest id.",
            f"QUESTION: {question}\n\nAVAILABLE DATASET IDS:\n{menu}\n\n"
            f"Reply with exactly one id from the list above.",
            max_out=24).strip().split()[0].strip(".,:`'\"")
        if out in CATALOG_BY_ID:
            return CATALOG_BY_ID[out], None
    except Exception as e:
        pass
    words = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
    best, score = None, 0
    for d in DATA_CATALOG:
        s = len(words & set(re.findall(r"[a-z]{4,}", d["label"].lower())))
        if s > score:
            best, score = d, s
    return (best or DATA_CATALOG[0]), "keyword match"

# ==========================================================================
# Streamlit UI
# ==========================================================================

st.set_page_config(page_title="Education Research Assistant",
                   page_icon="📖", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

LIT_EXAMPLES = [
    "Does retrieval practice improve K-12 science learning?",
    "What barriers do first-generation students face?",
    "Does one-to-one device access improve literacy outcomes?",
]
DATA_EXAMPLES = [
    "How has US education spending changed versus peer countries?",
    "Which states have the highest share of adults with a bachelor's degree?",
    "How has public school enrollment shifted across states?",
]


def _set_lit(text):
    st.session_state.q_input = text


def _set_data(text):
    st.session_state.dq_input = text


# ---- Sidebar: model provider + keys --------------------------------------
with st.sidebar:
    st.markdown("#### Model")
    _labels = {k: f"{PROVIDERS[k]['label']} · {PROVIDERS[k]['cost']}"
               for k in PROVIDER_ORDER}
    provider = st.selectbox(
        "Model provider", PROVIDER_ORDER, index=0,
        format_func=lambda k: _labels[k], label_visibility="collapsed")
    _p = PROVIDERS[provider]
    api_key = st.text_input(_p["key_label"], type="password",
                            value=os.environ.get(_p["env"], ""))
    st.caption(_p["note"])
    st.markdown(f"[Get a key]({_p['url']})")

    with st.expander("Backup model (recommended)", expanded=False):
        st.caption("Free tiers get busy. If the primary is overloaded or rate "
                   "limited, the app retries and then automatically uses this "
                   "backup instead — you don't have to do anything.")
        _b_opts = ["None"] + [k for k in PROVIDER_ORDER if k != provider]
        backup = st.selectbox(
            "Backup provider", _b_opts,
            format_func=lambda k: "None" if k == "None" else _labels[k])
        backup_key = ""
        if backup != "None":
            _bp = PROVIDERS[backup]
            backup_key = st.text_input(_bp["key_label"], type="password",
                                       value=os.environ.get(_bp["env"], ""),
                                       key="backup_key_input")
            st.caption(_bp["note"])
            st.markdown(f"[Get a key]({_bp['url']})")

    llm = {"provider": provider, "key": api_key,
           "backup": None if backup == "None" else backup,
           "backup_key": backup_key}

    st.divider()
    st.markdown("#### Optional keys")
    s2_key = st.text_input(
        "Semantic Scholar key", type="password",
        value=os.environ.get("S2_API_KEY", ""),
        help="Anonymous Semantic Scholar requests share one rate limit and "
             "often fail. A free key makes that source reliable.")
    census_key = st.text_input(
        "U.S. Census key", type="password",
        value=os.environ.get("CENSUS_API_KEY", ""),
        help="Optional. Census works without a key for moderate use.")

    st.divider()
    st.markdown("#### About")
    st.markdown(
        "**Literature review** searches ERIC, OpenAlex, Semantic Scholar, and "
        "CrossRef, then synthesizes a grounded evidence report.\n\n"
        "**Data explorer** pulls real statistics from the World Bank, U.S. "
        "Census, and NCES, then analyzes them. In both modes the model may "
        "only use what was retrieved.")

# ---- Header + mode switch ------------------------------------------------
st.markdown(
    '<div class="app-eyebrow">ERIC &middot; OpenAlex &middot; Semantic Scholar '
    '&middot; CrossRef &middot; World Bank &middot; Census &middot; NCES</div>'
    '<p class="app-title">Education Research Assistant</p>'
    '<p class="app-sub">Synthesize the literature, or analyze real statistics '
    '\u2014 both grounded strictly in what the APIs return.</p>',
    unsafe_allow_html=True)
st.write("")

app_mode = st.radio(
    "Mode", ["📚 Literature review", "📊 Data explorer"],
    horizontal=True, label_visibility="collapsed")
st.write("")

# ==========================================================================
# MODE 1 — Literature review
# ==========================================================================
if app_mode.endswith("Literature review"):
    question = st.text_area(
        "Research question", key="q_input", height=100,
        placeholder="e.g., Does retrieval practice improve K-12 science learning?",
        label_visibility="collapsed")

    ec1, ec2 = st.columns([3, 1])
    with ec1:
        bcols = st.columns(len(LIT_EXAMPLES))
        for col, ex in zip(bcols, LIT_EXAMPLES):
            with col:
                st.button(ex.split("?")[0][:36] + "…", key=f"ex_{ex[:12]}",
                          on_click=_set_lit, args=(ex,),
                          use_container_width=True, help=ex)
    with ec2:
        run = st.button("Run deep search", type="primary",
                        use_container_width=True)

    if run:
        if not question.strip():
            st.error("Type a research question first — or tap one of the examples.")
            st.stop()
        if not api_key:
            st.error("Add your API key in the sidebar to run the synthesis step.")
            st.stop()

        q = question.strip()
        lim = limits_for(provider)
        with st.status("Running deep search…", expanded=True) as status:
            st.write("Searching ERIC…")
            eric = fetch_eric(q)
            st.write(f"ERIC returned {len(eric)}. Searching OpenAlex…")
            oa = fetch_openalex(q)
            st.write(f"OpenAlex returned {len(oa)}. Searching Semantic Scholar…")
            s2 = fetch_semantic_scholar(q, s2_key)
            st.write(f"Semantic Scholar returned {len(s2)}. Searching CrossRef…")
            cr = fetch_crossref(q)
            st.write(f"CrossRef returned {len(cr)}. Screening and deduplicating…")

            all_papers = eric + oa + s2 + cr
            screened, included = dedupe_and_screen(
                all_papers, max_included=lim["max_papers"])
            counts = {"retrieved": len(all_papers), "eric": len(eric),
                      "openalex": len(oa), "s2": len(s2), "crossref": len(cr),
                      "screened": len(screened), "included": len(included)}

            if not included:
                status.update(label="No usable papers found", state="error")
                st.error("None of the retrieved records had usable abstracts. "
                         "Try broader wording — for example, drop grade levels "
                         "or specific program names.")
                st.stop()

            mode = question_mode(q)
            st.write(f"{counts['included']} papers included. Synthesizing "
                     f"with {PROVIDERS[provider]['label']} "
                     f"({'findings landscape' if mode == 'landscape' else 'consensus'} "
                     f"report)…")
            try:
                raw = clean_llm_output(synthesize(
                    llm, q,
                    build_context(included, abstract_chars=lim["abstract_chars"]),
                    counts, mode))
            except Exception as e:
                status.update(label="Synthesis failed", state="error")
                st.error(f"Synthesis failed: {e}")
                st.stop()
            status.update(label="Report ready", state="complete", expanded=False)

        meter, body = parse_meter(raw)
        findings, body = parse_findings(body)
        title, body = parse_title(body, q)
        for k in [k for k in st.session_state if str(k).startswith("sel_")]:
            del st.session_state[k]
        st.session_state.result = {
            "question": q, "title": title, "meter": meter, "findings": findings,
            "mode": mode, "body": body, "counts": counts, "included": included,
        }

    if "result" not in st.session_state:
        st.markdown(
            '<div class="how-row">'
            '<div class="how-card"><div class="how-step">Search</div>'
            '<div class="how-title">Four free databases</div>'
            '<div class="how-body">ERIC, OpenAlex, Semantic Scholar, and '
            'CrossRef — up to 60 records per search.</div></div>'
            '<div class="how-card"><div class="how-step">Screen</div>'
            '<div class="how-title">PRISMA-style screening</div>'
            '<div class="how-body">Duplicates merged, papers without usable '
            'abstracts excluded, ranked by citation count.</div></div>'
            '<div class="how-card"><div class="how-step">Synthesize</div>'
            '<div class="how-title">Grounded report</div>'
            '<div class="how-body">A five-section report where every claim '
            'cites a retrieved paper. Export to BibTeX or print-ready HTML.'
            '</div></div></div>', unsafe_allow_html=True)

    if "result" in st.session_state:
        r = st.session_state.result
        meter, counts = r["meter"], r["counts"]
        full_body = r["body"] + "\n\n" + make_references_md(r["included"])
        report_html = decorate_report_html(
            render_report_html(full_body), counts, r["included"])

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

        hero = (findings_hero_html(r["findings"])
                if r.get("mode") == "landscape" and r.get("findings")
                else verdict_html(meter))
        html_out = build_html_export(r["title"], r["question"], hero,
                                     report_html, counts)
        dl1, dl2, _sp = st.columns([1, 1, 1])
        with dl1:
            st.download_button("Download report (HTML)", html_out,
                               file_name="deep_search_report.html",
                               mime="text/html", use_container_width=True,
                               type="primary")
        with dl2:
            st.download_button("Download bibliography (.bib)",
                               make_bibtex(r["included"]),
                               file_name="literature_review.bib",
                               mime="text/plain", use_container_width=True)
        st.caption("Open the HTML report in a browser and press Ctrl+P / Cmd+P "
                   "for a clean PDF. To pick which papers go to Zotero, use "
                   "the Zotero export tab.")

        if "## 5" not in r["body"]:
            st.warning("This report looks shorter than expected — it may have "
                       "been truncated. Running the search again usually fixes it.")

        tab_report, tab_papers, tab_export = st.tabs(
            ["Report", f"Included papers ({counts['included']})", "Zotero export"])

        with tab_report:
            st.markdown(f'<div class="report-doc">{report_html}</div>',
                        unsafe_allow_html=True)

        with tab_papers:
            st.markdown("These are the only records the synthesis was allowed "
                        "to cite. Bracketed numbers in the report refer to "
                        "this list.")
            st.markdown(papers_html(r["included"]), unsafe_allow_html=True)

        with tab_export:
            st.markdown("Choose which papers to include, then download the "
                        ".bib file and import it into Zotero via "
                        "**File → Import**.")

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

        # ---- Follow-up questions: extend the report, never replace it ----
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
                st.error("Add your API key in the sidebar to run follow-ups.")
            else:
                lim = limits_for(provider)
                with st.status(f"Following up: {fu_q[:60]}…", expanded=True) as fst:
                    st.write("Searching ERIC…")
                    f_eric = fetch_eric(fu_q)
                    st.write(f"ERIC returned {len(f_eric)}. Searching OpenAlex…")
                    f_oa = fetch_openalex(fu_q)
                    st.write(f"OpenAlex returned {len(f_oa)}. Searching Semantic Scholar…")
                    f_s2 = fetch_semantic_scholar(fu_q, s2_key)
                    st.write(f"Semantic Scholar returned {len(f_s2)}. Searching CrossRef…")
                    f_cr = fetch_crossref(fu_q)
                    st.write(f"CrossRef returned {len(f_cr)}. Screening…")

                    prior_keys = set()
                    for p in r["included"]:
                        prior_keys.add(re.sub(r"\W+", "", p["title"].lower()))
                        if p["doi"]:
                            prior_keys.add(p["doi"].lower())
                    f_all = f_eric + f_oa + f_s2 + f_cr
                    f_screened, f_inc = dedupe_and_screen(
                        f_all, max_included=min(12, lim["max_papers"]),
                        prior_keys=prior_keys)

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
                                llm, fu_q, r["title"],
                                build_context(f_inc, start=start_n,
                                              abstract_chars=lim["abstract_chars"]),
                                start_n))
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
                            c["crossref"] = c.get("crossref", 0) + len(f_cr)
                            c["screened"] += len(f_screened)
                            c["included"] = len(r["included"])
                            fst.update(label="Follow-up added to the report",
                                       state="complete", expanded=False)
                            st.session_state.pop("fu_input", None)
                            st.rerun()

# ==========================================================================
# MODE 2 — Data explorer
# ==========================================================================
else:
    dq = st.text_area(
        "Data question", key="dq_input", height=90,
        placeholder="e.g., How has US education spending changed versus peer "
                    "countries?",
        label_visibility="collapsed")

    dc1, dc2 = st.columns([3, 1])
    with dc1:
        dcols = st.columns(len(DATA_EXAMPLES))
        for col, ex in zip(dcols, DATA_EXAMPLES):
            with col:
                st.button(ex[:36] + "…", key=f"dex_{ex[:12]}",
                          on_click=_set_data, args=(ex,),
                          use_container_width=True, help=ex)
    with dc2:
        drun = st.button("Analyze data", type="primary",
                         use_container_width=True)

    with st.expander("Dataset and parameters", expanded=False):
        o1, o2 = st.columns(2)
        with o1:
            auto = st.checkbox(
                "Let the model choose the dataset from my question", value=True,
                help="It must pick from the curated list below, so it can "
                     "never invent an indicator code.")
            labels = [f"{d['src']} — {d['label']}" for d in DATA_CATALOG]
            manual_idx = st.selectbox(
                "Dataset", range(len(DATA_CATALOG)),
                format_func=lambda i: labels[i], disabled=auto)
        with o2:
            y0, y1 = st.slider("Years (World Bank & NCES)", 1995,
                               date.today().year, (2010, 2022))
            census_year = st.selectbox(
                "Census ACS 5-year vintage",
                list(range(date.today().year - 2, 2014, -1)), index=1)
            country_set = st.selectbox("Countries (World Bank)",
                                       list(COUNTRY_SETS))

    if drun:
        if not dq.strip():
            st.error("Type a data question first — or tap one of the examples.")
            st.stop()
        if not api_key:
            st.error("Add your API key in the sidebar to run the analysis step.")
            st.stop()

        q = dq.strip()
        with st.status("Fetching data…", expanded=True) as status:
            if auto:
                st.write("Choosing the best dataset for your question…")
                entry, fallback = pick_indicator(llm, q)
                if fallback:
                    st.write(f"Selected by {fallback}: {entry['label']}")
            else:
                entry, fallback = DATA_CATALOG[manual_idx], None
            st.write(f"Querying {entry['src']}: {entry['label']}…")
            ds = load_dataset(entry, {
                "countries": COUNTRY_SETS[country_set], "y0": y0, "y1": y1,
                "census_year": census_year, "census_key": census_key})

            if not ds["rows"]:
                status.update(label="No data returned", state="error")
                st.error(f"{entry['src']} returned no observations for these "
                         "parameters. Open **Dataset and parameters** and try "
                         "a wider year range, an earlier Census vintage, or a "
                         "different dataset.")
                st.stop()

            st.write(f"{len(ds['rows'])} observations retrieved. Analyzing "
                     f"with {PROVIDERS[provider]['label']}…")
            try:
                raw = clean_llm_output(analyze_data(llm, q, ds))
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"Analysis failed: {e}")
                st.stop()
            status.update(label="Analysis ready", state="complete", expanded=False)

        title, body = parse_title(raw, entry["label"])
        st.session_state.data_result = {"question": q, "title": title,
                                        "body": body, "ds": ds}

    if "data_result" not in st.session_state:
        st.markdown(
            '<div class="how-row">'
            '<div class="how-card"><div class="how-step">Sources</div>'
            '<div class="how-title">World Bank, Census, NCES</div>'
            '<div class="how-body">Real statistics from official APIs — '
            'education spending, enrollment, attainment, income, and school '
            'finance.</div></div>'
            '<div class="how-card"><div class="how-step">Curated</div>'
            '<div class="how-title">No invented indicators</div>'
            '<div class="how-body">The model picks from a fixed catalog of '
            'verified indicator codes, so it can never fabricate a series.'
            '</div></div>'
            '<div class="how-card"><div class="how-step">Analyze</div>'
            '<div class="how-title">Grounded in the numbers</div>'
            '<div class="how-body">Charts are drawn from the data itself, and '
            'every figure cited must appear in it. Export HTML or CSV.'
            '</div></div></div>', unsafe_allow_html=True)

    if "data_result" in st.session_state:
        dr = st.session_state.data_result
        ds = dr["ds"]
        chart = (series_chart_html(ds) if ds["shape"] == "series"
                 else bar_chart_html(ds))
        if not chart:
            chart = bar_chart_html(ds)
        visuals = (data_stats_html(ds) + chart
                   + _figcap(1, html_lib.escape(ds["entry"]["label"])
                             + " &middot; " + html_lib.escape(ds["note"])))
        body_html = decorate_report_html(
            render_report_html(dr["body"]), {}, [], fig_start=2)

        st.divider()
        st.markdown(
            '<div class="result-head">'
            f'<div class="result-title">{html_lib.escape(dr["title"])}</div>'
            f'<div class="result-meta">{html_lib.escape(dr["question"])} '
            f'&middot; {ds["entry"]["src"]} &middot; {len(ds["rows"])} '
            f'observations &middot; Generated '
            f'{date.today().strftime("%B %d, %Y")}</div></div>',
            unsafe_allow_html=True)
        st.markdown(f'<div class="report-doc">{visuals}</div>',
                    unsafe_allow_html=True)

        sub = (f'{html_lib.escape(dr["question"])} &middot; '
               f'{html_lib.escape(ds["note"])} &middot; {len(ds["rows"])} '
               f'observations &middot; Generated '
               f'{date.today().strftime("%B %d, %Y")}')
        html_out = build_html_export(
            dr["title"], dr["question"], visuals, body_html, {},
            brand=f'Data Analysis &middot; {html_lib.escape(ds["entry"]["src"])}',
            subtitle=sub)
        e1, e2, _e3 = st.columns([1, 1, 1])
        with e1:
            st.download_button("Download analysis (HTML)", html_out,
                               file_name="data_analysis.html", mime="text/html",
                               use_container_width=True, type="primary")
        with e2:
            st.download_button("Download data (.csv)", make_csv(ds),
                               file_name="dataset.csv", mime="text/csv",
                               use_container_width=True)
        st.caption("Open the HTML analysis in a browser and press Ctrl+P / "
                   "Cmd+P for a clean PDF. The CSV contains every observation "
                   "retrieved.")

        t_an, t_data = st.tabs(["Analysis", f"Data ({len(ds['rows'])} rows)"])
        with t_an:
            st.markdown(f'<div class="report-doc">{body_html}</div>',
                        unsafe_allow_html=True)
        with t_data:
            st.markdown("Every figure in the analysis had to come from these "
                        "observations.")
            st.markdown(f'<div class="report-doc">{data_table_html(ds)}</div>',
                        unsafe_allow_html=True)
