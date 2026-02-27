#!/usr/bin/env python3
"""
Fetch topic-focused papers from arXiv, support multi-category parallel search,
and summarize abstracts with either extractive logic or a third-party
OpenAI-compatible LLM API.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, List, Tuple

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
}


@dataclass
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: List[str]
    published: str
    updated: str
    categories: List[str]
    matched_categories: List[str]
    url: str
    abstract: str
    concise_summary: str
    summary_source: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def split_keywords(raw_keywords: Iterable[str]) -> List[str]:
    keywords: List[str] = []
    seen = set()
    for item in raw_keywords:
        for part in item.split(","):
            term = normalize_space(part).lower()
            if not term or term in seen:
                continue
            seen.add(term)
            keywords.append(term)
    return keywords


def split_categories(raw_categories: Iterable[str]) -> List[str]:
    categories: List[str] = []
    seen = set()
    for item in raw_categories:
        for part in item.split(","):
            cat = normalize_space(part)
            if not cat:
                continue
            key = cat.lower()
            if key in seen:
                continue
            seen.add(key)
            categories.append(cat)
    if not categories:
        categories.append("cs.AI")
    return categories


def parse_version_number(version: str) -> int:
    if not version:
        return 0
    match = re.search(r"v(\d+)$", version)
    if not match:
        return 0
    return int(match.group(1))


def build_query(category: str, keywords: List[str], mode: str) -> str:
    if not keywords:
        return f"cat:{category}"
    terms = []
    for keyword in keywords:
        escaped = keyword.replace('"', "")
        if " " in escaped:
            terms.append(f'all:"{escaped}"')
        else:
            terms.append(f"all:{escaped}")
    joiner = " AND " if mode == "all" else " OR "
    keyword_clause = joiner.join(terms)
    return f"cat:{category} AND ({keyword_clause})"


def fetch_feed(query: str, start: int, max_results: int, timeout: float, user_agent: str) -> ET.Element:
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return ET.fromstring(payload)


def parse_arxiv_id(abs_url: str) -> tuple[str, str]:
    value = normalize_space(abs_url)
    match = re.search(r"/abs/([^/?#]+)", value)
    identifier = match.group(1) if match else value
    version_match = re.search(r"(v\d+)$", identifier)
    if version_match:
        version = version_match.group(1)
        return identifier[: -len(version)], version
    return identifier, ""


def parse_iso_date(raw: str) -> str:
    raw = normalize_space(raw)
    if not raw:
        return ""
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return raw[:10]


def tokenize(sentence: str) -> List[str]:
    return [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", sentence.lower()) if word not in STOPWORDS]


def split_sentences(text: str) -> List[str]:
    cleaned = normalize_space(text)
    if not cleaned:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", cleaned)
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder="...")


def summarize_abstract_extractive(
    abstract: str, keywords: List[str], max_sentences: int, max_chars: int
) -> str:
    sentences = split_sentences(abstract)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return clip_text(" ".join(sentences), max_chars)

    word_freq = {}
    for sentence in sentences:
        for token in tokenize(sentence):
            word_freq[token] = word_freq.get(token, 0) + 1

    scored = []
    for idx, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        score = sum(word_freq.get(token, 0) for token in tokens) / len(tokens)
        lowered = sentence.lower()
        if keywords and any(keyword in lowered for keyword in keywords):
            score += 1.0
        if idx == 0:
            score += 0.1
        scored.append((score, idx, sentence))

    if not scored:
        return clip_text(" ".join(sentences[:max_sentences]), max_chars)

    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    ordered = [item[2] for item in sorted(top, key=lambda item: item[1])]
    return clip_text(" ".join(ordered), max_chars)


def keyword_match(text: str, keywords: List[str], mode: str) -> bool:
    if not keywords:
        return True
    lowered = text.lower()
    if mode == "all":
        return all(keyword in lowered for keyword in keywords)
    return any(keyword in lowered for keyword in keywords)


def parse_entries(
    root: ET.Element,
    requested_category: str,
    keywords: List[str],
    mode: str,
    min_date: str,
    summary_sentences: int,
    summary_max_chars: int,
) -> List[Paper]:
    output: List[Paper] = []
    min_date_value = None
    if min_date:
        min_date_value = dt.date.fromisoformat(min_date)

    for entry in root.findall("atom:entry", ATOM_NS):
        title = normalize_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        abstract = normalize_space(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        abs_url = normalize_space(entry.findtext("atom:id", default="", namespaces=ATOM_NS))

        arxiv_id, version = parse_arxiv_id(abs_url)
        authors = [
            normalize_space(node.findtext("atom:name", default="", namespaces=ATOM_NS))
            for node in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author]
        categories = [cat.attrib.get("term", "").strip() for cat in entry.findall("atom:category", ATOM_NS)]
        categories = [cat for cat in categories if cat]
        published = parse_iso_date(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        updated = parse_iso_date(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))

        if requested_category and requested_category not in categories:
            continue

        if min_date_value and published:
            try:
                published_date = dt.date.fromisoformat(published)
                if published_date < min_date_value:
                    continue
            except ValueError:
                continue

        searchable_text = f"{title}\n{abstract}"
        if not keyword_match(searchable_text, keywords, mode):
            continue

        output.append(
            Paper(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                authors=authors,
                published=published,
                updated=updated,
                categories=categories,
                matched_categories=[requested_category] if requested_category else [],
                url=f"https://arxiv.org/abs/{arxiv_id}{version}",
                abstract=abstract,
                concise_summary=summarize_abstract_extractive(
                    abstract=abstract,
                    keywords=keywords,
                    max_sentences=summary_sentences,
                    max_chars=summary_max_chars,
                ),
                summary_source="extractive",
            )
        )

    return output


def fetch_category_papers(
    category: str,
    keywords: List[str],
    mode: str,
    start: int,
    max_results: int,
    min_date: str,
    summary_sentences: int,
    summary_max_chars: int,
    timeout: float,
    user_agent: str,
) -> List[Paper]:
    query = build_query(category=category, keywords=keywords, mode=mode)
    root = fetch_feed(
        query=query,
        start=start,
        max_results=max_results,
        timeout=timeout,
        user_agent=user_agent,
    )
    return parse_entries(
        root=root,
        requested_category=category,
        keywords=keywords,
        mode=mode,
        min_date=min_date,
        summary_sentences=summary_sentences,
        summary_max_chars=summary_max_chars,
    )


def should_replace_paper(existing: Paper, incoming: Paper) -> bool:
    existing_version = parse_version_number(existing.version)
    incoming_version = parse_version_number(incoming.version)
    if incoming_version != existing_version:
        return incoming_version > existing_version
    if incoming.published != existing.published:
        return incoming.published > existing.published
    if incoming.updated != existing.updated:
        return incoming.updated > existing.updated
    return len(incoming.abstract) > len(existing.abstract)


def merge_deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    merged = {}
    for paper in papers:
        key = paper.arxiv_id
        if key not in merged:
            merged[key] = paper
            continue
        existing = merged[key]
        matched_categories = sorted(set(existing.matched_categories + paper.matched_categories))
        preferred = paper if should_replace_paper(existing, paper) else existing
        merged[key] = replace(preferred, matched_categories=matched_categories)
    return list(merged.values())


def sort_papers(papers: List[Paper]) -> List[Paper]:
    def sort_key(paper: Paper) -> Tuple[str, str, int, str]:
        return (
            paper.published or "",
            paper.updated or "",
            parse_version_number(paper.version),
            paper.arxiv_id,
        )

    return sorted(papers, key=sort_key, reverse=True)


def build_chat_completions_url(api_base: str) -> str:
    base = normalize_space(api_base).rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_message_content(value) -> str:
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, list):
        chunks = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = normalize_space(str(item.get("text", "")))
                if text:
                    chunks.append(text)
        return normalize_space(" ".join(chunks))
    return ""


def summarize_with_openai_compatible(
    paper: Paper,
    keywords: List[str],
    summary_language: str,
    api_base: str,
    api_key: str,
    model: str,
    llm_timeout: float,
    llm_max_tokens: int,
    llm_temperature: float,
    user_agent: str,
    summary_max_chars: int,
) -> str:
    endpoint = build_chat_completions_url(api_base)
    if not endpoint:
        raise RuntimeError("Missing --llm-api-base.")
    if not model:
        raise RuntimeError("Missing --llm-model.")

    language_instruction = (
        "Use simplified Chinese." if summary_language == "zh" else "Use concise English."
    )
    focus = ", ".join(keywords) if keywords else "general relevance"
    user_prompt = (
        "Summarize this arXiv abstract into 2-3 concise sentences. "
        "Include problem, method, and key contribution. "
        f"{language_instruction} Focus on keywords: {focus}.\n\n"
        f"Title: {paper.title}\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Abstract: {paper.abstract}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise research assistant that writes compact paper summaries.",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": llm_temperature,
        "max_tokens": llm_max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=llm_timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)

    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError("No choices returned from LLM API.")
    message = choices[0].get("message", {})
    content = extract_message_content(message.get("content"))
    if not content:
        raise RuntimeError("LLM API returned empty content.")
    return clip_text(content, summary_max_chars)


def apply_llm_summaries(
    papers: List[Paper],
    keywords: List[str],
    summary_provider: str,
    summary_language: str,
    api_base: str,
    api_key: str,
    model: str,
    llm_timeout: float,
    llm_max_tokens: int,
    llm_temperature: float,
    llm_workers: int,
    llm_max_papers: int,
    user_agent: str,
    summary_max_chars: int,
) -> Tuple[List[Paper], List[str]]:
    if summary_provider != "openai-compatible":
        return papers, []

    warnings: List[str] = []
    if not api_base:
        warnings.append("LLM summary skipped: missing --llm-api-base.")
        return papers, warnings
    if not model:
        warnings.append("LLM summary skipped: missing --llm-model.")
        return papers, warnings

    target_count = min(len(papers), llm_max_papers)
    if target_count < len(papers):
        warnings.append(
            f"LLM summary only applied to first {target_count} papers due to --llm-max-papers."
        )

    output = list(papers)
    if target_count == 0:
        return output, warnings

    workers = max(1, min(llm_workers, target_count))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {}
        for index in range(target_count):
            paper = output[index]
            future = pool.submit(
                summarize_with_openai_compatible,
                paper=paper,
                keywords=keywords,
                summary_language=summary_language,
                api_base=api_base,
                api_key=api_key,
                model=model,
                llm_timeout=llm_timeout,
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                user_agent=user_agent,
                summary_max_chars=summary_max_chars,
            )
            future_to_index[future] = index

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            paper = output[index]
            try:
                summary = future.result()
                output[index] = replace(paper, concise_summary=summary, summary_source="llm")
            except Exception as exc:
                warnings.append(f"LLM summary fallback for {paper.arxiv_id}: {exc}")

    return output, warnings


def render_markdown(
    papers: List[Paper],
    categories: List[str],
    keywords: List[str],
    mode: str,
    summary_provider: str,
    warnings: List[str],
) -> str:
    category_text = ", ".join(categories) if categories else "(none)"
    keyword_text = ", ".join(keywords) if keywords else "(none)"
    lines = [
        "# arXiv Digest",
        "",
        f"- Categories: {category_text}",
        f"- Keywords: {keyword_text}",
        f"- Match mode: {mode}",
        f"- Summary provider: {summary_provider}",
        "",
    ]

    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if not papers:
        lines.append("No matching papers found.")
        lines.append("")
        return "\n".join(lines).strip() + "\n"

    for idx, paper in enumerate(papers, start=1):
        lines.append(f"## {idx}. {paper.title}")
        lines.append(f"- arXiv ID: `{paper.arxiv_id}`")
        if paper.version:
            lines.append(f"- Version: `{paper.version}`")
        lines.append(f"- Published: {paper.published or 'unknown'}")
        lines.append(f"- URL: {paper.url}")
        if paper.matched_categories:
            lines.append(f"- Matched categories: {', '.join(paper.matched_categories)}")
        if paper.authors:
            author_text = ", ".join(paper.authors[:8])
            if len(paper.authors) > 8:
                author_text += ", et al."
            lines.append(f"- Authors: {author_text}")
        lines.append(f"- Summary source: {paper.summary_source}")
        lines.append(f"- Summary: {paper.concise_summary or '(summary unavailable)'}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_json(
    papers: List[Paper],
    categories: List[str],
    keywords: List[str],
    mode: str,
    summary_provider: str,
    warnings: List[str],
) -> str:
    payload = {
        "categories": categories,
        "keywords": keywords,
        "match_mode": mode,
        "summary_provider": summary_provider,
        "count": len(papers),
        "warnings": warnings,
        "papers": [asdict(paper) for paper in papers],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch arXiv papers for topics with multi-category parallel search and "
            "optional third-party LLM summaries."
        )
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Topic keyword (repeatable). Comma-separated values are also allowed.",
    )
    parser.add_argument("--mode", choices=["any", "all"], default="any", help="Keyword match mode.")
    parser.add_argument(
        "--category",
        action="append",
        default=["cs.AI"],
        help="arXiv category (repeatable). Comma-separated values are also allowed.",
    )
    parser.add_argument(
        "--category-workers",
        type=int,
        default=4,
        help="Parallel workers for category-level arXiv search.",
    )
    parser.add_argument("--start", type=int, default=0, help="Result offset per category.")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum entries requested per category.")
    parser.add_argument(
        "--min-date",
        default="",
        help="Optional lower bound publish date in YYYY-MM-DD format.",
    )
    parser.add_argument("--summary-sentences", type=int, default=2, help="Sentences kept in extractive summary.")
    parser.add_argument("--summary-max-chars", type=int, default=320, help="Character cap for each summary.")
    parser.add_argument("--timeout", type=float, default=30.0, help="arXiv API HTTP timeout in seconds.")
    parser.add_argument(
        "--summary-provider",
        choices=["extractive", "openai-compatible"],
        default="extractive",
        help="Summary backend provider.",
    )
    parser.add_argument(
        "--summary-language",
        choices=["zh", "en"],
        default="zh",
        help="Language target for LLM summaries.",
    )
    parser.add_argument(
        "--llm-api-base",
        default=os.getenv("LLM_API_BASE", ""),
        help="OpenAI-compatible API base URL (for example https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="API key for the LLM provider. Can also be set via LLM_API_KEY.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", ""),
        help="Model name used by the OpenAI-compatible endpoint.",
    )
    parser.add_argument("--llm-timeout", type=float, default=45.0, help="LLM API timeout seconds.")
    parser.add_argument("--llm-max-tokens", type=int, default=220, help="Max output tokens per LLM summary.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="Sampling temperature for LLM summary.")
    parser.add_argument("--llm-workers", type=int, default=4, help="Parallel workers for LLM summarization.")
    parser.add_argument(
        "--llm-max-papers",
        type=int,
        default=50,
        help="Maximum paper count that LLM summarization will process.",
    )
    parser.add_argument(
        "--user-agent",
        default="arxiv-cs-ai-digest/2.0 (mailto:research@example.com)",
        help="User-Agent header for network requests.",
    )
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown", help="Output format.")
    parser.add_argument("--save", default="", help="Optional path to save output.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    keywords = split_keywords(args.keyword)
    categories = split_categories(args.category)
    warnings: List[str] = []

    if args.min_date:
        try:
            dt.date.fromisoformat(args.min_date)
        except ValueError:
            print("Invalid --min-date. Use YYYY-MM-DD.", file=sys.stderr)
            return 2

    if args.max_results < 1:
        print("--max-results must be >= 1.", file=sys.stderr)
        return 2
    if args.summary_sentences < 1:
        print("--summary-sentences must be >= 1.", file=sys.stderr)
        return 2
    if args.category_workers < 1:
        print("--category-workers must be >= 1.", file=sys.stderr)
        return 2
    if args.llm_workers < 1:
        print("--llm-workers must be >= 1.", file=sys.stderr)
        return 2
    if args.llm_max_papers < 0:
        print("--llm-max-papers must be >= 0.", file=sys.stderr)
        return 2

    papers_all: List[Paper] = []
    search_workers = max(1, min(args.category_workers, len(categories)))
    with ThreadPoolExecutor(max_workers=search_workers) as pool:
        future_to_category = {}
        for category in categories:
            future = pool.submit(
                fetch_category_papers,
                category=category,
                keywords=keywords,
                mode=args.mode,
                start=args.start,
                max_results=args.max_results,
                min_date=args.min_date,
                summary_sentences=args.summary_sentences,
                summary_max_chars=args.summary_max_chars,
                timeout=args.timeout,
                user_agent=args.user_agent,
            )
            future_to_category[future] = category

        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                papers_all.extend(future.result())
            except urllib.error.URLError as exc:
                warnings.append(f"Category {category} fetch failed: {exc}")
            except ET.ParseError as exc:
                warnings.append(f"Category {category} parse failed: {exc}")
            except Exception as exc:
                warnings.append(f"Category {category} failed: {exc}")

    papers = sort_papers(merge_deduplicate_papers(papers_all))
    papers, llm_warnings = apply_llm_summaries(
        papers=papers,
        keywords=keywords,
        summary_provider=args.summary_provider,
        summary_language=args.summary_language,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
        model=args.llm_model,
        llm_timeout=args.llm_timeout,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        llm_workers=args.llm_workers,
        llm_max_papers=args.llm_max_papers,
        user_agent=args.user_agent,
        summary_max_chars=args.summary_max_chars,
    )
    warnings.extend(llm_warnings)

    if args.output == "json":
        rendered = render_json(
            papers=papers,
            categories=categories,
            keywords=keywords,
            mode=args.mode,
            summary_provider=args.summary_provider,
            warnings=warnings,
        )
    else:
        rendered = render_markdown(
            papers=papers,
            categories=categories,
            keywords=keywords,
            mode=args.mode,
            summary_provider=args.summary_provider,
            warnings=warnings,
        )

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
