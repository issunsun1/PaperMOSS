#!/usr/bin/env python3
"""
Fetch topic-focused papers from arXiv with multi-category parallel search,
date-range filtering, optional OpenAI-compatible LLM summaries, and history
deduplication across runs.
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
from typing import Dict, Iterable, List, Set, Tuple

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


def log_verbose(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[INFO] {message}", file=sys.stderr, flush=True)


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


def normalize_key_text(text: str) -> str:
    lowered = normalize_space(text).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lowered)


def build_abstract_key(text: str, max_chars: int = 260) -> str:
    key = normalize_key_text(text)
    return key[:max_chars]


def build_summary_key(text: str, max_chars: int = 220) -> str:
    key = normalize_key_text(text)
    return key[:max_chars]


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


def in_date_window(published: str, date_from: str, date_to: str) -> bool:
    if not published:
        return True
    try:
        pub_date = dt.date.fromisoformat(published)
    except ValueError:
        return False
    if date_from:
        if pub_date < dt.date.fromisoformat(date_from):
            return False
    if date_to:
        if pub_date > dt.date.fromisoformat(date_to):
            return False
    return True


def parse_entries(
    root: ET.Element,
    requested_category: str,
    keywords: List[str],
    mode: str,
    date_from: str,
    date_to: str,
    summary_sentences: int,
    summary_max_chars: int,
) -> List[Paper]:
    output: List[Paper] = []

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

        if not in_date_window(published, date_from=date_from, date_to=date_to):
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
    date_from: str,
    date_to: str,
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
        date_from=date_from,
        date_to=date_to,
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


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def summarize_exception(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        code = getattr(exc, "code", "")
        reason = normalize_space(str(getattr(exc, "reason", "")))
        if reason:
            return f"HTTP {code}: {reason}"
        return f"HTTP {code}"
    if isinstance(exc, urllib.error.URLError):
        reason = normalize_space(str(getattr(exc, "reason", exc)))
        return f"URL error: {reason}"
    return normalize_space(str(exc)) or exc.__class__.__name__


def check_llm_endpoint_ready(
    api_base: str,
    api_key: str,
    model: str,
    llm_timeout: float,
    user_agent: str,
) -> Tuple[bool, str]:
    endpoint = build_chat_completions_url(api_base)
    if not endpoint:
        return False, "missing --llm-api-base"
    if not model:
        return False, "missing --llm-model"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=llm_timeout) as response:
            _ = response.read()
        return True, ""
    except Exception as exc:
        reason = summarize_exception(exc)
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
            return False, f"authorization failed ({reason}). Check API key/model permission."
        return False, f"endpoint precheck failed ({reason})"


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


def translate_summary_with_openai_compatible(
    paper: Paper,
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

    user_prompt = (
        "Translate the following paper summary into simplified Chinese. "
        "Keep it concise, accurate, and preserve technical terms.\n\n"
        f"Title: {paper.title}\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Summary: {paper.concise_summary}\n"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a technical translator. Translate faithfully into simplified Chinese.",
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
        raise RuntimeError("LLM API returned empty translation.")
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
    llm_ready: bool,
    verbose: bool,
) -> Tuple[List[Paper], List[str]]:
    if summary_provider != "openai-compatible":
        return papers, []

    warnings: List[str] = []
    if not llm_ready:
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
    log_verbose(verbose, f"Start LLM summarization for {target_count} papers with {workers} workers.")
    error_stats: Dict[str, dict] = {}
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
                reason = summarize_exception(exc)
                item = error_stats.setdefault(reason, {"count": 0, "ids": []})
                item["count"] += 1
                if len(item["ids"]) < 5:
                    item["ids"].append(paper.arxiv_id)

    for reason, item in sorted(error_stats.items(), key=lambda kv: kv[1]["count"], reverse=True):
        samples = ", ".join(item["ids"])
        sample_text = f" Sample IDs: {samples}." if samples else ""
        warnings.append(
            f"LLM summary fallback: {item['count']} papers failed ({reason}).{sample_text}"
        )

    return output, warnings


def apply_zh_translation_postprocess(
    papers: List[Paper],
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
    llm_ready: bool,
    verbose: bool,
) -> Tuple[List[Paper], List[str], int]:
    if summary_language != "zh":
        return papers, [], 0
    if not llm_ready:
        return papers, [], 0

    candidates = []
    for index, paper in enumerate(papers):
        summary_text = normalize_space(paper.concise_summary)
        if not summary_text:
            continue
        if contains_chinese(summary_text):
            continue
        candidates.append(index)

    if not candidates:
        return papers, [], 0

    warnings: List[str] = []
    target_count = min(len(candidates), llm_max_papers)
    if target_count < len(candidates):
        warnings.append(
            f"Chinese translation applied to first {target_count} summaries due to --llm-max-papers."
        )

    output = list(papers)
    selected = candidates[:target_count]
    workers = max(1, min(llm_workers, len(selected)))
    translated_count = 0
    log_verbose(verbose, f"Start zh translation for {len(selected)} summaries with {workers} workers.")

    error_stats: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {}
        for index in selected:
            paper = output[index]
            future = pool.submit(
                translate_summary_with_openai_compatible,
                paper=paper,
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
                translated = future.result()
                source = paper.summary_source
                if "+zh-llm" not in source:
                    source = f"{source}+zh-llm"
                output[index] = replace(paper, concise_summary=translated, summary_source=source)
                translated_count += 1
            except Exception as exc:
                reason = summarize_exception(exc)
                item = error_stats.setdefault(reason, {"count": 0, "ids": []})
                item["count"] += 1
                if len(item["ids"]) < 5:
                    item["ids"].append(paper.arxiv_id)

    for reason, item in sorted(error_stats.items(), key=lambda kv: kv[1]["count"], reverse=True):
        samples = ", ".join(item["ids"])
        sample_text = f" Sample IDs: {samples}." if samples else ""
        warnings.append(
            f"Chinese translation fallback: {item['count']} papers failed ({reason}).{sample_text}"
        )

    return output, warnings, translated_count


def load_history_records(history_file: str) -> Tuple[List[dict], List[str]]:
    path = Path(history_file)
    if not path.exists():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"History load failed; ignoring old history ({exc})."]

    if not isinstance(data, dict):
        return [], ["History format invalid; reset history in memory."]
    records = data.get("records", [])
    if not isinstance(records, list):
        return [], ["History records invalid; reset history in memory."]
    cleaned = [record for record in records if isinstance(record, dict)]
    return cleaned, []


def save_history_records(history_file: str, records: List[dict], max_entries: int) -> None:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    payload = {
        "updated_at": now,
        "records": records[:max_entries],
    }
    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_history_index(records: List[dict]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    arxiv_ids: Set[str] = set()
    title_keys: Set[str] = set()
    abstract_keys: Set[str] = set()
    summary_keys: Set[str] = set()
    for record in records:
        arxiv_id = normalize_space(str(record.get("arxiv_id", "")))
        if arxiv_id:
            arxiv_ids.add(arxiv_id)
        title_key = normalize_space(str(record.get("title_key", "")))
        if title_key:
            title_keys.add(title_key)
        abstract_key = normalize_space(str(record.get("abstract_key", "")))
        if abstract_key:
            abstract_keys.add(abstract_key)
        summary_key = normalize_space(str(record.get("summary_key", "")))
        if summary_key:
            summary_keys.add(summary_key)
    return arxiv_ids, title_keys, abstract_keys, summary_keys


def filter_papers_by_history(
    papers: List[Paper],
    history_arxiv_ids: Set[str],
    history_title_keys: Set[str],
    history_abstract_keys: Set[str],
    history_summary_keys: Set[str],
) -> Tuple[List[Paper], int]:
    kept: List[Paper] = []
    skipped = 0
    for paper in papers:
        title_key = normalize_key_text(paper.title)
        abstract_key = build_abstract_key(paper.abstract)
        summary_key = build_summary_key(paper.concise_summary)
        duplicate = (
            paper.arxiv_id in history_arxiv_ids
            or title_key in history_title_keys
            or abstract_key in history_abstract_keys
            or (summary_key and summary_key in history_summary_keys)
        )
        if duplicate:
            skipped += 1
            continue
        kept.append(paper)
    return kept, skipped


def update_history_records(records: List[dict], papers: List[Paper], max_entries: int) -> List[dict]:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    by_arxiv_id: Dict[str, dict] = {}
    extra_key = 0
    for record in records:
        record_id = normalize_space(str(record.get("arxiv_id", "")))
        if record_id:
            by_arxiv_id[record_id] = record
        else:
            by_arxiv_id[f"__extra_{extra_key}"] = record
            extra_key += 1

    for paper in papers:
        old = by_arxiv_id.get(paper.arxiv_id, {})
        existing_categories = old.get("matched_categories", [])
        if not isinstance(existing_categories, list):
            existing_categories = []
        merged_categories = sorted(
            set([normalize_space(str(item)) for item in existing_categories if normalize_space(str(item))]
                + paper.matched_categories)
        )

        by_arxiv_id[paper.arxiv_id] = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "title_key": normalize_key_text(paper.title),
            "abstract_key": build_abstract_key(paper.abstract),
            "summary_key": build_summary_key(paper.concise_summary),
            "published": paper.published,
            "updated": paper.updated,
            "url": paper.url,
            "matched_categories": merged_categories,
            "summary_source": paper.summary_source,
            "first_seen": old.get("first_seen", now),
            "last_seen": now,
        }

    merged = list(by_arxiv_id.values())
    merged.sort(key=lambda item: str(item.get("last_seen", "")), reverse=True)
    return merged[:max_entries]


def render_markdown_plain(
    papers: List[Paper],
    categories: List[str],
    keywords: List[str],
    mode: str,
    summary_provider: str,
    warnings: List[str],
    run_stats: dict,
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
        f"- Summary language target: {run_stats.get('summary_language_target', 'en')}",
        f"- Date range: {run_stats.get('date_window', '(unbounded)')}",
        f"- Retrieved before history dedup: {run_stats.get('retrieved_before_history', 0)}",
        f"- Removed by history dedup: {run_stats.get('history_removed', 0)}",
        f"- Translated to zh this run: {run_stats.get('translated_to_zh', 0)}",
        "",
    ]

    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if not papers:
        lines.append("No new papers found after filters and deduplication.")
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


def render_markdown_colorful(
    papers: List[Paper],
    categories: List[str],
    keywords: List[str],
    mode: str,
    summary_provider: str,
    warnings: List[str],
    run_stats: dict,
) -> str:
    category_text = ", ".join(categories) if categories else "(none)"
    keyword_text = ", ".join(keywords) if keywords else "(none)"
    lines = [
        "# arXiv Research Digest",
        "",
        '<div style="border-radius:16px;padding:18px 20px;background:linear-gradient(135deg,#0f172a,#1d4ed8,#0ea5e9);color:#eff6ff;">',
        '<div style="font-size:22px;font-weight:700;margin-bottom:8px;">Literature Snapshot</div>',
        f'<div style="line-height:1.7;"><b>Categories</b>: {category_text}<br/><b>Keywords</b>: {keyword_text}<br/>'
        f'<b>Match Mode</b>: {mode}<br/><b>Summary Provider</b>: {summary_provider}<br/>'
        f'<b>Summary Language</b>: {run_stats.get("summary_language_target", "en")}<br/>'
        f'<b>Date Range</b>: {run_stats.get("date_window", "(unbounded)")}<br/>'
        f'<b>Retrieved</b>: {run_stats.get("retrieved_before_history", 0)} | '
        f'<b>History Dedup Removed</b>: {run_stats.get("history_removed", 0)} | '
        f'<b>Translated to zh</b>: {run_stats.get("translated_to_zh", 0)}</div>',
        "</div>",
        "",
    ]

    if warnings:
        lines.append(
            '<div style="margin-top:12px;border-left:4px solid #f59e0b;background:#fffbeb;padding:10px 12px;border-radius:8px;">'
        )
        lines.append("<b>Warnings</b><br/>")
        for warning in warnings:
            lines.append(f"- {warning}<br/>")
        lines.append("</div>")
        lines.append("")

    if not papers:
        lines.append(
            '<div style="margin-top:14px;padding:14px;border-radius:10px;background:#f8fafc;border:1px solid #cbd5e1;">'
            "No new papers found after filters and deduplication."
            "</div>"
        )
        lines.append("")
        return "\n".join(lines).strip() + "\n"

    for idx, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:8]) if paper.authors else "unknown"
        if len(paper.authors) > 8:
            authors += ", et al."
        matched = ", ".join(paper.matched_categories) if paper.matched_categories else "(none)"
        lines.append(
            '<div style="margin-top:14px;border:1px solid #bfdbfe;border-radius:14px;padding:14px 16px;'
            'background:linear-gradient(165deg,#eff6ff,#ecfeff);">'
        )
        lines.append(f'<div style="font-size:18px;font-weight:700;color:#1e3a8a;margin-bottom:8px;">{idx}. {paper.title}</div>')
        lines.append(
            f'<div style="line-height:1.75;color:#0f172a;"><b>arXiv ID</b>: <code>{paper.arxiv_id}</code>'
            f' | <b>Published</b>: {paper.published or "unknown"}<br/>'
            f'<b>Matched Categories</b>: {matched}<br/>'
            f'<b>Authors</b>: {authors}<br/>'
            f'<b>Summary Source</b>: {paper.summary_source}<br/>'
            f'<b>Link</b>: <a href="{paper.url}">{paper.url}</a><br/>'
            f'<b>Summary</b>: {paper.concise_summary or "(summary unavailable)"}</div>'
        )
        lines.append("</div>")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_markdown(
    papers: List[Paper],
    categories: List[str],
    keywords: List[str],
    mode: str,
    summary_provider: str,
    warnings: List[str],
    run_stats: dict,
    markdown_theme: str,
) -> str:
    if markdown_theme == "colorful":
        return render_markdown_colorful(
            papers=papers,
            categories=categories,
            keywords=keywords,
            mode=mode,
            summary_provider=summary_provider,
            warnings=warnings,
            run_stats=run_stats,
        )
    return render_markdown_plain(
        papers=papers,
        categories=categories,
        keywords=keywords,
        mode=mode,
        summary_provider=summary_provider,
        warnings=warnings,
        run_stats=run_stats,
    )


def render_json(
    papers: List[Paper],
    categories: List[str],
    keywords: List[str],
    mode: str,
    summary_provider: str,
    warnings: List[str],
    run_stats: dict,
) -> str:
    payload = {
        "categories": categories,
        "keywords": keywords,
        "match_mode": mode,
        "summary_provider": summary_provider,
        "count": len(papers),
        "warnings": warnings,
        "run_stats": run_stats,
        "papers": [asdict(paper) for paper in papers],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def build_date_window_text(date_from: str, date_to: str) -> str:
    if date_from and date_to:
        return f"{date_from} to {date_to}"
    if date_from:
        return f"from {date_from}"
    if date_to:
        return f"until {date_to}"
    return "(unbounded)"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch arXiv papers for topics with multi-category parallel search, date-range "
            "filtering, optional third-party LLM summaries, and history deduplication."
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
        default=[],
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
    parser.add_argument("--date-from", default="", help="Lower publish date bound in YYYY-MM-DD.")
    parser.add_argument("--date-to", default="", help="Upper publish date bound in YYYY-MM-DD.")
    parser.add_argument(
        "--min-date",
        default="",
        help="Deprecated alias for --date-from (kept for backward compatibility).",
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
    parser.add_argument("--llm-workers", type=int, default=10, help="Parallel workers for LLM summarization.")
    parser.add_argument(
        "--llm-max-papers",
        type=int,
        default=30,
        help="Maximum paper count that LLM summarization will process.",
    )
    parser.add_argument(
        "--history-file",
        default=".cache/arxiv_cs_ai_digest_history.json",
        help="History index JSON path for cross-run deduplication.",
    )
    parser.add_argument(
        "--disable-history-dedup",
        action="store_true",
        help="Disable history dedup check and history file updates.",
    )
    parser.add_argument(
        "--history-max-entries",
        type=int,
        default=5000,
        help="Maximum records stored in history index.",
    )
    parser.add_argument(
        "--user-agent",
        default="arxiv-cs-ai-digest/3.0 (mailto:research@example.com)",
        help="User-Agent header for network requests.",
    )
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown", help="Output format.")
    parser.add_argument("--save", default="", help="Optional path to save output.")
    parser.add_argument("--save-markdown", default="", help="Optional markdown output file path (.md recommended).")
    parser.add_argument(
        "--markdown-theme",
        choices=["plain", "colorful"],
        default="colorful",
        help="Markdown visual style (used when --output markdown).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs to stderr.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    keywords = split_keywords(args.keyword)
    categories = split_categories(args.category)
    warnings: List[str] = []

    date_from = normalize_space(args.date_from)
    if not date_from and args.min_date:
        date_from = normalize_space(args.min_date)
    date_to = normalize_space(args.date_to)

    if date_from:
        try:
            dt.date.fromisoformat(date_from)
        except ValueError:
            print("Invalid --date-from. Use YYYY-MM-DD.", file=sys.stderr)
            return 2
    if date_to:
        try:
            dt.date.fromisoformat(date_to)
        except ValueError:
            print("Invalid --date-to. Use YYYY-MM-DD.", file=sys.stderr)
            return 2
    if date_from and date_to and date_from > date_to:
        print("--date-from cannot be later than --date-to.", file=sys.stderr)
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
    if args.history_max_entries < 1:
        print("--history-max-entries must be >= 1.", file=sys.stderr)
        return 2

    if args.save and args.save_markdown:
        print("Use either --save or --save-markdown, not both.", file=sys.stderr)
        return 2
    if args.save_markdown and args.output != "markdown":
        print("--save-markdown requires --output markdown.", file=sys.stderr)
        return 2

    log_verbose(
        args.verbose,
        f"Search start: categories={categories}, keywords={keywords}, date_window={build_date_window_text(date_from, date_to)}",
    )

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
                date_from=date_from,
                date_to=date_to,
                summary_sentences=args.summary_sentences,
                summary_max_chars=args.summary_max_chars,
                timeout=args.timeout,
                user_agent=args.user_agent,
            )
            future_to_category[future] = category

        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                result = future.result()
                papers_all.extend(result)
                log_verbose(args.verbose, f"Category {category}: fetched {len(result)} entries after filters.")
            except urllib.error.URLError as exc:
                warnings.append(f"Category {category} fetch failed: {exc}")
            except ET.ParseError as exc:
                warnings.append(f"Category {category} parse failed: {exc}")
            except Exception as exc:
                warnings.append(f"Category {category} failed: {exc}")

    papers = sort_papers(merge_deduplicate_papers(papers_all))
    retrieved_before_history = len(papers)

    history_removed = 0
    history_records: List[dict] = []
    if not args.disable_history_dedup:
        loaded_records, history_warnings = load_history_records(args.history_file)
        warnings.extend(history_warnings)
        history_records = loaded_records
        history_ids, history_titles, history_abstracts, history_summaries = build_history_index(history_records)
        papers, history_removed = filter_papers_by_history(
            papers=papers,
            history_arxiv_ids=history_ids,
            history_title_keys=history_titles,
            history_abstract_keys=history_abstracts,
            history_summary_keys=history_summaries,
        )
        if history_removed:
            warnings.append(f"History dedup removed {history_removed} previously retrieved papers.")
        log_verbose(
            args.verbose,
            f"History dedup finished: removed={history_removed}, remaining={len(papers)}.",
        )

    llm_needed = args.summary_provider == "openai-compatible" or args.summary_language == "zh"
    llm_ready = True
    if llm_needed:
        llm_ready, llm_reason = check_llm_endpoint_ready(
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            model=args.llm_model,
            llm_timeout=args.llm_timeout,
            user_agent=args.user_agent,
        )
        if not llm_ready:
            clean_reason = llm_reason.rstrip(". ")
            warnings.append(f"LLM endpoint unavailable: {clean_reason}.")

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
        llm_ready=llm_ready,
        verbose=args.verbose,
    )
    warnings.extend(llm_warnings)

    papers, zh_warnings, translated_to_zh = apply_zh_translation_postprocess(
        papers=papers,
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
        llm_ready=llm_ready,
        verbose=args.verbose,
    )
    warnings.extend(zh_warnings)

    if not args.disable_history_dedup:
        updated_records = update_history_records(history_records, papers, max_entries=args.history_max_entries)
        try:
            save_history_records(args.history_file, updated_records, max_entries=args.history_max_entries)
            log_verbose(args.verbose, f"History file updated: {args.history_file}")
        except Exception as exc:
            warnings.append(f"Failed to save history file: {exc}")

    run_stats = {
        "summary_language_target": args.summary_language,
        "date_window": build_date_window_text(date_from, date_to),
        "retrieved_before_history": retrieved_before_history,
        "history_removed": history_removed,
        "translated_to_zh": translated_to_zh,
        "history_file": args.history_file if not args.disable_history_dedup else "",
    }

    if args.output == "json":
        rendered = render_json(
            papers=papers,
            categories=categories,
            keywords=keywords,
            mode=args.mode,
            summary_provider=args.summary_provider,
            warnings=warnings,
            run_stats=run_stats,
        )
    else:
        rendered = render_markdown(
            papers=papers,
            categories=categories,
            keywords=keywords,
            mode=args.mode,
            summary_provider=args.summary_provider,
            warnings=warnings,
            run_stats=run_stats,
            markdown_theme=args.markdown_theme,
        )

    save_path = args.save_markdown or args.save
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        log_verbose(args.verbose, f"Output saved to {path}")

    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
