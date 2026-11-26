"""Pure grounded multiple-choice quiz helpers (no Streamlit or LangChain imports)."""

import json
import random
import re

from ui.catalog import CATALOG, DRIVE_LINKS, FALLBACK_CATEGORY, trang_label

QUIZ_TOPICS = [
    ("PEAS & Tác tử", "Bài giảng ELC3008", "PEAS tác tử môi trường"),
    ("Tìm kiếm DFS/BFS/A*", "Bài giảng ELC3008", "DFS BFS A* tìm kiếm heuristic"),
    ("CSP & Minimax", "UC Berkeley CS188", "CSP Minimax Alpha-Beta backtracking"),
    ("Học máy & SVM", "Bài giảng Học máy (Tiếng Việt)", "SVM cây quyết định KNN hồi quy"),
    ("GAN / VAE / LLM", "Large Language Models", "Transformer attention LLM embedding"),
]

_PAGE_RE = re.compile(r"<!--\s*page:\s*\d+\s*-->")
_WS_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_OPT_PREFIX_RE = re.compile(r"^[A-Da-d][\.\):]\s*")
_VALID_OR_LONE_BS_RE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_LETTERS = ("A", "B", "C", "D")


class QuizParseError(ValueError):
    """Raised when an LLM quiz response contains no valid multiple-choice items."""


def _clean_excerpt(text, max_len=850):
    cleaned = _PAGE_RE.sub(" ", text or "")
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _pick_diverse(entries, max_chunks, rng):
    """Pick chunks while diversifying (doc_id, trang) pairs."""
    picked = []
    seen_keys = set()
    doc_counts = {}
    for entry in entries:
        key = (entry["doc_id"], entry["trang"])
        if key in seen_keys or doc_counts.get(entry["doc_id"], 0) >= 2:
            continue
        picked.append(entry)
        seen_keys.add(key)
        doc_counts[entry["doc_id"]] = doc_counts.get(entry["doc_id"], 0) + 1
        if len(picked) >= max_chunks:
            return picked
    for entry in entries:
        if entry in picked:
            continue
        picked.append(entry)
        if len(picked) >= max_chunks:
            break
    return picked


def sample_quiz_chunks(docstore_dict, category="Tất cả", topic="", max_chunks=6, rng=None):
    """Sample diverse, substantial chunks from docstore_dict filtered by category and optional topic."""
    rng = rng or random.Random()
    cat_filter = str(category or "").strip()
    use_all = not cat_filter or cat_filter == "Tất cả" or cat_filter.startswith("Tất cả")

    pool = []
    for doc in (docstore_dict or {}).values():
        meta = getattr(doc, "metadata", None) or {}
        doc_id = meta.get("doc_id") or ""
        known = CATALOG.get(doc_id) or {}
        doc_cat = known.get("category") or FALLBACK_CATEGORY
        if not use_all and doc_cat != cat_filter:
            continue
        excerpt = _clean_excerpt(getattr(doc, "page_content", "") or "")
        if len(excerpt) < 200:
            continue
        pool.append({
            "doc_id": doc_id or "?",
            "doc_title": known.get("title") or meta.get("doc_title") or doc_id or "Tài liệu",
            "category": doc_cat,
            "trang": trang_label(meta),
            "url": DRIVE_LINKS.get(doc_id) or known.get("url"),
            "excerpt": excerpt,
        })

    if not pool:
        return []

    topic_tokens = [w for w in re.findall(r"[^\W_]+", str(topic or "").casefold()) if len(w) >= 2]
    chosen = []
    if topic_tokens:
        scored = []
        for idx, entry in enumerate(pool):
            hay = f"{entry['doc_title']} {entry['excerpt']}".casefold()
            hits = sum(1 for tok in topic_tokens if tok in hay)
            if hits > 0:
                scored.append((hits, rng.random(), idx, entry))
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = _pick_diverse([item[3] for item in scored], max_chunks, rng)

    if len(chosen) < max_chunks:
        remainder = [entry for entry in pool if entry not in chosen]
        rng.shuffle(remainder)
        chosen.extend(_pick_diverse(remainder, max_chunks - len(chosen), rng))

    return [
        {
            "source_idx": idx,
            "doc_id": item["doc_id"],
            "doc_title": item["doc_title"],
            "category": item["category"],
            "trang": item["trang"],
            "url": item["url"],
            "excerpt": item["excerpt"],
        }
        for idx, item in enumerate(chosen, 1)
    ]


def format_quiz_context(sampled_chunks):
    """Format sampled chunks with numbered citation tags matching prompt source_idx."""
    blocks = []
    for item in sampled_chunks or []:
        blocks.append(
            f"[Nguồn {item['source_idx']} | {item['doc_title']} | Trang {item['trang']}]\n"
            f"{item['excerpt']}"
        )
    return "\n\n---\n\n".join(blocks)


def _sanitize_json_text(text):
    """Strip LLM thinking tags/fences, trailing commas, and unescaped LaTeX backslashes."""
    cleaned = _THINK_RE.sub("", str(text or ""))
    cleaned = _THINK_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"```[A-Za-z0-9_-]*\s*", "", cleaned).replace("```", "").strip()
    cleaned = _VALID_OR_LONE_BS_RE.sub(
        lambda m: m.group(0) if m.group(1) else r"\\",
        cleaned,
    )
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
    return cleaned


def _extract_raw_items(sanitized):
    """Extract question dicts from full JSON or salvage complete items from a truncated array."""
    if not sanitized:
        return []
    decoder = json.JSONDecoder(strict=False)
    for m in re.finditer(r"[\[\{]", sanitized):
        try:
            obj, _ = decoder.raw_decode(sanitized, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and any(isinstance(it, dict) for it in obj):
            return obj
        if isinstance(obj, dict):
            nested = obj.get("questions") or obj.get("items") or obj.get("quiz")
            if isinstance(nested, list) and nested:
                return nested

    salvaged = []
    pos = 0
    while True:
        idx = sanitized.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(sanitized, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(obj, dict) and "question" in obj and ("options" in obj or "choices" in obj):
            salvaged.append(obj)
        pos = end
    return salvaged


def _normalize_options(raw_opts):
    if isinstance(raw_opts, dict):
        pairs = (
            (letter, raw_opts.get(letter) or raw_opts.get(letter.lower()) or "")
            for letter in _LETTERS
        )
    elif isinstance(raw_opts, list) and len(raw_opts) >= 4:
        pairs = zip(_LETTERS, raw_opts[:4])
    else:
        return {}

    options = {}
    for letter, raw_val in pairs:
        val = _OPT_PREFIX_RE.sub("", str(raw_val or "").strip()).strip()
        if val:
            options[letter] = val
    return options if len(options) == 4 else {}


def _normalize_answer(raw_ans, options):
    if isinstance(raw_ans, int) and not isinstance(raw_ans, bool) and 0 <= raw_ans < 4:
        cand = _LETTERS[raw_ans]
        return cand if cand in options else None
    ans = str(raw_ans or "").strip()
    if not ans:
        return None
    upper = ans.upper()
    if upper[0] in _LETTERS and (len(upper) == 1 or upper[1] in (".", ")", ":", " ")):
        return upper[0] if upper[0] in options else None
    stripped = _OPT_PREFIX_RE.sub("", ans).strip().casefold()
    return next((k for k, v in options.items() if v.casefold() == stripped), None)


def parse_quiz_response(raw_text, sampled_chunks, expected_count=5):
    """Parse and validate LLM JSON quiz output and bind verified PDF source metadata."""
    sanitized = _sanitize_json_text(raw_text)
    raw_items = _extract_raw_items(sanitized)
    if not raw_items:
        raise QuizParseError("Không trích xuất được dữ liệu JSON câu hỏi.")

    chunks = list(sampled_chunks or [])
    questions = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        q_text = str(item.get("question") or "").strip()
        options = _normalize_options(item.get("options") or item.get("choices"))
        ans = _normalize_answer(
            item.get("answer") if item.get("answer") is not None else (item.get("correct_answer") or item.get("correct")),
            options,
        )
        if not q_text or not options or not ans:
            continue

        explanation = str(item.get("explanation") or item.get("reason") or "").strip()
        src_meta = {}
        if chunks:
            raw_idx = item.get("source_idx")
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                idx = 1
            idx = max(1, min(len(chunks), idx))
            src_meta = chunks[idx - 1]

        questions.append({
            "question": q_text,
            "options": options,
            "answer": ans,
            "explanation": explanation,
            "doc_id": src_meta.get("doc_id") or "",
            "doc_title": src_meta.get("doc_title") or "Tài liệu môn học",
            "trang": src_meta.get("trang") or "?",
            "url": src_meta.get("url"),
        })
        if len(questions) >= expected_count:
            break

    if not questions:
        raise QuizParseError("Không trích xuất được câu hỏi trắc nghiệm hợp lệ.")
    return questions


def grade_quiz(questions, user_answers):
    """Grade selected answers (dict index->letter) and return score summary."""
    questions = list(questions or [])
    answers = user_answers or {}
    results = []
    correct_count = 0
    for idx, q in enumerate(questions):
        chosen = answers.get(idx)
        is_correct = chosen == q.get("answer")
        if is_correct:
            correct_count += 1
        results.append({
            "index": idx,
            "chosen": chosen,
            "correct": q.get("answer"),
            "is_correct": is_correct,
        })
    total = len(questions)
    pct = round((correct_count / total) * 100.0, 1) if total > 0 else 0.0
    return {
        "score": correct_count,
        "total": total,
        "pct": pct,
        "results": results,
    }
