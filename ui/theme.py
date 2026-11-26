"""Scientific Observatory design system & safe HTML renderers for ELC3008.

Pure functions (clean_snippet, latex_to_plain_text, normalize_latex_for_streamlit,
render_citation_box_html, render_system_card_html, render_turn_telemetry_html,
render_retrieval_panel_html, render_corpus_overview_html, render_doc_card_html)
are unit-testable without a Streamlit runtime. inject_global_css() imports streamlit
lazily so importing this module never requires Streamlit to be installed.
"""

import html
import math
import re
from string import Template

PALETTE = {
    "canvas": "#F7F8FC",
    "sidebar": "#EEF2F7",
    "card": "#FFFFFF",
    "primary": "#2447C7",
    "primary_end": "#1E3A8A",
    "primary_2": "#4338CA",
    "active_pill": "#EEF2FF",
    "callout": "#F8FAFC",
    "ink": "#0F172A",
    "muted": "#475569",
    "border": "rgba(15,23,42,0.12)",
    "link": "#1E40AF",
    "success": "#047857",
    "warning": "#B45309",
    "danger": "#B91C1C",
    "info": "#0F766E",
}

DARK_PALETTE = {
    "canvas": "#0B1120",
    "sidebar": "#0F172A",
    "card": "#162033",
    "primary": "#4F75FF",
    "primary_end": "#3B5BDB",
    "primary_2": "#6366F1",
    "active_pill": "#1E293B",
    "callout": "#111A2C",
    "ink": "#F1F5F9",
    "muted": "#94A3B8",
    "border": "rgba(148,163,184,0.18)",
    "link": "#7DA2FF",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "info": "#2DD4BF",
}

_CATEGORY_COLORS = (
    "#2447C7",
    "#0F766E",
    "#6D28D9",
    "#0369A1",
    "#B45309",
    "#BE185D",
    "#047857",
    "#334155",
)

_PAGE_COMMENT = re.compile(r"<!--\s*page:\s*\d+\s*-->")
_HL_STOPWORDS = frozenset({
    "là", "gì", "và", "của", "cho", "trong", "với", "các", "những", "một",
    "như", "thế", "nào", "ra", "sao", "có", "không", "được", "trên", "dưới",
    "theo", "từ", "đến", "khi", "nếu", "thì", "hay", "hoặc", "về", "hãy",
    "nêu", "so", "sánh", "giải", "thích", "trình", "bày", "ví", "dụ", "phân",
    "biệt", "câu", "hỏi", "phần", "chương", "bài", "mô", "hình", "thuật",
    "toán", "gồm", "thành", "tài", "liệu", "giữa", "khác", "nhau", "sự",
    "cách", "này", "đó", "tại", "bằng", "nhưng", "làm", "nên", "vào",
    "the", "and", "for", "with", "from", "what", "how", "why", "are", "is",
    "in", "of", "to", "on", "vs", "or", "an", "by",
})
_HTML_ENTITY_OR_TAG = re.compile(r"(&[a-zA-Z0-9#]+;|<[^>]+>)")
_WHITESPACE = re.compile(r"\s+")

_FENCED_CODE_RE = re.compile(r"```([^\n`]*)\n([\s\S]*?)```")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_DISPLAY_DOLLAR_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
_DISPLAY_BRACKET_RE = re.compile(r"\\\[([\s\S]+?)\\\]")
_INLINE_PAREN_RE = re.compile(r"\\\(([\s\S]+?)\\\)")
_INLINE_DOLLAR_RE = re.compile(r"(?<!\\)\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)")

_MATH_ENV_NAMES = (
    "aligned", "align", "align\\*", "equation", "equation\\*",
    "gather", "gather\\*", "cases", "split", "array",
    "matrix", "pmatrix", "bmatrix", "multline", "multline\\*",
)
_MATH_BEGIN_RE = re.compile(
    r"\\begin\{(?:" + "|".join(_MATH_ENV_NAMES) + r")\}(?:\{[^}]*\})?"
)
_MATH_END_RE = re.compile(
    r"\\end\{(?:" + "|".join(_MATH_ENV_NAMES) + r")\}"
)
_SIZE_NOISE_RE = re.compile(
    r"\\(?:left\.?|right\.?|bigl|bigr|big|Bigl|Bigr|Big|biggl|biggr|bigg|"
    r"Biggl|Biggr|Bigg|displaystyle|textstyle|limits|nolimits)(?![A-Za-z])"
)
_BARE_SIZE_NOISE_RE = re.compile(
    r"\\(?:left|right)(?=\s*(?:[()\[\]|.]|\\[{}|]))"
    r"|\\(?:displaystyle|textstyle|limits|nolimits)(?![A-Za-z])"
)
_TEXT_MACROS_RE = re.compile(
    r"\\(?:operatorname|boldsymbol|underline|textrm|textit|textbf|"
    r"mathrm|mathbf|mathit|mathcal|mathbb|mathsf|mathtt|text|bm)(?![A-Za-z])"
)
_ACCENT_MACROS = {
    "widehat": "hat",
    "hat": "hat",
    "overline": "bar",
    "bar": "bar",
    "widetilde": "tilde",
    "tilde": "tilde",
    "vec": "vec",
}
_ACCENT_RE = re.compile(
    r"\\(" + "|".join(sorted(_ACCENT_MACROS, key=len, reverse=True)) + r")(?![A-Za-z])"
)
_SQRT_RE = re.compile(r"\\sqrt(?![A-Za-z])")
_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|cfrac|frac)(?![A-Za-z])")
_SCRIPT_RE = re.compile(r"([_^])\{")
_BARE_SCRIPT_RE = re.compile(r"(?<=[A-Za-z0-9\u03b1-\u03c9\u0391-\u03a9)\]])([_^])\{")

_LATEX_TOKEN_MAP = {
    "cdot": "·", "times": "×", "div": "/", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "sim": "~", "propto": "∝",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "gets": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "implies": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "iff": "⇔",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "neg": "¬", "lnot": "¬",
    "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨",
    "infty": "∞", "sum": "Σ", "prod": "Π", "int": "∫",
    "partial": "∂", "nabla": "∇",
    "dots": "...", "ldots": "...", "cdots": "...",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "varsigma": "σ", "tau": "τ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "argmax": "argmax", "argmin": "argmin", "softmax": "softmax",
    "sigmoid": "sigmoid", "log": "log", "ln": "ln", "exp": "exp",
    "sin": "sin", "cos": "cos", "tan": "tan", "min": "min", "max": "max",
    "arg": "arg", "lim": "lim", "det": "det", "dim": "dim",
    "Pr": "Pr", "inf": "inf", "sup": "sup", "mod": "mod",
}
_LATEX_TOKEN_RE = re.compile(
    r"\\(" + "|".join(sorted(_LATEX_TOKEN_MAP, key=len, reverse=True)) + r")(?![A-Za-z])"
)
_SPACING_CMD_RE = re.compile(r"\\(?:qquad|quad|[,;:! ])")
_SIMPLE_SCRIPT_RE = re.compile(r"^[A-Za-z0-9\u03b1-\u03c9\u0391-\u03a9]+$")


def _extract_braced(s, start):
    """Extract balanced `{...}` group starting at or after optional horizontal space."""
    n = len(s)
    idx = start
    while idx < n and s[idx] in " \t":
        idx += 1
    if idx >= n or s[idx] != "{":
        return None
    depth = 0
    i = idx
    while i < n:
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[idx + 1:i], i + 1
        i += 1
    return None


def _is_outer_enclosed(expr):
    expr = expr.strip()
    if len(expr) < 2 or not (expr.startswith("(") and expr.endswith(")")):
        return False
    depth = 0
    for idx, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx < len(expr) - 1:
                return False
    return depth == 0


def _needs_frac_parens(expr, is_denom=False):
    expr = expr.strip()
    if not expr or _is_outer_enclosed(expr):
        return False
    depth = 0
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch == "\\":
            if depth == 0:
                for cmd in ("\\cdot", "\\times", "\\div", "\\pm", "\\mp"):
                    end = i + len(cmd)
                    if expr.startswith(cmd, i) and (end >= n or not expr[end].isalpha()):
                        return True
            i += 2
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            if ch.isspace() or ch in "+-*/·×":
                return True
            if is_denom and ch == "^":
                return True
        i += 1
    return False


def _unwrap_text_macros(s):
    while True:
        m = _TEXT_MACROS_RE.search(s)
        if not m:
            return s
        extracted = _extract_braced(s, m.end())
        if not extracted:
            s = s[:m.start()] + s[m.end():]
            continue
        inner, end_idx = extracted
        inner = _unwrap_text_macros(inner)
        s = s[:m.start()] + inner + s[end_idx:]


def _expand_accents(s):
    while True:
        m = _ACCENT_RE.search(s)
        if not m:
            return s
        extracted = _extract_braced(s, m.end())
        if not extracted:
            s = s[:m.start()] + m.group(1) + s[m.end():]
            continue
        inner, end_idx = extracted
        inner = _expand_accents(inner)
        fn = _ACCENT_MACROS[m.group(1)]
        s = s[:m.start()] + f"{fn}({inner})" + s[end_idx:]


def _expand_sqrts(s):
    while True:
        m = _SQRT_RE.search(s)
        if not m:
            return s
        pos = m.end()
        while pos < len(s) and s[pos] in " \t":
            pos += 1
        degree = None
        if pos < len(s) and s[pos] == "[":
            close_idx = s.find("]", pos + 1)
            if close_idx != -1:
                degree = s[pos + 1:close_idx].strip()
                pos = close_idx + 1
        extracted = _extract_braced(s, pos)
        if not extracted:
            s = s[:m.start()] + "√" + s[m.end():]
            continue
        inner, end_idx = extracted
        inner = _expand_sqrts(inner).strip()
        rep = f"({inner})^(1/{degree})" if degree else f"√({inner})"
        s = s[:m.start()] + rep + s[end_idx:]


def _expand_fracs(s):
    while True:
        m = _FRAC_RE.search(s)
        if not m:
            return s
        first = _extract_braced(s, m.end())
        if not first:
            s = s[:m.start()] + s[m.end():]
            continue
        num, pos = first
        second = _extract_braced(s, pos)
        if not second:
            s = s[:m.start()] + _expand_fracs(num) + s[pos:]
            continue
        den, end_idx = second
        num = _expand_fracs(num).strip()
        den = _expand_fracs(den).strip()
        if _needs_frac_parens(num, is_denom=False):
            num = f"({num})"
        if _needs_frac_parens(den, is_denom=True):
            den = f"({den})"
        s = s[:m.start()] + f"{num} / {den}" + s[end_idx:]


def _simplify_scripts(s, require_prefix=False):
    pattern = _BARE_SCRIPT_RE if require_prefix else _SCRIPT_RE
    out = []
    pos = 0
    while True:
        m = pattern.search(s, pos)
        if not m:
            out.append(s[pos:])
            break
        out.append(s[pos:m.start()])
        op = m.group(1)
        extracted = _extract_braced(s, m.end() - 1)
        if not extracted:
            out.append(s[m.start():m.end()])
            pos = m.end()
            continue
        inner, end_idx = extracted
        inner = _simplify_scripts(inner, require_prefix=False).strip()
        if _SIMPLE_SCRIPT_RE.fullmatch(inner):
            out.append(f"{op}{inner}")
        else:
            out.append(f"{op}({inner})")
        pos = end_idx
    return "".join(out)


def _convert_math_expr(expr, inline=False):
    s = str(expr)
    s = _MATH_BEGIN_RE.sub("", s)
    s = _MATH_END_RE.sub("", s)
    s = re.sub(r"\\\\(?:\[[^\]]*\])?", "; " if inline else "\n", s)
    s = s.replace(r"\&", "&")
    s = re.sub(r"(?<!\\)&", " ", s)
    s = _SIZE_NOISE_RE.sub("", s)
    s = _unwrap_text_macros(s)
    s = _expand_accents(s)
    s = _expand_sqrts(s)
    s = _expand_fracs(s)
    s = s.replace(r"\{", "{").replace(r"\}", "}").replace(r"\|", "||")
    s = s.replace(r"\%", "%").replace(r"\_", "_")
    s = _SPACING_CMD_RE.sub(" ", s)
    s = _LATEX_TOKEN_RE.sub(lambda m: _LATEX_TOKEN_MAP[m.group(1)], s)
    s = _simplify_scripts(s, require_prefix=False)
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in s.splitlines()]
    return "\n".join(line for line in lines if line) if not inline else " ".join(lines).strip()


def latex_to_plain_text(text):
    """Convert delimited and leaked bare LaTeX math into readable plain text while
    preserving non-math code blocks, inline code without backslashes, and currency."""
    if not text:
        return ""
    s = str(text)
    placeholders = []

    def _stash(value):
        token = f"\x00CODE{len(placeholders)}\x00"
        placeholders.append(value)
        return token

    def _handle_fenced(match):
        lang = match.group(1).strip().lower()
        body = match.group(2)
        if lang in ("math", "latex", "tex"):
            return _convert_math_expr(body, inline=False)
        return _stash(match.group(0))

    def _handle_inline_code(match):
        if "\\" in match.group(1):
            return match.group(0)
        return _stash(match.group(0))

    s = _FENCED_CODE_RE.sub(_handle_fenced, s)
    s = _INLINE_CODE_RE.sub(_handle_inline_code, s)
    s = _HTML_BR_RE.sub("\n", s)
    s = _DISPLAY_DOLLAR_RE.sub(lambda m: _convert_math_expr(m.group(1), inline=False), s)
    s = _DISPLAY_BRACKET_RE.sub(lambda m: _convert_math_expr(m.group(1), inline=False), s)
    s = _INLINE_PAREN_RE.sub(lambda m: _convert_math_expr(m.group(1), inline=True), s)
    s = _INLINE_DOLLAR_RE.sub(lambda m: _convert_math_expr(m.group(1), inline=True), s)

    # Fallback for leaked bare LaTeX commands outside math delimiters.
    s = _BARE_SIZE_NOISE_RE.sub("", s)
    s = _unwrap_text_macros(s)
    s = _expand_accents(s)
    s = _expand_sqrts(s)
    s = _expand_fracs(s)
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = _LATEX_TOKEN_RE.sub(lambda m: _LATEX_TOKEN_MAP[m.group(1)], s)
    s = _simplify_scripts(s, require_prefix=True)

    for idx, original in enumerate(placeholders):
        s = s.replace(f"\x00CODE{idx}\x00", original)
    return s


_INLINE_LATEX_SPACED_RE = re.compile(r"(?<!\\)\$[ \t]*([^$\n]*\\[^$\n]*?)[ \t]*\$(?!\d)")
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_BR_LEAD_RE = re.compile(r"^[ \t]*(?:<br\s*/?>[ \t]*)+", re.IGNORECASE)
_HTML_BR_TRAIL_RE = re.compile(r"(?:[ \t]*<br\s*/?>)+[ \t]*$", re.IGNORECASE)
_HTML_BR_MID_RE = re.compile(r"[ \t]*(?:<br\s*/?>[ \t]*)+", re.IGNORECASE)


def _replace_html_br(text):
    """Replace HTML <br> tags with Markdown line breaks (or '; ' inside table rows)."""
    if not _HTML_BR_RE.search(text):
        return text
    out_lines = []
    for line in text.split("\n"):
        if not _HTML_BR_RE.search(line):
            out_lines.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            cleaned = re.sub(r"\s*<br\s*/?>\s*", "; ", line, flags=re.IGNORECASE)
            out_lines.append(cleaned)
        else:
            cleaned = _HTML_BR_LEAD_RE.sub("", line)
            cleaned = _HTML_BR_TRAIL_RE.sub("", cleaned)
            cleaned = _HTML_BR_MID_RE.sub("  \n", cleaned)
            out_lines.append(cleaned)
    return "\n".join(out_lines)


def normalize_latex_for_streamlit(text):
    """Normalize LaTeX delimiters to Streamlit KaTeX (`$$...$$` / `$...$`) while
    protecting non-math code blocks, inline code spans, and currency."""
    if not text:
        return ""
    raw = str(text)
    placeholders = []

    def _stash(value):
        token = f"\x00CODE{len(placeholders)}\x00"
        placeholders.append(value)
        return token

    def _handle_fenced(match):
        lang = match.group(1).strip().lower()
        body = match.group(2).strip()
        if lang in ("math", "latex", "tex"):
            if body.startswith("$$") and body.endswith("$$") and len(body) >= 4:
                body = body[2:-2].strip()
            elif body.startswith(r"\[") and body.endswith(r"\]") and len(body) >= 4:
                body = body[2:-2].strip()
            return f"\n\n$$\n{body}\n$$\n\n"
        return _stash(match.group(0))

    s = _FENCED_CODE_RE.sub(_handle_fenced, raw)
    s = _INLINE_CODE_RE.sub(lambda m: _stash(m.group(0)), s)
    s = _DISPLAY_BRACKET_RE.sub(lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", s)
    s = _DISPLAY_DOLLAR_RE.sub(lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", s)
    s = _INLINE_PAREN_RE.sub(lambda m: f"${m.group(1).strip()}$", s)
    s = _INLINE_LATEX_SPACED_RE.sub(lambda m: f"${m.group(1).strip()}$", s)
    s = _replace_html_br(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    if not raw.startswith("\n"):
        s = s.lstrip("\n")
    if not raw.endswith("\n"):
        s = s.rstrip("\n")
    for idx, original in enumerate(placeholders):
        s = s.replace(f"\x00CODE{idx}\x00", original)
    return s


def clean_snippet(text, max_len=240):
    """Strip indexed `<!-- page: N -->` markers, convert LaTeX to plain text,
    collapse whitespace, truncate, then HTML-escape for custom cards."""
    if not text:
        return ""
    text = _PAGE_COMMENT.sub(" ", str(text))
    text = latex_to_plain_text(text)
    text = _WHITESPACE.sub(" ", text).strip()
    truncated = len(text) > max_len
    if truncated:
        text = text[:max_len].rstrip()
    escaped = html.escape(text)
    return (escaped + "…") if truncated else escaped


def highlight_query_terms(escaped_html, query):
    """Highlight query keywords inside already-escaped HTML without altering tags or entities."""
    if not escaped_html or not query:
        return escaped_html
    raw_tokens = re.findall(r"[^\W_]+", str(query), flags=re.UNICODE)
    seen = set()
    terms = []
    for tok in raw_tokens:
        if len(tok) < 2 or tok.isdigit():
            continue
        key = tok.casefold()
        if key in _HL_STOPWORDS or key in seen:
            continue
        seen.add(key)
        terms.append(tok)
    if not terms:
        return escaped_html
    terms.sort(key=len, reverse=True)
    terms = terms[:8]
    pattern = re.compile(
        r"(?<![^\W_])(" + "|".join(re.escape(t) for t in terms) + r")(?![^\W_])",
        flags=re.IGNORECASE | re.UNICODE,
    )
    parts = _HTML_ENTITY_OR_TAG.split(escaped_html)
    for idx in range(0, len(parts), 2):
        if parts[idx]:
            parts[idx] = pattern.sub(r'<mark class="rag-hl">\1</mark>', parts[idx])
    return "".join(parts)


def _fmt_score(score, decimals=3):
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)) and math.isfinite(score):
        return f"{float(score):.{decimals}f}"
    return None


def _score_pct(score):
    """Map a rerank relevance score or logit onto [0, 100] for visual meter width."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    val = float(score)
    if not math.isfinite(val):
        return None
    if 0.0 <= val <= 1.0:
        return round(val * 100.0, 1)
    clamped = max(-20.0, min(20.0, val))
    return round(100.0 / (1.0 + math.exp(-clamped)), 1)


def _safe_url_label(label_esc, url):
    if isinstance(url, str) and url.startswith("https://"):
        url_esc = html.escape(url, quote=True)
        return (
            f'<a href="{url_esc}" target="_blank" '
            f'rel="noopener noreferrer">{label_esc}</a>'
        )
    return label_esc


def _fmt_pages(page_min, page_max):
    if page_min is None and page_max is None:
        return "—"
    if page_min is None:
        return str(page_max)
    if page_max is None or page_min == page_max:
        return str(page_min)
    return f"{page_min}-{page_max}"


def _category_color(category):
    try:
        from ui.catalog import CATEGORY_ORDER
        idx = list(CATEGORY_ORDER).index(category)
        return _CATEGORY_COLORS[idx % len(_CATEGORY_COLORS)]
    except Exception:
        return _CATEGORY_COLORS[-1]


def render_page_header_html(title, subtitle="", chips=None):
    """Render unified page header with escaped title, subtitle, and status chips."""
    title_esc = html.escape(str(title or ""))
    sub_esc = html.escape(str(subtitle or ""))
    chips_html = ""
    if chips:
        chip_parts = []
        for chip in chips:
            if isinstance(chip, tuple):
                lbl, tone = chip[0], (chip[1] if len(chip) > 1 else "neutral")
            elif isinstance(chip, dict):
                lbl, tone = chip.get("label", ""), chip.get("tone", "neutral")
            else:
                lbl, tone = chip, "neutral"
            lbl_str = str(lbl or "").strip()
            if not lbl_str:
                continue
            tone_safe = tone if tone in ("primary", "success", "warn", "info", "neutral") else "neutral"
            chip_parts.append(
                f'<span class="rag-hdr-chip is-{tone_safe} rag-mono">{html.escape(lbl_str)}</span>'
            )
        if chip_parts:
            chips_html = f'<div class="rag-page-chips">{"".join(chip_parts)}</div>'
    sub_html = f'<div class="rag-page-sub">{sub_esc}</div>' if sub_esc else ""
    return (
        f'<div class="rag-page-header">'
        f'<div class="rag-page-header-main">'
        f'<div class="rag-page-title">{title_esc}</div>'
        f'{sub_html}'
        f'</div>'
        f'{chips_html}'
        f'</div>'
    )


def render_citation_box_html(sources, query=""):
    """Render cited chunks with visual confidence meter and collapsible verbatim snippet."""
    if not sources:
        return ""
    items = []
    for i, s in enumerate(sources, 1):
        doc_id = str(s.get("doc_id") or "")
        title = str(s.get("doc_title") or doc_id or "Tài liệu")
        title_esc = html.escape(title)
        doc_id_esc = html.escape(doc_id)
        trang = str(s.get("trang") or "").strip()
        label = _safe_url_label(title_esc, s.get("url"))

        raw_init = s.get("initial_rank")
        raw_final = s.get("final_rank")
        has_init = isinstance(raw_init, int) and not isinstance(raw_init, bool) and raw_init > 0
        has_final = isinstance(raw_final, int) and not isinstance(raw_final, bool) and raw_final > 0
        rank_num = raw_final if has_final else i
        rank_flow_html = ""
        if has_init and has_final:
            raw_delta = s.get("rank_delta")
            delta = (raw_delta if isinstance(raw_delta, int) and not isinstance(raw_delta, bool)
                     else raw_init - raw_final)
            rank_flow_html = (
                f'<span class="rag-rank-group">'
                f'<span class="rag-rank-flow rag-mono">#{raw_init} → #{raw_final}</span>'
                f'{_delta_badge(delta)}'
                f'</span>'
            )

        raw_score = s.get("rerank_score")
        score = _fmt_score(raw_score, decimals=3)
        pct = _score_pct(raw_score)
        meter_html = ""
        if pct is not None:
            meter_html = (
                f'<span class="rag-cite-meter" title="Độ liên quan">'
                f'<span class="rag-cite-fill" style="width:{pct:.1f}%;"></span>'
                f'</span>'
            )
        score_html = (
            f'<span class="rag-cite-score">↗ {score}</span>' if score else ""
        )
        score_wrap_html = (
            f'<span class="rag-cite-score-wrap">{meter_html}{score_html}</span>'
            if (meter_html or score_html) else ""
        )
        page_html = (
            f'<span class="rag-cite-page">Trang {html.escape(trang)}</span>'
            if trang else ""
        )
        snippet = highlight_query_terms(
            clean_snippet(s.get("snippet") or "", max_len=240),
            query,
        )
        snippet_html = (
            f'<details class="rag-cite-details">'
            f'<summary>Trích đoạn</summary>'
            f'<div class="rag-cite-snippet">{snippet}</div></details>'
            if snippet else ""
        )
        items.append(
            f'<li class="rag-cite-item">'
            f'<div class="rag-cite-row">'
            f'<span class="rag-rank-badge rag-mono">#{rank_num}</span>'
            f'<span class="rag-cite-title">{label}</span>'
            f'<span class="rag-cite-id">{doc_id_esc}</span>'
            f'{page_html}{rank_flow_html}{score_wrap_html}'
            f'</div>'
            f'{snippet_html}'
            f'</li>'
        )
    return (
        f'<div class="rag-citation" style="background:{PALETTE["callout"]};'
        f'border:1px solid {PALETTE["border"]};border-radius:12px;'
        f'padding:14px 16px;margin-top:4px;">'
        f'<div class="rag-citation-head" style="color:{PALETTE["link"]};'
        f'font-weight:600;font-size:0.78rem;letter-spacing:0.06em;margin-bottom:8px;">'
        f'<span class="rag-mono">NGUỒN TRÍCH DẪN · {len(sources)}</span></div>'
        f'<ol class="rag-cite-list" style="list-style:none;margin:0;padding:0;">'
        f'{"".join(items)}</ol></div>'
    )


def render_pipeline_stepper_html(
    phase=1,
    retrieve_ms=None,
    rerank_ms=None,
    selected_count=None,
    total_candidates=None,
    rerank_enabled=True,
):
    """Render a 3-step visual RAG pipeline progress stepper during live generation."""
    def _num(val):
        return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)

    if int(phase or 1) <= 1:
        steps = [
            ("is-active", '<span class="rag-step-dot"></span>', "1. Quét Hybrid BM25 + FAISS"),
            ("is-wait", '<span class="rag-step-idx">2</span>', "Cross-Encoder Rerank"),
            ("is-wait", '<span class="rag-step-idx">3</span>', "Tổng hợp câu trả lời"),
        ]
    else:
        ret_txt = f" ({float(retrieve_ms):.0f}ms)" if _num(retrieve_ms) else ""
        if rerank_enabled:
            rer_txt = f" {float(rerank_ms):.0f}ms" if _num(rerank_ms) else ""
            if _num(selected_count) and _num(total_candidates):
                cnt_txt = f" · {int(selected_count)}/{int(total_candidates)}"
            elif _num(selected_count):
                cnt_txt = f" · Top-{int(selected_count)}"
            else:
                cnt_txt = ""
            step2_lbl = f"2. Rerank{rer_txt}{cnt_txt}"
        else:
            cnt_txt = f" {int(selected_count)} đoạn" if _num(selected_count) else " Stage-1"
            step2_lbl = f"2. Chọn{cnt_txt}"
        steps = [
            ("is-done", '<span class="rag-step-check">✓</span>', f"1. Hybrid{ret_txt}"),
            ("is-done", '<span class="rag-step-check">✓</span>', step2_lbl),
            ("is-active is-synth", '<span class="rag-step-dot"></span>', "3. Đang viết câu trả lời..."),
        ]

    parts = []
    for idx, (cls_state, marker, label) in enumerate(steps):
        if idx > 0:
            parts.append('<span class="rag-step-sep" aria-hidden="true">→</span>')
        parts.append(f'<span class="rag-step {cls_state}">{marker}<span>{html.escape(label)}</span></span>')
    return f'<div class="rag-stepper rag-mono" role="status" aria-live="polite">{"".join(parts)}</div>'


def render_flashcards_html(questions):
    """Render a 2-column grid of pure HTML/CSS 3D flip flashcards for quiz review."""
    questions = list(questions or [])
    if not questions:
        return ""
    total = len(questions)
    cards = []
    for idx, q in enumerate(questions, 1):
        q_txt = html.escape(latex_to_plain_text(str(q.get("question") or "")))
        opts = q.get("options") or {}
        ans = str(q.get("answer") or "A").strip().upper()
        if ans not in opts:
            ans = next(iter(opts.keys()), "A")
        ans_esc = html.escape(ans)
        ans_body = html.escape(latex_to_plain_text(str(opts.get(ans) or "")))
        exp_raw = str(q.get("explanation") or "").strip()
        exp_html = (
            f'<div class="rag-fc-exp">{html.escape(latex_to_plain_text(exp_raw))}</div>'
            if exp_raw else ""
        )

        opt_items = []
        for letter in ("A", "B", "C", "D"):
            if letter in opts:
                opt_val = html.escape(latex_to_plain_text(str(opts[letter])))
                opt_items.append(
                    f'<div class="rag-fc-opt"><span class="rag-fc-opt-k rag-mono">{letter}.</span> {opt_val}</div>'
                )
        opts_html = f'<div class="rag-fc-opts">{"".join(opt_items)}</div>' if opt_items else ""

        title_esc = html.escape(str(q.get("doc_title") or "Tài liệu"))
        trang_esc = html.escape(str(q.get("trang") or "?"))
        label = f"{title_esc} · Trang {trang_esc}"
        url = q.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            url_esc = html.escape(url, quote=True)
            foot_html = (
                f'<a class="rag-source-pill" href="{url_esc}" target="_blank" '
                f'rel="noopener noreferrer"><span class="rag-source-name">{label}</span></a>'
            )
        else:
            foot_html = f'<span class="rag-source-pill"><span class="rag-source-name">{label}</span></span>'

        cards.append(
            f'<div class="rag-fc-wrap">'
            f'<label class="rag-fc-card">'
            f'<input type="checkbox" class="rag-fc-toggle" aria-label="Lật thẻ câu {idx}">'
            f'<div class="rag-fc-inner">'
            f'<div class="rag-fc-front">'
            f'<div class="rag-fc-top">'
            f'<span class="rag-rank-badge rag-mono">Thẻ {idx}/{total}</span>'
            f'<span class="rag-fc-hint rag-mono">Nhấn để lật ↻</span>'
            f'</div>'
            f'<div class="rag-fc-body">'
            f'<div class="rag-fc-q">{q_txt}</div>'
            f'{opts_html}'
            f'</div>'
            f'</div>'
            f'<div class="rag-fc-back">'
            f'<div class="rag-fc-top">'
            f'<span class="rag-quiz-status is-ok rag-mono">ĐÁP ÁN {ans_esc}</span>'
            f'<span class="rag-fc-hint rag-mono">Nhấn để úp lại ↺</span>'
            f'</div>'
            f'<div class="rag-fc-body">'
            f'<div class="rag-fc-ans"><strong>{ans_esc}.</strong> {ans_body}</div>'
            f'{exp_html}'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</label>'
            f'<div class="rag-fc-foot">{foot_html}</div>'
            f'</div>'
        )
    return f'<div class="rag-fc-grid">{"".join(cards)}</div>'


def render_source_pills_html(sources, max_pills=4):
    """Render compact scannable citation summary pills right above the detail expander."""
    sources = list(sources or [])
    if not sources:
        return ""
    pills = []
    for i, s in enumerate(sources[:max_pills], 1):
        doc_id = str(s.get("doc_id") or "")
        title = str(s.get("doc_title") or doc_id or "Tài liệu").strip()
        if len(title) > 34:
            title = title[:33].rstrip() + "…"
        raw_final = s.get("final_rank")
        rank_num = (
            raw_final
            if isinstance(raw_final, int) and not isinstance(raw_final, bool) and raw_final > 0
            else i
        )
        trang = str(s.get("trang") or "").strip()
        page_txt = f" · tr.{html.escape(trang)}" if trang else ""
        inner = (
            f'<span class="rag-source-rank rag-mono">#{rank_num}</span>'
            f'<span class="rag-source-name">{html.escape(title)}</span>'
            f'<span class="rag-source-pg rag-mono">{page_txt}</span>'
        )
        url = s.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            url_esc = html.escape(url, quote=True)
            pills.append(
                f'<a class="rag-source-pill" href="{url_esc}" target="_blank" '
                f'rel="noopener noreferrer">{inner}</a>'
            )
        else:
            pills.append(f'<span class="rag-source-pill">{inner}</span>')
    if len(sources) > max_pills:
        pills.append(
            f'<span class="rag-source-pill is-more rag-mono">+{len(sources) - max_pills} nguồn</span>'
        )
    return f'<div class="rag-source-pills">{"".join(pills)}</div>'


def render_system_card_html(cfg):
    """Render sidebar RAG pipeline architecture console without ever leaking API keys."""
    cfg = cfg or {}
    provider = str(cfg.get("rerank_provider") or "off").strip().lower()
    rerank_active = provider not in ("off", "none", "")
    rerank_model = str(cfg.get("rerank_model") or "").strip()
    top_k = html.escape(str(cfg.get("top_k") or 5))
    retrieve_n = html.escape(str(cfg.get("retrieve_n") or 20))
    embed_raw = str(cfg.get("embed_model") or "MiniLM-L12-v2")
    embed_short = html.escape(embed_raw.split("/")[-1])
    embed_dim = int(cfg.get("embed_dim") or 384)

    dot_cls = "rag-sys-dot is-on" if rerank_active else "rag-sys-dot is-off"
    if rerank_active:
        rerank_label = f'Rerank {html.escape(provider.upper())}'
        rerank_sub = html.escape(rerank_model or "active")
    else:
        rerank_label = "Rerank TẮT"
        rerank_sub = "Chỉ Stage-1 Hybrid"

    return (
        f'<div class="rag-sys-card">'
        f'<div class="rag-sys-head">'
        f'<span class="{dot_cls}"></span>'
        f'<span class="rag-sys-title">{rerank_label}</span>'
        f'</div>'
        f'<div class="rag-sys-sub rag-mono">{rerank_sub}</div>'
        f'<div class="rag-sys-grid">'
        f'<div class="rag-sys-row"><span>Stage-1</span><span class="rag-mono">0.3·BM25 + 0.7·FAISS ({retrieve_n})</span></div>'
        f'<div class="rag-sys-row"><span>Stage-2</span><span class="rag-mono">Top-{top_k}</span></div>'
        f'<div class="rag-sys-row"><span>LLM</span><span class="rag-mono">Groq</span></div>'
        f'<div class="rag-sys-row"><span>Embed</span><span class="rag-mono">{embed_dim}d · {embed_short}</span></div>'
        f'</div></div>'
    )


def render_turn_telemetry_html(telemetry):
    """Render compact per-turn scientific telemetry strip below an assistant response."""
    if not telemetry:
        return ""
    chips = []
    latency = telemetry.get("latency_s")
    if isinstance(latency, (int, float)) and math.isfinite(latency):
        chips.append(f'<span class="rag-tel-chip">⏱ {float(latency):.2f}s</span>')

    ret_ms = telemetry.get("retrieve_ms")
    if isinstance(ret_ms, (int, float)) and math.isfinite(ret_ms):
        chips.append(f'<span class="rag-tel-chip">Stage-1 {float(ret_ms):.0f}ms</span>')

    if telemetry.get("rerank_enabled"):
        rer_ms = telemetry.get("rerank_ms")
        if isinstance(rer_ms, (int, float)) and math.isfinite(rer_ms):
            chips.append(f'<span class="rag-tel-chip">Rerank {float(rer_ms):.0f}ms</span>')
    elif telemetry.get("rerank_fallback"):
        chips.append('<span class="rag-tel-chip rag-tel-warn">Rerank lỗi · Stage-1</span>')
    else:
        chips.append('<span class="rag-tel-chip">Rerank TẮT</span>')

    sel_cnt = telemetry.get("selected_count")
    tot_cnt = telemetry.get("total_candidates")
    if sel_cnt is not None and tot_cnt is not None:
        chips.append(f'<span class="rag-tel-chip">{int(sel_cnt)}/{int(tot_cnt)} chunks</span>')

    model = str(telemetry.get("model") or "").strip()
    if model:
        slot = str(telemetry.get("key_slot") or "").strip()
        slot_part = f" · key {html.escape(slot)}" if slot else ""
        chips.append(f'<span class="rag-tel-chip rag-tel-model">{html.escape(model)}{slot_part}</span>')

    if telemetry.get("fell_back"):
        cfg_model = html.escape(str(telemetry.get("configured_model") or "model chính"))
        chips.append(
            f'<span class="rag-tel-chip rag-tel-warn">'
            f'⇄ Dự phòng (hết hạn mức {cfg_model})</span>'
        )

    return f'<div class="rag-telemetry-strip rag-mono">{"".join(chips)}</div>'


def _delta_badge(delta):
    d = int(delta or 0)
    if d > 0:
        return f'<span class="rag-delta-up">▲ +{d}</span>'
    if d < 0:
        return f'<span class="rag-delta-down">▼ {d}</span>'
    return '<span class="rag-delta-flat">● 0</span>'


def render_retrieval_summary_html(res):
    """Render Stage-1/Stage-2 latency summary cards and split bar."""
    if not res:
        return ""
    ret_ms = float(res.get("retrieve_ms") or 0.0)
    rer_ms = float(res.get("rerank_ms") or 0.0)
    rer_on = bool(res.get("rerank_enabled"))
    rer_fallback = bool(res.get("rerank_fallback"))
    provider = str(res.get("rerank_provider") or "").strip().lower()
    candidates = list(res.get("candidates") or [])
    selected = list(res.get("selected") or [])
    total_cand = int(res.get("total_candidates") if res.get("total_candidates") is not None else len(candidates))

    sum_ms = ret_ms + (rer_ms if rer_on else 0.0)
    if sum_ms <= 0.0 or not rer_on or rer_ms <= 0.0:
        ret_w, rer_w = (100.0, 0.0)
    else:
        ret_w = max(8.0, min(92.0, (ret_ms / sum_ms) * 100.0))
        rer_w = round(100.0 - ret_w, 1)
        ret_w = round(ret_w, 1)

    rerank_val = f"{rer_ms:.0f} ms" if rer_on else ("LỖI" if rer_fallback else "TẮT")
    if rer_on:
        rerank_sub = "Local Cross-Encoder" if provider == "local" else "Cohere Cross-Encoder"
    elif rer_fallback:
        rerank_sub = "Rerank lỗi — giữ thứ tự Stage-1"
    else:
        rerank_sub = "Bỏ qua rerank"
    return (
        f'<div class="rag-lab-summary">'
        f'<div class="rag-lab-stat">'
        f'<div class="rag-lab-k">STAGE-1 HYBRID</div>'
        f'<div class="rag-lab-v rag-mono">{ret_ms:.0f} ms</div>'
        f'<div class="rag-lab-s">BM25 0.30 + FAISS 0.70</div>'
        f'</div>'
        f'<div class="rag-lab-stat">'
        f'<div class="rag-lab-k">STAGE-2 RERANK</div>'
        f'<div class="rag-lab-v rag-mono">{rerank_val}</div>'
        f'<div class="rag-lab-s">{rerank_sub}</div>'
        f'</div>'
        f'<div class="rag-lab-stat">'
        f'<div class="rag-lab-k">MẬT ĐỘ LỌC</div>'
        f'<div class="rag-lab-v rag-mono">{len(selected)} / {total_cand}</div>'
        f'<div class="rag-lab-s">Top-K giữ lại / ứng viên</div>'
        f'</div>'
        f'</div>'
        f'<div class="rag-latency-bar" title="Tỷ lệ thời gian Stage-1 vs Stage-2">'
        f'<span class="rag-lat-ret" style="width:{ret_w:.1f}%;"></span>'
        f'<span class="rag-lat-rer" style="width:{rer_w:.1f}%;"></span>'
        f'</div>'
    )


def render_retrieval_stage2_html(res):
    """Render Top-K reranked chunk cards for Stage 2 tab or combined panel."""
    selected = list((res or {}).get("selected") or [])
    if not selected:
        return '<div class="rag-lab-empty">Không tìm thấy ứng viên phù hợp cho truy vấn này.</div>'

    query = str((res or {}).get("question") or "")
    topk_cards = []
    for pos, item in enumerate(selected, 1):
        final_rank = int(item.get("final_rank") or pos)
        init_rank = int(item.get("initial_rank") or pos)
        title_esc = html.escape(str(item.get("doc_title") or item.get("doc_id") or "Tài liệu"))
        doc_id_esc = html.escape(str(item.get("doc_id") or "?"))
        trang_esc = html.escape(str(item.get("trang") or "?"))
        label_html = _safe_url_label(title_esc, item.get("url"))
        delta_html = _delta_badge(item.get("rank_delta"))

        raw_score = item.get("rerank_score")
        score_txt = _fmt_score(raw_score, decimals=4)
        pct = _score_pct(raw_score)
        score_block = ""
        if score_txt is not None and pct is not None:
            score_block = (
                f'<span class="rag-rerank-score-wrap">'
                f'<span class="rag-cite-meter"><span class="rag-cite-fill" style="width:{pct:.1f}%;"></span></span>'
                f'<span class="rag-mono rag-rerank-val">{score_txt}</span>'
                f'</span>'
            )

        snip = highlight_query_terms(
            clean_snippet(item.get("snippet") or "", max_len=220),
            query,
        )
        snip_html = f'<div class="rag-rerank-snip">{snip}</div>' if snip else ""
        topk_cards.append(
            f'<div class="rag-rerank-card">'
            f'<div class="rag-rerank-head">'
            f'<span class="rag-rank-badge rag-mono">#{final_rank}</span>'
            f'<span class="rag-rerank-title">{label_html}</span>'
            f'<span class="rag-cite-id">{doc_id_esc}</span>'
            f'<span class="rag-cite-page">Trang {trang_esc}</span>'
            f'<span class="rag-rank-group">'
            f'<span class="rag-rank-flow rag-mono">#{init_rank} → #{final_rank}</span>'
            f'{delta_html}'
            f'</span>'
            f'{score_block}'
            f'</div>'
            f'{snip_html}'
            f'</div>'
        )
    return (
        f'<div class="rag-lab-stack">'
        f'<div class="rag-lab-subhead">TOP-{len(selected)} SAU RERANK</div>'
        f'{"".join(topk_cards)}'
        f'</div>'
    )


def render_retrieval_stage1_html(res, collapsible=True):
    """Render Stage-1 Hybrid candidate comparison table."""
    candidates = list((res or {}).get("candidates") or [])
    selected = list((res or {}).get("selected") or [])
    total_cand = int(
        (res or {}).get("total_candidates")
        if (res or {}).get("total_candidates") is not None else len(candidates)
    )
    if not candidates:
        return '<div class="rag-lab-empty">Không có ứng viên Stage-1.</div>'

    selected_keys = {
        (it.get("doc_id"), it.get("trang"), it.get("initial_rank"))
        for it in selected
    }
    ordered_cands = sorted(candidates, key=lambda c: int(c.get("initial_rank") or 9999))
    rows_html = []
    for c in ordered_cands:
        init_r = int(c.get("initial_rank") or 0)
        fin_r = int(c.get("final_rank") or init_r)
        is_kept = (c.get("doc_id"), c.get("trang"), c.get("initial_rank")) in selected_keys
        row_cls = "rag-s1-kept" if is_kept else ""
        status_pill = (
            '<span class="rag-badge rag-badge-ok">TOP-K</span>'
            if is_kept else '<span class="rag-badge rag-badge-dim">—</span>'
        )
        c_title = html.escape(str(c.get("doc_title") or c.get("doc_id") or "?"))
        c_trang = html.escape(str(c.get("trang") or "?"))
        c_score = _fmt_score(c.get("rerank_score"), decimals=4) or "—"
        rows_html.append(
            f'<tr class="{row_cls}">'
            f'<td class="rag-mono">#{init_r}</td>'
            f'<td class="rag-mono">#{fin_r}</td>'
            f'<td class="rag-mono">{_delta_badge(c.get("rank_delta"))}</td>'
            f'<td class="rag-mono">{c_score}</td>'
            f'<td>{c_title}</td>'
            f'<td class="rag-mono">{c_trang}</td>'
            f'<td>{status_pill}</td>'
            f'</tr>'
        )

    table_wrap = (
        f'<div class="rag-stage1-wrap">'
        f'<table class="rag-stage1-table">'
        f'<thead><tr>'
        f'<th>S1</th><th>S2</th><th>Δ Hạng</th><th>Điểm</th>'
        f'<th>Tài liệu</th><th>Trang</th><th>Chọn</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    if collapsible:
        return (
            f'<details class="rag-stage1-details">'
            f'<summary>Toàn bộ {total_cand} ứng viên stage-1 (BM25 + FAISS)</summary>'
            f'{table_wrap}</details>'
        )
    return (
        f'<div class="rag-stage1-box">'
        f'<div class="rag-lab-subhead">TOÀN BỘ {total_cand} ỨNG VIÊN STAGE-1 (BM25 + FAISS)</div>'
        f'{table_wrap}</div>'
    )


def render_retrieval_empty_html(embed_dim=384):
    """Render balanced placeholder panel before running a retrieval probe."""
    dim = int(embed_dim or 384)
    svg_icon = (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="11" cy="11" r="7"></circle>'
        '<path d="m21 21-4.3-4.3"></path>'
        '<path d="M11 8v6"></path><path d="M8 11h6"></path>'
        '</svg>'
    )
    return (
        f'<div class="rag-lab-placeholder">'
        f'<div class="rag-lab-placeholder-icon">{svg_icon}</div>'
        f'<div class="rag-lab-placeholder-title">Sẵn sàng kiểm tra truy hồi 2 tầng</div>'
        f'<div class="rag-lab-placeholder-sub">'
        f'Chọn một mẫu chuẩn hoặc nhập câu hỏi ở phía trên để xem Stage-1 Hybrid '
        f'(BM25 + FAISS {dim}d) lấy ứng viên và Stage-2 Cross-Encoder sắp xếp lại thứ hạng.'
        f'</div>'
        f'<div class="rag-hero-flow">'
        f'<span class="rag-flow-step">① Câu hỏi</span>'
        f'<span class="rag-flow-arrow" aria-hidden="true">→</span>'
        f'<span class="rag-flow-step">② Hybrid BM25 + FAISS</span>'
        f'<span class="rag-flow-arrow" aria-hidden="true">→</span>'
        f'<span class="rag-flow-step">③ Cross-Encoder Rerank</span>'
        f'</div>'
        f'<div class="rag-lab-placeholder-pills">'
        f'<span class="rag-hero-pill">0 token LLM tiêu tốn</span>'
        f'<span class="rag-hero-pill">So sánh hạng S1 → S2</span>'
        f'<span class="rag-hero-pill">Xem trích đoạn gốc</span>'
        f'</div>'
        f'</div>'
    )


def render_retrieval_panel_html(res):
    """Render 2-stage retrieval diagnostics: latency split, Top-K cards, Stage-1 table."""
    if not res:
        return '<div class="rag-lab-empty">Chưa có dữ liệu truy hồi.</div>'

    summary_html = render_retrieval_summary_html(res)
    selected = list(res.get("selected") or [])
    if not selected:
        return (
            f'<div class="rag-lab-panel">{summary_html}'
            f'<div class="rag-lab-empty">Không tìm thấy ứng viên phù hợp cho truy vấn này.</div>'
            f'</div>'
        )

    stage2_html = render_retrieval_stage2_html(res)
    stage1_html = render_retrieval_stage1_html(res, collapsible=True)
    return (
        f'<div class="rag-lab-panel">'
        f'{summary_html}'
        f'{stage2_html}'
        f'{stage1_html}'
        f'</div>'
    )


def render_corpus_overview_html(rows, embed_dim=384,
                                 embed_model="paraphrase-multilingual-MiniLM-L12-v2"):
    """Render 4 corpus-level KPI cards and 8-category chunk distribution bar."""
    rows = list(rows or [])
    embed_dim = int(embed_dim or 384)
    total_docs = len(rows)
    total_chunks = sum(int(r.get("chunk_count") or 0) for r in rows)
    en_docs = sum(1 for r in rows if r.get("lang") == "EN")
    vi_docs = total_docs - en_docs
    drive_docs = sum(
        1 for r in rows
        if isinstance(r.get("url"), str) and r["url"].startswith("https://")
    )

    by_cat = {}
    for r in rows:
        cat = str(r.get("category") or "Khác")
        entry = by_cat.setdefault(cat, {"docs": 0, "chunks": 0})
        entry["docs"] += 1
        entry["chunks"] += int(r.get("chunk_count") or 0)

    try:
        from ui.catalog import CATEGORY_ORDER
        ordered_cats = [c for c in CATEGORY_ORDER if c in by_cat]
        ordered_cats += [c for c in by_cat if c not in ordered_cats]
    except Exception:
        ordered_cats = list(by_cat.keys())

    kpis_html = (
        f'<div class="rag-kpi-grid">'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">HỌC LIỆU CHỈ MỤC</div>'
        f'<div class="rag-kpi-val rag-mono">{total_docs}</div>'
        f'<div class="rag-kpi-sub">{vi_docs} Tiếng Việt · {en_docs} English</div>'
        f'</div>'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">TỔNG VECTOR CHUNKS</div>'
        f'<div class="rag-kpi-val rag-mono">{total_chunks:,}</div>'
        f'<div class="rag-kpi-sub">1,500 ký tự · overlap 300</div>'
        f'</div>'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">KHÔNG GIAN EMBEDDING</div>'
        f'<div class="rag-kpi-val rag-mono">{embed_dim}d</div>'
        f'<div class="rag-kpi-sub rag-mono" title="{html.escape(str(embed_model))}">{html.escape(str(embed_model).split("/")[-1])}</div>'
        f'</div>'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">NGUỒN PDF GỐC</div>'
        f'<div class="rag-kpi-val rag-mono">{drive_docs} / {total_docs}</div>'
        f'<div class="rag-kpi-sub">{len(ordered_cats)} nhóm học phần</div>'
        f'</div>'
        f'</div>'
    )

    segs = []
    legends = []
    for idx, cat in enumerate(ordered_cats):
        color = _CATEGORY_COLORS[idx % len(_CATEGORY_COLORS)]
        stats = by_cat[cat]
        pct = (stats["chunks"] / total_chunks * 100.0) if total_chunks > 0 else 0.0
        cat_esc = html.escape(cat)
        if pct > 0:
            segs.append(
                f'<span class="rag-dist-seg" '
                f'style="width:{pct:.2f}%;background:{color};" '
                f'title="{cat_esc}: {stats["docs"]} tài liệu · {stats["chunks"]:,} chunks ({pct:.1f}%)"></span>'
            )
        legends.append(
            f'<span class="rag-dist-leg">'
            f'<span class="rag-dist-dot" style="background:{color};"></span>'
            f'<span>{cat_esc}</span>'
            f'<span class="rag-mono rag-dist-num">{stats["docs"]} · {pct:.1f}%</span>'
            f'</span>'
        )

    dist_html = ""
    if ordered_cats:
        dist_html = (
            f'<div class="rag-dist-box">'
            f'<div class="rag-dist-head">'
            f'<span>PHÂN BỐ CHUNK THEO HỌC PHẦN</span>'
            f'<span class="rag-mono">{len(ordered_cats)} nhóm</span>'
            f'</div>'
            f'<div class="rag-dist-bar">{"".join(segs)}</div>'
            f'<div class="rag-dist-legend">{"".join(legends)}</div>'
            f'</div>'
        )

    return f'<div class="rag-corpus-overview">{kpis_html}{dist_html}</div>'


def render_doc_card_html(row, max_chunks, total_chunks):
    """Render metadata body and relative density bar inside a document card."""
    title_esc = html.escape(str(row.get("title") or row.get("doc_id") or "Tài liệu"))
    doc_id_esc = html.escape(str(row.get("doc_id") or ""))
    cat_raw = str(row.get("category") or "")
    cat_esc = html.escape(cat_raw)
    cat_color = _category_color(cat_raw)
    lang = str(row.get("lang") or "VI").upper()
    lang_esc = html.escape(lang)
    lang_cls = "rag-badge rag-badge-en" if lang == "EN" else "rag-badge rag-badge-vi"

    url = row.get("url")
    has_url = isinstance(url, str) and url.startswith("https://")
    if has_url and "drive.google.com" in url:
        src_badge = '<span class="rag-badge rag-badge-ok">PDF Drive</span>'
    elif has_url:
        src_badge = '<span class="rag-badge rag-badge-ok">PDF Gốc</span>'
    else:
        src_badge = '<span class="rag-badge rag-badge-dim">Vector Index</span>'

    chunks = int(row.get("chunk_count") or 0)
    pages = html.escape(_fmt_pages(row.get("page_min"), row.get("page_max")))
    share_pct = (chunks / total_chunks * 100.0) if total_chunks > 0 else 0.0
    rel_pct = (chunks / max_chunks * 100.0) if max_chunks > 0 else 0.0
    rel_pct = max(2.0, min(100.0, rel_pct)) if chunks > 0 else 0.0

    cat_style = (
        f'color:{cat_color};'
        f'background:color-mix(in srgb, {cat_color} 14%, transparent);'
        f'border-color:color-mix(in srgb, {cat_color} 34%, transparent);'
    )

    return (
        f'<div class="rag-doc-card" style="border-left:3px solid {cat_color};padding-left:10px;">'
        f'<div class="rag-doc-badges">'
        f'<span class="rag-badge rag-badge-cat" style="{cat_style}">{cat_esc}</span>'
        f'<span class="{lang_cls}">{lang_esc}</span>'
        f'{src_badge}'
        f'</div>'
        f'<div class="rag-doc-title" title="{title_esc}">{title_esc}</div>'
        f'<div class="rag-doc-id rag-mono">{doc_id_esc}</div>'
        f'<div class="rag-doc-stats rag-mono">'
        f'<span>{chunks:,} chunks ({share_pct:.1f}%)</span>'
        f'<span>Trang {pages}</span>'
        f'</div>'
        f'<div class="rag-doc-bar"><span style="width:{rel_pct:.1f}%;background:linear-gradient(90deg, {cat_color}, var(--info));"></span></div>'
        f'</div>'
    )


_CSS_TEMPLATE = Template("""
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root { color-scheme: light;
  --bg:$canvas; --surface:$card; --surface-2:$active_pill;
  --border:$border; --text:$ink; --muted:$muted;
  --primary:$primary; --primary-2:$primary_2; --primary-end:$primary_end;
  --success:$success; --warn:$warning; --danger:$danger; --info:$info;
  --radius:14px;
  --canvas:$canvas; --sidebar:$sidebar; --card:$card; --ink:$ink;
  --link:$link; --active-pill:$active_pill; --callout:$callout; --warning:$warning;
  --s-1:4px; --s-2:8px; --s-3:12px; --s-4:16px; --s-6:24px; --s-8:32px;
  --font-display:"Plus Jakarta Sans","Be Vietnam Pro",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --font-sans:"Be Vietnam Pro",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
  --ease:cubic-bezier(0.16,1,0.3,1); --speed:170ms; }

html, body, [data-testid="stApp"] {
  background:
    radial-gradient(ellipse 80% 38% at 50% -10%, rgba(36,71,199,0.055), transparent 72%),
    var(--bg);
  color:var(--text); font-family:var(--font-sans) !important; overflow-anchor:auto;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility; }

[data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th,
[data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"],
button, input, textarea, [data-baseweb="select"], [data-baseweb="tab"] {
  font-family:var(--font-sans) !important; }

h1, h2, h3, .rag-brand-title, .rag-page-title, .rag-hero-title,
.rag-kpi-val, .rag-lab-v, .rag-doc-title, .rag-lab-placeholder-title {
  font-family:var(--font-display) !important; }

/* Hide default Streamlit chrome while keeping stExpandSidebarButton reachable */
[data-testid="stHeader"] {
  background:linear-gradient(180deg, rgba(247,248,252,0.92) 55%, transparent);
  height:2.75rem; pointer-events:none; }
[data-testid="stHeader"] button,
[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] { pointer-events:auto; }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"] { visibility:hidden; display:none; }
[data-testid="stToolbarActions"] { display:none; }

[data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow-x:hidden; }
[data-testid="stMainBlockContainer"] {
  max-width:1240px; margin:0 auto; padding:2.1rem var(--s-8) var(--s-8); }
[data-testid="stAppViewContainer"]:has([data-testid="stChatInput"]) [data-testid="stMainBlockContainer"] {
  max-width:860px; padding-bottom:6rem; }
[data-testid="stBottomBlockContainer"] {
  max-width:860px; margin:0 auto; padding-left:var(--s-8); padding-right:var(--s-8); }
[data-testid="stElementContainer"]:has(style) { margin:0 !important; padding:0 !important; }

h1, h2, h3 { color:var(--text); letter-spacing:-0.02em; font-weight:700; }
[data-testid="stMarkdownContainer"] { color:var(--text); line-height:1.68; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
.rag-mono, [data-testid="stChatMessage"] code, [data-testid="stChatMessage"] pre {
  font-family:var(--font-mono) !important; font-variant-numeric:tabular-nums; }

a, a:visited { color:var(--link); text-underline-offset:3px;
  transition:color var(--speed) var(--ease); }
a:hover { color:var(--primary-end); }
::selection { background:var(--primary); color:#fff; }

@keyframes rag-rise { from { opacity:0; transform:translate3d(0,4px,0); }
  to { opacity:1; transform:translate3d(0,0,0); } }
@keyframes rag-fade { from { opacity:0; } to { opacity:1; } }
@keyframes rag-pulse { 0%, 100% { opacity:0.5; transform:scale(0.95); }
  50% { opacity:1; transform:scale(1.1); } }
@keyframes rag-flow { 0%, 100% { opacity:0.55; transform:translateX(0); }
  50% { opacity:1; transform:translateX(2px); } }

* { scrollbar-width:thin; scrollbar-color:rgba(15,23,42,0.16) var(--sidebar); }
::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-track { background:var(--sidebar); }
::-webkit-scrollbar-thumb { background:rgba(15,23,42,0.16); border-radius:999px;
  border:2px solid var(--sidebar); }
::-webkit-scrollbar-thumb:hover { background:var(--primary); }
::-webkit-scrollbar-corner { background:var(--sidebar); }

/* Sidebar (Light Notion/Perplexity style) */
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--border);
  color:var(--text); }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-top:0.75rem; padding-bottom:0.75rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); font-size:0.72rem;
  letter-spacing:0.05em; text-transform:uppercase; font-weight:600; margin-top:2px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:0.28rem; }
[data-testid="stSidebar"] [data-testid="stDivider"] { margin:0.28rem 0; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background:transparent; color:var(--muted); border:1px solid transparent;
  border-radius:10px; text-align:left; justify-content:flex-start;
  font-weight:500; font-size:0.88rem; line-height:1.3; padding:0.36rem 0.72rem; min-height:2.0rem;
  transition:border-color var(--speed) var(--ease), color var(--speed) var(--ease),
    background-color var(--speed) var(--ease); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--border); color:var(--text); background:rgba(255,255,255,0.85); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus:not(:focus-visible) {
  border-color:transparent; background:transparent; box-shadow:none; }

/* Keep white text/icon on blue primary buttons (CTA & Inspector form submit), separate active nav vs conv styles */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-primaryFormSubmit"] *,
[data-testid="stSidebar"] [class*="st-key-new_conv"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [class*="st-key-new_conv"] [data-testid="stBaseButton-primary"] * {
  color:#FFFFFF !important; }

[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"] {
  background:#FFFFFF !important;
  border:1px solid rgba(36,71,199,0.24);
  box-shadow:inset 3px 0 0 var(--primary), 0 1px 2px rgba(15,23,42,0.05);
  color:var(--text) !important; font-weight:600; font-size:0.88rem; justify-content:flex-start;
  text-align:left; transform:none;
  padding:0.36rem 0.72rem 0.36rem 0.85rem; min-height:2.0rem; }
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"] * {
  color:var(--text) !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"]:hover {
  background:#FFFFFF !important; filter:none; transform:none;
  box-shadow:inset 3px 0 0 var(--primary-end), 0 2px 6px rgba(15,23,42,0.06); }
[data-testid="stSidebar"] [class*="st-key-conv_"] [data-testid="stBaseButton-primary"] {
  background:rgba(36,71,199,0.10) !important;
  border:1px solid rgba(36,71,199,0.20); box-shadow:none;
  color:var(--link) !important; font-weight:600; font-size:0.86rem; justify-content:flex-start;
  text-align:left; transform:none;
  padding:0.34rem 0.72rem; min-height:2.0rem; }
[data-testid="stSidebar"] [class*="st-key-conv_"] [data-testid="stBaseButton-primary"] * {
  color:var(--link) !important; }
[data-testid="stSidebar"] [class*="st-key-conv_"] [data-testid="stBaseButton-primary"]:hover {
  background:rgba(36,71,199,0.15) !important; filter:none; transform:none; box-shadow:none; }

/* Ellipsis for long conversation titles in sidebar */
[data-testid="stSidebar"] [class*="st-key-conv_"] button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [class*="st-key-conv_"] button [data-testid="stMarkdownContainer"] p {
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100%; margin:0; text-align:left; }

/* Accessible delete button & subtle clear-all action */
[data-testid="stSidebar"] [class*="st-key-del_"] button {
  padding:0.25rem; min-height:2.0rem; min-width:2.0rem; justify-content:center;
  opacity:0.78; border:1px solid transparent; background:transparent; color:var(--muted); }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [class*="st-key-del_"] button {
  opacity:0.95; }
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover {
  opacity:1; color:var(--danger) !important; background:rgba(185,28,28,0.08);
  border-color:rgba(185,28,28,0.22); }
[data-testid="stSidebar"] [class*="st-key-clear_all"] button {
  background:transparent; border:1px solid transparent; color:var(--muted);
  font-size:0.78rem; padding:0.2rem 0.5rem; min-height:1.85rem; }
[data-testid="stSidebar"] [class*="st-key-clear_all"] button:hover {
  color:var(--danger); background:rgba(185,28,28,0.06); border-color:transparent; }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] {
  margin-top:6px !important; margin-bottom:2px !important; }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] [data-testid="stBaseButton-secondary"] {
  background:#FFFFFF; border:1px solid var(--border); color:var(--text);
  border-radius:999px; font-size:0.8rem; font-weight:600;
  padding:0.28rem 0.75rem; min-height:1.95rem;
  justify-content:center; text-align:center;
  box-shadow:0 1px 2px rgba(15,23,42,0.04); }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:var(--primary); }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:rgba(36,71,199,0.38); background:var(--surface-2); color:var(--link); }

[data-testid="stSidebar"] [class*="st-key-new_conv"] {
  margin-top:8px !important; }
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.rag-brand) {
  margin-bottom:8px !important; }
.rag-brand { background:#FFFFFF; border:1px solid var(--border); border-radius:12px;
  padding:9px 11px; box-shadow:0 1px 3px rgba(15,23,42,0.04); margin-bottom:2px; }
.rag-brand-top { display:flex; align-items:center; gap:9px; }
.rag-brand-logo { width:32px; height:32px; border-radius:9px; flex-shrink:0;
  background:linear-gradient(135deg,var(--primary),var(--primary-2));
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 2px 6px rgba(36,71,199,0.28); }
.rag-brand-kicker { display:inline-block; font-family:var(--font-mono); font-size:0.72rem;
  font-weight:600; letter-spacing:0.06em; color:var(--link);
  background:rgba(36,71,199,0.09); border:1px solid rgba(36,71,199,0.22);
  border-radius:999px; padding:1px 7px; margin-bottom:2px; }
.rag-brand-title { font-size:1.05rem; font-weight:700; letter-spacing:-0.02em; line-height:1.22;
  color:var(--text); }
.rag-brand-sub { color:var(--muted); font-size:0.75rem; margin-top:5px; line-height:1.35; }

.rag-sys-card { background:#FFFFFF; border:1px solid var(--border); border-radius:12px;
  padding:8px 10px; box-shadow:0 1px 2px rgba(15,23,42,0.03); }
.rag-sys-head { display:flex; align-items:center; gap:7px; font-size:0.8rem; font-weight:600;
  color:var(--text); }
.rag-sys-dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }
.rag-sys-dot.is-on { background:var(--success); box-shadow:0 0 0 3px rgba(4,120,87,0.16);
  animation:rag-pulse 2.2s ease-in-out infinite; }
.rag-sys-dot.is-off { background:var(--warn); }
.rag-sys-sub { color:var(--muted); font-size:0.73rem; margin:1px 0 6px 15px; }
.rag-sys-grid { display:flex; flex-direction:column; gap:2px; border-top:1px solid var(--border);
  padding-top:6px; }
.rag-sys-row { display:flex; justify-content:space-between; align-items:baseline;
  gap:6px; font-size:0.72rem; color:var(--muted); }
.rag-sys-row > span:first-child { white-space:nowrap; flex-shrink:0; }
.rag-sys-row .rag-mono { color:var(--text); text-align:right; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }

/* Unified Page Header */
.rag-page-header { display:flex; align-items:flex-end; justify-content:space-between;
  flex-wrap:wrap; gap:var(--s-3) var(--s-4);
  background:#FFFFFF; border:1px solid var(--border); border-left:4px solid var(--primary);
  border-radius:14px; padding:16px 20px; margin-bottom:var(--s-4);
  box-shadow:0 1px 3px rgba(15,23,42,0.04); }
.rag-page-header-main { min-width:220px; flex:1 1 auto; }
.rag-page-title { font-size:1.35rem; font-weight:700; letter-spacing:-0.025em;
  color:var(--text); line-height:1.25; text-wrap:balance; }
.rag-page-sub { color:var(--muted); font-size:0.86rem; margin-top:4px; line-height:1.45; }
.rag-page-chips { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.rag-hdr-chip { font-size:0.75rem; font-weight:600; padding:3px 10px; border-radius:999px;
  border:1px solid var(--border); background:#FFFFFF; color:var(--muted); }
.rag-hdr-chip.is-primary { color:var(--link); background:var(--surface-2);
  border-color:rgba(36,71,199,0.22); }
.rag-hdr-chip.is-success { color:var(--success); background:#ECFDF5;
  border-color:rgba(4,120,87,0.24); }
.rag-hdr-chip.is-warn { color:var(--warn); background:#FFFBEB;
  border-color:rgba(180,83,9,0.24); }
.rag-hdr-chip.is-info { color:var(--info); background:#F0FDFA;
  border-color:rgba(15,118,110,0.24); }

/* Chat Messages — Clean card for assistant, compact right-aligned pill for user */
[data-testid="stChatMessage"] { animation:rag-rise 180ms var(--ease) backwards;
  background:#FFFFFF; border:1px solid var(--border); border-radius:16px;
  padding:16px 20px; margin-bottom:10px;
  box-shadow:0 1px 2px rgba(15,23,42,0.04), 0 4px 12px rgba(15,23,42,0.02);
  will-change:opacity,transform; }
[data-testid="stChatMessage"]:not(:last-of-type) { animation:none; will-change:auto; }
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"]:not(:has([aria-label="Chat message from user"])) [data-testid^="stChatMessageAvatar"] {
  background:linear-gradient(135deg,var(--primary),var(--primary-2)) !important;
  border:none !important; color:#FFFFFF !important; border-radius:10px !important;
  box-shadow:0 2px 6px rgba(36,71,199,0.28); }
[data-testid="stChatMessageAvatarAssistant"] *,
[data-testid="stChatMessage"]:not(:has([aria-label="Chat message from user"])) [data-testid^="stChatMessageAvatar"] * {
  color:#FFFFFF !important; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] {
  background:linear-gradient(135deg,#E0E7FF,#C7D2FE) !important;
  color:var(--primary-end) !important;
  border:1px solid rgba(36,71,199,0.28) !important; border-radius:10px !important; }
[data-testid="stChatMessageAvatarUser"] *,
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] * {
  color:var(--primary-end) !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  font-size:0.95rem; line-height:1.7; }
[data-testid="stChatMessage"] table { width:100%; border-collapse:collapse;
  margin:10px 0; font-size:0.88rem; }
[data-testid="stChatMessage"] th { background:#F8FAFC; color:var(--text);
  font-weight:600; text-align:left; padding:8px 10px; border:1px solid var(--border); }
[data-testid="stChatMessage"] td { padding:8px 10px; border:1px solid var(--border);
  vertical-align:top; }
[data-testid="stChatMessage"] code { font-family:var(--font-mono); font-size:0.86em;
  background:#F1F5F9; color:var(--text); padding:1px 6px; border-radius:5px;
  border:1px solid var(--border); }
[data-testid="stChatMessage"] pre { background:#F8FAFC; border:1px solid var(--border);
  border-radius:10px; padding:12px 14px; overflow-x:auto; }
[data-testid="stChatMessage"] pre code { background:transparent; border:none; padding:0; }
[data-testid="stChatMessage"] blockquote { border-left:3px solid var(--primary);
  padding:4px 0 4px 12px; margin:8px 0; color:var(--muted); }
[data-testid="stLayoutWrapper"]:has([aria-label="Chat message from user"]),
[data-testid="stLayoutWrapper"]:has([data-testid="stChatMessageAvatarUser"]) {
  display:flex !important; justify-content:flex-end !important; width:100% !important; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  margin-left:auto !important; margin-right:0 !important;
  width:fit-content !important; min-width:140px; max-width:min(82%, 640px) !important;
  flex-direction:row-reverse !important; justify-content:flex-start !important; gap:12px;
  background:linear-gradient(180deg, #EEF2FF 0%, #E6ECFF 100%);
  border:1px solid rgba(36,71,199,0.22);
  border-radius:18px 18px 4px 18px; padding:11px 16px;
  box-shadow:0 1px 3px rgba(15,23,42,0.04); }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stVerticalBlock"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stElementContainer"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdown"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  flex:0 1 auto !important; width:fit-content !important; max-width:100% !important; min-width:0; }

/* Single-layer suggestion cards on Empty Chat & compact inline actions */
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] {
  background:#FFFFFF; border:1px solid var(--border); border-radius:14px;
  padding:16px 18px; min-height:104px; height:auto !important; text-align:left;
  justify-content:flex-start; align-items:flex-start;
  box-shadow:0 1px 2px rgba(15,23,42,0.04); white-space:normal !important; line-height:1.5; }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"]:hover {
  background:#FFFFFF; border-color:rgba(36,71,199,0.38); color:var(--text);
  box-shadow:inset 3px 0 0 var(--primary), 0 8px 20px rgba(15,23,42,0.07);
  transform:translate3d(0,-1px,0); }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] > div,
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"],
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
  white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important; max-width:100%; }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
  text-align:left; color:var(--muted); font-size:0.86rem; font-weight:400; margin:0; line-height:1.55; }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] strong {
  color:var(--text); font-weight:600; font-size:0.95rem; }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:var(--primary); }
[class*="st-key-sug_0"] [data-testid="stBaseButton-secondary"] {
  border-top:3px solid #2447C7; }
[class*="st-key-sug_0"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:#2447C7; }
[class*="st-key-sug_0"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:rgba(36,71,199,0.44);
  box-shadow:inset 3px 0 0 #2447C7, 0 8px 22px rgba(36,71,199,0.10); }
[class*="st-key-sug_1"] [data-testid="stBaseButton-secondary"] {
  border-top:3px solid #0F766E; }
[class*="st-key-sug_1"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:#0F766E; }
[class*="st-key-sug_1"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:rgba(15,118,110,0.44);
  box-shadow:inset 3px 0 0 #0F766E, 0 8px 22px rgba(15,118,110,0.10); }
[class*="st-key-sug_2"] [data-testid="stBaseButton-secondary"] {
  border-top:3px solid #6D28D9; }
[class*="st-key-sug_2"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:#6D28D9; }
[class*="st-key-sug_2"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:rgba(109,40,217,0.44);
  box-shadow:inset 3px 0 0 #6D28D9, 0 8px 22px rgba(109,40,217,0.10); }
[class*="st-key-sug_3"] [data-testid="stBaseButton-secondary"] {
  border-top:3px solid #B45309; }
[class*="st-key-sug_3"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:#B45309; }
[class*="st-key-sug_3"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:rgba(180,83,9,0.44);
  box-shadow:inset 3px 0 0 #B45309, 0 8px 22px rgba(180,83,9,0.10); }

[class*="st-key-regen"] [data-testid="stBaseButton-secondary"],
[class*="st-key-inspect_last"] [data-testid="stBaseButton-secondary"] {
  font-size:0.82rem; padding:0.35rem 0.85rem; margin-top:-2px; margin-bottom:8px;
  color:var(--muted); }

[data-testid="stForm"] { border:none !important; padding:0 !important; background:transparent !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF;
  border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:0 1px 3px rgba(15,23,42,0.04);
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease),
    transform var(--speed) var(--ease); }
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color:rgba(36,71,199,0.28);
  box-shadow:0 6px 20px rgba(15,23,42,0.06); }
[data-testid="stVerticalBlockBorderWrapper"]:has(.rag-doc-card):hover {
  transform:translate3d(0,-2px,0); }

[data-testid="stBaseButton-secondary"] { background:#FFFFFF; color:var(--text);
  border:1px solid rgba(15,23,42,0.14); border-radius:10px; font-weight:500;
  transition:border-color var(--speed) var(--ease), color var(--speed) var(--ease),
    background-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease),
    transform var(--speed) var(--ease); }
[data-testid="stBaseButton-secondary"]:hover { border-color:rgba(36,71,199,0.38);
  color:var(--link); background:var(--surface-2); }

[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
  background:linear-gradient(135deg,var(--primary),var(--primary-end));
  border:1px solid rgba(30,58,138,0.28); color:#FFFFFF; border-radius:10px; font-weight:600;
  box-shadow:0 1px 2px rgba(36,71,199,0.22);
  transition:transform var(--speed) var(--ease), box-shadow var(--speed) var(--ease),
    filter var(--speed) var(--ease); }
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  transform:translate3d(0,-1px,0); filter:brightness(1.04);
  box-shadow:0 6px 16px rgba(36,71,199,0.25); }
[data-testid="stBaseButton-primary"]:active,
[data-testid="stBaseButton-primaryFormSubmit"]:active,
[data-testid="stBaseButton-secondary"]:active { transform:translate3d(0,0,0); filter:brightness(0.97); }

button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible,
textarea:focus-visible, select:focus-visible, [role="tab"]:focus-visible {
  outline:2px solid var(--primary); outline-offset:2px; }

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea { background:#FFFFFF; color:var(--text);
  border:1px solid rgba(15,23,42,0.14); border-radius:10px; padding:0.55rem 0.85rem;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus { border-color:var(--primary);
  box-shadow:0 0 0 3px rgba(36,71,199,0.14); }

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background:#FFFFFF; color:var(--text); border:1px solid rgba(15,23,42,0.14); border-radius:10px; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
  border-color:var(--primary); box-shadow:0 0 0 3px rgba(36,71,199,0.14); }

[data-testid="stChatInput"] { background:#FFFFFF; border:1px solid rgba(15,23,42,0.15);
  border-radius:16px; box-shadow:0 4px 20px rgba(15,23,42,0.07);
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stChatInput"]:focus-within { border-color:var(--primary);
  box-shadow:0 0 0 3px rgba(36,71,199,0.14), 0 6px 24px rgba(15,23,42,0.08); }
[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"] { color:var(--text); }

[data-testid="stSpinner"] { color:var(--primary); }

[data-testid="stPills"],
[data-testid="stPills"] div,
[data-testid="stButtonGroup"],
[data-testid="stButtonGroup"] div,
[role="radiogroup"],
[role="radiogroup"] > div {
  flex-wrap:wrap !important; overflow:visible !important;
  max-width:100% !important; height:auto !important; }
[data-testid="stPills"] [role="radiogroup"],
[data-testid="stButtonGroup"] [role="radiogroup"],
[data-testid="stPills"] > div,
[data-testid="stButtonGroup"] > div {
  display:flex !important; flex-wrap:wrap !important; gap:6px !important; }
[data-testid="stPills"] button,
[data-testid="stButtonGroup"] button { border-radius:999px; font-size:0.8rem;
  white-space:nowrap; flex:0 0 auto !important; width:auto !important;
  transition:background-color var(--speed) var(--ease), color var(--speed) var(--ease),
    border-color var(--speed) var(--ease); }

[data-testid="stTabs"], [role="tabpanel"], [role="tabpanel"] > div {
  max-width:100%; min-width:0; width:100%; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--border); }
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:8px 14px;
  font-weight:600; font-size:0.86rem; color:var(--muted); }
[data-testid="stTabs"] [aria-selected="true"] { color:var(--link); }

[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stAlert"] [kind="error"] { border-left:3px solid var(--danger); }
[data-testid="stAlert"] [kind="warning"] { border-left:3px solid var(--warn); }
[data-testid="stAlert"] [kind="success"] { border-left:3px solid var(--success); }
[data-testid="stAlert"] [kind="info"] { border-left:3px solid var(--info); }

[data-testid="stExpander"] > details { border:1px solid var(--border); border-radius:12px;
  background:#FFFFFF; box-shadow:0 1px 2px rgba(15,23,42,0.03); }
[data-testid="stExpander"] > details > summary { transition:color var(--speed) var(--ease);
  color:var(--text); font-weight:500; }
[data-testid="stExpander"] > details > summary:hover { color:var(--link); }
[data-testid="stChatMessage"] [data-testid="stExpander"] { margin-top:8px; }
.katex-display { overflow-x:auto; overflow-y:hidden; padding:4px 0; }
[data-testid="stExpander"] [data-testid="stVerticalBlock"],
[data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {
  animation:rag-fade 180ms var(--ease) backwards; }
[data-testid="stDivider"] hr { border-color:var(--border); }

/* Conversation Empty Hero & Pipeline Strip */
.rag-hero { text-align:center; padding:2rem 1.75rem 1.65rem; margin-bottom:1.15rem;
  background:radial-gradient(circle at 1px 1px, rgba(36,71,199,0.085) 1px, transparent 0),
    linear-gradient(180deg, #FFFFFF 0%, #F7F8FC 100%);
  background-size:20px 20px, 100% 100%;
  border:1px solid var(--border); border-radius:20px;
  box-shadow:0 2px 12px rgba(15,23,42,0.04);
  animation:rag-fade 220ms var(--ease) backwards; }
.rag-hero-badge { display:inline-block; font-family:var(--font-mono); font-size:0.75rem;
  font-weight:600; letter-spacing:0.06em; color:var(--link);
  background:rgba(36,71,199,0.08); border:1px solid rgba(36,71,199,0.2);
  border-radius:999px; padding:4px 12px; margin-bottom:10px; }
.rag-hero-title { font-size:2rem; font-weight:800; letter-spacing:-0.03em; margin:0;
  color:var(--text); text-wrap:balance;
  background:linear-gradient(135deg, #0F172A 18%, var(--primary) 62%, var(--info) 100%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.rag-hero-sub { color:var(--muted); margin-top:8px; font-size:0.95rem; max-width:580px;
  margin-left:auto; margin-right:auto; line-height:1.55; }
.rag-hero-pills { display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-top:14px; }
.rag-hero-pill { font-family:var(--font-mono); font-size:0.75rem; color:var(--text);
  background:#FFFFFF; border:1px solid var(--border); border-radius:999px; padding:4px 12px;
  box-shadow:0 1px 2px rgba(15,23,42,0.03); }
.rag-hero-flow { display:flex; flex-wrap:wrap; justify-content:center; align-items:center;
  gap:6px; margin-top:14px; padding-top:12px; border-top:1px dashed var(--border); }
.rag-flow-step { font-family:var(--font-mono); font-size:0.75rem; font-weight:500;
  color:var(--muted); background:#FFFFFF; border:1px solid var(--border);
  border-radius:8px; padding:3px 10px; }
.rag-flow-arrow { color:var(--primary); font-weight:700; font-size:0.85rem;
  display:inline-block; animation:rag-flow 1.8s ease-in-out infinite; }
.rag-flow-arrow:last-of-type { animation-delay:0.4s; }

/* Live Retrieval Indicator & 3-Step Pipeline Stepper */
.rag-stepper { display:flex; flex-wrap:wrap; align-items:center; gap:6px;
  padding:8px 12px; border-radius:12px; background:#F8FAFC;
  border:1px solid var(--border); font-size:0.78rem; margin-bottom:6px; }
.rag-step { display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
  border-radius:999px; border:1px solid var(--border); font-weight:500; line-height:1.35; }
.rag-step.is-done { background:#ECFDF5; color:var(--success);
  border-color:rgba(4,120,87,0.24); font-weight:600; }
.rag-step.is-active { background:#EEF2FF; color:var(--link);
  border-color:rgba(36,71,199,0.28); font-weight:600; }
.rag-step.is-active.is-synth { background:#F0FDFA; color:var(--info);
  border-color:rgba(15,118,110,0.28); }
.rag-step.is-wait { background:#FFFFFF; color:var(--muted); opacity:0.78; }
.rag-step-dot { width:7px; height:7px; border-radius:50%; background:var(--primary);
  box-shadow:0 0 0 3px rgba(36,71,199,0.2);
  animation:rag-pulse 1.15s ease-in-out infinite; flex-shrink:0; }
.rag-step.is-synth .rag-step-dot { background:var(--info);
  box-shadow:0 0 0 3px rgba(15,118,110,0.2); }
.rag-step-check { font-weight:700; }
.rag-step-idx { display:inline-flex; align-items:center; justify-content:center;
  width:15px; height:15px; border-radius:50%; background:rgba(15,23,42,0.08);
  font-size:0.68rem; font-weight:700; }
.rag-step-sep { color:var(--muted); font-weight:600; font-size:0.78rem; }
mark.rag-hl { background:#FEF08A; color:#0F172A; padding:0 3px; border-radius:4px;
  font-weight:600; box-shadow:0 0 0 1px rgba(202,138,4,0.22); }
.rag-stage-live { display:flex; align-items:center; gap:10px; min-height:38px;
  padding:8px 14px; border-radius:10px; background:#EEF2FF; border:1px solid rgba(36,71,199,0.2);
  color:var(--link); font-size:0.83rem; font-family:var(--font-mono); }
.rag-stage-live.is-synth { background:#F0FDFA; border-color:rgba(15,118,110,0.24);
  color:var(--info); }
.rag-stage-dot { width:8px; height:8px; border-radius:50%; background:var(--primary);
  box-shadow:0 0 0 3px rgba(36,71,199,0.2);
  animation:rag-pulse 1.15s ease-in-out infinite; flex-shrink:0; }
.rag-stage-live.is-synth .rag-stage-dot { background:var(--info);
  box-shadow:0 0 0 3px rgba(15,118,110,0.2); }

/* Telemetry Strip & Compact Source Pills */
.rag-telemetry-strip { display:flex; flex-wrap:wrap; gap:8px 6px; margin:10px 0 14px; }
.rag-tel-chip { font-size:0.75rem; color:var(--muted); background:#FFFFFF;
  border:1px solid var(--border); border-radius:999px; padding:2px 9px; }
.rag-tel-model { color:var(--link); border-color:rgba(36,71,199,0.22); background:#EEF2FF; }
.rag-tel-warn { color:var(--warn); border-color:rgba(180,83,9,0.32); background:#FFFBEB; }

.rag-source-pills { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:0 0 10px; }
.rag-source-pill, a.rag-source-pill { display:inline-flex; align-items:center; gap:5px;
  max-width:100%; font-size:0.76rem; color:var(--text); background:#F0FDFA;
  border:1px solid rgba(15,118,110,0.24); border-radius:999px; padding:3px 10px;
  text-decoration:none; transition:border-color var(--speed) var(--ease),
    background-color var(--speed) var(--ease); }
a.rag-source-pill:hover { border-color:var(--info); background:#CCFBF1; color:var(--text); }
.rag-source-rank { font-weight:700; font-size:0.72rem; color:var(--info); }
.rag-source-name { font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  max-width:300px; }
.rag-source-pg { color:var(--muted); font-size:0.72rem; white-space:nowrap; }
.rag-source-pill.is-more { background:#FFFFFF; border-color:var(--border); color:var(--muted); }

/* Citations */
.rag-citation { animation:rag-fade 180ms var(--ease) backwards; }
.rag-cite-item { padding:12px 0; border-bottom:1px dashed var(--border); font-size:0.89rem; }
.rag-cite-item:last-child { border-bottom:none; padding-bottom:2px; }
.rag-cite-row { display:flex; flex-wrap:wrap; align-items:center; gap:6px 8px; }
.rag-rank-group { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }
.rag-cite-title, .rag-cite-title a { color:var(--link); font-weight:600; text-decoration:none; }
.rag-cite-title a:hover { text-decoration:underline; }
.rag-cite-id { font-family:var(--font-mono); color:var(--muted); font-size:0.75rem;
  background:#F1F5F9; border:1px solid var(--border); border-radius:6px; padding:1px 6px; }
.rag-cite-page { color:var(--muted); font-size:0.78rem; white-space:nowrap; }
.rag-cite-score-wrap { display:inline-flex; align-items:center; gap:6px; margin-left:auto; white-space:nowrap; }
.rag-cite-meter { display:inline-block; width:56px; height:5px; background:rgba(15,23,42,0.09);
  border-radius:999px; overflow:hidden; vertical-align:middle; margin-left:auto; }
.rag-cite-score-wrap .rag-cite-meter { margin-left:0; }
.rag-cite-fill { display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg, var(--primary), var(--info)); }
.rag-cite-score { font-family:var(--font-mono); background:#F0FDFA; color:var(--info);
  border:1px solid rgba(15,118,110,0.22); border-radius:999px; padding:1px 8px;
  font-size:0.75rem; font-weight:600; }

[data-testid="stExpander"] .rag-cite-details,
.rag-cite-details { margin-top:6px; border:none !important; border-radius:0 !important;
  background:transparent !important; box-shadow:none !important; padding:0 !important; }
[data-testid="stExpander"] .rag-cite-details > summary,
.rag-cite-details > summary { color:var(--muted); font-size:0.78rem; cursor:pointer;
  list-style:none; display:inline-flex; align-items:center; gap:4px;
  padding:2px 0 !important; margin:0; background:transparent !important;
  border:none !important; border-radius:0 !important; min-height:0 !important;
  transition:color var(--speed) var(--ease); }
.rag-cite-details > summary::-webkit-details-marker { display:none; }
.rag-cite-details > summary::before { content:"▸"; font-size:0.75rem; line-height:1; }
.rag-cite-details[open] > summary::before { content:"▾"; }
.rag-cite-details > summary:hover { color:var(--text); }
.rag-cite-details[open] .rag-cite-snippet { animation:rag-fade 160ms var(--ease); }
.rag-cite-snippet { color:#334155; font-size:0.83rem; margin:8px 0 6px; line-height:1.55;
  border-left:2px solid var(--primary); padding-left:10px; }

/* Inspector / Retrieval Lab */
.rag-lab-panel, .rag-lab-stack { display:flex; flex-direction:column; gap:10px; }
.rag-lab-summary { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px;
  margin-bottom:8px; }
.rag-lab-stat { background:#FFFFFF; border:1px solid var(--border); border-radius:12px;
  padding:12px 14px; box-shadow:0 1px 2px rgba(15,23,42,0.03); border-top:3px solid var(--primary); }
.rag-lab-stat:nth-child(1) { border-top-color:var(--info); }
.rag-lab-stat:nth-child(2) { border-top-color:var(--primary); }
.rag-lab-stat:nth-child(3) { border-top-color:var(--success); }
.rag-lab-k { font-family:var(--font-mono); font-size:0.75rem; color:var(--muted); letter-spacing:0.04em; }
.rag-lab-v { font-size:1.22rem; font-weight:700; color:var(--text); margin:3px 0; white-space:nowrap; }
.rag-lab-s { font-size:0.76rem; color:var(--muted); }
.rag-latency-bar { display:flex; height:6px; border-radius:999px; overflow:hidden;
  background:#E2E8F0; gap:2px; margin-bottom:12px; }
.rag-lat-ret { display:block; height:100%; background:var(--info); }
.rag-lat-rer { display:block; height:100%; background:var(--primary); }
.rag-lab-subhead { font-family:var(--font-mono); font-size:0.75rem; letter-spacing:0.05em;
  color:var(--link); font-weight:600; margin-top:2px; }
.rag-rerank-card { background:#FFFFFF; border:1px solid var(--border); border-radius:12px;
  padding:12px 14px; box-shadow:0 1px 2px rgba(15,23,42,0.03);
  transition:border-color var(--speed) var(--ease); }
.rag-rerank-card:hover { border-color:rgba(36,71,199,0.28); }
.rag-rerank-head { display:flex; flex-wrap:wrap; align-items:center; gap:7px; }
.rag-rank-badge { font-size:0.76rem; font-weight:700; color:#FFFFFF; background:var(--info);
  border-radius:6px; padding:1px 7px; }
.rag-rerank-title, .rag-rerank-title a { font-weight:600; color:var(--text); text-decoration:none; }
.rag-rerank-title a:hover { color:var(--link); text-decoration:underline; }
.rag-rank-flow { font-size:0.75rem; color:var(--muted); }
.rag-rerank-score-wrap { display:inline-flex; align-items:center; gap:6px; margin-left:auto; }
.rag-rerank-val { font-size:0.76rem; font-weight:600; color:var(--link); }
.rag-delta-up { font-family:var(--font-mono); font-size:0.75rem; font-weight:600; color:var(--success); }
.rag-delta-down { font-family:var(--font-mono); font-size:0.75rem; font-weight:600; color:var(--danger); }
.rag-delta-flat { font-family:var(--font-mono); font-size:0.75rem; color:var(--muted); }
.rag-rerank-snip { color:#334155; font-size:0.83rem; line-height:1.5; margin-top:8px;
  border-left:2px solid var(--primary); padding-left:10px; }

.rag-stage1-details, .rag-stage1-box { border:1px solid var(--border); border-radius:12px;
  padding:10px 12px; background:#FFFFFF; box-shadow:0 1px 2px rgba(15,23,42,0.03);
  max-width:100%; min-width:0; overflow:hidden; }
.rag-stage1-details > summary { cursor:pointer; font-size:0.84rem; font-weight:600; color:var(--link);
  list-style:none; }
.rag-stage1-details > summary::before { content:"▸ "; }
.rag-stage1-details[open] > summary::before { content:"▾ "; }
.rag-stage1-wrap { width:100%; max-width:100%; overflow-x:auto; margin-top:8px;
  animation:rag-fade 160ms var(--ease); }
.rag-stage1-table { width:100%; border-collapse:collapse; font-size:0.78rem;
  font-variant-numeric:tabular-nums; }
.rag-stage1-table th { text-align:left; color:var(--muted); font-weight:600; padding:6px 4px;
  border-bottom:1px solid var(--border); font-family:var(--font-mono); font-size:0.73rem;
  white-space:nowrap; }
.rag-stage1-table td { padding:6px 4px; border-bottom:1px solid rgba(15,23,42,0.07); color:var(--text); }
.rag-stage1-table td:nth-child(5) { max-width:168px; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.rag-stage1-table td.rag-mono,
.rag-stage1-table td:last-child { white-space:nowrap; }
.rag-s1-kept { background:rgba(15,118,110,0.06); }
.rag-lab-empty { color:var(--muted); font-size:0.86rem; padding:12px 0; }
.rag-lab-placeholder { background:#FFFFFF; border:1px dashed rgba(15,23,42,0.18);
  border-radius:var(--radius); padding:32px 24px; text-align:center;
  display:flex; flex-direction:column; align-items:center; gap:10px; }
.rag-lab-placeholder-icon { width:44px; height:44px; border-radius:12px; display:flex;
  align-items:center; justify-content:center; font-size:1.35rem; color:var(--link);
  background:var(--surface-2); border:1px solid rgba(36,71,199,0.22); }
.rag-lab-placeholder-title { font-size:1.05rem; font-weight:700; color:var(--text); }
.rag-lab-placeholder-sub { color:var(--muted); font-size:0.86rem; max-width:460px; line-height:1.5; }
.rag-lab-placeholder-pills { display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-top:4px; }

/* Documents Corpus Overview & 2-Column Card Grid */
.rag-corpus-overview { display:flex; flex-direction:column; gap:12px; margin-bottom:16px; }
.rag-kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; }
.rag-kpi-card { background:#FFFFFF; border:1px solid var(--border); border-radius:var(--radius);
  padding:14px 16px; min-width:0; overflow:hidden; box-shadow:0 1px 2px rgba(15,23,42,0.03);
  border-top:3px solid var(--primary); }
.rag-kpi-card:nth-child(2) { border-top-color:var(--info); }
.rag-kpi-card:nth-child(3) { border-top-color:var(--primary-2); }
.rag-kpi-card:nth-child(4) { border-top-color:var(--success); }
.rag-kpi-lbl { font-family:var(--font-mono); font-size:0.75rem; letter-spacing:0.04em; color:var(--muted); }
.rag-kpi-val { font-size:1.6rem; font-weight:700; color:var(--text); margin:4px 0 2px; }
.rag-kpi-sub { font-size:0.78rem; color:var(--muted); overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; overflow-wrap:anywhere; }
.rag-dist-box { background:#FFFFFF; border:1px solid var(--border); border-radius:var(--radius);
  padding:14px 16px; min-width:0; overflow:hidden; box-shadow:0 1px 2px rgba(15,23,42,0.03); }
.rag-dist-head { display:flex; justify-content:space-between; font-family:var(--font-mono);
  font-size:0.75rem; letter-spacing:0.04em; color:var(--muted); margin-bottom:8px; }
.rag-dist-bar { display:flex; height:10px; border-radius:999px; overflow:hidden; gap:2px;
  background:#E2E8F0; }
.rag-dist-seg { display:block; height:100%; }
.rag-dist-legend { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:10px; }
.rag-dist-leg { display:inline-flex; align-items:center; gap:6px; font-size:0.78rem; color:var(--text); }
.rag-dist-dot { width:8px; height:8px; border-radius:2px; display:inline-block; }
.rag-dist-num { color:var(--muted); font-size:0.75rem; }

.rag-doc-card { margin-bottom:10px; }
.rag-doc-badges { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:8px; }
.rag-badge { display:inline-block; font-family:var(--font-mono); font-size:0.75rem; font-weight:600;
  border-radius:999px; padding:2px 9px; border:1px solid var(--border); white-space:nowrap; }
.rag-badge-cat { color:var(--link); background:var(--surface-2); border-color:rgba(37,99,235,0.2);
  max-width:100%; overflow:hidden; text-overflow:ellipsis; }
.rag-badge-en { color:#0369A1; background:rgba(3,105,161,0.09); border-color:rgba(3,105,161,0.24); }
.rag-badge-vi { color:#6D28D9; background:rgba(124,58,237,0.09); border-color:rgba(124,58,237,0.24); }
.rag-badge-ok { color:#047857; background:rgba(4,120,87,0.09); border-color:rgba(4,120,87,0.24); }
.rag-badge-dim { color:var(--muted); background:rgba(15,23,42,0.04); }
.rag-doc-title { font-weight:600; font-size:0.95rem; color:var(--text); line-height:1.4;
  min-height:2.8em; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden; }
.rag-doc-id { color:var(--muted); font-size:0.75rem; margin-top:4px;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rag-doc-stats { display:flex; justify-content:space-between; color:var(--muted); font-size:0.76rem;
  margin-top:8px; }
.rag-doc-bar { height:4px; background:#E2E8F0; border-radius:999px; overflow:hidden;
  margin-top:6px; }
.rag-doc-bar > span { display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg, var(--primary), var(--info)); }
[data-testid="stVerticalBlockBorderWrapper"]:has(.rag-doc-card) [data-testid="stBaseButton-secondary"],
[data-testid="stVerticalBlockBorderWrapper"]:has(.rag-doc-card) a {
  font-size:0.82rem; padding:0.44rem 0.6rem; white-space:nowrap; }

/* Follow-up Suggestion Chips & Mini-Quiz */
.rag-followup-label { font-size:0.72rem; font-weight:600; letter-spacing:0.06em;
  color:var(--muted); margin:6px 0 4px; }
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"] {
  background:#FFFFFF; border:1px solid var(--border); border-radius:12px;
  padding:0.48rem 0.85rem; min-height:2.25rem; height:auto !important;
  text-align:left; justify-content:flex-start;
  color:var(--text); font-size:0.86rem; font-weight:500;
  white-space:normal !important; line-height:1.4;
  box-shadow:0 1px 2px rgba(15,23,42,0.03); }
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"] > div,
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"],
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
  white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important; text-align:left; margin:0; }
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"]:hover {
  background:#F0FDFA; border-color:rgba(15,118,110,0.34); color:var(--info);
  transform:translate3d(2px,0,0); }
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:var(--info); }
[class*="st-key-quiz_preset_"] [data-testid="stBaseButton-secondary"] {
  font-size:0.78rem; padding:0.35rem 0.5rem; min-height:2.85rem; height:auto !important;
  display:flex; align-items:center; justify-content:center;
  white-space:normal !important; line-height:1.25; }
[class*="st-key-quiz_preset_"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important;
  font-size:0.78rem; margin:0; text-align:center; }

.rag-quiz-score { background:#FFFFFF; border:1px solid var(--border); border-radius:14px;
  padding:14px 18px; margin-bottom:14px; box-shadow:0 2px 8px rgba(15,23,42,0.04); }
.rag-quiz-score.is-ok { border-left:4px solid var(--success); }
.rag-quiz-score.is-warn { border-left:4px solid var(--warn); }
.rag-quiz-score.is-danger { border-left:4px solid var(--danger); }
.rag-quiz-score-top { display:flex; flex-wrap:wrap; justify-content:space-between;
  align-items:flex-end; gap:8px; margin-bottom:8px; }
.rag-quiz-score-num { font-size:1.35rem; font-weight:800; color:var(--text); margin-top:2px; }
.rag-quiz-score-note { font-size:0.82rem; color:var(--muted); }
.rag-quiz-score.is-ok .rag-doc-bar > span { background:var(--success); }
.rag-quiz-score.is-warn .rag-doc-bar > span { background:var(--warn); }
.rag-quiz-score.is-danger .rag-doc-bar > span { background:var(--danger); }
.rag-quiz-qhead { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:8px; }
.rag-quiz-src { font-size:0.75rem; color:var(--muted); margin-left:auto; }
.rag-quiz-status { font-size:0.73rem; font-weight:700; border-radius:999px; padding:2px 9px; }
.rag-quiz-status.is-ok { color:var(--success); background:#ECFDF5;
  border:1px solid rgba(4,120,87,0.24); }
.rag-quiz-status.is-wrong { color:var(--danger); background:#FEF2F2;
  border:1px solid rgba(185,28,28,0.24); }
.rag-quiz-ans { font-size:0.75rem; color:var(--muted); margin-left:auto; }
.rag-quiz-fb { font-size:0.86rem; line-height:1.55; padding:8px 12px; border-radius:10px;
  margin:8px 0; color:var(--text); }
.rag-quiz-fb.is-ok { background:#ECFDF5; border-left:3px solid var(--success); }
.rag-quiz-fb.is-wrong { background:#FEF2F2; border-left:3px solid var(--danger); }

/* 3D Flip Flashcards */
.rag-fc-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px;
  margin:10px 0 14px; }
.rag-fc-wrap { display:flex; flex-direction:column; gap:6px; min-width:0; }
.rag-fc-card { display:block; perspective:1100px; cursor:pointer; user-select:none;
  position:relative; margin:0; }
.rag-fc-toggle { position:absolute; opacity:0; width:1px; height:1px; pointer-events:none; }
.rag-fc-inner { display:grid; grid-template-columns:minmax(0, 1fr); transform-style:preserve-3d;
  transition:transform 0.52s cubic-bezier(0.22, 1, 0.36, 1); }
.rag-fc-toggle:checked + .rag-fc-inner { transform:rotateY(180deg); }
.rag-fc-front, .rag-fc-back { grid-area:1 / 1; backface-visibility:hidden;
  -webkit-backface-visibility:hidden; border-radius:14px; padding:16px 18px; min-height:220px;
  display:flex; flex-direction:column; justify-content:flex-start; gap:10px;
  box-shadow:0 2px 8px rgba(15,23,42,0.05); }
.rag-fc-front { background:#FFFFFF; border:1px solid var(--border); border-top:3px solid var(--primary); }
.rag-fc-card:hover .rag-fc-front { border-color:rgba(36,71,199,0.38);
  box-shadow:0 8px 22px rgba(15,23,42,0.08); }
.rag-fc-back { transform:rotateY(180deg);
  background:linear-gradient(180deg, #F0FDFA 0%, #FFFFFF 100%);
  border:1px solid rgba(15,118,110,0.32); border-top:3px solid var(--info); }
.rag-fc-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.rag-fc-hint { font-size:0.72rem; color:var(--muted); }
.rag-fc-body { display:flex; flex-direction:column; gap:8px; }
.rag-fc-q { font-size:0.93rem; font-weight:600; color:var(--text); line-height:1.5; }
.rag-fc-opts { display:flex; flex-direction:column; gap:4px; padding-top:4px;
  border-top:1px dashed var(--border); }
.rag-fc-opt { font-size:0.81rem; color:var(--muted); line-height:1.4; }
.rag-fc-opt-k { font-weight:700; color:var(--link); }
.rag-fc-ans { font-size:0.95rem; font-weight:600; color:var(--success); line-height:1.5; }
.rag-fc-exp { font-size:0.84rem; color:#334155; line-height:1.55; padding:8px 10px;
  background:rgba(15,118,110,0.06); border-left:3px solid var(--info); border-radius:8px; }
.rag-fc-foot { display:flex; align-items:center; }

@media (max-width: 1120px) {
  .rag-kpi-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] { padding:2.5rem var(--s-4) 6rem; }
  [data-testid="stBottomBlockContainer"] { padding-left:var(--s-4); padding-right:var(--s-4); }
  .rag-lab-summary, .rag-kpi-grid, .rag-fc-grid { grid-template-columns:1fr; }
  .rag-kpi-sub { white-space:normal; overflow-wrap:anywhere; }
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
    align-items:center !important; gap:4px !important; width:100% !important; }
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    flex:1 1 auto !important; width:calc(100% - 38px) !important; min-width:0 !important; }
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > div,
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stElementContainer"],
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stButton"],
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child button {
    flex:none !important; width:100% !important; min-width:0 !important; }
  [data-testid="stSidebar"] [class*="st-key-conv_"] button [data-testid="stMarkdownContainer"] {
    width:100% !important; min-width:0 !important; }
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    flex:0 0 34px !important; width:34px !important; min-width:34px !important; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-direction:column; gap:10px; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    width:100% !important; flex:1 1 100% !important; }
  [data-testid="stChatMessage"] { padding:12px 12px; gap:8px; }
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    overflow-x:auto !important; max-width:100% !important; -webkit-overflow-scrolling:touch;
    scrollbar-width:thin; }
  [data-testid="stChatMessage"] table { display:table !important; width:max-content !important;
    min-width:100% !important; }
  [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {
    min-width:140px !important; max-width:230px !important; white-space:normal !important; }
  [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    max-width:94% !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration:0.01ms !important;
    animation-duration:0.01ms !important; animation-iteration-count:1 !important; }
}
""")

_DARK_CSS_TEMPLATE = Template("""
:root { color-scheme: dark;
  --bg:$canvas; --surface:$card; --surface-2:$active_pill;
  --border:$border; --text:$ink; --muted:$muted;
  --primary:$primary; --primary-2:$primary_2; --primary-end:$primary_end;
  --success:$success; --warn:$warning; --danger:$danger; --info:$info;
  --canvas:$canvas; --sidebar:$sidebar; --card:$card; --ink:$ink;
  --link:$link; --active-pill:$active_pill; --callout:$callout; --warning:$warning; }
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:
    radial-gradient(ellipse 80% 40% at 50% -10%, rgba(79,117,255,0.13), transparent 72%),
    var(--bg) !important;
  color:var(--text) !important; }
[data-testid="stHeader"] {
  background:linear-gradient(180deg, rgba(11,17,32,0.92) 55%, transparent) !important; }
[data-testid="stHeader"] button,
[data-testid="stSidebarCollapseButton"] button { color:var(--text) !important; }
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"], [data-testid="stBottomBlockContainer"] > div {
  background:transparent !important; }
* { scrollbar-color:rgba(148,163,184,0.28) var(--sidebar); }
::-webkit-scrollbar-thumb { background:rgba(148,163,184,0.28); border-color:var(--sidebar); }
[data-testid="stSidebar"] { background:var(--sidebar) !important;
  border-right:1px solid var(--border) !important; color:var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:active {
  background:rgba(255,255,255,0.07) !important; color:var(--text) !important;
  border-color:var(--border) !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"] {
  background:var(--surface) !important; border-color:rgba(125,162,255,0.34) !important;
  box-shadow:inset 3px 0 0 var(--primary), 0 1px 4px rgba(0,0,0,0.35) !important;
  color:var(--text) !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"]:focus,
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"]:active {
  background:var(--surface) !important; color:var(--text) !important;
  border-color:rgba(125,162,255,0.44) !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stBaseButton-primary"] * {
  color:var(--text) !important; }
[data-testid="stSidebar"] [class*="st-key-conv_"] [data-testid="stBaseButton-primary"] {
  background:rgba(79,117,255,0.18) !important; border-color:rgba(125,162,255,0.32) !important;
  color:var(--link) !important; }
[data-testid="stSidebar"] [class*="st-key-conv_"] [data-testid="stBaseButton-primary"] * {
  color:var(--link) !important; }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] [data-testid="stBaseButton-secondary"] {
  background:var(--surface) !important; border-color:var(--border) !important;
  color:var(--text) !important; box-shadow:none !important; }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
  color:#FBBF24 !important; }
[data-testid="stSidebar"] [class*="st-key-theme_toggle"] [data-testid="stBaseButton-secondary"]:hover {
  background:var(--surface-2) !important; border-color:rgba(125,162,255,0.38) !important;
  color:var(--link) !important; }
.rag-brand, .rag-sys-card, .rag-page-header, .rag-lab-stat, .rag-rerank-card,
.rag-stage1-details, .rag-stage1-box, .rag-lab-placeholder, .rag-kpi-card,
.rag-dist-box, .rag-quiz-score,
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] > details {
  background:var(--surface) !important; border-color:var(--border) !important;
  box-shadow:0 2px 10px rgba(0,0,0,0.28) !important; color:var(--text) !important; }
[data-testid="stExpander"] > details > summary { color:var(--text) !important; }
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[role="radiogroup"] label, [role="radiogroup"] label p {
  color:var(--text) !important; }
[data-testid="stCaptionContainer"], .rag-page-sub, .rag-brand-sub, .rag-sys-sub,
.rag-lab-s, .rag-kpi-sub, .rag-fc-hint, .rag-fc-opt, .rag-quiz-src,
.rag-quiz-ans, .rag-quiz-score-note, .rag-cite-page, .rag-rank-flow {
  color:var(--muted) !important; }
[data-testid="stChatMessage"] {
  background:var(--surface) !important; border-color:var(--border) !important;
  box-shadow:0 2px 12px rgba(0,0,0,0.3) !important; }
[data-testid="stChatMessage"] th {
  background:#1E293B !important; color:var(--text) !important; border-color:var(--border) !important; }
[data-testid="stChatMessage"] td {
  color:var(--text) !important; border-color:var(--border) !important; }
[data-testid="stChatMessage"] code {
  background:#0F172A !important; color:#E2E8F0 !important; border-color:var(--border) !important; }
[data-testid="stChatMessage"] pre {
  background:#0F172A !important; border-color:var(--border) !important; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:linear-gradient(180deg, #1E2B4B 0%, #17223B 100%) !important;
  border-color:rgba(125,162,255,0.34) !important; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] {
  background:linear-gradient(135deg, #27375F, #1E2B4B) !important;
  color:#C7D2FE !important; border-color:rgba(125,162,255,0.4) !important; }
[data-testid="stChatMessageAvatarUser"] *,
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] * {
  color:#C7D2FE !important; }
.katex { color:var(--text) !important; }
.rag-hero {
  background:radial-gradient(circle at 1px 1px, rgba(125,162,255,0.14) 1px, transparent 0),
    linear-gradient(180deg, #162033 0%, #0F172A 100%) !important;
  background-size:20px 20px, 100% 100% !important;
  border-color:var(--border) !important; box-shadow:0 4px 20px rgba(0,0,0,0.35) !important; }
.rag-hero-title {
  background:linear-gradient(135deg, #F8FAFC 18%, #93B4FF 62%, #2DD4BF 100%) !important;
  -webkit-background-clip:text !important; background-clip:text !important;
  -webkit-text-fill-color:transparent !important; }
.rag-hero-badge {
  background:rgba(79,117,255,0.16) !important; border-color:rgba(125,162,255,0.34) !important; }
.rag-hero-pill, .rag-flow-step, .rag-hdr-chip, .rag-tel-chip {
  background:#0F172A !important; color:var(--text) !important; border-color:var(--border) !important; }
.rag-hdr-chip.is-primary, .rag-tel-model {
  background:rgba(79,117,255,0.18) !important; color:var(--link) !important;
  border-color:rgba(125,162,255,0.32) !important; }
.rag-hdr-chip.is-success {
  background:rgba(52,211,153,0.14) !important; color:var(--success) !important;
  border-color:rgba(52,211,153,0.3) !important; }
.rag-hdr-chip.is-warn, .rag-tel-warn {
  background:rgba(251,191,36,0.14) !important; color:var(--warn) !important;
  border-color:rgba(251,191,36,0.3) !important; }
.rag-hdr-chip.is-info {
  background:rgba(45,212,191,0.14) !important; color:var(--info) !important;
  border-color:rgba(45,212,191,0.3) !important; }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"] {
  background:var(--surface) !important; border-color:var(--border) !important; }
[class*="st-key-sug_"] [data-testid="stBaseButton-secondary"]:hover {
  background:#1B2740 !important; }
[class*="st-key-sug_0"] [data-testid="stBaseButton-secondary"] { border-top-color:#60A5FA !important; }
[class*="st-key-sug_0"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] { color:#60A5FA !important; }
[class*="st-key-sug_1"] [data-testid="stBaseButton-secondary"] { border-top-color:#2DD4BF !important; }
[class*="st-key-sug_1"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] { color:#2DD4BF !important; }
[class*="st-key-sug_2"] [data-testid="stBaseButton-secondary"] { border-top-color:#A78BFA !important; }
[class*="st-key-sug_2"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] { color:#A78BFA !important; }
[class*="st-key-sug_3"] [data-testid="stBaseButton-secondary"] { border-top-color:#FBBF24 !important; }
[class*="st-key-sug_3"] [data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] { color:#FBBF24 !important; }
[data-testid="stBaseButton-secondary"], a[data-testid="stBaseLinkButton-secondary"] {
  background:var(--surface) !important; color:var(--text) !important;
  border-color:var(--border) !important; }
[data-testid="stBaseButton-secondary"]:hover, a[data-testid="stBaseLinkButton-secondary"]:hover {
  background:var(--surface-2) !important; border-color:rgba(125,162,255,0.4) !important;
  color:var(--link) !important; }
[class*="st-key-fup_"] [data-testid="stBaseButton-secondary"]:hover {
  background:rgba(45,212,191,0.12) !important; border-color:rgba(45,212,191,0.38) !important;
  color:var(--info) !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
[data-testid="stSelectbox"] .react-aria-ComboBox,
[data-testid="stSelectbox"] .react-aria-ComboBox > div,
[data-testid="stSelectbox"] [role="group"],
[data-testid="stSelectbox"] input[role="combobox"] {
  background-color:#0F172A !important; color:var(--text) !important;
  border-color:var(--border) !important; caret-color:var(--text) !important; }
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stSelectbox"] input[role="combobox"]::placeholder {
  color:var(--muted) !important; opacity:0.85 !important; }
[data-testid="stSelectbox"] [role="group"]:focus-within {
  border-color:var(--primary) !important;
  box-shadow:0 0 0 3px rgba(79,117,255,0.22) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
  color:var(--text) !important; }
[data-testid="stSelectbox"] button[aria-haspopup="listbox"],
[data-testid="stSelectbox"] button[aria-label="Open"],
[data-testid="stSelectbox"] [role="group"] button {
  background-color:#0F172A !important; color:var(--muted) !important;
  border-color:var(--border) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stSelectbox"] [role="group"] svg {
  fill:var(--muted) !important; color:var(--muted) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="menu"], ul[role="listbox"],
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {
  background-color:#162033 !important; color:var(--text) !important;
  border:1px solid var(--border) !important;
  box-shadow:0 12px 32px rgba(0,0,0,0.5) !important; }
li[role="option"], li[role="option"] > div, li[role="option"] span,
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] > div {
  background-color:transparent !important; color:var(--text) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"] {
  background-color:#1E293B !important; color:var(--link) !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] > div,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"] > div {
  color:var(--link) !important; font-weight:600; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
  background-color:#162033 !important; border-color:rgba(148,163,184,0.28) !important;
  color:var(--text) !important; }
[data-testid="stChatInput"] { box-shadow:0 6px 24px rgba(0,0,0,0.4) !important; }
[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"] {
  background:transparent !important; color:var(--text) !important; caret-color:var(--text) !important; }
[data-testid="stChatInput"] textarea::placeholder { color:var(--muted) !important; opacity:0.85 !important; }
[data-testid="stChatInput"] button {
  background:transparent !important; color:var(--muted) !important; }
[data-testid="stChatInput"] button:not(:disabled) {
  color:var(--link) !important; }
[data-testid="stButtonGroup"] button {
  background:#0F172A !important; color:var(--muted) !important; border-color:var(--border) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background:rgba(79,117,255,0.22) !important; color:var(--link) !important;
  border-color:rgba(125,162,255,0.45) !important; }
[data-testid="stAlert"] > div {
  background:var(--surface) !important; color:var(--text) !important; border-color:var(--border) !important; }
.rag-citation {
  background:var(--callout) !important; border-color:var(--border) !important; }
.rag-citation-head { color:var(--link) !important; }
.rag-cite-id { background:#1E293B !important; color:var(--muted) !important; border-color:var(--border) !important; }
.rag-cite-score { background:rgba(45,212,191,0.14) !important; color:var(--info) !important;
  border-color:rgba(45,212,191,0.3) !important; }
.rag-cite-meter { background:rgba(148,163,184,0.22) !important; }
.rag-cite-snippet, .rag-rerank-snip { color:#CBD5E1 !important; }
.rag-source-pill, a.rag-source-pill {
  background:rgba(45,212,191,0.12) !important; color:var(--text) !important;
  border-color:rgba(45,212,191,0.3) !important; }
a.rag-source-pill:hover { background:rgba(45,212,191,0.22) !important; color:#FFFFFF !important; }
.rag-source-pill.is-more { background:#0F172A !important; color:var(--muted) !important; }
.rag-stepper { background:#0F172A !important; border-color:var(--border) !important; }
.rag-step.is-done { background:rgba(52,211,153,0.14) !important; border-color:rgba(52,211,153,0.3) !important; }
.rag-step.is-active { background:rgba(79,117,255,0.18) !important; border-color:rgba(125,162,255,0.36) !important; }
.rag-step.is-active.is-synth { background:rgba(45,212,191,0.16) !important; border-color:rgba(45,212,191,0.34) !important; }
.rag-step.is-wait { background:#162033 !important; color:var(--muted) !important; }
mark.rag-hl { background:rgba(250,204,21,0.26) !important; color:#FEF08A !important;
  box-shadow:0 0 0 1px rgba(250,204,21,0.35) !important; }
.rag-stage1-table td { color:var(--text) !important; border-bottom-color:rgba(148,163,184,0.14) !important; }
.rag-s1-kept { background:rgba(45,212,191,0.1) !important; }
.rag-latency-bar, .rag-dist-bar, .rag-doc-bar { background:#1E293B !important; }
.rag-badge-en { color:#38BDF8 !important; background:rgba(56,189,248,0.14) !important; border-color:rgba(56,189,248,0.3) !important; }
.rag-badge-vi { color:#C084FC !important; background:rgba(192,132,252,0.14) !important; border-color:rgba(192,132,252,0.3) !important; }
.rag-badge-ok { color:var(--success) !important; background:rgba(52,211,153,0.14) !important; border-color:rgba(52,211,153,0.3) !important; }
.rag-badge-dim { color:var(--muted) !important; background:rgba(148,163,184,0.12) !important; }
.rag-doc-card .rag-badge-cat { filter:brightness(1.38); }
.rag-quiz-status.is-ok { background:rgba(52,211,153,0.15) !important; border-color:rgba(52,211,153,0.32) !important; }
.rag-quiz-status.is-wrong { background:rgba(248,113,113,0.15) !important; border-color:rgba(248,113,113,0.32) !important; }
.rag-quiz-fb { color:var(--text) !important; }
.rag-quiz-fb.is-ok { background:rgba(52,211,153,0.1) !important; }
.rag-quiz-fb.is-wrong { background:rgba(248,113,113,0.1) !important; }
.rag-fc-front { background:var(--surface) !important; border-color:var(--border) !important;
  border-top:3px solid var(--primary) !important; }
.rag-fc-back { background:linear-gradient(180deg, #132833 0%, #162033 100%) !important;
  border-color:rgba(45,212,191,0.36) !important; border-top:3px solid var(--info) !important; }
.rag-fc-exp { color:#E2E8F0 !important; background:rgba(45,212,191,0.1) !important; }
""")

_FONT_LINKS_HTML = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Be+Vietnam+Pro:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@400;500;600'
    '&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap">'
)


def _global_css(theme_mode="light"):
    raw = _CSS_TEMPLATE.substitute(**PALETTE)
    if str(theme_mode or "light").strip().lower() == "dark":
        raw = raw + "\n" + _DARK_CSS_TEMPLATE.substitute(**DARK_PALETTE)
    return "\n".join(line for line in raw.splitlines() if line.strip())


def inject_global_css(theme_mode=None):
    import streamlit as st

    if theme_mode is None:
        sess = getattr(st, "session_state", None)
        theme_mode = sess.get("theme_mode", "light") if hasattr(sess, "get") else "light"
    st.markdown(_FONT_LINKS_HTML, unsafe_allow_html=True)
    st.markdown(f"<style>\n{_global_css(theme_mode)}\n</style>", unsafe_allow_html=True)
