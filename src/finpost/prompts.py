"""Prompt construction and output cleanup.

The same ``build_messages`` feeds the baseline eval, the SFT data (with the
reference summary as the assistant turn), and DPO, so every stage sees an
identical prompt. Bump ``PROMPT_VERSION`` whenever the wording changes.
"""

from __future__ import annotations

import re

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = "You are a financial analyst who writes concise, factual earnings summaries."

INSTRUCTION = (
    "Summarize the key financial results and guidance from the earnings call transcript below "
    "as 3 to 7 short bullet points, one per line. Report figures such as EPS, revenue, and "
    "guidance exactly as stated. Use only facts from the transcript.\n\n"
    "Transcript:\n{transcript}"
)

# "- ", "* ", "• ", "1. ", "2) " at the start of a line. Whitespace after the
# marker is required so "1.5% growth" or "-2%" are left alone.
_BULLET_RE = re.compile(r"^(?:[-*•·]|\d{1,2}[.)])\s+")


def build_messages(transcript: str, summary: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION.format(transcript=transcript)},
    ]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary})
    return messages


def truncate_lines(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Keep whole lines from the start of ``text`` within ``max_tokens``.

    Cutting at line (sentence) boundaries avoids leaving half a number at the
    end. The prepared remarks, where most figures are stated, come first in a
    transcript, so a head cut mostly drops Q&A.

    Returns the kept text and whether anything was dropped.
    """
    lines = text.split("\n")
    lengths = [len(ids) for ids in tokenizer(lines, add_special_tokens=False)["input_ids"]]
    kept, used = [], 0
    for line, n in zip(lines, lengths):
        cost = n + (1 if kept else 0)  # the joining newline
        if used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    if not kept and lines:
        # A single line longer than the budget: fall back to a token cut.
        ids = tokenizer(lines[0], add_special_tokens=False)["input_ids"][:max_tokens]
        return tokenizer.decode(ids), True
    return "\n".join(kept), len(kept) < len(lines)


def build_prompt(transcript: str, tokenizer, max_input_tokens: int) -> tuple[str, bool]:
    """Chat-formatted generation prompt with the transcript cut to ``max_input_tokens``."""
    text, truncated = truncate_lines(transcript, tokenizer, max_input_tokens)
    prompt = tokenizer.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
    return prompt, truncated


def postprocess_summary(text: str) -> str:
    """One bullet per line, markers and markdown bold removed, blank lines dropped.

    Must run before scoring: a leftover "1." marker would count as a number.
    """
    lines = []
    for line in text.replace("**", "").splitlines():
        line = _BULLET_RE.sub("", line.strip()).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
