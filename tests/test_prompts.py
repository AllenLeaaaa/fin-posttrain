import pytest

from finpost import prompts


class WordTokenizer:
    """Minimal stand-in for an HF tokenizer: one token per whitespace-separated word."""

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": list(range(len(texts.split())))}
        return {"input_ids": [list(range(len(t.split()))) for t in texts]}

    def decode(self, ids):
        return " ".join("w" for _ in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


def test_truncate_keeps_whole_lines():
    text = "a b c\nd e\nf g h i"
    # line costs: 3, then 2 + 1 newline = 3, then 4 + 1 = 5
    assert prompts.truncate_lines(text, WordTokenizer(), 6) == ("a b c\nd e", True)
    assert prompts.truncate_lines(text, WordTokenizer(), 100) == (text, False)


def test_truncate_falls_back_to_token_cut_for_long_first_line():
    kept, truncated = prompts.truncate_lines("a b c d e", WordTokenizer(), 2)
    assert truncated and len(kept.split()) == 2


def test_build_prompt_contains_instruction_and_generation_marker():
    prompt, truncated = prompts.build_prompt("revenue was $5", WordTokenizer(), 100)
    assert not truncated
    assert "revenue was $5" in prompt and "bullet points" in prompt
    assert prompt.endswith("<assistant>")


def test_build_messages_with_summary_adds_assistant_turn():
    messages = prompts.build_messages("t", summary="s")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("- eps $0.83\n- revenue $5 million", "eps $0.83\nrevenue $5 million"),
        ("1. **EPS**: $0.83\n2) revenue up\n\n", "EPS: $0.83\nrevenue up"),
        ("• margin 5%", "margin 5%"),
        # not bullet markers
        ("1.5% growth\n-2% decline", "1.5% growth\n-2% decline"),
    ],
)
def test_postprocess_summary(raw, expected):
    assert prompts.postprocess_summary(raw) == expected
