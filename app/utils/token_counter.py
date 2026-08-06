"""
Token counter utility.

Approximate token counting for enforcing the world slice budget.
Uses the ~4 chars/token heuristic which is close enough for English text.
No external tokenizer dependency needed.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses the common heuristic of ~4 characters per token for English text.
    This is an approximation — actual tokenization varies by model.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # ~4 chars per token is a reasonable heuristic for English
    return max(1, len(text) // 4)
