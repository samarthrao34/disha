def token_count(text: str) -> int:
    return 0 if text is None else len(text.strip().split())

def is_empty_or_short(text: str, min_tokens: int = 2) -> bool:
    return token_count(text) < min_tokens
