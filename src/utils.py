import re
import string

def normalize_answer(answer: str) -> str:
    """Normalizes the answer."""

    # 1. lowercase
    # 2. remove punctuations
    # 3. remove articles (a, an, the)
    # 4. fix whitespaces
    return " ".join(
        re.sub(
            re.compile(r"\b(a|an|the)\b", re.UNICODE),
            " ",
            "".join(ch for ch in answer.lower() if ch not in string.punctuation)
        ).split()
    )

def is_correct_prediction(prediction: str, gold: str) -> bool:
    """Checks if the prediction matches the gold answer after normalization."""

    return normalize_answer(gold) in normalize_answer(prediction)
