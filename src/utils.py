import re
import string

def normalize_answer(answer):
    """Normalizes the answer."""

    # 1. lowercase
    lower = answer.lower()

    # 2. remove punctuations
    no_punc = "".join(ch for ch in lower if ch not in string.punctuation)

    # 3. remove articles (a, an, the)
    no_articles = re.sub(re.compile(r'\b(a|an|the)\b', re.UNICODE), " ", no_punc)

    # 4. fix whitespaces
    normalized = " ".join(no_articles.split())

    return normalized

def is_correct_prediction(prediction, gold):
    """Checks if the prediction matches the gold answer after normalization."""

    return normalize_answer(gold) in normalize_answer(prediction)
