from transformers import pipeline

from preprocess import preprocess

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    max_length=128,
    truncation=True,
)


def predict(text: str) -> dict:
    """Preprocess and run sentiment classification on a single text.

    Returns a dict with keys:
        - label: str  ("Negative", "Neutral", or "Positive")
        - score: float (confidence score)
    """
    cleaned = preprocess(text)
    result = classifier(cleaned)[0]
    return {
        "label": LABEL_MAP[result["label"]],
        "score": result["score"],
    }


def predict_batch(texts: list[str], batch_size: int = 64) -> list[dict]:
    """Preprocess and run sentiment classification on a list of texts."""
    cleaned = [preprocess(t) for t in texts]
    results = classifier(cleaned, batch_size=batch_size)
    return [{"label": LABEL_MAP[r["label"]], "score": r["score"]} for r in results]
