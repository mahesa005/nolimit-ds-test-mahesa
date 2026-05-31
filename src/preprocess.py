import re


def preprocess(text: str) -> str:
    """Preprocess a single text input for sentiment classification.

    Steps mirror the notebook pipeline:
    1. Cast to string
    2. Remap null-like values
    3. Lowercase
    4. Remove URLs
    5. Replace @mentions with @user
    """
    text = str(text).strip()

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Replace @mentions with @user
    text = re.sub(r"@\w+", "@user", text)

    return text
