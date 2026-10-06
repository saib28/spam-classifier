import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import MODEL_PATH, classify  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(),
                                reason="run `python src/train.py` first")


@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.mark.parametrize("message", [
    "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt CLAIM to 81010",
    "Congratulations, you've been selected for a free cash prize. Call 09061701461 now to claim",
])
def test_detects_spam(model, message):
    assert classify(model, message) == "SPAM"


@pytest.mark.parametrize("message", [
    "Hey, are we still on for lunch tomorrow?",
    "I'll be home late tonight, can you feed the cat",
])
def test_allows_ham(model, message):
    assert classify(model, message) == "ham"
