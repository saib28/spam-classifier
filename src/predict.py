"""Classify messages as spam or ham with the trained model.

Usage:
    python src/predict.py "WINNER! Claim your free prize now"
    python src/predict.py            # interactive mode
"""

import sys
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_classifier.joblib"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def classify(model, message: str) -> str:
    return "SPAM" if model.predict([message])[0] == 1 else "ham"


def main() -> None:
    model = load_model()
    if len(sys.argv) > 1:
        for message in sys.argv[1:]:
            print(f"{classify(model, message):>4}  |  {message}")
        return

    print("Type a message to classify (empty line to quit).")
    while message := input("> ").strip():
        print(f"  -> {classify(model, message)}")


if __name__ == "__main__":
    main()
