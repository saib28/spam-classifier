"""Download and load the UCI SMS Spam Collection dataset."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection"


def download(path: Path = DATA_PATH) -> Path:
    """Download the dataset if it isn't already on disk."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path.write_bytes(archive.read("SMSSpamCollection"))
    return path


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return a DataFrame with columns `label` ("ham"/"spam") and `text`."""
    download(path)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    return df.drop_duplicates(subset="text").reset_index(drop=True)
