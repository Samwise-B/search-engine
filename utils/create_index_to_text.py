from datasets import load_dataset
from data_preprocessing import load_word_to_int, clean_text, tokenize
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

INTERNET_FILEPATH = Path(__file__).parent.parent / "data/all_docs"
INTERNET_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    dataset = load_dataset("microsoft/ms_marco", "v1.1")

    splits = ["train", "test", "validation"]
    all_documents: set[str] = set()

    for split in splits:

        passages = dataset[split]["passages"]  # [:10000]
        for passage in tqdm(passages, desc=split + " passages"):
            all_documents.update(set(passage["passage_text"]))

    with open(INTERNET_FILEPATH, "w", encoding="utf-8") as f:
        for index, doc in tqdm(
            enumerate(all_documents),
            desc="Writing into document",
            total=len(all_documents),
        ):
            f.write(f"{doc}\n")