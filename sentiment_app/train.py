from __future__ import annotations

import argparse
from itertools import islice

from datasets import load_dataset

from sentiment_app.model import IncrementalSentimentModel, MODEL_PATH


def batched(iterator, batch_size: int):
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def train(dataset_name: str, split: str, max_samples: int | None, batch_size: int) -> None:
    print(f"Loading dataset: {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    model = IncrementalSentimentModel()

    stream = iter(dataset)
    trained = 0

    if max_samples is not None:
        stream = islice(stream, max_samples)

    for batch in batched(stream, batch_size=batch_size):
        texts = [item["text"] for item in batch]
        labels = [int(item["label"]) for item in batch]
        model.partial_fit(texts, labels)
        trained += len(batch)
        print(f"Trained on {trained} samples", end="\r")

    print(f"\nTraining complete. Total samples: {trained}")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an incremental sentiment model.")
    parser.add_argument("--dataset", default="yelp_polarity", help="HF dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=200000, help="Limit for quick experiments")
    parser.add_argument("--batch-size", type=int, default=2048, help="Streaming batch size")
    args = parser.parse_args()

    train(args.dataset, args.split, args.max_samples, args.batch_size)


if __name__ == "__main__":
    main()
