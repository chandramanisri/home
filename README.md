# Customer Sentiment Trainer

This project now includes a Python sentiment-analysis application that can:

1. Stream a **large internet dataset** (default: `yelp_polarity` from Hugging Face).
2. Train an incremental sentiment model without loading all data into memory.
3. Serve predictions through a simple Flask API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train on a large dataset

```bash
python -m sentiment_app.train --dataset yelp_polarity --split train --max-samples 200000 --batch-size 2048
```

Notes:
- Increase `--max-samples` (or set it very high) for larger training runs.
- Training uses streaming mode so memory usage remains stable.

## Run the API

```bash
python -m sentiment_app.app
```

API endpoints:
- `GET /health` – service status.
- `POST /predict` – infer customer sentiment.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Support resolved my issue quickly and politely."}'
```

Example response:

```json
{
  "text": "Support resolved my issue quickly and politely.",
  "sentiment_score": 0.9123,
  "sentiment": "very_positive"
}
```

## How sentiment is interpreted

The model returns a positive probability (`sentiment_score` from 0 to 1) and maps it to labels:

- `very_negative` (<= 0.30)
- `negative` (<= 0.45)
- `neutral` (0.45 - 0.55)
- `positive` (>= 0.55)
- `very_positive` (>= 0.70)
