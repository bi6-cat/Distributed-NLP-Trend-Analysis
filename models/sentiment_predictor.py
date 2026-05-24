"""
Task 2.5 — Inference wrapper: predict_batch(texts, batch_size=32)

Dùng cho:
  - Spark mapPartitions (Phase 3)
  - API serving
  - Standalone prediction

Cách dùng:
    from models.sentiment_predictor import SentimentPredictor

    predictor = SentimentPredictor("models/phobert_finetuned/final")

    # Single text
    result = predictor.predict("Giáo viên giảng rất hay!")
    # → {"label": "Positive", "label_id": 2, "confidence": 0.95, "probs": [0.02, 0.03, 0.95]}

    # Batch (dùng trong Spark)
    results = predictor.predict_batch(["câu 1", "câu 2", ...], batch_size=32)
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


class SentimentPredictor:
    """
    Wrapper inference cho PhoBERT đã fine-tune.

    Args:
        model_path  : đường dẫn tới thư mục chứa model (có config.json, pytorch_model.bin)
        device      : "cuda" / "cpu" / None (tự detect)
        max_length  : max token length (phải khớp với lúc train, mặc định 256)
        preprocessor: instance TextPreprocessor (nếu muốn clean text trước khi predict)
    """

    def __init__(
        self,
        model_path: str = "models/phobert_finetuned/final",
        device: Optional[str] = None,
        max_length: int = 256,
        preprocessor=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.max_length = max_length
        self.preprocessor = preprocessor

        print(f"[SentimentPredictor] Loading model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[SentimentPredictor] Ready ✅")

    #  Public API 

    def predict(self, text: str) -> Dict:
        """
        Predict cho 1 câu.

        Returns:
            {
                "text":       str,
                "label":      "Negative" | "Neutral" | "Positive",
                "label_id":   0 | 1 | 2,
                "confidence": float,
                "probs":      [neg_prob, neu_prob, pos_prob]
            }
        """
        results = self.predict_batch([text], batch_size=1)
        return results[0]

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Predict cho danh sách câu (dùng trong Spark mapPartitions).

        Args:
            texts     : danh sách văn bản gốc
            batch_size: số câu xử lý mỗi lần (tuỳ VRAM)

        Returns:
            List[Dict] — cùng thứ tự với texts đầu vào
        """
        if not texts:
            return []

        # Optional preprocessing
        if self.preprocessor:
            texts = self.preprocessor.preprocess_batch(texts, remove_stopwords=False)

        results = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_results = self._predict_batch_raw(batch_texts)
            results.extend(batch_results)

        return results

    #  Internal 

    @torch.no_grad()
    def _predict_batch_raw(self, texts: List[str]) -> List[Dict]:
        """Tokenize + forward pass cho 1 batch."""
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # (B, 3)
        preds   = probs.argmax(axis=-1)                                # (B,)

        results = []
        for i, text in enumerate(texts):
            label_id = int(preds[i])
            results.append({
                "text":       text,
                "label":      LABEL_MAP[label_id],
                "label_id":   label_id,
                "confidence": float(probs[i][label_id]),
                "probs": {
                    "Negative": float(probs[i][0]),
                    "Neutral":  float(probs[i][1]),
                    "Positive": float(probs[i][2]),
                },
            })
        return results


#  Spark integration helper 

def make_spark_predict_fn(model_path: str, batch_size: int = 32):
    """
    Factory tạo function dùng trong Spark mapPartitions.

    Dùng lazy init để tránh serialize model qua mạng.

    Cách dùng trong spark_jobs/sentiment_job.py:
        predict_fn = make_spark_predict_fn("models/phobert_finetuned/final")
        result_rdd = text_rdd.mapPartitions(predict_fn)
    """
    def predict_partition(partition):
        # Lazy load — chỉ init 1 lần trên mỗi worker
        predictor = SentimentPredictor(model_path=model_path)
        rows = list(partition)
        texts = [r["text"] for r in rows]

        predictions = predictor.predict_batch(texts, batch_size=batch_size)

        for row, pred in zip(rows, predictions):
            yield {**row, **pred}

    return predict_partition


#  CLI quick test 

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phobert_finetuned/final")
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model)

    if args.text:
        result = predictor.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Demo
        samples = [
            "Giáo viên giảng rất hay và nhiệt tình!",
            "Bài giảng bình thường, không có gì đặc biệt.",
            "Môn học này thực sự nhàm chán và vô ích.",
            "Thầy giải thích rõ ràng, dễ hiểu.",
            "Chưa áp dụng công nghệ thông tin vào giảng dạy.",
        ]
        results = predictor.predict_batch(samples, batch_size=4)
        for r in results:
            print(f"[{r['label']:8s} {r['confidence']:.2f}] {r['text']}")
