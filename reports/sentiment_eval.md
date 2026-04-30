# Sentiment Model Evaluation Report

**Model:** `vinai/phobert-base` (fine-tuned)  
**Task:** 3-class Sentiment Classification (Negative / Neutral / Positive)  
**Date:** 2026-04-05

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `vinai/phobert-base` |
| Num labels | 3 (Negative, Neutral, Positive) |
| Max sequence length | 256 |
| Batch size | 32 (train) / 64 (eval) |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Gradient accumulation steps | 1 |
| Max grad norm | 1.0 |
| FP16 | Yes |
| Class weights | [1.0, 12.0, 1.0] (upweight Neutral) |
| Seed | 42 |

---

## Test Set Results

| Metric | Score | Target | Status |
|---|---|---|---|
| Accuracy | **93.27%** | ≥ 80% | PASS |
| Macro F1 | **0.8378** | ≥ 0.75 | PASS |

---

## Training History (per Epoch)

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro F1 |
|---|---|---|---|---|
| 1 | 0.7079 | 0.5411 | 93.75% | 0.8523 |
| 2 | 0.3970 | 0.4312 | 93.30% | 0.8350 |
| 3 | 0.2912 | 0.5361 | 94.31% | 0.8508 |
| 4 | 0.2370 | 0.6376 | 94.38% | 0.8597 |
| 5 | 0.1748 | 0.6837 | 94.63% | **0.8674** |

Best Val Macro F1: **0.8674** (Epoch 5)

---

## Model Output

| Artifact | Path |
|---|---|
| Fine-tuned model | `models/phobert_finetuned/final/` |
| Eval results (JSON) | `models/phobert_finetuned/eval_results.json` |
| Training config | `models/train_config.json` |
| Inference wrapper | `models/sentiment_predictor.py` |

---

## Notes

- Neutral class được upweight (12.0) do mất cân bằng dữ liệu đầu vào.
- Train loss giảm đều qua 5 epoch (0.71 → 0.17), model hội tụ tốt.
- Val loss tăng nhẹ từ epoch 3 trở đi (dấu hiệu nhẹ của overfitting), nhưng Val Macro F1 vẫn tăng — best checkpoint được giữ lại.
- Kết quả vượt cả hai target (Accuracy ≥ 80%, Macro F1 ≥ 0.75).
