"""
Task 2.1 — Fine-tune PhoBERT cho Vietnamese Sentiment Analysis (3-class)

Pipeline:
  1. Load & preprocess data (text_cleaner.py)
  2. Tokenize bằng PhoBERT tokenizer
  3. Fine-tune với class weights (xử lý mất cân bằng nhãn Neutral)
  4. Đánh giá: Accuracy, Macro F1, Confusion Matrix
  5. Lưu model checkpoint

Cách chạy:
    python train_phobert.py                          # dùng config mặc định
    python train_phobert.py --config path/config.json
    python train_phobert.py --no-preprocess          # bỏ qua bước text cleaning
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Preprocessing pipeline (từ task 1.2) ──────────────────────────────────────
# Chỉ import khi cần, tránh crash nếu VnCoreNLP chưa cài
def get_preprocessor(config):
    try:
        from preprocessing.text_cleaner import TextPreprocessor
        return TextPreprocessor(
            slang_dict_path="data/slang_dict.json",
            stopwords_path="data/stopwords_vi.txt",
            use_vncorenlp=True,
            vncorenlp_jar="vncorenlp/VnCoreNLP-1.1.1.jar",
        )
    except Exception as e:
        print(f"[WARNING] Không load được TextPreprocessor: {e}")
        print("[WARNING] Chạy không có preprocessing step.")
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    Dataset cho PhoBERT fine-tuning.

    Args:
        texts   : list[str] — văn bản đã preprocess
        labels  : list[int] — nhãn 0/1/2
        tokenizer: PhoBERT tokenizer
        max_length: độ dài tối đa (token)
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(sentences_path, sentiments_path, preprocessor=None):
    """Load một split, trả về (texts, labels)."""
    df_texts  = pd.read_csv(sentences_path,  encoding="utf-8-sig")
    df_labels = pd.read_csv(sentiments_path, encoding="utf-8-sig")

    texts  = df_texts.iloc[:, 0].astype(str).tolist()
    labels = df_labels.iloc[:, 0].astype(int).tolist()

    if preprocessor:
        print(f"  Preprocessing {len(texts)} samples...")
        # remove_stopwords=False vì PhoBERT dùng full context tốt hơn
        texts = preprocessor.preprocess_batch(texts, remove_stopwords=False)

    return texts, labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels):
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report   = classification_report(
        labels, preds,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy":       acc,
        "eval_macro_f1":  macro_f1,
        "report":         report,
        "confusion_matrix": cm.tolist(),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids      = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, label_ids)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids      = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, label_ids)
        preds   = outputs.logits.argmax(dim=-1)

        total_loss   += loss.item()
        all_preds    += preds.cpu().tolist()
        all_labels   += label_ids.cpu().tolist()

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Seed
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Preprocessor
    preprocessor = None if args.no_preprocess else get_preprocessor(cfg)

    # Load data
    print("\n[1/5] Loading data...")
    train_texts, train_labels = load_split(
        cfg["data"]["train_sentences"], cfg["data"]["train_sentiments"], preprocessor
    )
    val_texts, val_labels = load_split(
        cfg["data"]["val_sentences"], cfg["data"]["val_sentiments"], preprocessor
    )
    test_texts, test_labels = load_split(
        cfg["data"]["test_sentences"], cfg["data"]["test_sentiments"], preprocessor
    )
    print(f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Tokenizer & model
    print("\n[2/5] Loading PhoBERT...")
    model_name = cfg["model_name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=cfg["num_labels"]
    ).to(device)

    # Datasets & loaders
    max_len    = cfg["max_length"]
    batch_size = cfg["batch_size"]

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds   = SentimentDataset(val_texts,   val_labels,   tokenizer, max_len)
    test_ds  = SentimentDataset(test_texts,  test_labels,  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=2)

    # Loss với class weights (xử lý nhãn Neutral ít)
    class_weights = torch.tensor(cfg["class_weights"], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    num_epochs     = cfg["num_epochs"]
    total_steps    = len(train_loader) * num_epochs
    warmup_steps   = int(total_steps * cfg["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # fp16 scaler
    use_fp16 = cfg.get("fp16", False) and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler() if use_fp16 else None

    # Output dirs
    os.makedirs(cfg["output_dir"],     exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Training
    print(f"\n[3/5] Training {num_epochs} epochs...")
    best_macro_f1  = 0.0
    best_ckpt_path = None
    history        = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch":          epoch,
            "train_loss":     round(train_loss, 4),
            "val_loss":       round(val_metrics["loss"], 4),
            "val_accuracy":   round(val_metrics["accuracy"], 4),
            "val_macro_f1":   round(val_metrics["eval_macro_f1"], 4),
        }
        history.append(row)

        print(
            f"  Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {row['train_loss']} | "
            f"Val Loss: {row['val_loss']} | "
            f"Val Acc: {row['val_accuracy']} | "
            f"Val Macro-F1: {row['val_macro_f1']}"
        )

        # Lưu checkpoint tốt nhất theo Macro F1
        if val_metrics["eval_macro_f1"] > best_macro_f1:
            best_macro_f1  = val_metrics["eval_macro_f1"]
            best_ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch}_f1_{best_macro_f1:.4f}")
            model.save_pretrained(best_ckpt_path)
            tokenizer.save_pretrained(best_ckpt_path)
            print(f"  ✅ Best checkpoint saved → {best_ckpt_path}")

    # Lưu training history
    history_path = os.path.join(cfg["output_dir"], "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Evaluate trên test set bằng best checkpoint
    print(f"\n[4/5] Evaluating on TEST set (best checkpoint: {best_ckpt_path})...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt_path).to(device)
    test_metrics = evaluate(best_model, test_loader, device, criterion)

    print(f"\n{'='*60}")
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro-F1 : {test_metrics['eval_macro_f1']:.4f}")
    print(f"\n{test_metrics['report']}")
    print(f"  Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    print(f"{'='*60}\n")

    # Lưu kết quả evaluation
    eval_result = {
        "best_checkpoint": best_ckpt_path,
        "best_val_macro_f1": best_macro_f1,
        "test_accuracy":   test_metrics["accuracy"],
        "test_macro_f1":   test_metrics["eval_macro_f1"],
        "test_report":     test_metrics["report"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "training_history": history,
    }
    eval_path = os.path.join(cfg["output_dir"], "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    # Copy best model → output_dir/final
    final_path = os.path.join(cfg["output_dir"], "final")
    best_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"[5/5] Done!")
    print(f"  Final model  → {final_path}")
    print(f"  Eval results → {eval_path}")
    print(f"  History      → {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="models/train_config.json",
        help="Đường dẫn tới train_config.json"
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Bỏ qua bước text preprocessing (dùng khi data đã clean)"
    )
    args = parser.parse_args()
    main(args)
