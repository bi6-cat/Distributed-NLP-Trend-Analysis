#!/usr/bin/env python3
"""
spark_jobs/label_topics_llm.py
------------------------------
Đọc topics.csv từ LDA output → gọi Ollama API (Mistral 7B) để sinh label
thông minh → ghi đè file topics.csv + topics.parquet với label mới.

Giữ nguyên hoàn toàn cấu trúc cột:
    topic_id | label | top_keywords | coherence_score | model_version | created_at

Cách chạy:
    # Từ máy host (Ollama chạy trên localhost:11434)
    python spark_jobs/label_topics_llm.py

    # Tuỳ chỉnh đường dẫn / endpoint
    python spark_jobs/label_topics_llm.py \\
        --topics-csv output/lda_staged_m2_run/topics.csv \\
        --ollama-url http://localhost:11434 \\
        --model mistral-local \\
        --dry-run   # chỉ in kết quả, không ghi đè

    # Từ bên trong Docker (spark-master gọi sang ollama service)
    python /opt/spark/work-dir/spark_jobs/label_topics_llm.py \\
        --ollama-url http://ollama:11434
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("label_topics_llm")

# ─────────────────────────────────────────────────────────────────────────────
# Mặc định khớp với docker-compose.yml
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TOPICS_CSV   = "output/lda_staged_m2_run/topics.csv"
DEFAULT_OLLAMA_URL   = "http://localhost:11434"
DEFAULT_MODEL        = "mistral-local"
DEFAULT_TIMEOUT      = 120          # giây, đủ lâu cho CPU inference
MAX_KEYWORDS         = 15           # số từ khoá gửi vào prompt
MAX_RETRIES          = 3


# ─────────────────────────────────────────────────────────────────────────────
# Xây dựng Prompt tiếng Việt
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(keywords_raw: str) -> str:
    """
    keywords_raw: chuỗi kiểu "['màn_hình' 'điện_thoại' 'camera' ...]"
    Trả về prompt theo Mistral Instruct format.
    """
    # Parse keywords từ string representation của numpy/Python list
    kws: List[str] = []
    for token in keywords_raw.replace("[", "").replace("]", "").replace("'", "").split():
        word = token.strip().rstrip(",")
        if word:
            kws.append(word)

    top_kws = ", ".join(kws[:MAX_KEYWORDS])

    return (
        f"[INST] Bạn là chuyên gia phân tích mạng xã hội Việt Nam. "
        f"Dựa vào các từ khóa sau của một chủ đề thảo luận trên mạng xã hội: "
        f"{top_kws}. "
        f"Hãy đặt tên chủ đề này thật ngắn gọn, tối đa 4 từ tiếng Việt, "
        f"phản ánh đúng nội dung. "
        f"Chỉ trả về tên chủ đề, không giải thích gì thêm. [/INST]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra Ollama sẵn sàng
# ─────────────────────────────────────────────────────────────────────────────
def wait_for_ollama(base_url: str, timeout: int = 60) -> bool:
    """Chờ Ollama service sẵn sàng (tối đa timeout giây)."""
    url = f"{base_url}/api/tags"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info(f"✅ Ollama is ready at {base_url}")
                    return True
        except Exception:
            pass
        logger.info("Waiting for Ollama to start...")
        time.sleep(3)
    return False


def list_models(base_url: str) -> List[str]:
    """Lấy danh sách models đã load trong Ollama."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.warning(f"Cannot list models: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Gọi Ollama API
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """
    Gọi Ollama /api/generate endpoint (non-streaming).
    Trả về text label hoặc None nếu lỗi.
    """
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":    0.1,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_predict":    25,
            "stop":           ["</s>", "[INST]", "\n\n"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
                text = result.get("response", "").strip()
                # Loại bỏ ký tự thừa nếu model trả về nhiều hơn cần
                text = text.split("\n")[0].strip().rstrip(".")
                return text if text else None
        except urllib.error.URLError as exc:
            logger.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline chính
# ─────────────────────────────────────────────────────────────────────────────
def label_topics(
    topics_csv: str,
    ollama_url: str,
    model: str,
    dry_run: bool = False,
) -> None:
    """
    Đọc topics.csv → label bằng LLM → ghi đè (hoặc in ra nếu dry_run).
    Cấu trúc cột giữ nguyên 100%:
        topic_id, label, top_keywords, coherence_score, model_version, created_at
    """
    # ── Kiểm tra file đầu vào ──
    if not os.path.exists(topics_csv):
        logger.error(f"topics.csv không tìm thấy: {topics_csv}")
        sys.exit(1)

    # ── Kiểm tra Ollama ──
    if not wait_for_ollama(ollama_url, timeout=30):
        logger.error(f"Ollama không phản hồi tại {ollama_url}")
        sys.exit(1)

    # ── Kiểm tra model đã tồn tại trong Ollama chưa ──
    available = list_models(ollama_url)
    logger.info(f"Models có sẵn trong Ollama: {available}")

    if not any(model in m for m in available):
        logger.warning(
            f"Model '{model}' chưa được import vào Ollama!\n"
            f"Chạy lệnh sau để import:\n"
            f"  docker exec ollama ollama create {model} -f /models/Modelfile"
        )
        # Thử tự import nếu có Modelfile
        logger.info("Thử import model tự động...")
        import subprocess
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "create", model, "-f", "/models/Modelfile"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Import thất bại: {result.stderr}")
            sys.exit(1)
        logger.info("Import model thành công!")

    # ── Đọc CSV ──
    with open(topics_csv, newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    logger.info(f"Đọc {len(rows)} topics từ {topics_csv}")
    logger.info(f"Columns: {fieldnames}")

    # ── Label từng topic ──
    labeled_rows = []
    total = len(rows)
    for i, row in enumerate(rows):
        topic_id      = row.get("topic_id", i)
        keywords_raw  = row.get("top_keywords", "")
        old_label     = row.get("label", "")

        prompt = build_prompt(keywords_raw)

        logger.info(f"[{i+1}/{total}] Topic {topic_id}: '{old_label}' → calling LLM...")
        t0 = time.time()

        if dry_run:
            new_label = f"[DRY-RUN] Topic {topic_id}"
        else:
            new_label = call_ollama(ollama_url, model, prompt)

        elapsed = time.time() - t0

        if new_label:
            logger.info(f"  ✅ '{old_label}' → '{new_label}'  ({elapsed:.1f}s)")
        else:
            # Fallback: giữ label cũ nếu LLM không trả kết quả
            new_label = old_label
            logger.warning(f"  ⚠️  LLM không trả kết quả, giữ label cũ: '{old_label}'")

        # Cập nhật chỉ cột 'label', giữ nguyên toàn bộ cột còn lại
        updated_row = dict(row)
        updated_row["label"] = new_label
        labeled_rows.append(updated_row)

    # ── Ghi đè CSV ──
    if not dry_run:
        with open(topics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(labeled_rows)
        logger.info(f"✅ Đã ghi đè {topics_csv}")

        # ── Cập nhật topics.parquet nếu tồn tại ──
        parquet_path = topics_csv.replace(".csv", ".parquet")
        if os.path.exists(parquet_path):
            try:
                import pandas as pd
                df = pd.read_csv(topics_csv)
                df.to_parquet(parquet_path, index=False)
                logger.info(f"✅ Đã cập nhật {parquet_path}")
            except ImportError:
                logger.warning("pandas không có, bỏ qua cập nhật .parquet")
            except Exception as exc:
                logger.warning(f"Lỗi cập nhật parquet: {exc}")
    else:
        logger.info("\n=== DRY RUN RESULTS ===")
        for row in labeled_rows:
            logger.info(f"  Topic {row['topic_id']}: {row['label']}")

    # ── In tóm tắt ──
    logger.info("\n" + "=" * 60)
    logger.info("LABELING HOÀN TẤT!")
    logger.info(f"  Topics đã label: {total}")
    logger.info("  Kết quả:")
    for row in labeled_rows:
        tid = row.get("topic_id", row.get("\ufefftopic_id", "?"))
        logger.info(f"    [{str(tid):>2}] {row.get('label', '')}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label LDA topics bằng Mistral 7B qua Ollama API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topics-csv", default=DEFAULT_TOPICS_CSV,
        help="Đường dẫn tới topics.csv từ LDA output.",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL,
        help="Base URL của Ollama service (http://localhost:11434 hoặc http://ollama:11434 trong Docker).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Tên model trong Ollama registry (sau khi đã ollama create).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in kết quả, không ghi đè file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_topics(
        topics_csv  = args.topics_csv,
        ollama_url  = args.ollama_url,
        model       = args.model,
        dry_run     = args.dry_run,
    )
