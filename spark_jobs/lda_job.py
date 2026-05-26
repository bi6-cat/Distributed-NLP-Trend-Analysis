#!/usr/bin/env python3
"""
spark_jobs/lda_job.py — Triển khai LDA Topic Modeling trên Spark MLlib

Task 2.1 (Member 3 — Phase 2)
Dự án: Vietnamese Social Media Trend & Controversy Analysis System

Pipeline:
    HDFS Parquet → Vietnamese Text Preprocessing → TF-IDF (CountVectorizer + IDF)
    → LDA (k=15-25, maxIter=50, optimizer="em") → Topic Descriptions → ClickHouse

Chuẩn tiền xử lý (References):
    ① Lowercase normalization — Manning, Raghavan, Schütze (2008).
       "Introduction to Information Retrieval". Cambridge University Press. Chương 2.2.
    ② HTML/URL/emoji removal — Vijayarani et al. (2015).
       "Preprocessing Techniques for Text Mining". Chuẩn UGC preprocessing.
    ③ Teencode/slang normalization — Nguyen et al. (2018).
       "UIT-VSFC: Vietnamese Students' Feedback Corpus". KSE 2018.
       → Chuẩn VLSP Shared Tasks, Zalo AI Challenge community.
    ④ Vietnamese word segmentation — underthesea (BiLSTM-CRF, F1≈97.5%).
       Dựa trên kiến trúc VnCoreNLP (Vu T. Nguyen et al., NAACL 2018).
    ⑤ Stopword removal — 1,942 từ tổng hợp từ VLSP community,
       stopwords-iso/vietnamese, Lê Thanh et al. (2003).

Thuật toán LDA:
    Blei, Ng, Jordan (2003). "Latent Dirichlet Allocation". JMLR 3, 993-1022.
    Optimizer "em" (Expectation-Maximization) cho batch processing.

Cách chạy:
    # Trên cluster (đọc Parquet từ HDFS — phụ thuộc Member 2 infra)
    spark-submit --master spark://master-node:7077 spark_jobs/lda_job.py \\
        --input-path hdfs:///data/staged/ \\
        --output-path hdfs:///data/results/lda/ \\
        --k 20 --max-iter 50

    # Local dev/test (đọc CSV từ data/real/ hoặc data/fake/)
    spark-submit spark_jobs/lda_job.py --local \\
        --input-path data/real/ \\
        --output-path output/lda/ \\
        --k 10 --max-iter 20

Phụ thuộc:
    - Member 2: Spark cluster + HDFS sẵn sàng (Hard blocker)
    - Member 1: Dữ liệu staged Parquet trên HDFS ≥ 50K records
    - Member 4: text_cleaner.py (Soft dep — tự viết preprocessing tạm)
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, FloatType, IntegerType, StringType,
    StructField, StructType, TimestampType,
)
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF
from pyspark.ml.clustering import LDA
from pyspark.ml.functions import vector_to_array

# ============================================================================
# CẤU HÌNH LOGGING
# ============================================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("lda_job")


def log_progress(percent: int, message: str) -> None:
    logger.info(f"[PROGRESS] {percent:>3}% | {message}")


# Cache tokenizer tại process hiện tại để tránh import lặp lại.
_VI_TOKENIZER = None
# Cache reusable preprocessor từ module preprocessing (nếu import được).
_TEXT_PREPROCESSOR = None

# ============================================================================
# HẰNG SỐ MẶC ĐỊNH
# ============================================================================
DEFAULT_K: int = 10              # Tune trên data/preprocessed/result.csv (coherence tốt)
DEFAULT_MAX_ITER: int = 60       # Tăng nhẹ để hội tụ ổn định hơn trên local processed data
DEFAULT_OPTIMIZER: str = "em"    # "em" ổn định hơn "online" cho batch
DEFAULT_VOCAB_SIZE: int = 4_000  # Giảm nhiễu từ hiếm trên tập processed hiện tại
DEFAULT_MIN_DF: int = 3          # Giữ đủ từ khóa quan trọng khi corpus còn nhỏ
MAX_TERMS_PER_TOPIC: int = 15    # Số từ hiển thị cho mỗi topic
DEFAULT_EVAL_MAX_DOCS: int = 5000

# Đường dẫn tài nguyên NLP (Member 3 — Phase 1 deliverables)
STOPWORDS_PATH: str = "data/stopwords_vi.txt"
SLANG_DICT_PATH: str = "data/slang_dict.json"


# ============================================================================
# LOAD TÀI NGUYÊN NLP
# ============================================================================

def _read_reference_text(filepath: str, spark: Optional[SparkSession] = None) -> str:
    if filepath.startswith("hdfs://"):
        if spark is None:
            raise FileNotFoundError(filepath)
        rows = spark.sparkContext.wholeTextFiles(filepath, minPartitions=1).take(1)
        if not rows:
            raise FileNotFoundError(filepath)
        return rows[0][1]

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_stopwords(filepath: str, spark: Optional[SparkSession] = None) -> Set[str]:
    """
    Đọc danh sách Vietnamese stopwords từ file text.

    Danh sách 1,942 từ tổng hợp từ:
        - stopwords-iso/vietnamese (GitHub community)
        - VLSP Shared Tasks (2016-2024)
        - Lê Thanh et al. (2003) — Vietnamese Information Retrieval

    Args:
        filepath: Đường dẫn tới file stopwords (mỗi dòng một từ).

    Returns:
        Set các stopword đã lowercase.
    """
    stopwords: Set[str] = set()
    try:
        content = _read_reference_text(filepath, spark)
        for line in content.splitlines():
            word = line.strip().lower()
            if word:
                stopwords.add(word)
        logger.info(f"Loaded {len(stopwords):,} stopwords from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found: {filepath} — using empty set.")
    return stopwords


def load_slang_dict(filepath: str, spark: Optional[SparkSession] = None) -> Dict[str, str]:
    """
    Đọc bảng chuẩn hóa teencode/slang từ file JSON.

    Format: {"ko": "không", "bth": "bình thường", ...}
    Dự án hiện có ~2,000+ mục, theo chuẩn VLSP/UIT-NLP community.

    Cơ sở: Nguyen et al. (2018), "UIT-VSFC". KSE 2018.
    Hầu hết đội thi VLSP Shared Tasks đều dùng bước normalize slang
    trước khi train model trên dữ liệu mạng xã hội.

    Args:
        filepath: Đường dẫn tới file JSON slang dict.

    Returns:
        Dict mapping {từ_slang: từ_chuẩn}.
    """
    slang_dict: Dict[str, str] = {}
    try:
        content = _read_reference_text(filepath, spark)
        slang_dict = json.loads(content)
        logger.info(f"Loaded {len(slang_dict):,} slang entries from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Slang dict not found: {filepath} — skipping normalization.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format: {filepath}")
    return slang_dict


# ============================================================================
# TIỀN XỬ LÝ VĂN BẢN TIẾNG VIỆT (Spark UDF)
# ============================================================================

def create_preprocessing_udf(stopwords_bc, slang_dict_bc):
    """
    Tạo PySpark UDF tiền xử lý văn bản tiếng Việt.

    Pipeline 5 bước (theo chuẩn VLSP + IR kinh điển):
        ① Lowercase — Manning et al. (2008), Chương 2.2
        ② Remove HTML/URL/emoji/ký tự đặc biệt — chuẩn UGC preprocessing
        ③ Normalize teencode qua slang_dict — VLSP, UIT-VSFC (Nguyen 2018)
        ④ Word segmentation bằng underthesea — BiLSTM-CRF, F1 ≈ 97.5%
           Kiến trúc tương đương VnCoreNLP (Vu T. Nguyen et al., NAACL 2018)
        ⑤ Stopword removal + lọc token — chuẩn IR (Manning 2008)

    Tại sao dùng broadcast variables?
        - stopwords (1,942 từ) và slang_dict (2,000+ mục) cần có mặt trên
          MỌI worker node của Spark cluster.
        - Broadcast giúp gửi 1 lần → cache tại mỗi executor, tránh serialize
          lặp lại cho mỗi task (tiết kiệm network I/O đáng kể trên cluster).

    Args:
        stopwords_bc: Spark broadcast variable chứa Set[str] stopwords.
        slang_dict_bc: Spark broadcast variable chứa Dict[str, str] slang.

    Returns:
        PySpark UDF: (text: str) → List[str] (danh sách token sạch)
    """

    @F.udf(returnType=ArrayType(StringType()))
    def preprocess_vietnamese(text: Optional[str]) -> List[str]:
        """Tiền xử lý 1 văn bản tiếng Việt → danh sách token sạch."""
        return preprocess_vietnamese_text(
            text,
            stopwords_bc.value,
            slang_dict_bc.value,
        )

    return preprocess_vietnamese


def preprocess_vietnamese_text(
    text: Optional[str],
    stopwords: Set[str],
    slang_dict: Dict[str, str],
) -> List[str]:
    """Tiền xử lý tiếng Việt thuần Python cho cả UDF và local fallback."""
    global _VI_TOKENIZER
    global _TEXT_PREPROCESSOR

    if not text or not isinstance(text, str):
        return []

    # Ưu tiên tái sử dụng pipeline chuẩn trong preprocessing/text_cleaner.py
    # để tránh duplicated logic giữa các job.
    if _TEXT_PREPROCESSOR is None:
        try:
            from preprocessing.text_cleaner import TextPreprocessor

            _TEXT_PREPROCESSOR = TextPreprocessor(use_vncorenlp=False)
            logger.info("Using shared TextPreprocessor from preprocessing/.")
        except Exception as exc:
            logger.warning(f"Shared TextPreprocessor unavailable, fallback internal: {exc}")
            _TEXT_PREPROCESSOR = False

    if _TEXT_PREPROCESSOR:
        try:
            normalized_text = _TEXT_PREPROCESSOR.preprocess(text, remove_stopwords=False)
            tokens = normalized_text.split()
        except Exception:
            tokens = text.split()
    else:
        # ① Lowercase — chuẩn hóa chữ hoa/thường
        text = text.lower()

        # ② Loại bỏ nhiễu HTML/URL/emoji/ký tự đặc biệt
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(
            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
            r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
            r"ỳýỷỹỵđ_]",
            " ",
            text,
        )

        # ③ Chuẩn hóa teencode/slang
        words = text.split()
        words = [slang_dict.get(w, w) for w in words]
        text = " ".join(words)

        # ④ Word segmentation — import lazy để giảm overhead
        if _VI_TOKENIZER is None:
            from underthesea import word_tokenize as vi_tokenize

            _VI_TOKENIZER = vi_tokenize

        try:
            tokens = _VI_TOKENIZER(text, format="text").split()
        except Exception:
            # Fallback: split đơn giản nếu underthesea gặp lỗi runtime
            tokens = text.split()

    # ⑤ Stopword removal + lọc token ngắn/số
    cleaned: List[str] = [
        t
        for t in tokens
        if t not in stopwords
        and not t.isdigit()
        and len(t) >= 2
    ]

    return cleaned


# ============================================================================
# KHỞI TẠO SPARK SESSION
# ============================================================================

def create_spark_session(
    app_name: str = "LDA_TopicModeling",
    local: bool = False,
) -> SparkSession:
    """
    Khởi tạo SparkSession cho LDA job.

    Cluster mode nhận executor/driver memory từ spark-submit trong Airflow DAG.
    Job chỉ giữ các cấu hình runtime không phụ thuộc tài nguyên cụ thể.

    Args:
        app_name: Tên ứng dụng Spark (hiển thị trên Spark UI).
        local: True → local[*] mode cho dev/test.

    Returns:
        SparkSession đã cấu hình.
    """
    builder = SparkSession.builder.appName(app_name)

    if local:
        logger.info("Mode: LOCAL (dev/test)")

        # Windows thường map lệnh `python` sang Microsoft Store alias,
        # khiến Spark Python worker không connect lại driver.
        # Ép cả driver/worker dùng đúng interpreter của venv hiện tại.
        python_exec = sys.executable
        os.environ["PYSPARK_PYTHON"] = python_exec
        os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

        builder = builder.master("local[*]")
        builder = (
            builder
            .config("spark.pyspark.python", python_exec)
            .config("spark.pyspark.driver.python", python_exec)
            .config("spark.python.worker.reuse", "false")
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        )
    else:
        # TODO [Member 2]: Cấu hình master URL từ spark_config.py
        # Mặc định dùng cấu hình từ spark-submit --master
        logger.info("Mode: CLUSTER (sử dụng cấu hình từ spark-submit)")

    spark = (
        builder
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession ready — master: {spark.sparkContext.master}")
    return spark


# ============================================================================
# ĐỌC DỮ LIỆU
# ============================================================================

def load_data(
    spark: SparkSession,
    input_path: str,
    local: bool = False,
) -> DataFrame:
    """
    Đọc dữ liệu text từ HDFS (Parquet) hoặc local CSV.

    Cluster mode:
        Đọc Parquet từ HDFS /data/staged/ (output từ Member 2 cleaning_job.py)
        Schema kỳ vọng: [post_id, title, content, source, date, ...]

    Local mode:
        Đọc CSV từ data/real/ hoặc data/fake/
            - Ưu tiên: voz_posts.csv + voz_comments.csv (data/real)
            - Fallback: posts_5k.csv + comments_5k.csv (data/fake)
        Dùng cho dev/test khi chưa có HDFS
        → Workaround cho Hard Blocker từ Member 1 & 2

    Args:
        spark: SparkSession.
        input_path: Đường dẫn dữ liệu (HDFS path hoặc local dir).
        local: True nếu chạy local mode.

    Returns:
        DataFrame với schema canonical:
            - post_id (str)
            - source (str)
            - author (str)
            - created_at (long, unix timestamp)
            - url (str)
            - text (str)
            - preprocessed_text (str, optional)
    """
    def _pick_col_or_lit(df: DataFrame, candidates: List[str], default: str = ""):
        """Chọn cột đầu tiên tồn tại trong candidates, nếu không có thì trả literal."""
        for c in candidates:
            if c in df.columns:
                return F.col(c)
        return F.lit(default)

    def _build_created_at_expr(df: DataFrame):
        """Chuẩn hóa thời gian về unix timestamp (giây) nếu có thể parse."""
        if "created_at" in df.columns:
            return F.col("created_at").cast("long")
        if "time" in df.columns:
            # Ví dụ: "Feb 23, 2026 at 4:00 PM"
            return F.unix_timestamp(F.col("time"), "MMM d, yyyy 'at' h:mm a").cast("long")
        if "time_post" in df.columns:
            # Ví dụ: "Feb 13, 2026"
            return F.unix_timestamp(F.col("time_post"), "MMM d, yyyy").cast("long")
        return F.lit(None).cast("long")

    def _build_id_expr(df: DataFrame, candidates: List[str], prefix: str):
        for c in candidates:
            if c in df.columns:
                return F.col(c).cast("string")
        return F.concat(F.lit(prefix), F.monotonically_increasing_id().cast("string"))

    if local:
        # ── LOCAL MODE: parquet (stg_posts_core từ cleaning_job) hoặc CSV fallback ──
        def _has_parquet(path: str) -> bool:
            if not os.path.isdir(path):
                return False
            for entry in os.listdir(path):
                if entry.startswith("_"):
                    continue
                full = os.path.join(path, entry)
                if entry.endswith(".parquet"):
                    return True
                # Hive-style partition subdirs (e.g. source=voz/)
                if os.path.isdir(full) and "=" in entry:
                    return True
            return False
        has_parquet = _has_parquet(input_path)
        if has_parquet:
            logger.info(f"[LOCAL] Reading parquet from {input_path}")
            df = spark.read.parquet(input_path)
            # Map stg_posts_core schema → canonical schema
            text_col = next(
                (c for c in ["segmented_text", "clean_text", "body", "title", "text"] if c in df.columns),
                None,
            )
            if text_col is None:
                logger.error(f"No text column found in parquet. Columns: {df.columns}")
                sys.exit(1)
            df = df.select(
                _pick_col_or_lit(df, ["post_id"], "").cast("string").alias("post_id"),
                _pick_col_or_lit(df, ["source"], "unknown").cast("string").alias("source"),
                _pick_col_or_lit(df, ["author"], "unknown").cast("string").alias("author"),
                _build_created_at_expr(df).alias("created_at"),
                F.lit("").cast("string").alias("url"),
                F.col(text_col).cast("string").alias("text"),
                _pick_col_or_lit(df, ["clean_text", "segmented_text"], "").cast("string").alias("preprocessed_text"),
            )
            df = df.filter(F.col("text").isNotNull() & (F.trim(F.col("text")) != ""))
            row_count = df.count()
            logger.info(f"Total documents from parquet: {row_count:,}")
            if row_count == 0:
                logger.error("No documents after filtering!")
                sys.exit(1)
            return df

        logger.info(f"Reading CSV from local: {input_path}")
        dfs: List[DataFrame] = []

        # Ưu tiên đọc file đã processed (schema unified).
        processed_path = os.path.join(input_path, "result.csv")
        if os.path.exists(processed_path):
            processed_df = spark.read.csv(processed_path, header=True, inferSchema=True)
            logger.info(f"  result.csv: {processed_df.count():,} rows")
            processed_df = processed_df.select(
                _build_id_expr(processed_df, ["post_id", "id_post", "thread_id"], "r_").alias("post_id"),
                _pick_col_or_lit(processed_df, ["source"], "voz").cast("string").alias("source"),
                _pick_col_or_lit(processed_df, ["author", "author_name", "user"], "unknown").cast("string").alias("author"),
                _build_created_at_expr(processed_df).alias("created_at"),
                _pick_col_or_lit(processed_df, ["url"], "").cast("string").alias("url"),
                _pick_col_or_lit(processed_df, ["content", "text", "noi_dung"], "").cast("string").alias("text"),
                _pick_col_or_lit(processed_df, ["clean_text"], "").cast("string").alias("preprocessed_text"),
            )
            dfs.append(processed_df)

        # Đọc posts (ưu tiên data/real)
        posts_path = None
        for filename in ["posts_preprocessed.csv", "voz_posts.csv", "posts_5k.csv"]:
            candidate = os.path.join(input_path, filename)
            if os.path.exists(candidate):
                posts_path = candidate
                break

        if posts_path and os.path.exists(posts_path):
            posts_df = spark.read.csv(posts_path, header=True, inferSchema=True)
            logger.info(f"  {os.path.basename(posts_path)}: {posts_df.count():,} rows")
            # Ghép các cột text có thể có (tieu_de, noi_dung, title, content)
            text_cols = [
                c for c in ["title_clean", "tieu_de", "noi_dung", "title", "content"]
                if c in posts_df.columns
            ]
            if text_cols:
                posts_df = posts_df.withColumn(
                    "text",
                    F.concat_ws(
                        " ",
                        *[F.coalesce(F.col(c), F.lit("")) for c in text_cols],
                    ),
                )
                posts_df = posts_df.select(
                    _build_id_expr(posts_df, ["id_post", "post_id", "thread_id"], "p_").alias("post_id"),
                    F.lit("voz").alias("source"),
                    _pick_col_or_lit(posts_df, ["author_name", "author", "user"], "unknown").cast("string").alias("author"),
                    _build_created_at_expr(posts_df).alias("created_at"),
                    _pick_col_or_lit(posts_df, ["url"], "").cast("string").alias("url"),
                    F.col("text").cast("string").alias("text"),
                    F.lit(None).cast("string").alias("preprocessed_text"),
                )
                dfs.append(posts_df)

        # Đọc comments (ưu tiên data/real)
        comments_path = None
        for filename in ["comments_preprocessed.csv", "voz_comments.csv", "comments_5k.csv"]:
            candidate = os.path.join(input_path, filename)
            if os.path.exists(candidate):
                comments_path = candidate
                break

        if comments_path and os.path.exists(comments_path):
            comments_df = spark.read.csv(
                comments_path, header=True, inferSchema=True
            )
            logger.info(f"  {os.path.basename(comments_path)}: {comments_df.count():,} rows")
            comment_col = next(
                (
                    c
                    for c in ["comment_clean", "comment", "noi_dung", "content"]
                    if c in comments_df.columns
                ),
                None,
            )
            if comment_col:
                comments_df = comments_df.withColumn("text", F.col(comment_col))
                comments_df = comments_df.select(
                    _build_id_expr(comments_df, ["comment_id", "post_id", "id_post"], "c_").alias("post_id"),
                    F.lit("voz").alias("source"),
                    _pick_col_or_lit(comments_df, ["user", "author", "author_name"], "unknown").cast("string").alias("author"),
                    _build_created_at_expr(comments_df).alias("created_at"),
                    _pick_col_or_lit(comments_df, ["url"], "").cast("string").alias("url"),
                    F.col("text").cast("string").alias("text"),
                    F.lit(None).cast("string").alias("preprocessed_text"),
                )
                dfs.append(comments_df)

        if not dfs:
            logger.error(f"No CSV files found in {input_path}")
            sys.exit(1)

        df = dfs[0]
        for extra_df in dfs[1:]:
            df = df.unionByName(extra_df)
    else:
        # ── CLUSTER MODE: đọc Parquet từ HDFS ──
        # TODO [Member 2]: Xác nhận schema Parquet sau cleaning_job.py
        logger.info(f"Reading Parquet from HDFS: {input_path}")
        df = spark.read.parquet(input_path)

        text_cols = [
            c
            for c in ["title", "segmented_text", "clean_text", "body", "content", "tieu_de", "noi_dung"]
            if c in df.columns
        ]
        if text_cols:
            df = df.withColumn(
                "text",
                F.concat_ws(
                    " ",
                    *[F.coalesce(F.col(c), F.lit("")) for c in text_cols],
                ),
            )
        elif "text" not in df.columns:
            logger.error(
                f"No text column found. Available: {df.columns}"
            )
            sys.exit(1)

        df = df.select(
            _pick_col_or_lit(df, ["post_id", "thread_id", "videoId", "id_post"]).cast("string").alias("post_id"),
            _pick_col_or_lit(df, ["source"], "unknown").cast("string").alias("source"),
            _pick_col_or_lit(df, ["author", "username", "authorDisplayName", "user"], "unknown").cast("string").alias("author"),
            _build_created_at_expr(df).alias("created_at"),
            _pick_col_or_lit(df, ["url", "video_url", "article_url"], "").cast("string").alias("url"),
            F.col("text").cast("string").alias("text"),
            F.lit(None).cast("string").alias("preprocessed_text"),
        )

    # Loại bỏ rows rỗng
    df = df.filter(F.col("text").isNotNull() & (F.trim(F.col("text")) != ""))
    row_count = df.count()
    logger.info(f"Total documents: {row_count:,}")

    if row_count == 0:
        logger.error("No documents after filtering!")
        sys.exit(1)

    return df


def infer_post_topic_assignment(
    lda_model,
    tfidf_df: DataFrame,
) -> DataFrame:
    """
    Gán topic tốt nhất cho từng document sau khi train LDA.

    Output columns:
        post_id, source, author, content, created_at, url,
        topic_id, topic_label, topic_prob
    """
    transformed = lda_model.transform(tfidf_df)

    assignments = (
        transformed
        .withColumn("topic_probs", vector_to_array(F.col("topicDistribution")))
        .withColumn("topic_probability", F.array_max(F.col("topic_probs")))
        .withColumn(
            "topic_id",
            (F.expr("array_position(topic_probs, array_max(topic_probs))") - F.lit(1)).cast("int"),
        )
        .select(
            F.col("post_id"),
            F.col("topic_id"),
            F.col("topic_probability"),
            F.lit("lda").alias("model_type"),
            F.current_timestamp().alias("predicted_at"),
        )
    )
    return assignments


# ============================================================================
# XÂY DỰNG TF-IDF FEATURES
# ============================================================================

def build_tfidf_features(
    df: DataFrame,
    tokens_col: str = "tokens",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_df: int = DEFAULT_MIN_DF,
) -> Tuple[DataFrame, CountVectorizerModel, List[str]]:
    """
    Xây dựng TF-IDF feature vectors từ tokens đã tiền xử lý.

    Sử dụng Spark MLlib pipeline:
        CountVectorizer → term-frequency vectors
        IDF → normalize bằng Inverse Document Frequency

    Lý thuyết TF-IDF:
        TF-IDF(t, d) = TF(t, d) x log(N / DF(t))
        Salton & Buckley (1988), "Term-weighting approaches in automatic text retrieval"
        → Chuẩn mực cho mọi hệ thống IR và Topic Modeling

    Tại sao dùng CountVectorizer thay vì HashingTF?
        - CountVectorizer giữ mapping index→word (cần cho topic interpretation)
        - HashingTF nhanh hơn nhưng mất khả năng truy xuất ngược từ vựng
        - Với vocab_size=10K, CountVectorizer đủ nhanh cho 500K docs

    Args:
        df: DataFrame chứa cột tokens (ArrayType(StringType)).
        tokens_col: Tên cột token.
        vocab_size: Kích thước từ vựng tối đa.
        min_df: Document frequency tối thiểu để giữ từ.

    Returns:
        (tfidf_df, cv_model, vocabulary):
            - tfidf_df: DataFrame chứa cột 'tfidf_features' (SparseVector)
            - cv_model: CountVectorizerModel (lưu lại cho evaluation)
            - vocabulary: List[str] mapping index→word
    """
    logger.info(
        f"Building TF-IDF — vocab_size={vocab_size}, min_df={min_df}"
    )

    # Bước 1: CountVectorizer → Term Frequency
    cv = CountVectorizer(
        inputCol=tokens_col,
        outputCol="tf_features",
        vocabSize=vocab_size,
        minDF=float(min_df),
    )
    cv_model: CountVectorizerModel = cv.fit(df)
    tf_df = cv_model.transform(df)

    vocabulary: List[str] = list(cv_model.vocabulary)
    logger.info(f"  Vocabulary size: {len(vocabulary):,} terms")

    # Bước 2: IDF → Inverse Document Frequency normalization
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
    idf_model = idf.fit(tf_df)
    tfidf_df = idf_model.transform(tf_df)

    logger.info("TF-IDF features ready.")
    return tfidf_df, cv_model, vocabulary


# ============================================================================
# HUẤN LUYỆN LDA
# ============================================================================

def train_lda(
    features_df: DataFrame,
    k: int = DEFAULT_K,
    max_iter: int = DEFAULT_MAX_ITER,
    optimizer: str = DEFAULT_OPTIMIZER,
    seed: int = 42,
) -> Tuple:
    """
    Huấn luyện mô hình LDA trên Spark MLlib.

    Thuật toán: Latent Dirichlet Allocation
        Blei, Ng, Jordan (2003). "Latent Dirichlet Allocation". JMLR 3.
        Mỗi document = hỗn hợp K topics (Dirichlet prior α)
        Mỗi topic = phân phối xác suất trên vocabulary (Dirichlet prior β)

    Optimizer EM vs Online:
        "em" (Expectation-Maximization):
            → Ổn định, hội tụ tốt cho batch processing trên corpus tĩnh.
            → Tốt cho dữ liệu đã crawl xong (Phase 2).
        "online" (Online Variational Bayes):
            → Nhanh hơn, phù hợp cho incremental learning.
            → Dùng ở Phase 3 nếu cần update liên tục.

    Chọn k = 15-25 (theo TECH_STACK.md):
        → Dữ liệu tiếng Việt mạng xã hội (VOZ, VnExpress, YouTube) có
          đa dạng chủ đề nhưng không quá phân tán.
        → k quá nhỏ (<10) → topics quá tổng quát, k quá lớn (>30) → fragmented.
        → Giá trị tối ưu xác định qua coherence score (Task 2.2).

    Args:
        features_df: DataFrame chứa cột 'tf_features'.
        k: Số topics (khuyến nghị 15-25).
        max_iter: Số vòng lặp tối đa.
        optimizer: "em" hoặc "online".
        seed: Random seed cho reproducibility.

    Returns:
        (lda_model, log_likelihood, log_perplexity)
    """
    logger.info(
        f"Training LDA — k={k}, maxIter={max_iter}, optimizer={optimizer}"
    )

    lda = LDA(
        k=k,
        maxIter=max_iter,
        featuresCol="tf_features",
        optimizer=optimizer,
        seed=seed,
        # subsamplingRate chỉ áp dụng cho "online" optimizer
        subsamplingRate=0.05 if optimizer == "online" else 1.0,
    )

    lda_model = lda.fit(features_df)

    # Đánh giá nội tại (intrinsic evaluation)
    # Log-Likelihood: cao hơn → model fit data tốt hơn
    # Log-Perplexity: thấp hơn → model dự đoán tốt hơn
    ll = lda_model.logLikelihood(features_df)
    lp = lda_model.logPerplexity(features_df)

    logger.info(f"  Log-Likelihood:  {ll:,.2f}")
    logger.info(f"  Log-Perplexity:  {lp:.4f}")
    logger.info("LDA training complete.")

    return lda_model, ll, lp


def evaluate_k_sweep(
    features_df: DataFrame,
    vocabulary: List[str],
    token_texts: List[List[str]],
    k_values: List[int],
    max_iter: int,
    optimizer: str,
    output_path: str,
) -> None:
    """Chạy sweep k và lưu metrics (LL/Perplexity/Coherence) ra CSV."""
    if not k_values:
        return

    coherence_ready = bool(token_texts)
    dictionary = None
    corpus = None
    if coherence_ready:
        try:
            from gensim.corpora import Dictionary

            dictionary = Dictionary(token_texts)
            corpus = [dictionary.doc2bow(t) for t in token_texts]
        except Exception as exc:
            logger.warning(f"Coherence disabled (gensim unavailable/error): {exc}")
            coherence_ready = False

    metrics_rows: List[Dict[str, object]] = []
    for k in k_values:
        logger.info(f"[EVAL] Running k={k}")
        lda_model, ll, lp = train_lda(
            features_df,
            k=k,
            max_iter=max_iter,
            optimizer=optimizer,
        )
        topics = extract_topics(lda_model, vocabulary)

        row: Dict[str, object] = {
            "k": k,
            "log_likelihood": float(ll),
            "log_perplexity": float(lp),
            "coherence_c_v": None,
            "coherence_u_mass": None,
        }

        if coherence_ready and dictionary is not None and corpus is not None:
            try:
                from gensim.models import CoherenceModel

                topic_words = [[kw["word"] for kw in t["keywords"]] for t in topics]
                cv = CoherenceModel(
                    topics=topic_words,
                    texts=token_texts,
                    dictionary=dictionary,
                    coherence="c_v",
                ).get_coherence()
                umass = CoherenceModel(
                    topics=topic_words,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence="u_mass",
                ).get_coherence()
                row["coherence_c_v"] = float(cv)
                row["coherence_u_mass"] = float(umass)
            except Exception as exc:
                logger.warning(f"[EVAL] Coherence failed for k={k}: {exc}")

        metrics_rows.append(row)

    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "lda_k_sweep_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "log_likelihood", "log_perplexity", "coherence_c_v", "coherence_u_mass"],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    logger.info(f"[EVAL] Saved k-sweep metrics → {csv_path}")


# ============================================================================
# TRÍCH XUẤT & MÔ TẢ TOPICS
# ============================================================================

def extract_topics(
    lda_model,
    vocabulary: List[str],
    max_terms: int = MAX_TERMS_PER_TOPIC,
) -> List[Dict]:
    """
    Trích xuất top từ khóa đại diện cho mỗi topic từ LDA model.

    LDA model lưu topic-word distribution dưới dạng ma trận K×V.
    Hàm này convert indices → readable words qua vocabulary mapping.

    Args:
        lda_model: Mô hình LDA đã train (DistributedLDAModel).
        vocabulary: Danh sách từ vựng (index→word) từ CountVectorizer.
        max_terms: Số từ khóa tối đa cho mỗi topic.

    Returns:
        List[Dict] — mỗi topic chứa:
            topic_id (int), topic_label (str), keywords (List[Dict])
    """
    topics_df = lda_model.describeTopics(maxTermsPerTopic=max_terms)
    topics_list = topics_df.collect()

    results: List[Dict] = []
    for row in topics_list:
        topic_id: int = row["topic"]
        term_indices: List[int] = row["termIndices"]
        term_weights: List[float] = row["termWeights"]

        keywords: List[Dict] = []
        for idx, weight in zip(term_indices, term_weights):
            word = vocabulary[idx] if idx < len(vocabulary) else f"UNK_{idx}"
            keywords.append({"word": word, "weight": float(weight)})

        # Tạo label tự động từ top-3 keywords
        # Ví dụ: "iphone | giá | camera" → dễ đọc trên dashboard
        top3_words = [kw["word"] for kw in keywords[:3]]
        topic_label = " | ".join(top3_words)

        results.append({
            "topic_id": topic_id,
            "topic_label": topic_label,
            "keywords": keywords,
        })

        # Log top-5 keywords
        kw_str = ", ".join(
            f'{kw["word"]}({kw["weight"]:.3f})' for kw in keywords[:5]
        )
        logger.info(f"  Topic {topic_id}: [{topic_label}] — {kw_str}")

    return results


# ============================================================================
# LƯU KẾT QUẢ
# ============================================================================

def save_results(
    spark: SparkSession,
    lda_model,
    topics: List[Dict],
    assignments_df: DataFrame,
    output_path: str,
    local: bool = False,
    k: int = None,
) -> None:
    """
    Lưu mô hình LDA và mô tả topics.

    Cluster mode:
        - LDA model → HDFS: {output_path}/lda_model/
        - Topics Parquet → HDFS: {output_path}/topics/
        - TODO: Topics → ClickHouse stg_topic_labels (cần Member 2 JDBC config)

    Local mode:
        - LDA model → local: {output_path}/lda_model/
        - Topics JSON → local: {output_path}/topics.json
        - Post-topic CSV → local: {output_path}/post_topic_assignment.csv

    Args:
        spark: SparkSession.
        lda_model: Mô hình LDA đã train.
        topics: Danh sách topic descriptions từ extract_topics().
        output_path: Thư mục output.
        local: True nếu local mode.
    """
    if local:
        # Trên Windows không có winutils.exe, thao tác ghi Spark model có thể lỗi.
        model_path = os.path.join(output_path, "lda_model")
        try:
            logger.info(f"Saving LDA model → {model_path}")
            lda_model.write().overwrite().save(model_path)
        except Exception as exc:
            logger.warning(f"Skipping LDA model save in local mode: {exc}")

        # ── Local: lưu parquet đơn (cùng format với BERTopic, đọc được bởi save_topics_to_ch) ──
        import pandas as _pd
        model_version = f"lda_k{k}" if k else f"lda_k{len(topics)}"
        now_ts = datetime.now(tz=timezone.utc)
        os.makedirs(output_path, exist_ok=True)

        topics_export = [
            {
                "topic_id":       int(t["topic_id"]),
                "label":          str(t["topic_label"]),
                "top_keywords":   [kw["word"] for kw in t["keywords"]],
                "coherence_score": None,
                "model_version":  model_version,
                "created_at":     now_ts,
            }
            for t in topics
        ]
        topics_parquet = os.path.join(output_path, "topics.parquet")
        _pd.DataFrame(topics_export).to_parquet(topics_parquet, index=False)
        logger.info(f"Saving topics → {topics_parquet}")

        assignment_parquet = os.path.join(output_path, "post_topic_assignment.parquet")
        assignments_df.toPandas().to_parquet(assignment_parquet, index=False)
        logger.info(f"Saving post-topic assignments → {assignment_parquet}")
    else:
        # ── Lưu LDA model (Spark MLlib format) ──
        model_path = os.path.join(output_path, "lda_model")
        logger.info(f"Saving LDA model → {model_path}")
        lda_model.write().overwrite().save(model_path)

        # ── Cluster: lưu topics Parquet (spec: stg_topics schema) ──
        model_version = f"lda_k{k}" if k else f"lda_k{len(topics)}"
        now = datetime.now(tz=timezone.utc)
        topics_schema = StructType([
            StructField("topic_id", IntegerType(), False),
            StructField("label", StringType(), False),
            StructField("top_keywords", ArrayType(StringType()), False),
            StructField("coherence_score", FloatType(), True),
            StructField("model_version", StringType(), False),
            StructField("created_at", TimestampType(), False),
        ])
        topics_rows = [
            (
                int(t["topic_id"]),
                str(t["topic_label"]),
                [kw["word"] for kw in t["keywords"]],
                None,
                model_version,
                now,
            )
            for t in topics
        ]
        topics_df = spark.createDataFrame(topics_rows, schema=topics_schema)
        topics_parquet_path = os.path.join(output_path, "topics")
        topics_df.write.mode("overwrite").parquet(topics_parquet_path)
        logger.info(f"Saving topics Parquet → {topics_parquet_path}")

        # post_topics schema: post_id, topic_id, topic_probability, model_type, predicted_at
        assignment_parquet_path = os.path.join(output_path, "post_topic_assignment")
        assignments_df.write.mode("overwrite").parquet(assignment_parquet_path)
        logger.info(f"Saving post-topic assignments Parquet → {assignment_parquet_path}")

        # TODO [Member 2/5]: Ghi vào ClickHouse stg_topic_labels
        # Cần ClickHouse JDBC driver cấu hình trong Spark (Task 2.2 Member 2)
        # ─────────────────────────────────────────────────────────
        # topics_df.write \
        #     .format("jdbc") \
        #     .option("url", "jdbc:clickhouse://storage-node:8123/nlp_db") \
        #     .option("dbtable", "stg_topic_labels") \
        #     .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        #     .mode("append") \
        #     .save()
        # ─────────────────────────────────────────────────────────

    logger.info("All results saved.")


def save_vocabulary(
    cv_model: CountVectorizerModel,
    output_path: str,
    local: bool = False,
) -> None:
    """
    Lưu CountVectorizerModel và vocabulary JSON.

    Notebook đánh giá (Task 2.2) cần vocabulary để:
        1. Map term indices → readable words cho CoherenceModel
        2. Xây dựng cấu trúc tương thích gensim

    Args:
        cv_model: CountVectorizerModel đã fit.
        output_path: Thư mục output.
        local: True nếu local mode.
    """
    if local:
        os.makedirs(output_path, exist_ok=True)

        cv_path = os.path.join(output_path, "cv_model")
        try:
            logger.info(f"Saving CountVectorizerModel → {cv_path}")
            cv_model.write().overwrite().save(cv_path)
        except Exception as exc:
            logger.warning(f"Skipping CountVectorizerModel save in local mode: {exc}")

        vocab_json_path = os.path.join(output_path, "vocabulary.json")
        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(list(cv_model.vocabulary), f, ensure_ascii=False, indent=2)
        logger.info(f"Saving vocabulary JSON → {vocab_json_path}")
    else:
        # Lưu Spark CountVectorizerModel
        cv_path = os.path.join(output_path, "cv_model")
        logger.info(f"Saving CountVectorizerModel → {cv_path}")
        cv_model.write().overwrite().save(cv_path)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "LDA Topic Modeling trên Spark MLlib "
            "— Vietnamese Social Media Trend Analysis"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path", type=str, default="data/real/",
        help="Đường dẫn dữ liệu đầu vào (HDFS hoặc local dir).",
    )
    parser.add_argument(
        "--output-path", type=str, default="output/lda/",
        help="Đường dẫn lưu kết quả (HDFS hoặc local dir).",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help="Số lượng topics (khuyến nghị 15-25).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=DEFAULT_MAX_ITER,
        help="Số vòng lặp EM tối đa.",
    )
    parser.add_argument(
        "--optimizer", type=str, default=DEFAULT_OPTIMIZER,
        choices=["em", "online"],
        help="Thuật toán tối ưu LDA.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
        help="Kích thước từ vựng tối đa.",
    )
    parser.add_argument(
        "--min-df", type=int, default=DEFAULT_MIN_DF,
        help="Document frequency tối thiểu.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Chạy local mode (dev/test với data/real/ hoặc data/fake/).",
    )
    parser.add_argument(
        "--stopwords-path", type=str, default=STOPWORDS_PATH,
        help="Đường dẫn file stopwords.",
    )
    parser.add_argument(
        "--slang-dict-path", type=str, default=SLANG_DICT_PATH,
        help="Đường dẫn file slang dictionary.",
    )
    parser.add_argument(
        "--eval-k-values", type=str, default="",
        help="Danh sách k để sweep, ví dụ: '10,12,15,18,20,25'.",
    )
    parser.add_argument(
        "--eval-max-docs", type=int, default=DEFAULT_EVAL_MAX_DOCS,
        help="Số documents tối đa dùng để tính coherence trong k-sweep.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point — chạy toàn bộ LDA pipeline.

    Pipeline end-to-end:
        1. Khởi tạo Spark
        2. Load tài nguyên NLP (stopwords, slang dict) → broadcast
        3. Đọc dữ liệu (HDFS Parquet hoặc local CSV)
        4. Tiền xử lý tiếng Việt (5 bước theo chuẩn VLSP + IR)
        5. Xây dựng TF-IDF features (CountVectorizer + IDF)
        6. Huấn luyện LDA (k topics, maxIter vòng)
        7. Trích xuất & hiển thị topics
        8. Lưu kết quả (model + topics + vocabulary)
    """
    args = parse_args()

    logger.info("=" * 70)
    logger.info("LDA TOPIC MODELING — Vietnamese Social Media")
    logger.info(f"  k={args.k}, maxIter={args.max_iter}, optimizer={args.optimizer}")
    logger.info(f"  input={args.input_path}, output={args.output_path}")
    logger.info(f"  mode={'LOCAL' if args.local else 'CLUSTER'}")
    logger.info("=" * 70)

    # 1. Khởi tạo Spark
    log_progress(5, "Starting Spark session")
    spark = create_spark_session(local=args.local)

    try:
        # 2. Load & broadcast tài nguyên NLP
        log_progress(10, "Loading NLP reference files")
        stopwords = load_stopwords(args.stopwords_path, spark)
        slang_dict = load_slang_dict(args.slang_dict_path, spark)
        stopwords_bc = spark.sparkContext.broadcast(stopwords)
        slang_dict_bc = spark.sparkContext.broadcast(slang_dict)

        # 3. Đọc dữ liệu
        log_progress(15, "Reading staged documents")
        df = load_data(spark, args.input_path, local=args.local)

        # 4. Tiền xử lý tiếng Việt
        log_progress(25, "Preprocessing Vietnamese text")
        logger.info("Preprocessing Vietnamese text...")
        if args.local:
            # Local fallback: dùng hoàn toàn Spark SQL built-in để tránh
            # crash Python worker trên Windows.
            logger.info("Using Spark SQL local preprocessing path (no Python UDF workers).")
            has_preprocessed = "preprocessed_text" in df.columns
            if has_preprocessed:
                preprocessed_count = df.filter(
                    F.col("preprocessed_text").isNotNull()
                    & (F.trim(F.col("preprocessed_text")) != "")
                ).count()
            else:
                preprocessed_count = 0

            if preprocessed_count > 0:
                logger.info(
                    f"Using preprocessed_text directly for tokenization ({preprocessed_count:,} docs)."
                )
                df = (
                    df
                    .withColumn("preprocessed_text", F.lower(F.col("preprocessed_text")))
                    .withColumn("tokens", F.split(F.trim(F.col("preprocessed_text")), r"\s+"))
                    .withColumn(
                        "tokens",
                        F.expr("filter(tokens, x -> length(x) >= 2 AND NOT x rlike '^[0-9]+$')"),
                    )
                    .filter(F.size(F.col("tokens")) > 0)
                )
            else:
                stopwords_lit = F.array(*[F.lit(w) for w in sorted(stopwords)])
                df = (
                    df
                    .withColumn("clean_text", F.lower(F.col("text")))
                    .withColumn("clean_text", F.regexp_replace(F.col("clean_text"), r"<[^>]+>", " "))
                    .withColumn("clean_text", F.regexp_replace(F.col("clean_text"), r"https?://\S+|www\.\S+", " "))
                    .withColumn(
                        "clean_text",
                        F.regexp_replace(
                            F.col("clean_text"),
                            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ_]",
                            " ",
                        ),
                    )
                    .withColumn("words", F.split(F.trim(F.col("clean_text")), r"\s+"))
                    .withColumn("_stopwords", stopwords_lit)
                    .withColumn(
                        "tokens",
                        F.expr(
                            "filter(words, x -> length(x) >= 2 AND NOT x rlike '^[0-9]+$' AND NOT array_contains(_stopwords, x))"
                        ),
                    )
                    .drop("clean_text", "words", "_stopwords")
                    .filter(F.size(F.col("tokens")) > 0)
                )
        else:
            preprocess_udf = create_preprocessing_udf(stopwords_bc, slang_dict_bc)
            df = df.withColumn("tokens", preprocess_udf(F.col("text")))
            df = df.filter(F.size(F.col("tokens")) > 0)

        doc_count = df.count()
        logger.info(f"  Documents after preprocessing: {doc_count:,}")
        df.cache()
        log_progress(40, f"Preprocessing complete: {doc_count:,} documents")

        # 5. TF-IDF
        log_progress(45, "Building TF-IDF features")
        tfidf_df, cv_model, vocabulary = build_tfidf_features(
            df,
            tokens_col="tokens",
            vocab_size=args.vocab_size,
            min_df=args.min_df,
        )
        log_progress(60, f"TF-IDF ready: {len(vocabulary):,} vocabulary terms")

        # 6. Huấn luyện LDA (train trên TF counts)
        log_progress(65, f"Training LDA model with k={args.k}")
        lda_model, ll, lp = train_lda(
            tfidf_df,
            k=args.k,
            max_iter=args.max_iter,
            optimizer=args.optimizer,
        )
        log_progress(78, "LDA training complete")

        # 7. Trích xuất topics
        log_progress(82, "Extracting topic descriptions")
        logger.info("Extracting topic descriptions:")
        topics = extract_topics(lda_model, vocabulary)

        # 8. Gán topic cho từng bài viết/comment
        log_progress(86, "Inferring best topic per document")
        logger.info("Inferring best topic per document...")
        assignments_df = infer_post_topic_assignment(lda_model, tfidf_df)

        # 8.5 Sweep k tự động (nếu được bật)
        if args.eval_k_values.strip():
            try:
                k_values = [int(x.strip()) for x in args.eval_k_values.split(",") if x.strip()]
                k_values = sorted(set(k_values))
            except ValueError:
                logger.error("Invalid --eval-k-values format. Example: 10,12,15,18")
                k_values = []

            token_rows = (
                df.select("tokens")
                .limit(max(1, int(args.eval_max_docs)))
                .collect()
            )
            token_texts: List[List[str]] = [r["tokens"] for r in token_rows if r["tokens"]]

            evaluate_k_sweep(
                tfidf_df,
                vocabulary,
                token_texts,
                k_values,
                args.max_iter,
                args.optimizer,
                args.output_path,
            )

        # 9. Lưu kết quả
        log_progress(92, "Saving LDA outputs")
        save_results(
            spark,
            lda_model,
            topics,
            assignments_df,
            args.output_path,
            local=args.local,
            k=args.k,
        )
        save_vocabulary(cv_model, args.output_path, local=args.local)
        log_progress(98, "Saved LDA outputs")

        logger.info("=" * 70)
        logger.info("LDA PIPELINE COMPLETE!")
        logger.info(f"  Topics: {len(topics)}")
        logger.info(f"  Post-topic rows: {assignments_df.count():,}")
        logger.info(f"  Log-Likelihood: {ll:,.2f}")
        logger.info(f"  Log-Perplexity: {lp:.4f}")
        logger.info(f"  Output: {args.output_path}")
        logger.info("=" * 70)
        log_progress(100, "LDA job complete")

    finally:
        spark.stop()
        logger.info("SparkSession stopped.")


if __name__ == "__main__":
    main()
