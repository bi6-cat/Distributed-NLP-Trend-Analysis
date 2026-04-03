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

    # Local dev/test (đọc CSV từ data/fake/)
    spark-submit spark_jobs/lda_job.py --local \\
        --input-path data/fake/ \\
        --output-path output/lda/ \\
        --k 10 --max-iter 20

Phụ thuộc:
    - Member 2: Spark cluster + HDFS sẵn sàng (Hard blocker)
    - Member 1: Dữ liệu staged Parquet trên HDFS ≥ 50K records
    - Member 4: text_cleaner.py (Soft dep — tự viết preprocessing tạm)
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF
from pyspark.ml.clustering import LDA

# ============================================================================
# CẤU HÌNH LOGGING
# ============================================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("lda_job")

# ============================================================================
# HẰNG SỐ MẶC ĐỊNH
# ============================================================================
DEFAULT_K: int = 20              # Số topics (thử nghiệm 15-25 theo TECH_STACK)
DEFAULT_MAX_ITER: int = 50       # Số vòng lặp EM (Blei et al., 2003)
DEFAULT_OPTIMIZER: str = "em"    # "em" ổn định hơn "online" cho batch
DEFAULT_VOCAB_SIZE: int = 10_000 # Kích thước từ vựng tối đa
DEFAULT_MIN_DF: int = 5          # Document frequency tối thiểu
MAX_TERMS_PER_TOPIC: int = 15    # Số từ hiển thị cho mỗi topic

# Đường dẫn tài nguyên NLP (Member 3 — Phase 1 deliverables)
STOPWORDS_PATH: str = "data/stopwords_vi.txt"
SLANG_DICT_PATH: str = "data/slang_dict.json"


# ============================================================================
# LOAD TÀI NGUYÊN NLP
# ============================================================================

def load_stopwords(filepath: str) -> Set[str]:
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
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
        logger.info(f"Loaded {len(stopwords):,} stopwords from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found: {filepath} — using empty set.")
    return stopwords


def load_slang_dict(filepath: str) -> Dict[str, str]:
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
        with open(filepath, "r", encoding="utf-8") as f:
            slang_dict = json.load(f)
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
        if not text or not isinstance(text, str):
            return []

        # Import underthesea bên trong UDF vì:
        # - Module không serializable qua pickle
        # - Cần import tại mỗi executor (worker) riêng biệt
        from underthesea import word_tokenize as vi_tokenize

        _stopwords: Set[str] = stopwords_bc.value
        _slang_dict: Dict[str, str] = slang_dict_bc.value

        # ① Lowercase — chuẩn hóa chữ hoa/thường
        # Lý do: "Việt Nam" và "việt nam" phải là cùng entity
        # Ref: Manning et al. (2008), Chương 2.2.1
        text = text.lower()

        # ② Loại bỏ nhiễu HTML/URL/emoji/ký tự đặc biệt
        # Lý do: Dữ liệu crawl từ VOZ, VnExpress chứa HTML tags, URLs
        # Ref: Vijayarani et al. (2015), "Preprocessing Techniques for Text Mining"
        text = re.sub(r"<[^>]+>", " ", text)                # HTML tags
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
        # Giữ lại chữ cái tiếng Việt (có dấu), số, và khoảng trắng
        text = re.sub(
            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
            r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
            r"ỳýỷỹỵđ_]",
            " ",
            text,
        )

        # ③ Chuẩn hóa teencode/slang
        # Lý do: Mạng xã hội VN chứa rất nhiều teencode ("ko"→"không")
        # Ref: Nguyen et al. (2018), UIT-VSFC, KSE 2018
        words = text.split()
        words = [_slang_dict.get(w, w) for w in words]
        text = " ".join(words)

        # ④ Word segmentation — underthesea (BiLSTM-CRF)
        # Lý do: Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không rõ ràng
        # "học sinh" → "học_sinh" (1 từ ghép, không phải 2 từ đơn)
        # Ref: Vu T. Nguyen et al. (2018), VnCoreNLP, NAACL 2018
        try:
            tokens = vi_tokenize(text, format="text").split()
        except Exception:
            # Fallback: split đơn giản nếu underthesea gặp lỗi
            tokens = text.split()

        # ⑤ Stopword removal + lọc token ngắn/số
        # Lý do: Từ chức năng ("của", "và", "là") không mang ngữ nghĩa topic
        # Ref: Manning et al. (2008), Chương 2.2.2
        cleaned: List[str] = [
            t
            for t in tokens
            if t not in _stopwords  # Loại 1,942 stopwords
            and not t.isdigit()     # Loại token toàn số
            and len(t) >= 2         # Loại token quá ngắn (1 ký tự)
        ]

        return cleaned

    return preprocess_vietnamese


# ============================================================================
# KHỞI TẠO SPARK SESSION
# ============================================================================

def create_spark_session(
    app_name: str = "LDA_TopicModeling",
    local: bool = False,
) -> SparkSession:
    """
    Khởi tạo SparkSession cho LDA job.

    Cấu hình theo TECH_STACK.md (Section 4.2):
        - executor.memory = 8g (mỗi worker node 8GB RAM)
        - driver.memory = 4g
        - shuffle.partitions = 200 (tối ưu cho cluster 3 workers)

    Args:
        app_name: Tên ứng dụng Spark (hiển thị trên Spark UI).
        local: True → local[*] mode cho dev/test.

    Returns:
        SparkSession đã cấu hình.
    """
    builder = SparkSession.builder.appName(app_name)

    if local:
        logger.info("Mode: LOCAL (dev/test)")
        builder = builder.master("local[*]")
    else:
        # TODO [Member 2]: Cấu hình master URL từ spark_config.py
        # Mặc định dùng cấu hình từ spark-submit --master
        logger.info("Mode: CLUSTER (sử dụng cấu hình từ spark-submit)")

    spark = (
        builder
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "200")
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
        Đọc CSV từ data/fake/ (posts_5k.csv + comments_5k.csv)
        Dùng cho dev/test khi chưa có HDFS
        → Workaround cho Hard Blocker từ Member 1 & 2

    Args:
        spark: SparkSession.
        input_path: Đường dẫn dữ liệu (HDFS path hoặc local dir).
        local: True nếu chạy local mode.

    Returns:
        DataFrame với cột 'text' chứa nội dung văn bản gốc.
    """
    if local:
        # ── LOCAL MODE: đọc CSV từ data/fake/ ──
        logger.info(f"Reading CSV from local: {input_path}")
        dfs: List[DataFrame] = []

        # Đọc posts
        posts_path = os.path.join(input_path, "posts_5k.csv")
        if os.path.exists(posts_path):
            posts_df = spark.read.csv(posts_path, header=True, inferSchema=True)
            logger.info(f"  posts_5k.csv: {posts_df.count():,} rows")
            # Ghép các cột text có thể có (tieu_de, noi_dung, title, content)
            text_cols = [
                c for c in ["tieu_de", "noi_dung", "title", "content"]
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
                dfs.append(posts_df.select("text"))

        # Đọc comments
        comments_path = os.path.join(input_path, "comments_5k.csv")
        if os.path.exists(comments_path):
            comments_df = spark.read.csv(
                comments_path, header=True, inferSchema=True
            )
            logger.info(f"  comments_5k.csv: {comments_df.count():,} rows")
            comment_col = next(
                (
                    c
                    for c in ["comment", "noi_dung", "content"]
                    if c in comments_df.columns
                ),
                None,
            )
            if comment_col:
                comments_df = comments_df.withColumn("text", F.col(comment_col))
                dfs.append(comments_df.select("text"))

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
            c for c in ["title", "content", "tieu_de", "noi_dung"]
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

    # Loại bỏ rows rỗng
    df = df.filter(F.col("text").isNotNull() & (F.trim(F.col("text")) != ""))
    row_count = df.count()
    logger.info(f"Total documents: {row_count:,}")

    if row_count == 0:
        logger.error("No documents after filtering!")
        sys.exit(1)

    return df


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
    tfidf_df: DataFrame,
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
        tfidf_df: DataFrame chứa cột 'tfidf_features'.
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
        featuresCol="tfidf_features",
        optimizer=optimizer,
        seed=seed,
        # subsamplingRate chỉ áp dụng cho "online" optimizer
        subsamplingRate=0.05 if optimizer == "online" else 1.0,
    )

    lda_model = lda.fit(tfidf_df)

    # Đánh giá nội tại (intrinsic evaluation)
    # Log-Likelihood: cao hơn → model fit data tốt hơn
    # Log-Perplexity: thấp hơn → model dự đoán tốt hơn
    ll = lda_model.logLikelihood(tfidf_df)
    lp = lda_model.logPerplexity(tfidf_df)

    logger.info(f"  Log-Likelihood:  {ll:,.2f}")
    logger.info(f"  Log-Perplexity:  {lp:.4f}")
    logger.info("LDA training complete.")

    return lda_model, ll, lp


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
    output_path: str,
    local: bool = False,
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

    Args:
        spark: SparkSession.
        lda_model: Mô hình LDA đã train.
        topics: Danh sách topic descriptions từ extract_topics().
        output_path: Thư mục output.
        local: True nếu local mode.
    """
    # ── Lưu LDA model (Spark MLlib format) ──
    model_path = os.path.join(output_path, "lda_model")
    logger.info(f"Saving LDA model → {model_path}")
    lda_model.write().overwrite().save(model_path)

    if local:
        # ── Local: lưu topics JSON ──
        topics_path = os.path.join(output_path, "topics.json")
        os.makedirs(output_path, exist_ok=True)
        with open(topics_path, "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saving topics → {topics_path}")
    else:
        # ── Cluster: lưu topics Parquet trên HDFS ──
        topics_flat: List[Dict] = []
        for t in topics:
            for kw in t["keywords"]:
                topics_flat.append({
                    "topic_id": t["topic_id"],
                    "topic_label": t["topic_label"],
                    "keyword": kw["word"],
                    "weight": kw["weight"],
                })
        topics_df = spark.createDataFrame(topics_flat)
        topics_parquet_path = os.path.join(output_path, "topics")
        topics_df.write.mode("overwrite").parquet(topics_parquet_path)
        logger.info(f"Saving topics Parquet → {topics_parquet_path}")

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
    # Lưu Spark CountVectorizerModel
    cv_path = os.path.join(output_path, "cv_model")
    logger.info(f"Saving CountVectorizerModel → {cv_path}")
    cv_model.write().overwrite().save(cv_path)

    # Lưu vocabulary JSON (tiện cho notebook evaluation)
    if local:
        os.makedirs(output_path, exist_ok=True)
        vocab_json_path = os.path.join(output_path, "vocabulary.json")
        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(list(cv_model.vocabulary), f, ensure_ascii=False, indent=2)
        logger.info(f"Saving vocabulary JSON → {vocab_json_path}")


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
        "--input-path", type=str, default="data/fake/",
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
        help="Chạy local mode (dev/test với data/fake/).",
    )
    parser.add_argument(
        "--stopwords-path", type=str, default=STOPWORDS_PATH,
        help="Đường dẫn file stopwords.",
    )
    parser.add_argument(
        "--slang-dict-path", type=str, default=SLANG_DICT_PATH,
        help="Đường dẫn file slang dictionary.",
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
    spark = create_spark_session(local=args.local)

    try:
        # 2. Load & broadcast tài nguyên NLP
        stopwords = load_stopwords(args.stopwords_path)
        slang_dict = load_slang_dict(args.slang_dict_path)
        stopwords_bc = spark.sparkContext.broadcast(stopwords)
        slang_dict_bc = spark.sparkContext.broadcast(slang_dict)

        # 3. Đọc dữ liệu
        df = load_data(spark, args.input_path, local=args.local)

        # 4. Tiền xử lý tiếng Việt
        logger.info("Preprocessing Vietnamese text...")
        preprocess_udf = create_preprocessing_udf(stopwords_bc, slang_dict_bc)
        df = df.withColumn("tokens", preprocess_udf(F.col("text")))
        df = df.filter(F.size(F.col("tokens")) > 0)
        doc_count = df.count()
        logger.info(f"  Documents after preprocessing: {doc_count:,}")
        df.cache()

        # 5. TF-IDF
        tfidf_df, cv_model, vocabulary = build_tfidf_features(
            df,
            tokens_col="tokens",
            vocab_size=args.vocab_size,
            min_df=args.min_df,
        )

        # 6. Huấn luyện LDA
        lda_model, ll, lp = train_lda(
            tfidf_df,
            k=args.k,
            max_iter=args.max_iter,
            optimizer=args.optimizer,
        )

        # 7. Trích xuất topics
        logger.info("Extracting topic descriptions:")
        topics = extract_topics(lda_model, vocabulary)

        # 8. Lưu kết quả
        save_results(spark, lda_model, topics, args.output_path, local=args.local)
        save_vocabulary(cv_model, args.output_path, local=args.local)

        logger.info("=" * 70)
        logger.info("LDA PIPELINE COMPLETE!")
        logger.info(f"  Topics: {len(topics)}")
        logger.info(f"  Log-Likelihood: {ll:,.2f}")
        logger.info(f"  Log-Perplexity: {lp:.4f}")
        logger.info(f"  Output: {args.output_path}")
        logger.info("=" * 70)

    finally:
        spark.stop()
        logger.info("SparkSession stopped.")


if __name__ == "__main__":
    main()
