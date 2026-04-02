"""
Spark job: chạy text preprocessing phân tán trên cluster.

Cách chạy local:
    python spark_jobs/preprocess_job.py

Cách chạy trên cluster (sau khi M2 setup xong):
    spark-submit scripts/spark_submit_cluster.sh
    # hoặc xem scripts/spark_submit_cluster.sh để biết các tham số
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

# ── Project root (dùng trên driver) ──────────────────────────────
_DRIVER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Đường dẫn model ───────────────────────────────────────────────
# Trên cluster: override bằng biến môi trường NLP_MODEL_PATH
# hoặc truyền qua spark-submit --conf spark.driver.extraJavaOptions
# Ví dụ: export NLP_MODEL_PATH=hdfs:///models/phobert_finetuned/final
MODEL_PATH = os.environ.get(
    "NLP_MODEL_PATH",
    os.path.join(_DRIVER_ROOT, "models", "phobert_finetuned", "final"),
)

# ── Đường dẫn data files ──────────────────────────────────────────
# Trên cluster: các file này phải được distribute qua --files
# hoặc đặt trên HDFS/shared storage
# Sau khi distribute bằng --files, dùng SparkFiles.get() để lấy path
SLANG_DICT_PATH = os.environ.get(
    "NLP_SLANG_DICT",
    os.path.join(_DRIVER_ROOT, "data", "slang_dict.json"),
)
STOPWORDS_PATH = os.environ.get(
    "NLP_STOPWORDS",
    os.path.join(_DRIVER_ROOT, "data", "stopwords_vi.txt"),
)


# ── process_partition — đặt ở đây, NGOÀI mọi hàm main ───────────
def process_partition(iterator):
    """
    Lazy init — chạy trên mỗi Spark worker.

    Hoạt động trên cả local mode lẫn cluster mode:
      - sys.path.insert đảm bảo import được package preprocessing/
      - Path tới model/data đọc từ env vars (set bởi spark-submit)
        hoặc fallback về SparkFiles nếu distribute qua --files
    """
    import os
    import sys
    import traceback

    # ── Đảm bảo worker tìm được package preprocessing/ ───────────
    # Trên cluster: project root phải có trong PYTHONPATH
    # hoặc package được distribute qua --py-files project.zip
    try:
        from pyspark import SparkFiles
        _spark_root = SparkFiles.getRootDirectory()
        if _spark_root not in sys.path:
            sys.path.insert(0, _spark_root)
    except Exception:
        pass

    # Fallback: thêm thư mục cha của file này (hoạt động trên local)
    _job_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _job_root not in sys.path:
        sys.path.insert(0, _job_root)

    try:
        from preprocessing.text_cleaner import TextPreprocessor
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception as e:
        print(f"[WORKER IMPORT ERROR] {e}")
        traceback.print_exc()
        return

    # ── Resolve paths: ưu tiên SparkFiles (cluster), fallback env var ──
    def _resolve_path(env_var, filename):
        """Tìm file theo thứ tự: SparkFiles → env var → đường dẫn gốc."""
        try:
            from pyspark import SparkFiles
            spark_path = SparkFiles.get(filename)
            if os.path.exists(spark_path):
                return spark_path
        except Exception:
            pass
        return os.environ.get(env_var, os.path.join(_job_root, "data", filename))

    slang_path     = _resolve_path("NLP_SLANG_DICT",  "slang_dict.json")
    stopwords_path = _resolve_path("NLP_STOPWORDS",   "stopwords_vi.txt")
    model_path     = os.environ.get(
        "NLP_MODEL_PATH",
        os.path.join(_job_root, "models", "phobert_finetuned", "final")
    )

    # Preprocessor
    preprocessor = TextPreprocessor(
        slang_dict_path=slang_path,
        stopwords_path=stopwords_path,
        use_vncorenlp=False,
    )

    # Sentiment model — dùng model_path đã resolve ở trên (không dùng global MODEL_PATH)
    LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    BATCH_SIZE = 32
    rows       = list(iterator)

    for start in range(0, len(rows), BATCH_SIZE):
        batch       = rows[start : start + BATCH_SIZE]
        clean_texts = []

        for row in batch:
            try:
                if row.content is None:
                    clean = ""
                elif not isinstance(row.content, str):
                    clean = str(row.content)
                else:
                    clean = preprocessor.preprocess(row.content)

            except Exception as e:
                print("[PREPROCESS ERROR]", e)
                clean = ""

            clean_texts.append(clean)

        encoding = tokenizer(
            clean_texts,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device)
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

        for i, row in enumerate(batch):
            yield (
                row.post_id,
                row.source,
                row.content,
                clean_texts[i],
                LABEL_MAP[int(preds[i])],
                int(preds[i]),
                float(probs[i][int(preds[i])]),
                getattr(row, "post_type",      None),
                getattr(row, "author",         None),
                getattr(row, "author_id",      None),
                getattr(row, "created_at",     None),
                getattr(row, "reaction_count", None),
                getattr(row, "view_count",     None),
                getattr(row, "comment_count",  None),
                getattr(row, "parent_post_id", None),
                getattr(row, "title",          None),
                getattr(row, "tags",           None),
            )


# ── Schema dùng chung cho cả 2 hàm main ──────────────────────────
OUTPUT_SCHEMA = StructType([
    StructField("post_id",         StringType(),  True),
    StructField("source",          StringType(),  True),
    StructField("content",         StringType(),  True),
    StructField("clean_text",      StringType(),  True),
    StructField("sentiment_label", StringType(),  True),
    StructField("sentiment_id",    IntegerType(), True),
    StructField("confidence",      FloatType(),   True),
    StructField("post_type",       StringType(),  True),
    StructField("author",          StringType(),  True),
    StructField("author_id",       StringType(),  True),
    StructField("created_at",      IntegerType(), True),
    StructField("reaction_count",  IntegerType(), True),
    StructField("view_count",      IntegerType(), True),
    StructField("comment_count",   IntegerType(), True),
    StructField("parent_post_id",  StringType(),  True),
    StructField("title",           StringType(),  True),
    StructField("tags",            StringType(),  True),
])


# ── main_with_map_partitions — dùng khi có HDFS (cluster mode) ───
def main_with_map_partitions():
    """
    Production entry point — chạy trên Spark cluster sau khi M2 setup xong.

    Yêu cầu trước khi chạy:
      1. Set env vars (hoặc truyền qua spark-submit --conf):
           export NLP_MODEL_PATH=hdfs:///models/phobert_finetuned/final
           export NLP_SLANG_DICT=hdfs:///data/ref/slang_dict.json
           export NLP_STOPWORDS=hdfs:///data/ref/stopwords_vi.txt
      2. HDFS_NAMENODE được set đúng trong hadoop-site.xml của cluster

    Xem scripts/spark_submit_cluster.sh để biết cách chạy đầy đủ.
    """
    # SparkSession.builder.getOrCreate() đọc config từ spark-submit
    # không cần set master/memory ở đây — để spark-submit quản lý
    spark = SparkSession.builder \
        .appName("NLP_SentimentJob_MapPartitions") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # HDFS_INPUT: override bằng env var nếu cần đổi source
    hdfs_input  = os.environ.get("HDFS_INPUT",  "hdfs:///data/raw/voz/date=*/*.json")
    hdfs_output = os.environ.get("HDFS_OUTPUT", "hdfs:///data/staged/voz_sentiment/")

    print(f"[INFO] Reading from : {hdfs_input}")
    print(f"[INFO] Writing to   : {hdfs_output}")

    raw_df = spark.read.json(hdfs_input)

    # Đổi tên cột về đúng schema nếu data từ UniversalSocialPost
    if "post_id" not in raw_df.columns and "thread_id" in raw_df.columns:
        raw_df = raw_df.withColumnRenamed("thread_id", "post_id")
    if "content" not in raw_df.columns and "message" in raw_df.columns:
        raw_df = raw_df.withColumnRenamed("message", "content")

    # Đảm bảo các cột metadata tồn tại (null nếu nguồn không có)
    from pyspark.sql.functions import lit
    for col_name in ("post_type", "author", "author_id", "created_at",
                     "reaction_count", "view_count", "comment_count",
                     "parent_post_id", "title", "tags"):
        if col_name not in raw_df.columns:
            raw_df = raw_df.withColumn(col_name, lit(None))

    processed_rdd = raw_df.rdd.mapPartitions(process_partition)
    processed_df  = spark.createDataFrame(processed_rdd, OUTPUT_SCHEMA)

    processed_df.write.mode("overwrite").parquet(hdfs_output)
    print(f"[INFO] Done. Written to {hdfs_output}")
    spark.stop()


# ── main_local_test — dùng khi test local ────────────────────────
def main_local_test():
    """Test local — không cần cluster, đọc từ CSV."""
    import sys

    # Fix encoding tiếng Việt trên Windows terminal (cp1252 → utf-8)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    # Tự detect Python executable hiện tại (cross-platform: Windows + Linux)
    python_exe = sys.executable
    os.environ["PYSPARK_PYTHON"]        = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

    # ── Kiểm tra 1: Python path ──
    print(f"[CHECK 1] Python: {python_exe}")

    spark = SparkSession.builder \
        .appName("TextPreprocessing_Local") \
        .master("local[1]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # ── Kiểm tra 2: File CSV có tồn tại không ──
    csv_path = os.path.join(DATA_DIR, "comments.csv")
    print(f"[CHECK 2] CSV exists: {os.path.exists(csv_path)}")

    try:
        raw_df = spark.read \
            .option("multiLine", True) \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("encoding", "UTF-8") \
            .csv(csv_path, header=True)
    except Exception as e:
        print(f"[ERROR] Đọc CSV thất bại: {e}")
        spark.stop()
        return

    # ── Kiểm tra 3: Các cột có tồn tại trong CSV không ──
    print(f"[CHECK 3] Columns in CSV: {raw_df.columns}")

    required_cols = ["id_post", "comment"]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        print(f"[ERROR] Thiếu cột: {missing}")
        print(f"[ERROR] Các cột hiện có: {raw_df.columns}")
        spark.stop()
        return

    raw_df = raw_df.withColumn("post_id", col("id_post")) \
                   .withColumn("source",  col("url")) \
                   .withColumn("content", col("comment"))

    raw_df = raw_df.limit(50)
    print(f"\n[INFO] Input: {raw_df.count()} rows")
    raw_df.show(5, truncate=50)

    # ── Kiểm tra 4: Test preprocessing KHÔNG có PhoBERT (tránh OOM) ──
    # PhoBERT (~500MB) + PyTorch (~300MB) > memory worker cho phép trên local
    # → Test riêng PhoBERT bên ngoài Spark bằng: python -c "..."
    print(f"[CHECK 4] Model path: {MODEL_PATH} | exists: {os.path.exists(MODEL_PATH)}")
    print("[INFO] Spark local test: chỉ test preprocessing (không load PhoBERT trong worker)")

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    # Resolve project root trên driver — truyền vào UDF qua closure
    _project_root = BASE_DIR
    _slang_path   = os.path.join(BASE_DIR, "data", "slang_dict.json")

    def preprocess_only(text):
        """
        UDF thuần Python — không dùng underthesea/VnCoreNLP (tránh C-extension
        crash trong Spark worker subprocess).
        Thực hiện: lowercase → bỏ HTML/URL/emoji → chuẩn hoá slang.
        Word tokenization được bỏ qua khi test local.
        """
        if not text:
            return ""
        try:
            import re, json, unicodedata, sys
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)

            # Lowercase
            t = text.lower()
            # Bỏ HTML tags
            t = re.sub(r"<[^>]+>", " ", t)
            # Bỏ URLs
            t = re.sub(r"https?://\S+|www\.\S+", " ", t)
            # Bỏ email
            t = re.sub(r"\S+@\S+", " ", t)
            # Bỏ emoji (unicode ranges)
            t = re.sub(r"[\U00010000-\U0010ffff]", " ", t, flags=re.UNICODE)
            # Chuẩn hoá unicode NFC
            t = unicodedata.normalize("NFC", t)
            # Chuẩn hoá khoảng trắng
            t = re.sub(r"\s+", " ", t).strip()
            # Chuẩn hoá slang
            with open(_slang_path, encoding="utf-8") as f:
                slang = json.load(f)
            words = t.split()
            t = " ".join(slang.get(w, w) for w in words)
            # Bỏ ký tự đặc biệt (giữ dấu tiếng Việt + chữ + số)
            t = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", t)
            return re.sub(r"\s+", " ", t).strip()
        except Exception as e:
            return f"[ERR] {e}"

    preprocess_udf = udf(preprocess_only, StringType())

    try:
        result_df = raw_df.withColumn("clean_text", preprocess_udf(col("content")))
        result_df.select("post_id", "content", "clean_text").show(10, truncate=50)
        print("[OK] Spark preprocessing pipeline hoạt động!")
    except Exception as e:
        print(f"\n[ERROR] UDF thất bại: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        return

    # ── Kiểm tra 5: Ghi output (CSV cho local test, Parquet chỉ dùng trên cluster) ──
    output_path = os.path.join(BASE_DIR, "output", "preprocess_local.csv")
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    try:
        result_df.select("post_id", "content", "clean_text") \
                 .toPandas() \
                 .to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n[OK] Lưu tại: {output_path}")
    except Exception as e:
        print(f"[ERROR] Ghi CSV thất bại: {e}")

    print("\n[NEXT] Test PhoBERT inference riêng (ngoài Spark):")
    print("  python -c \"from models.sentiment_predictor import SentimentPredictor; "
          "p=SentimentPredictor('models/phobert_finetuned/final'); "
          "print(p.predict('Sản phẩm rất tệ'))\"")

    spark.stop()


# ── run_local_pipeline — chạy full luồng KHÔNG cần Spark ─────────
def run_local_pipeline(
    comments_csv: str = None,
    posts_csv: str = None,
    output_csv: str = None,
    batch_size: int = 32,
    limit: int = None,
):
    """
    Chạy full pipeline (VozAdapter → preprocess → PhoBERT sentiment) thuần Python,
    không cần Spark. Dùng để dev/debug nhanh trên local.

    Luồng:
        comments.csv + posts.csv
             ↓  VozAdapter  (validate schema, chuẩn hoá tên cột, parse time)
             ↓  TextPreprocessor  (clean + tokenize + stopwords)
             ↓  PhoBERT inference  (sentiment_label, confidence)
             ↓  CSV output  (đúng OUTPUT_SCHEMA)

    Args:
        comments_csv : path tới voz comments CSV (mặc định data/comments.csv).
        posts_csv    : path tới voz posts CSV    (mặc định data/posts.csv).
        output_csv   : path file kết quả         (mặc định output/pipeline_local.csv).
        batch_size   : số dòng mỗi batch PhoBERT (giữ nguyên 32 như Spark job).
        limit        : giới hạn tổng số dòng xử lý (None = tất cả).
    """
    import sys

    # Fix encoding tiếng Việt trên Windows terminal
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    BASE_DIR = _DRIVER_ROOT
    sys.path.insert(0, BASE_DIR)

    if comments_csv is None:
        comments_csv = os.path.join(BASE_DIR, "data", "comments.csv")
    if posts_csv is None:
        posts_csv = os.path.join(BASE_DIR, "data", "posts.csv")
    if output_csv is None:
        os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
        output_csv = os.path.join(BASE_DIR, "output", "pipeline_local.csv")

    # ── Bước 1: Load + validate qua VozAdapter ───────────────────────
    print(f"[1/4] VozAdapter: doc comments={comments_csv}")
    print(f"                        posts={posts_csv}")
    from schemas.voz_adapter import VozAdapter
    adapter = VozAdapter()
    comments_df, posts_df = adapter.from_csv(comments_csv, posts_csv)
    print(f"    → comments: {len(comments_df)} dong | posts: {len(posts_df)} dong")

    # Map comments → OUTPUT_SCHEMA columns
    comments_mapped = pd.DataFrame({
        "post_id":        comments_df["comment_id"],
        "source":         comments_df["source"],
        "content":        comments_df["comment"],
        "post_type":      comments_df["post_type"],
        "author":         comments_df["user"],
        "author_id":      comments_df["id_user"].astype(str),
        "created_at":     comments_df["created_at"],
        "reaction_count": comments_df["reaction_count"],
        "view_count":     None,
        "comment_count":  None,
        "parent_post_id": comments_df["id_post"].astype(str),
        "title":          None,
        "tags":           None,
    })

    # Map posts → OUTPUT_SCHEMA columns
    posts_mapped = pd.DataFrame({
        "post_id":        posts_df["id_post"].astype(str),
        "source":         posts_df["source"],
        "content":        posts_df["title"],
        "post_type":      posts_df["post_type"],
        "author":         posts_df["author_name"],
        "author_id":      posts_df["id_author"].astype(str),
        "created_at":     posts_df["created_at"],
        "reaction_count": None,
        "view_count":     posts_df["view_count"],
        "comment_count":  posts_df["comment_count"],
        "parent_post_id": None,
        "title":          posts_df["title"],
        "tags":           posts_df["tags"],
    })

    df = pd.concat([comments_mapped, posts_mapped], ignore_index=True)
    if limit:
        df = df.head(limit)
    df["content"] = df["content"].fillna("")
    print(f"    → Tong: {len(df)} dong")

    # ── Bước 2: Text preprocessing ───────────────────────────────────
    print("[2/4] Preprocessing van ban (slang + underthesea + stopwords)...")
    from preprocessing.text_cleaner import TextPreprocessor
    preprocessor = TextPreprocessor(
        slang_dict_path=SLANG_DICT_PATH,
        stopwords_path=STOPWORDS_PATH,
        use_vncorenlp=False,
    )
    df["clean_text"] = preprocessor.preprocess_batch(df["content"].tolist())
    print(f"    → Done.")

    # ── Bước 3: PhoBERT sentiment inference ─────────────────────────
    print(f"[3/4] Load PhoBERT model: {MODEL_PATH}")
    LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    → Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    sentiment_labels, sentiment_ids, confidences = [], [], []
    texts = df["clean_text"].tolist()
    print(f"    → Inference {len(texts)} dong, batch_size={batch_size}...")

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

        for i in range(len(batch_texts)):
            sentiment_ids.append(int(preds[i]))
            sentiment_labels.append(LABEL_MAP[int(preds[i])])
            confidences.append(float(probs[i][int(preds[i])]))

        done = min(start + batch_size, len(texts))
        print(f"    → {done}/{len(texts)} dong", end="\r")
    print()

    # ── Bước 4: Ghi output theo OUTPUT_SCHEMA ────────────────────────
    df["sentiment_label"] = sentiment_labels
    df["sentiment_id"]    = sentiment_ids
    df["confidence"]      = confidences

    # Đúng thứ tự cột như OUTPUT_SCHEMA
    final_cols = [
        "post_id", "source", "content", "clean_text",
        "sentiment_label", "sentiment_id", "confidence",
        "post_type", "author", "author_id", "created_at",
        "reaction_count", "view_count", "comment_count",
        "parent_post_id", "title", "tags",
    ]
    df[final_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[4/4] Luu ket qua: {output_csv}")

    counts = df["sentiment_label"].value_counts()
    print("\n── Phan bo sentiment ──")
    for label, cnt in counts.items():
        print(f"  {label:10s}: {cnt:4d}  ({cnt/len(df)*100:.1f}%)")
    print(f"  {'TOTAL':10s}: {len(df):4d}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess + Sentiment pipeline")
    parser.add_argument("--mode",     choices=["spark", "local"], default="local",
                        help="spark = dung Spark local[1] | local = thuan Python (default)")
    parser.add_argument("--comments", default=None, help="Path comments CSV (default: data/comments.csv)")
    parser.add_argument("--posts",    default=None, help="Path posts CSV    (default: data/posts.csv)")
    parser.add_argument("--output",   default=None, help="Path CSV dau ra   (default: output/pipeline_local.csv)")
    parser.add_argument("--limit",    type=int, default=None, help="Gioi han so dong xu ly")
    args = parser.parse_args()

    if args.mode == "spark":
        main_local_test()
    else:
        run_local_pipeline(
            comments_csv=args.comments,
            posts_csv=args.posts,
            output_csv=args.output,
            limit=args.limit,
        )