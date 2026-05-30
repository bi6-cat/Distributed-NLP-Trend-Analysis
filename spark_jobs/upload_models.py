"""
upload_models.py  —  Upload models lên S3/GCS sau khi retrain xong

Chạy trong weekly job, SAU bước retrain isolation_forest + crisis_classifier.

Cách dùng:
    # S3
    python spark_jobs/upload_models.py --backend s3 --bucket your-bucket

    # GCS
    python spark_jobs/upload_models.py --backend gcs --bucket your-gcs-bucket

    # Dry-run (in danh sách, không upload)
    python spark_jobs/upload_models.py --backend s3 --bucket your-bucket --dry-run

Biến môi trường (thay thế cho --args):
    UPLOAD_BACKEND   : "s3" | "gcs"            (mặc định "s3")
    UPLOAD_BUCKET    : tên bucket
    MODEL_DIR        : thư mục chứa models     (mặc định "models/")
    S3_PREFIX        : prefix trong bucket     (mặc định "models/crisis_detection/")
    AWS_PROFILE      : AWS profile (tuỳ chọn)
"""

import argparse
import os
import sys

# Danh sách file cần upload — thêm/bớt tại đây
MODELS = [
    "isolation_forest_hourly.pkl",
    "crisis_classifier.pkl",
    "features_tier1.json",
    "features_tier2.json",
]

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
DEFAULT_PREFIX = "models/crisis_detection/"


def upload_s3(bucket: str, model_dir: str, prefix: str, dry_run: bool) -> None:
    try:
        import boto3
    except ImportError:
        print("[upload] boto3 chưa cài. Chạy: pip install boto3")
        sys.exit(1)

    profile = os.environ.get("AWS_PROFILE")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    for model_file in MODELS:
        local_path = os.path.join(model_dir, model_file)
        if not os.path.exists(local_path):
            print(f"[upload] SKIP (không tìm thấy): {local_path}")
            continue

        s3_key = prefix + model_file
        size_mb = os.path.getsize(local_path) / (1024 * 1024)

        if dry_run:
            print(f"[dry-run] Would upload: {local_path} → s3://{bucket}/{s3_key}  ({size_mb:.1f} MB)")
        else:
            print(f"[upload] {local_path} → s3://{bucket}/{s3_key}  ({size_mb:.1f} MB)")
            s3.upload_file(local_path, bucket, s3_key)
            print(f"[upload] OK: {s3_key}")


def upload_gcs(bucket: str, model_dir: str, prefix: str, dry_run: bool) -> None:
    try:
        from google.cloud import storage
    except ImportError:
        print("[upload] google-cloud-storage chưa cài. Chạy: pip install google-cloud-storage")
        sys.exit(1)

    client = storage.Client()
    gcs_bucket = client.bucket(bucket)

    for model_file in MODELS:
        local_path = os.path.join(model_dir, model_file)
        if not os.path.exists(local_path):
            print(f"[upload] SKIP (không tìm thấy): {local_path}")
            continue

        blob_name = prefix + model_file
        size_mb = os.path.getsize(local_path) / (1024 * 1024)

        if dry_run:
            print(f"[dry-run] Would upload: {local_path} → gs://{bucket}/{blob_name}  ({size_mb:.1f} MB)")
        else:
            print(f"[upload] {local_path} → gs://{bucket}/{blob_name}  ({size_mb:.1f} MB)")
            blob = gcs_bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            print(f"[upload] OK: {blob_name}")


def add_to_spark_context(spark, bucket: str, prefix: str, backend: str) -> None:
    """
    Thêm model files vào SparkContext để executor có thể dùng SparkFiles.get().
    Gọi hàm này từ weekly Spark job SAU khi upload.

    Ví dụ:
        from spark_jobs.upload_models import add_to_spark_context
        add_to_spark_context(spark, bucket="my-bucket", prefix="models/crisis_detection/", backend="s3")
    """
    for model_file in MODELS:
        if backend == "s3":
            uri = f"s3a://{bucket}/{prefix}{model_file}"
        else:
            uri = f"gs://{bucket}/{prefix}{model_file}"
        spark.sparkContext.addFile(uri)
        print(f"[spark] addFile: {uri}")


def main():
    parser = argparse.ArgumentParser(description="Upload crisis detection models lên cloud storage")
    parser.add_argument("--backend",   default=os.environ.get("UPLOAD_BACKEND", "s3"),
                        choices=["s3", "gcs"], help="Cloud storage backend")
    parser.add_argument("--bucket",    default=os.environ.get("UPLOAD_BUCKET", ""),
                        help="Tên bucket")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
                        help=f"Thư mục models (mặc định: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--prefix",    default=os.environ.get("S3_PREFIX", DEFAULT_PREFIX),
                        help=f"Prefix trong bucket (mặc định: {DEFAULT_PREFIX})")
    parser.add_argument("--dry-run",   action="store_true",
                        help="In danh sách file sẽ upload, không upload thật")
    args = parser.parse_args()

    if not args.bucket:
        print("[upload] Thiếu --bucket. Dùng: --bucket <tên-bucket> hoặc set UPLOAD_BUCKET")
        sys.exit(1)

    print(f"[upload] Backend : {args.backend}")
    print(f"[upload] Bucket  : {args.bucket}")
    print(f"[upload] Prefix  : {args.prefix}")
    print(f"[upload] Models  : {args.model_dir}")
    print(f"[upload] Dry-run : {args.dry_run}")
    print()

    if args.backend == "s3":
        upload_s3(args.bucket, args.model_dir, args.prefix, args.dry_run)
    else:
        upload_gcs(args.bucket, args.model_dir, args.prefix, args.dry_run)

    print("\n[upload] Xong.")


if __name__ == "__main__":
    main()
