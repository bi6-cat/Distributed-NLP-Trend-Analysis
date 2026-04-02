"""
scripts/preprocess_for_m3.py — Tiền xử lý voz_comments.csv + voz_posts.csv
và xuất ra CSV sạch cho Member 3 (Topic Modeling).

Cách chạy:
    python scripts/preprocess_for_m3.py

Output:
    output/m3_comments_clean.csv
    output/m3_posts_clean.csv

Các cột output:
    comments: id_post, id_user, user, time, comment, clean_text
    posts:    id_post, title, time_post, category, subcategory, clean_text
"""

import os
import sys
import io
import pandas as pd

# Fix encoding tieng Viet tren Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Đảm bảo import được package preprocessing/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.text_cleaner import TextPreprocessor

#  Paths 
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMENTS_INPUT  = os.path.join(DATA_DIR, "comments.csv")
POSTS_INPUT     = os.path.join(DATA_DIR, "posts.csv")
COMMENTS_OUTPUT = os.path.join(OUTPUT_DIR, "m3_comments_clean.csv")
POSTS_OUTPUT    = os.path.join(OUTPUT_DIR, "m3_posts_clean.csv")


def build_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(
        slang_dict_path=os.path.join(DATA_DIR, "slang_dict.json"),
        stopwords_path=os.path.join(DATA_DIR, "stopwords_vi.txt"),
        use_vncorenlp=False,   # dùng underthesea để chạy local không cần Java
    )


def preprocess_series(series: pd.Series, preprocessor: TextPreprocessor) -> pd.Series:
    """Áp dụng preprocessor lên từng cell, bắt lỗi từng dòng."""
    results = []
    for i, text in enumerate(series):
        try:
            if pd.isna(text) or str(text).strip() == "":
                results.append("")
            else:
                results.append(preprocessor.preprocess(str(text)))
        except Exception as e:
            print(f"  [WARN] Dòng {i} lỗi: {e}")
            results.append("")
    return pd.Series(results, index=series.index)


def drop_duplicates(df: pd.DataFrame, text_col: str, id_col: str) -> pd.DataFrame:
    """
    Lọc trùng lặp theo 2 tiêu chí:
      1. Trùng ID record (comment_id / id_post) — giữ lần đầu
      2. Trùng nội dung clean_text — giữ lần đầu
    """
    before = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    n_dup_id = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first")
    n_dup_text = before - len(df)

    print(f"      Loại trùng ID  : {n_dup_id} rows")
    print(f"      Loại trùng text: {n_dup_text} rows")
    return df


def process_comments(preprocessor: TextPreprocessor) -> None:
    print(f"\n[1/2] Đọc comments: {COMMENTS_INPUT}")
    df = pd.read_csv(COMMENTS_INPUT)
    print(f"      {len(df)} rows, columns: {df.columns.tolist()}")

    # Tạo comment_id unique từ (id_post, id_user, time) để dedup theo ID record
    df["comment_id"] = (
        df["id_post"].astype(str) + "_" +
        df["id_user"].astype(str) + "_" +
        df["time"].astype(str)
    )

    print("      Đang tiền xử lý cột 'comment'...")
    df["clean_text"] = preprocess_series(df["comment"], preprocessor)

    # Lọc clean_text rỗng trước khi dedup
    df = df[df["clean_text"] != ""].copy()

    print("      Lọc trùng lặp...")
    df = drop_duplicates(df, text_col="clean_text", id_col="comment_id")

    out = df[["comment_id", "id_post", "id_user", "user", "time", "comment", "clean_text"]]
    out.to_csv(COMMENTS_OUTPUT, index=False, encoding="utf-8")
    print(f"      Done. {len(out)} rows → {COMMENTS_OUTPUT}")


def process_posts(preprocessor: TextPreprocessor) -> None:
    print(f"\n[2/2] Đọc posts: {POSTS_INPUT}")
    df = pd.read_csv(POSTS_INPUT)
    print(f"      {len(df)} rows, columns: {df.columns.tolist()}")

    print("      Đang tiền xử lý cột 'title'...")
    df["clean_text"] = preprocess_series(df["title"], preprocessor)

    # Lọc clean_text rỗng trước khi dedup
    df = df[df["clean_text"] != ""].copy()

    print("      Lọc trùng lặp...")
    df = drop_duplicates(df, text_col="clean_text", id_col="id_post")

    out = df[["id_post", "title", "time_post", "category", "subcategory", "clean_text"]]
    out.to_csv(POSTS_OUTPUT, index=False, encoding="utf-8")
    print(f"      Done. {len(out)} rows → {POSTS_OUTPUT}")


def main():
    print("[INFO] Khởi tạo TextPreprocessor...")
    preprocessor = build_preprocessor()
    print("[INFO] OK\n")

    process_comments(preprocessor)
    process_posts(preprocessor)

    print("\n[DONE] Output files:")
    print(f"  → {COMMENTS_OUTPUT}")
    print(f"  → {POSTS_OUTPUT}")
    print("\n[NOTE] Gửi 2 file này cho M3 để chạy topic modeling.")


if __name__ == "__main__":
    main()
