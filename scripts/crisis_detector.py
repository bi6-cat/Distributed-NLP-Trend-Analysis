"""
Task 3.5 — Crisis Detector: tổng hợp ≥ 2/3 điều kiện → Crisis Alert

Ba điều kiện để xác định khủng hoảng trên một post:

  Cond 1 — ISOLATION FOREST (post-level)
      Post có chỉ số bất thường tổng thể: velocity cao, engagement đột biến,
      neg_ratio vượt ngưỡng → IsolationForest đánh dấu is_anomaly=True.

  Cond 2 — GLOBAL SPIKE (platform-level)
      Ít nhất 1 comment của post này rơi vào giờ mà toàn platform có spike
      (comment_count > rolling_mean + 2σ trên tất cả posts).

  Cond 3 — PER-POST SPIKE (post-level time series)
      Bản thân post đang nhận comments với tốc độ bất thường trong cửa sổ
      thời gian của chính nó (per-post rolling threshold spike).

  → crisis_score = số điều kiện thoả (0-3)
  → is_crisis = crisis_score ≥ min_conditions  (mặc định 2)
  → alert_level: HIGH (3/3) | MEDIUM (2/3) | NORMAL (< 2)

Output schema (stg_crisis_alerts):
  post_id, detected_at, alert_level, crisis_score, is_crisis,
  cond_isolation_forest, cond_global_spike, cond_post_spike,
  neg_ratio, velocity, comment_count, engagement_score, anomaly_score,
  global_spike_buckets, post_spike_buckets

Cách dùng:
  1. CLI:
       python scripts/crisis_detector.py \\
           --comments data/comments.csv \\
           --output output/crisis_alerts.csv

  2. Import:
       from scripts.crisis_detector import CrisisDetector

       detector = CrisisDetector(min_conditions=2)
       alerts   = detector.run(comments_df, sentiments_df)
       # alerts: DataFrame với schema stg_crisis_alerts

  3. Ghi ClickHouse (sau khi M2 setup xong):
       detector.write_clickhouse(alerts, host="...", user="...", password="...")
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

# Import từ models của Member 4
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isolation_forest import CrisisDetector as IsoDetector, extract_features
from models.rolling_threshold import RollingThreshold



ALERT_LEVEL = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "NORMAL"}

OUTPUT_COLUMNS = [
    "post_id",
    "detected_at",
    "alert_level",
    "crisis_score", # là tổng số điều kiện thoả (0-3)
    "is_crisis", 
    "cond_isolation_forest",
    "cond_global_spike",
    "cond_post_spike",
    # Features từ IsolationForest
    "neg_ratio",
    "pos_ratio",
    "velocity",
    "mention_count",
    "engagement_score",
    "comment_count",
    "anomaly_score",
    # Thống kê từ RollingThreshold
    "global_spike_buckets",
    "post_spike_buckets",
]


# ── CrisisDetector 

class CrisisDetector:
    """
    Pipeline tổng hợp IsolationForest + RollingThreshold → Crisis Alert.

    Args:
        min_conditions : số điều kiện tối thiểu để phát alert (mặc định 2)
        contamination  : tỉ lệ outlier cho IsolationForest (mặc định 0.05)
        window         : rolling window tính bằng số buckets (mặc định 24)
        n_sigma        : hệ số sigma cho rolling threshold (mặc định 2.0)
        freq           : kích thước time bucket (mặc định '1h')
    """

    def __init__(
        self,
        min_conditions: int = 2,
        contamination: float = 0.05,
        window: int = 24,
        n_sigma: float = 2.0,
        freq: str = "1h",
    ):
        self.min_conditions = min_conditions
        self.iso_detector   = IsoDetector(contamination=contamination)
        self.rolling        = RollingThreshold(window=window, n_sigma=n_sigma, freq=freq)

    # Public API 
    def run(
        self,
        comments_df: pd.DataFrame,
        sentiments_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline phát hiện khủng hoảng.

        Args:
            comments_df  : raw comments (id_post, comment, time, reactions, ...)
            sentiments_df: optional — output từ SentimentPredictor
                           cần cột [post_id, sentiment_label]

        Returns:
            DataFrame theo schema stg_crisis_alerts, sắp xếp theo crisis_score giảm dần
        """
        print("[CrisisDetector] Bước 1/3: IsolationForest...")
        iso_results = self._run_isolation_forest(comments_df, sentiments_df)

        print("[CrisisDetector] Bước 2/3: Rolling Threshold (global + per-post)...")
        global_spikes, per_post_spikes = self._run_rolling(comments_df, sentiments_df)

        print("[CrisisDetector] Bước 3/3: Tổng hợp điều kiện → Crisis Alert...")
        alerts = self._combine(iso_results, global_spikes, per_post_spikes, comments_df)

        return alerts

    def run_from_files(
        self,
        comments_path: str,
        posts_path: Optional[str] = None,
        sentiments_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load CSV từ M1 → validate qua VozAdapter → run().

        Args:
            comments_path  : đường dẫn comments.csv
            posts_path     : đường dẫn posts.csv (tuỳ chọn)
                             nếu truyền → id_author, author_name được join vào output
            sentiments_path: file sentiments CSV có cột [post_id, sentiment_label]
        """
        from schemas.voz_adapter import VozAdapter

        adapter = VozAdapter()
        print(f"[CrisisDetector] Đọc và validate data từ M1...")
        comments_df, posts_df = adapter.from_csv(comments_path, posts_path)
        print(f"[CrisisDetector] Comments hợp lệ: {len(comments_df)} rows")
        if not posts_df.empty:
            print(f"[CrisisDetector] Posts hợp lệ  : {len(posts_df)} rows")

        sentiments_df = None
        if sentiments_path and os.path.exists(sentiments_path):
            print(f"[CrisisDetector] Đọc sentiments: {sentiments_path}")
            sentiments_df = pd.read_csv(sentiments_path)

        alerts = self.run(comments_df, sentiments_df)

        # Nếu có posts_df, join thêm id_author + author_name vào output
        if not posts_df.empty and "id_author" in posts_df.columns:
            author_info = (
                posts_df[["id_post", "id_author", "author_name"]]
                .drop_duplicates(subset=["id_post"])   # tránh tạo duplicate rows
                .copy()
            )
            author_info = author_info.rename(columns={"id_post": "post_id"})
            author_info["post_id"] = author_info["post_id"].astype(str)
            alerts["post_id"] = alerts["post_id"].astype(str)
            alerts = alerts.merge(author_info, on="post_id", how="left")

        return alerts

    def write_clickhouse(
        self,
        alerts: pd.DataFrame,
        host: str,
        database: str = "default",
        user: str = "default",
        password: str = "",
        table: str = "stg_crisis_alerts",
    ) -> None:
        """
        Ghi alerts vào ClickHouse (dùng sau khi M2 setup xong).

        Requires: pip install clickhouse-connect
        """
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError(
                "clickhouse-connect chưa cài.\n"
                "Chạy: pip install clickhouse-connect\n"
                "Hoặc yêu cầu Member 2 setup ClickHouse trước."
            )

        client = clickhouse_connect.get_client(
            host=host, database=database, username=user, password=password
        )
        # Đảm bảo detected_at là string ISO để ClickHouse nhận
        df = alerts.copy()
        if "detected_at" in df.columns:
            df["detected_at"] = df["detected_at"].astype(str)

        client.insert_df(table, df[OUTPUT_COLUMNS])
        print(f"[CrisisDetector] Đã ghi {len(df)} alerts vào ClickHouse {table}")

    # Internal steps

    def _run_isolation_forest(
        self,
        comments_df: pd.DataFrame,
        sentiments_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Chạy IsolationForest → trả DataFrame với index=post_id.
        Columns: neg_ratio, pos_ratio, velocity, mention_count,
                 engagement_score, comment_count, anomaly_score, is_anomaly
        """
        return self.iso_detector.detect(comments_df, sentiments_df)

    def _run_rolling(
        self,
        comments_df: pd.DataFrame,
        sentiments_df: Optional[pd.DataFrame],
    ):
        """
        Chạy RollingThreshold global + per-post.

        Returns:
            global_spikes    : DataFrame index=bucket, cột is_spike
            per_post_spikes  : dict {post_id: DataFrame}
        """
        global_spikes = self.rolling.detect_global(
            comments_df, sentiments_df, metric="comment_count"
        )
        per_post_spikes = self.rolling.detect_per_post(
            comments_df, sentiments_df=sentiments_df, metric="comment_count"
        )
        return global_spikes, per_post_spikes

    def _combine(
        self,
        iso_results: pd.DataFrame,
        global_spikes: pd.DataFrame,
        per_post_spikes: dict,
        comments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Kết hợp 3 nguồn tín hiệu → bảng crisis_alerts.
        """
        from models.rolling_threshold import _parse_time_series

        df = comments_df.copy()
        df = df.rename(columns={"id_post": "post_id"})
        df["dt"]     = _parse_time_series(df["time"])
        df["bucket"] = df["dt"].dt.floor(self.rolling.freq)

        # Tập hợp các spike buckets toàn cục
        global_spike_buckets = set(
            global_spikes[global_spikes["is_spike"]].index.tolist()
        )

        records = []
        for post_id in iso_results.index:
            # --- Condition 1: IsolationForest ---
            cond1 = bool(iso_results.loc[post_id, "is_anomaly"])

            # --- Condition 2: Global spike ---
            # Post có ít nhất 1 comment trong giờ bị đánh dấu global spike
            post_buckets = set(
                df[df["post_id"] == post_id]["bucket"].dropna().tolist()
            )
            global_hits = len(post_buckets & global_spike_buckets)
            cond2 = global_hits > 0

            # --- Condition 3: Per-post spike ---
            post_spike_count = 0
            if post_id in per_post_spikes:
                post_spike_count = int(per_post_spikes[post_id]["is_spike"].sum())
            cond3 = post_spike_count > 0

            # --- Crisis score & level ---
            score = int(cond1) + int(cond2) + int(cond3)
            is_crisis = score >= self.min_conditions

            records.append({
                "post_id":               post_id,
                "detected_at":           datetime.now(timezone.utc).isoformat(),
                "alert_level":           ALERT_LEVEL.get(score, "NORMAL"),
                "crisis_score":          score,
                "is_crisis":             is_crisis,
                "cond_isolation_forest": cond1,
                "cond_global_spike":     cond2,
                "cond_post_spike":       cond3,
                # Features từ IsolationForest
                "neg_ratio":             float(iso_results.loc[post_id, "neg_ratio"]),
                "pos_ratio":             float(iso_results.loc[post_id, "pos_ratio"]),
                "velocity":              float(iso_results.loc[post_id, "velocity"]),
                "mention_count":         float(iso_results.loc[post_id, "mention_count"]),
                "engagement_score":      float(iso_results.loc[post_id, "engagement_score"]),
                "comment_count":         float(iso_results.loc[post_id, "comment_count"]),
                "anomaly_score":         float(iso_results.loc[post_id, "anomaly_score"]),
                # Thống kê rolling
                "global_spike_buckets":  global_hits,
                "post_spike_buckets":    post_spike_count,
            })

        alerts = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
        alerts = alerts.sort_values(["crisis_score", "anomaly_score"],
                                    ascending=[False, True])
        return alerts.reset_index(drop=True)


#  Report 

def print_report(alerts: pd.DataFrame, top_n: int = 10) -> None:
    """In báo cáo tổng hợp crisis alerts."""
    total      = len(alerts)
    n_crisis   = alerts["is_crisis"].sum()
    n_high     = (alerts["alert_level"] == "HIGH").sum()
    n_medium   = (alerts["alert_level"] == "MEDIUM").sum()

    print(f"\n{'='*65}")
    print(f"  CRISIS DETECTION REPORT — TỔNG HỢP 3 ĐIỀU KIỆN")
    print(f"{'='*65}")
    print(f"  Tổng posts phân tích : {total}")
    print(f"  Posts là crisis      : {n_crisis}  ({n_crisis/total*100:.1f}%)")
    print(f"    └─ HIGH  (3/3)     : {n_high}")
    print(f"    └─ MEDIUM(2/3)     : {n_medium}")
    print(f"{'='*65}")

    crisis_posts = alerts[alerts["is_crisis"]]
    if crisis_posts.empty:
        print("  Không phát hiện crisis nào.\n")
        return

    print(f"\n  Top {min(top_n, len(crisis_posts))} crisis posts:\n")
    display_cols = [
        "post_id", "alert_level", "crisis_score",
        "cond_isolation_forest", "cond_global_spike", "cond_post_spike",
        "velocity", "comment_count", "global_spike_buckets", "post_spike_buckets",
    ]
    disp = crisis_posts[display_cols].head(top_n)
    # Rename cho gọn
    disp = disp.rename(columns={
        "cond_isolation_forest": "cond_iso",
        "cond_global_spike":     "cond_glob",
        "cond_post_spike":       "cond_post",
        "global_spike_buckets":  "g_spikes",
        "post_spike_buckets":    "p_spikes",
    })
    print(disp.to_string(index=False))
    print()

    # Phân tích điều kiện
    print("  Phân tích tần suất điều kiện (trên crisis posts):")
    for cond_col, label in [
        ("cond_isolation_forest", "IsolationForest"),
        ("cond_global_spike",     "GlobalSpike"),
        ("cond_post_spike",       "PostSpike"),
    ]:
        n = crisis_posts[cond_col].sum()
        print(f"    {label:20s}: {n}/{n_crisis} ({n/n_crisis*100:.0f}%)")
    print()


#  CLI 

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Crisis Detector — tổng hợp IsolationForest + RollingThreshold"
    )
    parser.add_argument(
        "--comments",
        default="data/comments.csv",
        help="Đường dẫn tới comments.csv",
    )
    parser.add_argument(
        "--posts",
        default=None,
        help="Đường dẫn tới posts.csv (tuỳ chọn, để join id_author + author_name)",
    )
    parser.add_argument(
        "--sentiments",
        default=None,
        help="File sentiments CSV có cột [post_id, sentiment_label] (tuỳ chọn)",
    )
    parser.add_argument(
        "--min_conditions",
        type=int,
        default=2,
        help="Số điều kiện tối thiểu để phát crisis alert (mặc định: 2)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Contamination cho IsolationForest (mặc định: 0.05)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Rolling window (mặc định: 24 buckets)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Hệ số sigma cho rolling threshold (mặc định: 2.0)",
    )
    parser.add_argument(
        "--freq",
        default="1h",
        help="Time bucket size (mặc định: 1h)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Lưu kết quả ra CSV (tuỳ chọn)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Số crisis posts in ra (mặc định: 10)",
    )
    args = parser.parse_args()

    detector = CrisisDetector(
        min_conditions = args.min_conditions,
        contamination  = args.contamination,
        window         = args.window,
        n_sigma        = args.sigma,
        freq           = args.freq,
    )

    alerts = detector.run_from_files(
        comments_path   = args.comments,
        posts_path      = args.posts,
        sentiments_path = args.sentiments,
    )

    print_report(alerts, top_n=args.top)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        alerts.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"[CrisisDetector] Đã lưu: {args.output}")
    else:
        # Luôn lưu mặc định vào output/
        default_out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "crisis_alerts.csv"
        )
        os.makedirs(os.path.dirname(default_out), exist_ok=True)
        alerts.to_csv(default_out, index=False, encoding="utf-8-sig")
        print(f"[CrisisDetector] Đã lưu mặc định: {default_out}")
