"""
Task 3.3 — Isolation Forest: phát hiện bất thường / khủng hoảng

Mỗi post được biểu diễn bằng vector 3 features tổng hợp từ comments:
  - neg_ratio    : tỉ lệ comment Negative / tổng
  - velocity     : số comments/giờ cao nhất (peak hourly rate)
  - comment_count: tổng số comments của post

Isolation Forest từ sklearn sẽ đánh dấu post nào là bất thường (is_anomaly=True).

Cách dùng:
  1. Standalone — chạy trên CSV local (không cần model sentiment):
       python models/isolation_forest.py \\
           --comments data/voz_comments.csv \\
           --sentiments output/sentiments.csv   # cột: post_id, comment_idx, sentiment_label
                                                # nếu bỏ qua → random mock cho demo

  2. Import trong pipeline:
       from models.isolation_forest import CrisisDetector

       detector = CrisisDetector(contamination=0.05)
       results_df = detector.detect(comments_df, sentiments_df)
       # results_df columns: post_id, neg_ratio, pos_ratio, velocity,
       #                     comment_count, anomaly_score, is_anomaly
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── Hằng số ───────────────────────────────────────────────────────────────────

FEATURES = [
    "neg_ratio",
    "velocity",
    "comment_count",
]

COMMENT_TIME_FORMAT = "%b %d, %Y at %I:%M %p"   # "Feb 23, 2026 at 4:00 PM"


# ── Feature engineering ───────────────────────────────────────────────────────

def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse chuỗi thời gian VOZ → datetime. Trả None nếu thất bại."""
    if not isinstance(time_str, str):
        return None
    try:
        return datetime.strptime(time_str.strip(), COMMENT_TIME_FORMAT)
    except ValueError:
        return None



def _compute_velocity(times: pd.Series) -> float:
    """
    Tính peak hourly velocity: số comments trong cửa sổ 1 giờ bận nhất.
    Trả 0.0 nếu không đủ dữ liệu thời gian.
    """
    parsed = times.apply(_parse_time).dropna()
    if len(parsed) < 2:
        return float(len(parsed))

    parsed = parsed.sort_values().reset_index(drop=True)
    max_count = 1
    left = 0
    for right in range(len(parsed)):
        # Dịch con trỏ trái ra ngoài cửa sổ 1 giờ
        while (parsed[right] - parsed[left]).total_seconds() > 3600:
            left += 1
        max_count = max(max_count, right - left + 1)
    return float(max_count)


def extract_features(
    comments_df: pd.DataFrame,
    sentiments_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Tổng hợp features theo từng post_id.

    Args:
        comments_df   : DataFrame với cột [id_post, comment, time]
        sentiments_df : DataFrame với cột [post_id, sentiment_label]
                        Nếu None → neg_ratio = 0 (dùng khi không có model)

    Returns:
        DataFrame index=post_id với 3 cột FEATURES
    """
    df = comments_df.copy()
    df = df.rename(columns={"id_post": "post_id"})

    # ── aggregate per post ────────────────────────────────────────────────────
    agg = df.groupby("post_id").agg(
        comment_count = ("comment", "count"),
    )

    # velocity (peak hourly rate)
    velocity_map = (
        df.groupby("post_id")["time"]
        .apply(_compute_velocity)
        .rename("velocity")
    )
    agg = agg.join(velocity_map)

    # ── sentiment ratios ──────────────────────────────────────────────────────
    if sentiments_df is not None and not sentiments_df.empty:
        sent = sentiments_df.copy()
        # Hỗ trợ cả cột 'post_id' lẫn 'id_post'
        if "id_post" in sent.columns and "post_id" not in sent.columns:
            sent = sent.rename(columns={"id_post": "post_id"})

        sent["sentiment_label"] = sent["sentiment_label"].str.lower()
        sent_agg = sent.groupby("post_id")["sentiment_label"].value_counts(
            normalize=True
        ).unstack(fill_value=0.0)

        if "negative" not in sent_agg.columns:
            sent_agg["negative"] = 0.0

        agg["neg_ratio"] = sent_agg["negative"].reindex(agg.index, fill_value=0.0).round(5)
    else:
        agg["neg_ratio"] = 0.0

    return agg[FEATURES].fillna(0.0)


# ── Isolation Forest detector ─────────────────────────────────────────────────

class CrisisDetector:
    """
    Phát hiện post bất thường / khủng hoảng bằng Isolation Forest.

    Args:
        contamination : tỉ lệ ước tính outlier trong dữ liệu (0.0 < x ≤ 0.5)
        n_estimators  : số cây trong rừng (mặc định 200)
        random_state  : seed cho reproducibility
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators  = n_estimators,
            contamination = contamination,
            random_state  = random_state,
            n_jobs        = -1,
        )
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, feature_df: pd.DataFrame) -> "CrisisDetector":
        """
        Train Isolation Forest trên tập feature đã tổng hợp.

        Args:
            feature_df: DataFrame với FEATURES columns, index = post_id
        """
        X = self.scaler.fit_transform(feature_df[FEATURES])
        self.model.fit(X)
        self._fitted = True
        print(f"[CrisisDetector] Fitted on {len(feature_df)} posts.")
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Chạy anomaly detection trên feature_df.

        Returns:
            DataFrame với các cột gốc + anomaly_score + is_anomaly
              anomaly_score : float, càng âm càng bất thường (sklearn convention)
              is_anomaly    : bool, True = khủng hoảng tiềm năng
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() hoặc detect() trước khi predict().")

        X = self.scaler.transform(feature_df[FEATURES])
        scores  = self.model.decision_function(X)  # higher = more normal
        labels  = self.model.predict(X)             # -1 = anomaly, 1 = normal

        result = feature_df.copy()
        result["anomaly_score"] = scores
        result["is_anomaly"]    = labels == -1
        return result

    def detect(
        self,
        comments_df: pd.DataFrame,
        sentiments_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Hàm tiện ích: tổng hợp feature → fit → predict trong một bước.
        Phù hợp khi train và inference trên cùng một batch.

        Args:
            comments_df  : raw VOZ comments DataFrame
            sentiments_df: sentiment predictions (có thể None)

        Returns:
            DataFrame với post_id làm index, đầy đủ features + anomaly_score + is_anomaly
        """
        features = extract_features(comments_df, sentiments_df)
        self.fit(features)
        return self.predict(features)

    def fit_predict_from_files(
        self,
        comments_path: str,
        sentiments_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load CSV → detect. Tiện dùng từ CLI hoặc script.

        Args:
            comments_path  : đường dẫn tới voz_comments.csv
            sentiments_path: đường dẫn tới sentiments CSV (cột post_id, sentiment_label)
                             Có thể None nếu chưa có model.
        """
        print(f"[CrisisDetector] Đọc comments: {comments_path}")
        comments_df = pd.read_csv(comments_path)

        sentiments_df = None
        if sentiments_path and os.path.exists(sentiments_path):
            print(f"[CrisisDetector] Đọc sentiments: {sentiments_path}")
            sentiments_df = pd.read_csv(sentiments_path)
        else:
            print("[CrisisDetector] Không có sentiment file → neg_ratio/pos_ratio = 0")

        return self.detect(comments_df, sentiments_df)


# ── Report helper ─────────────────────────────────────────────────────────────

def print_report(results: pd.DataFrame, top_n: int = 10) -> None:
    """In tóm tắt kết quả anomaly detection."""
    n_anomaly = results["is_anomaly"].sum()
    total     = len(results)

    print(f"\n{'='*60}")
    print(f"  CRISIS DETECTION REPORT")
    print(f"{'='*60}")
    print(f"  Tổng số posts phân tích : {total}")
    print(f"  Posts bất thường        : {n_anomaly} ({n_anomaly/total*100:.1f}%)")
    print(f"  Contamination rate      : {n_anomaly/total:.3f}")
    print(f"{'='*60}")

    anomalies = results[results["is_anomaly"]].sort_values("anomaly_score")

    if anomalies.empty:
        print("  Không phát hiện bất thường nào.\n")
        return

    print(f"\n  Top {min(top_n, len(anomalies))} posts bất thường nhất:\n")
    cols_display = ["anomaly_score", "neg_ratio", "velocity", "comment_count"]
    cols_display = [c for c in cols_display if c in anomalies.columns]
    print(anomalies[cols_display].head(top_n).to_string(float_format="{:.3f}".format))
    print()

    print("  Thống kê features (anomaly vs normal):")
    for feat in FEATURES:
        a_mean = anomalies[feat].mean()
        n_mean = results[~results["is_anomaly"]][feat].mean()
        print(f"    {feat:20s}  anomaly={a_mean:.3f}  normal={n_mean:.3f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Isolation Forest — phát hiện post bất thường từ VOZ data"
    )
    parser.add_argument(
        "--comments",
        default="data/comments.csv",
        help="Đường dẫn tới file comments CSV (mặc định: data/comments.csv)",
    )
    parser.add_argument(
        "--sentiments",
        default=None,
        help="Đường dẫn tới file sentiments CSV có cột [post_id, sentiment_label] (tuỳ chọn)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Tỉ lệ outlier ước tính (mặc định: 0.05 = 5%%)",
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
        help="Số posts bất thường in ra màn hình (mặc định: 10)",
    )
    args = parser.parse_args()

    detector = CrisisDetector(contamination=args.contamination)
    results  = detector.fit_predict_from_files(
        comments_path   = args.comments,
        sentiments_path = args.sentiments,
    )

    print_report(results, top_n=args.top)

    if args.output:
        results.reset_index().to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"[CrisisDetector] Đã lưu kết quả: {args.output}")
