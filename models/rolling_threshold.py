"""
Task 3.4 — Rolling Mean Threshold: phát hiện spike hoạt động

Nguyên lý:
    spike = value(t) > rolling_mean(t) + n_sigma * rolling_std(t)

Hai chế độ phân tích:
  1. GLOBAL  — toàn bộ platform, theo giờ/ngày
               Ví dụ: số comments/giờ trên VOZ tăng đột biến
  2. PER-POST — từng post riêng lẻ
               Ví dụ: post X đang nhận bình luận gấp 3x bình thường

Metrics được hỗ trợ:
  - comment_count : số comments trong mỗi time bucket
  - neg_count     : số comments Negative (cần sentiment_df)
  - neg_ratio     : tỉ lệ Negative (cần sentiment_df)
  - engagement    : tổng reactions trong bucket

Cách dùng:
  1. CLI:
       python models/rolling_threshold.py \\
           --comments data/voz_comments.csv \\
           --window 24 --sigma 2.0 --freq 1h

  2. Import:
       from models.rolling_threshold import RollingThreshold

       rt = RollingThreshold(window=24, n_sigma=2.0)
       spikes = rt.detect_global(comments_df)
       per_post = rt.detect_per_post(comments_df, post_id=1208381)
"""

import re
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Union


# ── Hằng số ───────────────────────────────────────────────────────────────────

COMMENT_TIME_FORMAT = "%b %d, %Y at %I:%M %p"   # "Feb 23, 2026 at 4:00 PM"

SUPPORTED_METRICS = ("comment_count", "neg_count", "neg_ratio", "engagement")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time_series(time_col: pd.Series) -> pd.Series:
    """Parse cột thời gian VOZ → datetime. Các giá trị lỗi trả NaT."""
    def _safe_parse(s):
        if not isinstance(s, str):
            return pd.NaT
        try:
            return datetime.strptime(s.strip(), COMMENT_TIME_FORMAT)
        except ValueError:
            return pd.NaT

    return time_col.apply(_safe_parse)


def _parse_reactions(reactions_str) -> int:
    """Trích tổng số reactions từ chuỗi 'Ưng (3) | Haha (1)'."""
    if not isinstance(reactions_str, str):
        return 0
    return sum(int(c) for c in re.findall(r"\((\d+)\)", reactions_str))


def _prepare_df(
    comments_df: pd.DataFrame,
    sentiments_df: Optional[pd.DataFrame],
    freq: str,
) -> pd.DataFrame:
    """
    Chuẩn bị DataFrame với cột datetime, bucket time, reactions, sentiment.

    Args:
        comments_df  : raw VOZ comments (id_post, comment, time, reactions, ...)
        sentiments_df: optional DataFrame với cột [post_id/id_post, sentiment_label]
        freq         : pandas offset alias cho time bucket ('1h', '6h', '1D', ...)

    Returns:
        DataFrame với cột: post_id, dt, bucket, reactions_count, sentiment_label
    """
    df = comments_df.copy()
    df = df.rename(columns={"id_post": "post_id"})

    # Parse time
    df["dt"] = _parse_time_series(df["time"])
    df = df.dropna(subset=["dt"])
    df["bucket"] = df["dt"].dt.floor(freq)

    # Reactions
    df["reactions_count"] = df["reactions"].apply(_parse_reactions)

    # Merge sentiments nếu có
    if sentiments_df is not None and not sentiments_df.empty:
        sent = sentiments_df.copy()
        if "id_post" in sent.columns and "post_id" not in sent.columns:
            sent = sent.rename(columns={"id_post": "post_id"})
        # Merge theo index hoặc post_id + thứ tự comment
        if "comment_idx" in sent.columns:
            df = df.reset_index(drop=True)
            df["comment_idx"] = df.index
            df = df.merge(sent[["post_id", "comment_idx", "sentiment_label"]],
                          on=["post_id", "comment_idx"], how="left")
        else:
            # Fallback: merge theo thứ tự trong từng post
            sent_grouped = (
                sent.groupby("post_id")["sentiment_label"]
                .apply(list)
                .reset_index()
            )
            df = df.sort_values(["post_id", "dt"]).reset_index(drop=True)
            labels = []
            for _, grp in df.groupby("post_id", sort=False):
                post_id = grp["post_id"].iloc[0]
                match = sent_grouped[sent_grouped["post_id"] == post_id]
                if not match.empty:
                    s_list = match["sentiment_label"].iloc[0]
                    for i in range(len(grp)):
                        labels.append(s_list[i] if i < len(s_list) else None)
                else:
                    labels.extend([None] * len(grp))
            df["sentiment_label"] = labels
    else:
        df["sentiment_label"] = None

    return df[["post_id", "dt", "bucket", "reactions_count", "sentiment_label", "comment"]]


def _aggregate_metric(
    df: pd.DataFrame,
    metric: str,
    group_cols: list,
) -> pd.Series:
    """
    Tổng hợp metric theo group_cols (ví dụ: ['bucket'] hoặc ['post_id', 'bucket']).

    Returns:
        Series với index = group_cols, values = metric value
    """
    grp = df.groupby(group_cols)

    if metric == "comment_count":
        return grp["comment"].count().rename(metric)

    elif metric == "neg_count":
        return (
            grp["sentiment_label"]
            .apply(lambda s: (s == "Negative").sum())
            .rename(metric)
        )

    elif metric == "neg_ratio":
        def _neg_ratio(s):
            total = len(s)
            if total == 0:
                return 0.0
            return (s == "Negative").sum() / total
        return grp["sentiment_label"].apply(_neg_ratio).rename(metric)

    elif metric == "engagement":
        return grp["reactions_count"].sum().rename(metric)

    else:
        raise ValueError(f"Metric không hỗ trợ: {metric}. Chọn từ: {SUPPORTED_METRICS}")


# ── Rolling Threshold ─────────────────────────────────────────────────────────

class RollingThreshold:
    """
    Phát hiện spike bằng rolling mean + n*sigma.

    Args:
        window    : số time buckets dùng cho rolling window (mặc định 24 = 24h nếu freq='1h')
        n_sigma   : hệ số độ lệch chuẩn (mặc định 2.0)
        freq      : kích thước time bucket — pandas offset alias (mặc định '1h')
        min_periods: số tối thiểu buckets để tính rolling stats (mặc định window//2)
    """

    def __init__(
        self,
        window: int = 24,
        n_sigma: float = 2.0,
        freq: str = "1h",
        min_periods: Optional[int] = None,
    ):
        self.window      = window
        self.n_sigma     = n_sigma
        self.freq        = freq
        self.min_periods = min_periods if min_periods is not None else max(window // 2, 2)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_global(
        self,
        comments_df: pd.DataFrame,
        sentiments_df: Optional[pd.DataFrame] = None,
        metric: str = "comment_count",
    ) -> pd.DataFrame:
        """
        Phát hiện spike trên toàn bộ platform (tổng hợp mọi post).

        Args:
            comments_df  : raw VOZ comments
            sentiments_df: optional sentiment predictions
            metric       : comment_count | neg_count | neg_ratio | engagement

        Returns:
            DataFrame index=bucket với cột:
              value, rolling_mean, rolling_std, threshold, is_spike
        """
        df = _prepare_df(comments_df, sentiments_df, self.freq)
        series = _aggregate_metric(df, metric, group_cols=["bucket"])

        # Reindex để lấp đầy khoảng trống (giờ không có comment → 0)
        full_idx = pd.date_range(series.index.min(), series.index.max(), freq=self.freq)
        series = series.reindex(full_idx, fill_value=0)

        return self._apply_rolling(series, label=metric)

    def detect_per_post(
        self,
        comments_df: pd.DataFrame,
        post_id: Optional[int] = None,
        sentiments_df: Optional[pd.DataFrame] = None,
        metric: str = "comment_count",
    ) -> Union[pd.DataFrame, dict]:
        """
        Phát hiện spike cho từng post riêng lẻ.

        Args:
            comments_df  : raw VOZ comments
            post_id      : nếu chỉ định → trả DataFrame của post đó
                           nếu None → trả dict {post_id: DataFrame} cho mọi post
            sentiments_df: optional sentiment predictions
            metric       : comment_count | neg_count | neg_ratio | engagement

        Returns:
            DataFrame (nếu post_id được chỉ định) hoặc dict[post_id → DataFrame]
        """
        df = _prepare_df(comments_df, sentiments_df, self.freq)

        if post_id is not None:
            sub = df[df["post_id"] == post_id]
            if sub.empty:
                raise ValueError(f"Không tìm thấy post_id={post_id} trong dữ liệu.")
            series = _aggregate_metric(sub, metric, group_cols=["bucket"])
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq=self.freq)
            series = series.reindex(full_idx, fill_value=0)
            return self._apply_rolling(series, label=metric)

        # Tất cả posts
        results = {}
        for pid, grp in df.groupby("post_id"):
            series = _aggregate_metric(grp, metric, group_cols=["bucket"])
            if len(series) < self.min_periods:
                continue   # quá ít data points
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq=self.freq)
            series = series.reindex(full_idx, fill_value=0)
            results[pid] = self._apply_rolling(series, label=metric)
        return results

    def summarize_spikes(
        self,
        spike_df: pd.DataFrame,
        label: str = "",
    ) -> pd.DataFrame:
        """
        Trích xuất các điểm spike từ kết quả detect_global / detect_per_post.

        Returns:
            DataFrame chỉ chứa các time bucket là spike, sắp xếp theo mức độ vượt threshold.
        """
        spikes = spike_df[spike_df["is_spike"]].copy()
        spikes["excess"] = spikes["value"] - spikes["threshold"]
        spikes = spikes.sort_values("excess", ascending=False)
        return spikes

    # ── Internal ──────────────────────────────────────────────────────────────

    def _apply_rolling(self, series: pd.Series, label: str) -> pd.DataFrame:
        """
        Áp dụng rolling mean + n*sigma lên time series.

        Args:
            series: pd.Series index=datetime, values=metric
            label : tên metric (dùng để đặt tên cột)

        Returns:
            DataFrame với cột: value, rolling_mean, rolling_std, threshold, is_spike
        """
        roll       = series.rolling(window=self.window, min_periods=self.min_periods)
        r_mean     = roll.mean()
        r_std      = roll.std().fillna(0)
        threshold  = r_mean + self.n_sigma * r_std

        result = pd.DataFrame({
            "value":        series,
            "rolling_mean": r_mean,
            "rolling_std":  r_std,
            "threshold":    threshold,
            "is_spike":     series > threshold,
        })
        result.index.name = "bucket"
        return result


# ── Report helper ─────────────────────────────────────────────────────────────

def print_global_report(
    spike_df: pd.DataFrame,
    metric: str,
    window: int,
    n_sigma: float,
    top_n: int = 10,
) -> None:
    """In báo cáo kết quả detect_global."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    n_spike  = spike_df["is_spike"].sum()
    n_total  = len(spike_df)
    n_nonnull = spike_df["rolling_mean"].notna().sum()

    print(f"\n{'='*62}")
    print(f"  ROLLING THRESHOLD REPORT — metric: {metric}")
    print(f"{'='*62}")
    print(f"  Window         : {window} buckets")
    print(f"  Sigma          : {n_sigma}")
    print(f"  Total buckets  : {n_total}")
    print(f"  Buckets với đủ data : {n_nonnull}")
    print(f"  Spike buckets  : {n_spike}")
    if n_nonnull > 0:
        print(f"  Spike rate     : {n_spike/n_nonnull*100:.1f}%")
    print(f"{'='*62}")

    spikes = spike_df[spike_df["is_spike"]].copy()
    spikes["excess"] = spikes["value"] - spikes["threshold"]
    spikes = spikes.sort_values("excess", ascending=False)

    if spikes.empty:
        print("  Không phát hiện spike nào.\n")
        return

    print(f"\n  Top {min(top_n, len(spikes))} spike mạnh nhất:\n")
    display = spikes[["value", "rolling_mean", "rolling_std", "threshold", "excess"]].head(top_n)
    print(display.to_string(float_format="{:.2f}".format))
    print()

    # Thống kê tổng quát
    print("  Thống kê tổng quát:")
    print(f"    Giá trị max   : {spike_df['value'].max():.1f}")
    print(f"    Giá trị mean  : {spike_df['value'].mean():.2f}")
    print(f"    Spike max     : {spikes['value'].max():.1f}  "
          f"(threshold={spikes.loc[spikes['value'].idxmax(), 'threshold']:.2f})")
    print()


def print_per_post_report(
    per_post_results: dict,
    top_n_posts: int = 5,
    top_n_spikes: int = 3,
) -> None:
    """In báo cáo kết quả detect_per_post (tất cả posts)."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    posts_with_spikes = {
        pid: df for pid, df in per_post_results.items()
        if df["is_spike"].any()
    }

    print(f"\n{'='*62}")
    print(f"  PER-POST ROLLING THRESHOLD REPORT")
    print(f"{'='*62}")
    print(f"  Posts phân tích    : {len(per_post_results)}")
    print(f"  Posts có spike     : {len(posts_with_spikes)}")
    print(f"{'='*62}")

    if not posts_with_spikes:
        print("  Không post nào có spike.\n")
        return

    # Xếp hạng theo số spike giảm dần
    ranked = sorted(
        posts_with_spikes.items(),
        key=lambda kv: kv[1]["is_spike"].sum(),
        reverse=True,
    )

    print(f"\n  Top {min(top_n_posts, len(ranked))} posts nhiều spike nhất:\n")
    for i, (pid, df) in enumerate(ranked[:top_n_posts]):
        n_sp = df["is_spike"].sum()
        spikes = df[df["is_spike"]].copy()
        spikes["excess"] = spikes["value"] - spikes["threshold"]
        max_spike = spikes["excess"].max()
        print(f"  [{i+1}] post_id={pid} — {n_sp} spikes, max_excess={max_spike:.2f}")
        top_sp = spikes.sort_values("excess", ascending=False).head(top_n_spikes)
        for ts, row in top_sp.iterrows():
            print(f"        {ts}  value={row['value']:.0f}  "
                  f"threshold={row['threshold']:.2f}  excess=+{row['excess']:.2f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Rolling Mean Threshold — phát hiện spike trên VOZ comment stream"
    )
    parser.add_argument(
        "--comments",
        default="data/comments.csv",
        help="Đường dẫn tới comments.csv",
    )
    parser.add_argument(
        "--sentiments",
        default=None,
        help="File sentiments CSV có cột [post_id, sentiment_label] (tuỳ chọn)",
    )
    parser.add_argument(
        "--metric",
        default="comment_count",
        choices=list(SUPPORTED_METRICS),
        help="Metric để phân tích (mặc định: comment_count)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Kích thước rolling window tính bằng số buckets (mặc định: 24)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Hệ số sigma cho ngưỡng spike (mặc định: 2.0)",
    )
    parser.add_argument(
        "--freq",
        default="1h",
        help="Kích thước time bucket — pandas offset alias (mặc định: 1h)",
    )
    parser.add_argument(
        "--mode",
        default="global",
        choices=["global", "per_post"],
        help="Chế độ phân tích: global (toàn platform) hoặc per_post (từng post)",
    )
    parser.add_argument(
        "--post_id",
        type=int,
        default=None,
        help="(chỉ dùng với --mode per_post) Post ID cụ thể cần phân tích",
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
        help="Số spike in ra màn hình (mặc định: 10)",
    )
    args = parser.parse_args()

    # Load data
    print(f"[RollingThreshold] Đọc: {args.comments}")
    comments_df = pd.read_csv(args.comments)

    sentiments_df = None
    if args.sentiments:
        import os
        if os.path.exists(args.sentiments):
            print(f"[RollingThreshold] Đọc sentiments: {args.sentiments}")
            sentiments_df = pd.read_csv(args.sentiments)
        else:
            print(f"[RollingThreshold] WARN: không tìm thấy {args.sentiments}")

    rt = RollingThreshold(
        window=args.window,
        n_sigma=args.sigma,
        freq=args.freq,
    )

    if args.mode == "global":
        result = rt.detect_global(comments_df, sentiments_df, metric=args.metric)
        print_global_report(result, args.metric, args.window, args.sigma, args.top)

        if args.output:
            result.reset_index().to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"[RollingThreshold] Đã lưu: {args.output}")

    else:  # per_post
        if args.post_id:
            result = rt.detect_per_post(
                comments_df, post_id=args.post_id,
                sentiments_df=sentiments_df, metric=args.metric,
            )
            print(f"\n[RollingThreshold] Per-post phân tích post_id={args.post_id}")
            print_global_report(result, args.metric, args.window, args.sigma, args.top)

            if args.output:
                result.reset_index().to_csv(args.output, index=False, encoding="utf-8-sig")
                print(f"[RollingThreshold] Đã lưu: {args.output}")
        else:
            results = rt.detect_per_post(
                comments_df, sentiments_df=sentiments_df, metric=args.metric,
            )
            print_per_post_report(results, top_n_posts=args.top, top_n_spikes=3)
