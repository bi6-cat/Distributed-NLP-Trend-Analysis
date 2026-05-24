import os
import sys
import argparse
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isolation_forest import CrisisDetector as IsoDetector
from models.rolling_threshold import RollingThreshold

SEVERITY_MAP = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "NORMAL"}

OUTPUT_COLUMNS = [
    "event_id",
    "detected_at",
    "severity",
    "anomaly_score",
    "trigger_conditions",
    "affected_topics",
    "neg_ratio",
    "mention_velocity",
    "evidence_post_ids",
]


class CrisisDetector:

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

    def run(
        self,
        comments_df: pd.DataFrame,
        sentiments_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        print("[CrisisDetector] Bước 1/3: IsolationForest...")
        iso_results = self.iso_detector.detect(comments_df, sentiments_df)

        print("[CrisisDetector] Bước 2/3: Rolling Threshold...")
        global_spikes = self.rolling.detect_global(comments_df, sentiments_df, metric="comment_count")
        per_post_spikes = self.rolling.detect_per_post(comments_df, sentiments_df=sentiments_df, metric="comment_count")

        print("[CrisisDetector] Bước 3/3: Tổng hợp điều kiện...")
        return self._combine(iso_results, global_spikes, per_post_spikes, comments_df)

    def run_from_files(
        self,
        comments_path: str,
        posts_path: Optional[str] = None,
        sentiments_path: Optional[str] = None,
    ) -> pd.DataFrame:
        from schemas.voz_adapter import VozAdapter

        adapter = VozAdapter()
        comments_df, _ = adapter.from_csv(comments_path, posts_path)
        print(f"[CrisisDetector] Comments: {len(comments_df)} rows")

        sentiments_df = None
        if sentiments_path and os.path.exists(sentiments_path):
            sentiments_df = pd.read_csv(sentiments_path)

        return self.run(comments_df, sentiments_df)

    def run_from_clickhouse(
        self,
        host: str,
        database: str = "tech_radar",
        user: str = "default",
        password: str = "",
        write_back: bool = True,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError("Chạy: pip install clickhouse-connect")

        client = clickhouse_connect.get_client(host=host, database=database, username=user, password=password)

        limit_clause = f"LIMIT {limit}" if limit else ""

        print("[CrisisDetector] Đọc stg_posts_core (comments only)...")
        posts_df = client.query_df(f"""
            SELECT
                parent_id      AS id_post,
                body           AS comment,
                created_at     AS time,
                reaction_count AS reactions,
                comment_count
            FROM stg_posts_core
            WHERE parent_id IS NOT NULL
            {limit_clause}
        """)
        posts_df = posts_df.dropna(subset=["time"])
        posts_df["time"] = pd.to_datetime(posts_df["time"]).dt.strftime("%b %d, %Y at %I:%M %p")
        # _parse_reactions expect string "Ưng (3) | Haha (1)" — convert int → string
        posts_df["reactions"] = posts_df["reactions"].apply(lambda x: f"({int(x)})" if pd.notna(x) else "")
        print(f"[CrisisDetector] comments: {len(posts_df)} rows")

        print("[CrisisDetector] Đọc stg_posts_nlp (comments only)...")
        sentiments_df = client.query_df(f"""
            SELECT c.parent_id AS post_id, n.sentiment_label
            FROM stg_posts_nlp n
            JOIN stg_posts_core c ON c.post_id = n.post_id
            WHERE c.parent_id IS NOT NULL
            {limit_clause}
        """)
        print(f"[CrisisDetector] stg_posts_nlp comments: {len(sentiments_df)} rows")

        events = self.run(posts_df, sentiments_df if not sentiments_df.empty else None)

        if write_back and not events.empty:
            self.write_clickhouse(events, host=host, database=database, user=user, password=password)

        return events

    def write_clickhouse(
        self,
        alerts: pd.DataFrame,
        host: str,
        database: str = "tech_radar",
        user: str = "default",
        password: str = "",
        table: str = "stg_crisis_events",
    ) -> None:
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError("Chạy: pip install clickhouse-connect")

        client = clickhouse_connect.get_client(host=host, database=database, username=user, password=password)
        df = alerts[OUTPUT_COLUMNS].copy()
        if "detected_at" in df.columns:
            df["detected_at"] = pd.to_datetime(df["detected_at"]).dt.tz_localize(None)
        client.insert_df(table, df)
        print(f"[CrisisDetector] Đã ghi {len(df)} events vào {table}")

    def write_hourly_events(
        self,
        hourly_df: pd.DataFrame,
        host: str,
        database: str = "tech_radar",
        user: str = "default",
        password: str = "",
        table: str = "stg_crisis_events",
    ) -> None:
        """
        Ghi DataFrame hourly aggregation vào bảng stg_crisis_events.

        hourly_df phải có các cột:
            date, hour, comment_count, z_score,
            global_spike, if_spike, is_spike, is_crisis,
            neg_ratio, neg_score_avg
        """
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError("Chạy: pip install clickhouse-connect")

        HOURLY_COLUMNS = [
            "date", "hour", "comment_count", "z_score",
            "global_spike", "if_spike", "is_spike", "is_crisis",
            "neg_ratio", "neg_score_avg",
        ]
        missing = [c for c in HOURLY_COLUMNS if c not in hourly_df.columns]
        if missing:
            raise ValueError(f"hourly_df thiếu cột: {missing}")

        client = clickhouse_connect.get_client(host=host, database=database, username=user, password=password)
        df = hourly_df[HOURLY_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["hour"] = df["hour"].astype("uint8")
        df["comment_count"] = df["comment_count"].astype("uint32")
        for col in ("global_spike", "if_spike", "is_spike", "is_crisis"):
            df[col] = df[col].astype("uint8")
        client.insert_df(table, df)
        print(f"[CrisisDetector] Đã ghi {len(df)} hourly events vào {table}")

    def _combine(
        self,
        iso_results: pd.DataFrame,
        global_spikes: pd.DataFrame,
        per_post_spikes: dict,
        comments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        from models.rolling_threshold import _parse_time_series

        df = comments_df.copy().rename(columns={"id_post": "post_id"})
        df["bucket"] = _parse_time_series(df["time"]).dt.floor(self.rolling.freq)

        global_spike_buckets = set(global_spikes[global_spikes["is_spike"]].index.tolist())

        records = []
        for post_id in iso_results.index:
            cond1 = bool(iso_results.loc[post_id, "is_anomaly"])
            cond2 = len(set(df[df["post_id"] == post_id]["bucket"].dropna()) & global_spike_buckets) > 0
            cond3 = post_id in per_post_spikes and int(per_post_spikes[post_id]["is_spike"].sum()) > 0

            score    = int(cond1) + int(cond2) + int(cond3)
            severity = SEVERITY_MAP.get(score, "NORMAL")
            triggers = (["isolation_forest"] if cond1 else []) + \
                       (["global_spike"]     if cond2 else []) + \
                       (["per_post_spike"]   if cond3 else [])

            records.append({
                "event_id":           str(uuid.uuid4()),
                "detected_at":        datetime.now(timezone.utc).isoformat(),
                "severity":           severity,
                "anomaly_score":      float(iso_results.loc[post_id, "anomaly_score"]),
                "trigger_conditions": triggers,
                "affected_topics":    [],
                "neg_ratio":          float(iso_results.loc[post_id, "neg_ratio"]),
                "mention_velocity":   float(iso_results.loc[post_id, "velocity"]),
                "evidence_post_ids":  [str(post_id)],
                "_crisis_score":      score,
                "_is_crisis":         score >= self.min_conditions,
            })

        events = pd.DataFrame(records)
        return events.sort_values(["_crisis_score", "anomaly_score"], ascending=[False, True]).reset_index(drop=True)


def print_report(events: pd.DataFrame, top_n: int = 10) -> None:
    total    = len(events)
    n_crisis = events["_is_crisis"].sum() if "_is_crisis" in events.columns else events["severity"].isin(["HIGH", "MEDIUM"]).sum()
    n_high   = (events["severity"] == "HIGH").sum()
    n_medium = (events["severity"] == "MEDIUM").sum()

    print(f"\n{'='*65}")
    print(f"  CRISIS DETECTION REPORT")
    print(f"{'='*65}")
    print(f"  Tổng posts  : {total}")
    print(f"  Crisis      : {n_crisis} ({n_crisis/total*100:.1f}%)")
    print(f"    HIGH (3/3): {n_high}")
    print(f"    MED  (2/3): {n_medium}")
    print(f"{'='*65}")

    crisis_events = events[events["severity"].isin(["HIGH", "MEDIUM"])]
    if crisis_events.empty:
        print("  Không phát hiện crisis nào.\n")
        return

    disp = crisis_events[["event_id", "severity", "anomaly_score", "neg_ratio", "mention_velocity", "trigger_conditions"]].head(top_n).copy()
    disp["event_id"] = disp["event_id"].str[:8] + "..."
    disp["trigger_conditions"] = disp["trigger_conditions"].apply(lambda x: ",".join(x) if isinstance(x, list) else x)
    print(f"\n  Top {min(top_n, len(crisis_events))} crisis events:\n")
    print(disp.to_string(index=False))

    print("\n  Tần suất trigger conditions:")
    for name, label in [("isolation_forest", "IsolationForest"), ("global_spike", "GlobalSpike"), ("per_post_spike", "PerPostSpike")]:
        n = crisis_events["trigger_conditions"].apply(lambda x: name in x if isinstance(x, list) else False).sum()
        print(f"    {label:20s}: {n}/{n_crisis} ({n/max(n_crisis,1)*100:.0f}%)")
    print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--comments",       default="data/comments.csv")
    parser.add_argument("--posts",          default=None)
    parser.add_argument("--sentiments",     default=None)
    parser.add_argument("--min_conditions", type=int,   default=2)
    parser.add_argument("--contamination",  type=float, default=0.05)
    parser.add_argument("--window",         type=int,   default=24)
    parser.add_argument("--sigma",          type=float, default=2.0)
    parser.add_argument("--freq",           default="1h")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--top",            type=int,   default=10)
    
    # Clickhouse args
    parser.add_argument("--clickhouse-host", default=None, help="Host của ClickHouse để đọc/ghi trực tiếp (thay vì CSV)")
    parser.add_argument("--clickhouse-db",   default="tech_radar")
    parser.add_argument("--clickhouse-user", default="root")
    parser.add_argument("--clickhouse-pass", default="root")
    parser.add_argument("--clickhouse-limit", type=int, default=None, help="Giới hạn số lượng records khi truy vấn ClickHouse")
    args = parser.parse_args()

    detector = CrisisDetector(
        min_conditions=args.min_conditions,
        contamination=args.contamination,
        window=args.window,
        n_sigma=args.sigma,
        freq=args.freq,
    )

    if args.clickhouse_host:
        print(f"[CrisisDetector] Chế độ ClickHouse (Host: {args.clickhouse_host})")
        events = detector.run_from_clickhouse(
            host=args.clickhouse_host,
            database=args.clickhouse_db,
            user=args.clickhouse_user,
            password=args.clickhouse_pass,
            write_back=True,
            limit=args.clickhouse_limit
        )
    else:
        print("[CrisisDetector] Chế độ File CSV/Local")
        events = detector.run_from_files(args.comments, args.posts, args.sentiments)
        
    print_report(events, top_n=args.top)

    out = args.output or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "crisis_events.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    events.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[CrisisDetector] Đã lưu: {out}")
