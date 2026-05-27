"""
algorithms/minhash_dedup.py — MinHash/LSH Deduplication cho Spark Cleaning Job

Thuật toán:
    segmented_text
        → k=5 character shingles (set)
        → MinHash(num_perm=128) — signature vector
        → LSH bucket (threshold=0.8, 16 bands × 8 rows)
        → Candidate duplicate pairs
        → Giữ post_id có created_at nhỏ nhất (oldest) mỗi cluster

Approach: Driver-side LSH
    - Collect MinHash signatures về driver → chạy datasketch LSH
    - Broadcast set post_ids_to_keep về Spark executors
    - df.filter(col("post_id").isin(...)) → Spark xử lý filter phân tán
    - An toàn với dataset ≤ 2 triệu records (mỗi sig ~1KB, 2M = ~2GB)

Tham số (khớp data_flow_schema_evolution.md Stage 3, Transformation #7):
    k          = 5      # kích thước shingle (ký tự)
    num_perm   = 128    # số hash functions
    bands      = 16     # LSH bands (= num_perm / rows_per_band)
    rows/band  = 8      # rows per band (16 × 8 = 128 ✓)
    threshold  = 0.80   # Jaccard similarity ngưỡng duplicate

Phụ thuộc:
    pip install datasketch   (đã thêm vào ansible/roles/conda/tasks/main.yml)
"""

from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


class MinHashDeduplicator:
    """
    MinHash + LSH deduplicator tích hợp với PySpark DataFrame.

    Ví dụ:
        >>> deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.8, k=5)
        >>> clean_df = deduplicator.fit_transform(result_df, spark)
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.80,
        k: int = 5,
    ) -> None:
        """
        Khởi tạo MinHashDeduplicator.

        Args:
            num_perm:  Số hash functions MinHash (= bands × rows_per_band).
                       128 → 16 bands × 8 rows theo spec.
            threshold: Ngưỡng Jaccard similarity để coi là duplicate (0.0–1.0).
                       0.8 = ≥ 80% overlap mới bị loại.
            k:         Kích thước shingle (ký tự). k=5 phù hợp tiếng Việt.
        """
        self.num_perm  = num_perm
        self.threshold = threshold
        self.k         = k

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _shingling(self, text: str) -> set:
        """
        Tạo tập k-character shingles từ text.

        Ví dụ (k=3): "hello" → {"hel", "ell", "llo"}

        Args:
            text: Văn bản đã được segmented (dùng segmented_text).

        Returns:
            Set of k-char shingle strings. Trả set rỗng nếu text ngắn hơn k.
        """
        if not text or len(text) < self.k:
            return set()
        return {text[i: i + self.k] for i in range(len(text) - self.k + 1)}

    def _compute_minhash(self, text: str):
        """
        Tính MinHash signature từ văn bản.

        Args:
            text: Văn bản nguồn.

        Returns:
            datasketch.MinHash object, hoặc None nếu text rỗng.
        """
        from datasketch import MinHash  # lazy import — chỉ cần ở driver

        shingles = self._shingling(text)
        if not shingles:
            return None

        mh = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        return mh

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df, spark) -> object:
        """
        Loại bỏ duplicate khỏi Spark DataFrame.

        Pipeline:
            1. Collect (post_id, segmented_text, created_at) về driver
            2. Tính MinHash signature cho mỗi bản ghi
            3. Build datasketch MinHashLSH, phát hiện duplicate pairs
            4. Union-Find: gom cluster, giữ post_id có created_at nhỏ nhất
            5. Broadcast set post_ids_to_keep
            6. df.filter(col("post_id").isin(...)) → trả về DataFrame đã dedup

        Args:
            df:    Spark DataFrame có các cột: post_id (String),
                   segmented_text (String), created_at (Long/timestamp).
            spark: SparkSession hiện tại.

        Returns:
            Spark DataFrame đã loại bỏ duplicate.
        """
        from datasketch import MinHashLSH
        from pyspark.sql.functions import col, expr

        logger.info("[Dedup] Bắt đầu MinHash LSH deduplication "
                    f"(num_perm={self.num_perm}, threshold={self.threshold}, k={self.k})")

        # Gắn row id ổn định để dedup theo từng row thay vì chỉ theo post_id.
        working_df = df.withColumn("__dedup_row_id", expr("uuid()")).cache()
        working_df.count()

        # ── Bước 1: Collect metadata về driver ───────────────────────────────
        rows = (
            working_df.select("__dedup_row_id", "post_id", "segmented_text", "created_at")
              .rdd
              .map(
                  lambda r: (
                      r["__dedup_row_id"],
                      r["post_id"],
                      r["segmented_text"] or "",
                      r["created_at"] or 0,
                  )
              )
              .collect()
        )
        logger.info(f"[Dedup] Collected {len(rows):,} records về driver")

        # ── Bước 2: Build MinHash signatures ─────────────────────────────────
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # row_id → created_at (để so sánh oldest)
        created_at_map: dict[str, int] = {}
        # row_id → MinHash (để insert vào LSH)
        minhash_map: dict[str, object] = {}

        skipped = 0
        for row_id, _, text, created_at in rows:
            mh = self._compute_minhash(text)
            if mh is None:
                skipped += 1
                continue
            minhash_map[row_id] = mh
            if hasattr(created_at, 'timestamp'):
                created_at_map[row_id] = int(created_at.timestamp())
            else:
                created_at_map[row_id] = int(created_at)

        logger.info(f"[Dedup] Đã tính signature cho {len(minhash_map):,} records "
                    f"(bỏ qua {skipped:,} records không có text)")

        # ── Bước 3: Insert vào LSH và phát hiện duplicate pairs ──────────────
        # Dùng Union-Find để gom clusters
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            """Path-compressed Union-Find."""
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)  # path compression
                x = parent.get(x, x)
            return x

        def union(a: str, b: str) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                # Giữ node có created_at nhỏ hơn (oldest) làm root
                if created_at_map.get(pa, 0) <= created_at_map.get(pb, 0):
                    parent[pb] = pa
                else:
                    parent[pa] = pb

        # Insert và query LSH — O(n) amortized
        inserted_ids = []
        for row_id, mh in minhash_map.items():
            # Query trước khi insert để tìm gần đúng (approximate neighbors)
            try:
                neighbors = lsh.query(mh)
                for neighbor_row_id in neighbors:
                    if neighbor_row_id != row_id:
                        union(row_id, neighbor_row_id)
            except Exception:
                pass  # row_id chưa có trong LSH, bỏ qua

            # Insert vào LSH (dùng row_id làm key)
            try:
                lsh.insert(row_id, mh)
                inserted_ids.append(row_id)
                parent.setdefault(row_id, row_id)
            except ValueError:
                # Trùng key — có thể xảy ra với dữ liệu test nhỏ
                pass

        logger.info(f"[Dedup] Đã insert {len(inserted_ids):,} signatures vào LSH index")

        # ── Bước 4: Xác định post_ids cần giữ (1 per cluster = oldest) ───────
        # Với mỗi row_id, root của Union-Find là representative được giữ lại
        row_ids_to_keep: Set[str] = {find(row_id) for row_id in inserted_ids}

        # Thêm lại các records bị skip (không có text) — không ảnh hưởng dedup
        for row_id, _, _, _ in rows:
            if row_id not in minhash_map:
                row_ids_to_keep.add(row_id)

        n_removed = len(rows) - len(row_ids_to_keep)
        pct = n_removed / max(len(rows), 1) * 100
        logger.info(f"[Dedup] Phát hiện {n_removed:,} duplicates ({pct:.1f}% records loại bỏ)")
        logger.info(f"[Dedup] Giữ lại {len(row_ids_to_keep):,} records duy nhất")

        # ── Bước 5: Broadcast + filter ────────────────────────────────────────
        bc_keep = spark.sparkContext.broadcast(row_ids_to_keep)

        deduped_df = (
            working_df
            .filter(col("__dedup_row_id").isin(list(bc_keep.value)))
            .drop("__dedup_row_id")
        )

        bc_keep.unpersist()
        working_df.unpersist()

        return deduped_df

    def __repr__(self) -> str:
        return (
            f"MinHashDeduplicator("
            f"num_perm={self.num_perm}, "
            f"threshold={self.threshold}, "
            f"k={self.k})"
        )
