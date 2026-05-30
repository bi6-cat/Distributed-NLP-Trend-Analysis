"""Driver-side MinHash/LSH deduplication for stg_posts_core cleaning."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MinHashDeduplicator:
    """Remove near-duplicate rows using k-shingles, MinHash, and LSH."""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.80,
        k: int = 5,
    ) -> None:
        self.num_perm  = num_perm
        self.threshold = threshold
        self.k         = k

    def _shingling(self, text: str) -> set:
        if not text or len(text) < self.k:
            return set()
        return {text[i: i + self.k] for i in range(len(text) - self.k + 1)}

    def _compute_minhash(self, text: str):
        from datasketch import MinHash

        shingles = self._shingling(text)
        if not shingles:
            return None

        mh = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        return mh

    def fit_transform(self, df, spark) -> object:
        """Return a DataFrame with one representative row per near-duplicate cluster."""
        from datasketch import MinHashLSH
        from pyspark.sql.functions import broadcast, expr

        logger.info(
            "[Dedup] MinHashLSH start num_perm=%s threshold=%s k=%s",
            self.num_perm,
            self.threshold,
            self.k,
        )

        working_df = df.withColumn("__dedup_row_id", expr("uuid()")).cache()
        working_df.count()

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
        if not rows:
            return df

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        created_at_map: dict[str, int] = {}
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

        logger.info("[Dedup] Signatures=%s skipped=%s", f"{len(minhash_map):,}", f"{skipped:,}")

        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)
                x = parent.get(x, x)
            return x

        def union(a: str, b: str) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                if created_at_map.get(pa, 0) <= created_at_map.get(pb, 0):
                    parent[pb] = pa
                else:
                    parent[pa] = pb

        inserted_ids = []
        for row_id, mh in minhash_map.items():
            for neighbor_row_id in lsh.query(mh):
                if neighbor_row_id != row_id:
                    union(row_id, neighbor_row_id)

            try:
                lsh.insert(row_id, mh)
                inserted_ids.append(row_id)
                parent.setdefault(row_id, row_id)
            except ValueError:
                # Trùng key — có thể xảy ra với dữ liệu test nhỏ
                pass

        logger.info(f"[Dedup] Đã insert {len(inserted_ids):,} signatures vào LSH index")

        row_ids_to_keep: set[str] = {find(row_id) for row_id in inserted_ids}

        for row_id, _, _, _ in rows:
            if row_id not in minhash_map:
                row_ids_to_keep.add(row_id)

        n_removed = len(rows) - len(row_ids_to_keep)
        pct = n_removed / max(len(rows), 1) * 100
        logger.info(f"[Dedup] Phát hiện {n_removed:,} duplicates ({pct:.1f}% records loại bỏ)")
        logger.info(f"[Dedup] Giữ lại {len(row_ids_to_keep):,} records duy nhất")

        keep_df = spark.createDataFrame([(row_id,) for row_id in row_ids_to_keep], ["__dedup_row_id"])
        deduped_df = working_df.join(broadcast(keep_df), "__dedup_row_id", "inner").drop("__dedup_row_id")

        working_df.unpersist()
        return deduped_df

    def __repr__(self) -> str:
        return (
            f"MinHashDeduplicator("
            f"num_perm={self.num_perm}, "
            f"threshold={self.threshold}, "
            f"k={self.k})"
        )
