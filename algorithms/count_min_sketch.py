"""
algorithms/count_min_sketch.py — Count-Min Sketch cho Keyword Frequency Streaming

Task 2.3 (Member 3 — Phase 2)
Dự án: Vietnamese Social Media Trend & Controversy Analysis System

Count-Min Sketch (CMS) là cấu trúc dữ liệu xác suất dùng để ước lượng
tần suất (frequency) của phần tử trong luồng dữ liệu (data stream) với
bộ nhớ cố định, bất kể kích thước dữ liệu.

Lý thuyết (CS246 — Mining Massive Datasets, Stanford):
    Cấu trúc: Ma trận d × w (depth × width)
        d = số hàm hash (depth)     → kiểm soát xác suất sai (δ)
        w = chiều rộng (width)      → kiểm soát biên sai số (ε)

    Giới hạn sai số:
        ε (epsilon) = e / w         → sai số tối đa so với tổng count
        δ (delta) = (1/2)^d         → xác suất vượt quá sai số ε

    Với d=5, w=2048 (cấu hình mặc định theo TECH_STACK.md):
        ε = e / 2048 ≈ 0.00133     → sai số < 0.13% tổng count
        δ = (1/2)^5 = 0.03125      → xác suất sai > ε chỉ 3.1%

    Tính chất quan trọng:
        ① KHÔNG BAO GIỜ undercount (luôn ≥ exact count)
        ② Có thể overcount, nhưng bị giới hạn bởi ε × N
        ③ Bộ nhớ O(d × w) — cố định, không phụ thuộc N
        ④ Có thể merge 2 sketch (cộng element-wise)

Tham chiếu:
    [1] Cormode & Muthukrishnan (2005). "An Improved Data Stream Summary:
        The Count-Min Sketch and its Applications". Journal of Algorithms.
    [2] Leskovec, Rajaraman, Ullman. "Mining of Massive Datasets" (CS246).
        Chapter 4: Mining Data Streams.

Cách sử dụng:
    >>> from algorithms.count_min_sketch import CountMinSketch
    >>> cms = CountMinSketch(d=5, w=2048)
    >>> cms.add("iphone")
    >>> cms.add("iphone")
    >>> cms.add("samsung")
    >>> cms.query("iphone")   # → 2 (hoặc >= 2, never undercount)
    >>> cms.query("samsung")  # → 1 (hoặc >= 1)
    >>> cms.query("pixel")    # → 0 (hoặc >= 0, có thể false positive)

Phụ thuộc:
    - mmh3 (MurmurHash3): Hash phi mã hóa, nhanh, phân phối đều
    - Cài đặt: pip install mmh3
"""

import hashlib
import math
import pickle
import sys
from typing import Generator, Iterator, List, Optional, Tuple

try:
    import mmh3  # type: ignore  # noqa: F401
except ImportError:
    mmh3 = None


class CountMinSketch:
    """
    Count-Min Sketch — Cấu trúc dữ liệu xác suất cho đếm tần suất streaming.

    Thuật toán (Cormode & Muthukrishnan, 2005):
        1. Khởi tạo ma trận d × w toàn 0
        2. ADD(item): với mỗi hàng i (0..d-1), tăng table[i][hash_i(item) % w]
        3. QUERY(item): trả về min(table[i][hash_i(item) % w]) cho i = 0..d-1

    Attributes:
        d (int): Số hàm hash (depth). Mặc định 5.
        w (int): Chiều rộng mỗi hàng (width). Mặc định 2048.
        table (List[List[int]]): Ma trận counter d × w.
        total_count (int): Tổng số lần add (dùng cho tính error rate).
    """

    def __init__(self, d: int = 5, w: int = 2048) -> None:
        """
        Khởi tạo Count-Min Sketch.

        Cấu hình mặc định d=5, w=2048 theo TECH_STACK.md (Section 5.3):
            ε = e/2048 ≈ 0.00133 → sai số < 0.13% tổng count
            δ = (1/2)^5 = 0.03125 → xác suất vượt quá ε chỉ 3.1%

        Args:
            d: Số hàm hash (depth). Tăng d → giảm δ (ít sai hơn).
            w: Chiều rộng (width). Tăng w → giảm ε (chính xác hơn).

        Raises:
            ValueError: Nếu d hoặc w <= 0.
        """
        if d <= 0 or w <= 0:
            raise ValueError(f"d và w phải > 0. Nhận d={d}, w={w}")

        self.d: int = d
        self.w: int = w
        self.table: List[List[int]] = [[0] * w for _ in range(d)]
        self.total_count: int = 0

    # ── Thuộc tính toán học ──────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """
        Biên sai số ε = e / w.

        Ý nghĩa: Với xác suất ≥ (1 - δ), giá trị ước lượng query(x)
        thỏa mãn: exact(x) ≤ query(x) ≤ exact(x) + ε × N
        trong đó N = tổng số lần add.

        Returns:
            float: Giá trị epsilon.
        """
        return math.e / self.w

    @property
    def delta(self) -> float:
        """
        Xác suất sai vượt ε: δ = (1/2)^d.

        Ý nghĩa: Xác suất query(x) > exact(x) + ε×N là ≤ δ.

        Returns:
            float: Giá trị delta.
        """
        return (0.5) ** self.d

    # ── Hàm hash ────────────────────────────────────────────────

    def _hash(self, item: str, seed: int) -> int:
        """
        Hash item với MurmurHash3 và seed cho trước.

        Tại sao MurmurHash3?
            - Phi mã hóa (non-cryptographic) → nhanh hơn SHA/MD5
            - Phân phối đều (uniform distribution) → ít collision
            - Hỗ trợ seed → tạo d hàm hash độc lập từ 1 hàm

        Args:
            item: Chuỗi cần hash (keyword).
            seed: Seed tạo hàm hash khác nhau cho mỗi hàng.

        Returns:
            int: Chỉ số cột trong [0, w).
        """
        if mmh3 is not None:
            return mmh3.hash(item, seed=seed) % self.w

        # Fallback ổn định khi container chưa cài mmh3.
        # blake2b deterministic theo item+seed, đủ tốt cho CMS trong pipeline.
        payload = f"{seed}:{item}".encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % self.w

    # ── Thao tác chính ──────────────────────────────────────────

    def add(self, item: str, count: int = 1) -> None:
        """
        Thêm item vào sketch với count lần.

        Thuật toán: Với mỗi hàng i, tăng table[i][hash_i(item)] += count

        Args:
            item: Keyword/token cần đếm (VD: "iphone", "macbook_pro").
            count: Số lần xuất hiện (mặc định 1). Phải > 0.

        Raises:
            ValueError: Nếu count <= 0.
        """
        if count <= 0:
            raise ValueError(f"count phải > 0. Nhận count={count}")

        for i in range(self.d):
            col = self._hash(item, seed=i)
            self.table[i][col] += count
        self.total_count += count

    def query(self, item: str) -> int:
        """
        Ước lượng tần suất của item.

        Thuật toán: Trả về min(table[i][hash_i(item)]) cho i = 0..d-1

        Tính chất QUAN TRỌNG:
            - Kết quả LUÔN ≥ exact count (never undercount)
            - Kết quả ≤ exact count + ε × total_count (giới hạn overcount)

        Args:
            item: Keyword cần truy vấn.

        Returns:
            int: Tần suất ước lượng (≥ exact count thực tế).
        """
        return min(
            self.table[i][self._hash(item, seed=i)]
            for i in range(self.d)
        )

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """
        Gộp hai sketch thành một (element-wise addition).

        Ứng dụng: Combine kết quả từ nhiều Spark partition/worker.
        Mỗi worker có 1 CMS local → merge tất cả thành 1 CMS global.

        Tính chất: merged.query(x) = cms1.query(x) + cms2.query(x)
        (bảo toàn giới hạn sai số ε, δ không đổi)

        Args:
            other: CountMinSketch khác cùng kích thước (d, w).

        Returns:
            CountMinSketch mới chứa tổng của cả hai.

        Raises:
            ValueError: Nếu d hoặc w không khớp.
        """
        if self.d != other.d or self.w != other.w:
            raise ValueError(
                f"Không thể merge CMS kích thước khác nhau: "
                f"({self.d}x{self.w}) vs ({other.d}x{other.w})"
            )

        merged = CountMinSketch(d=self.d, w=self.w)
        for i in range(self.d):
            for j in range(self.w):
                merged.table[i][j] = self.table[i][j] + other.table[i][j]
        merged.total_count = self.total_count + other.total_count

        return merged

    def top_k(
        self, candidates: List[str], k: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Lấy top-K keyword có tần suất cao nhất từ tập ứng viên.

        Lưu ý: CMS không lưu danh sách item → cần cung cấp tập candidates
        (ví dụ: toàn bộ unique keywords từ vocabulary).

        Args:
            candidates: Danh sách keyword ứng viên để truy vấn.
            k: Số lượng top keywords cần lấy.

        Returns:
            List[(keyword, estimated_count)] sắp xếp giảm dần.
        """
        scored: List[Tuple[str, int]] = [
            (item, self.query(item)) for item in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── Bộ nhớ & Serialization ──────────────────────────────────

    def __sizeof__(self) -> int:
        """
        Ước tính bộ nhớ sử dụng (bytes).

        CMS sử dụng O(d × w × sizeof(int)) bộ nhớ cố định,
        bất kể bao nhiêu item đã add.

        Returns:
            int: Số bytes ước tính.
        """
        return self.memory_bytes()

    def memory_bytes(self) -> int:
        """
        Tính chính xác bộ nhớ sử dụng (bytes).

        Mỗi cell trong table là Python int (28 bytes cho int nhỏ).
        Tổng: d × w × 28 + overhead cấu trúc list.

        Returns:
            int: Số bytes.
        """
        # sys.getsizeof cho list lồng list
        total = sys.getsizeof(self.table)
        for row in self.table:
            total += sys.getsizeof(row)
            for cell in row:
                total += sys.getsizeof(cell)
        return total

    def serialize(self) -> bytes:
        """
        Serialize CMS thành bytes (dùng pickle).

        Ứng dụng: Lưu trạng thái CMS ra HDFS hoặc local file
        để DAG Airflow load lại ở lần chạy tiếp (mỗi 15 phút).

        Returns:
            bytes: Dữ liệu pickle.
        """
        return pickle.dumps({
            "d": self.d,
            "w": self.w,
            "table": self.table,
            "total_count": self.total_count,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "CountMinSketch":
        """
        Khôi phục CMS từ bytes đã serialize.

        Args:
            data: Bytes từ serialize().

        Returns:
            CountMinSketch đã khôi phục trạng thái.
        """
        obj = pickle.loads(data)
        cms = cls(d=obj["d"], w=obj["w"])
        cms.table = obj["table"]
        cms.total_count = obj["total_count"]
        return cms

    def __repr__(self) -> str:
        """Mô tả ngắn gọn CMS."""
        return (
            f"CountMinSketch(d={self.d}, w={self.w}, "
            f"total_count={self.total_count:,}, "
            f"epsilon={self.epsilon:.6f}, delta={self.delta:.6f})"
        )


# ============================================================================
# WRAPPER TÍCH HỢP VỚI PYSPARK
# ============================================================================

class CountMinSketchSpark:
    """
    Wrapper tích hợp Count-Min Sketch với PySpark.

    Hỗ trợ 2 pattern sử dụng:

    Pattern 1: mapPartitions + merge (phổ biến nhất)
        → Mỗi partition tạo CMS local, sau đó merge tất cả
        → Phù hợp cho batch processing trên Spark

    Pattern 2: Aggregate trên RDD
        → Dùng RDD.aggregate() với CMS làm accumulator
        → Tiện lợi hơn nhưng ít linh hoạt

    Ví dụ:
        >>> from algorithms.count_min_sketch import CountMinSketchSpark
        >>> # Giả sử có RDD chứa keywords
        >>> keywords_rdd = spark.sparkContext.parallelize(["a", "b", "a", "c"])
        >>> cms = CountMinSketchSpark.process_rdd(keywords_rdd, d=5, w=2048)
        >>> cms.query("a")  # → 2
    """

    @staticmethod
    def aggregate_partition(
        iterator: Iterator,
        d: int = 5,
        w: int = 2048,
    ) -> Generator["CountMinSketch", None, None]:
        """
        Tạo CMS cho một partition (dùng trong mapPartitions).

        Args:
            iterator: Iterator chứa keywords (strings) trong partition.
            d: Depth cho CMS.
            w: Width cho CMS.

        Yields:
            CountMinSketch chứa tần suất keywords trong partition này.
        """
        cms = CountMinSketch(d=d, w=w)
        for keyword in iterator:
            if isinstance(keyword, str) and keyword.strip():
                cms.add(keyword.strip())
        yield cms

    @staticmethod
    def merge_sketches(sketches: List[CountMinSketch]) -> CountMinSketch:
        """
        Merge danh sách CMS thành một CMS tổng hợp.

        Dùng sau mapPartitions để combine kết quả từ tất cả partitions.

        Args:
            sketches: Danh sách CMS cần merge.

        Returns:
            CountMinSketch tổng hợp.

        Raises:
            ValueError: Nếu danh sách rỗng hoặc kích thước không đồng nhất.
        """
        if not sketches:
            raise ValueError("Danh sách sketches rỗng.")

        result = sketches[0]
        for cms in sketches[1:]:
            result = result.merge(cms)
        return result

    @staticmethod
    def process_rdd(rdd, d: int = 5, w: int = 2048) -> CountMinSketch:
        """
        Tạo CMS từ RDD chứa keywords (end-to-end).

        Pipeline:
            1. mapPartitions → mỗi partition tạo 1 CMS local
            2. collect() → thu tất cả CMS về driver
            3. merge → combine thành 1 CMS global

        Lưu ý về bộ nhớ:
            - Mỗi partition CMS chỉ ~80KB (d=5, w=2048)
            - 200 partitions → collect ~16MB về driver → chấp nhận được

        Args:
            rdd: PySpark RDD chứa strings (keywords).
            d: Depth cho CMS.
            w: Width cho CMS.

        Returns:
            CountMinSketch tổng hợp chứa tần suất toàn bộ RDD.
        """
        partition_sketches: List[CountMinSketch] = (
            rdd.mapPartitions(
                lambda it: CountMinSketchSpark.aggregate_partition(it, d, w)
            )
            .collect()
        )

        if not partition_sketches:
            return CountMinSketch(d=d, w=w)

        return CountMinSketchSpark.merge_sketches(partition_sketches)

    @staticmethod
    def process_dataframe_column(
        df,
        column: str,
        d: int = 5,
        w: int = 2048,
    ) -> CountMinSketch:
        """
        Tạo CMS từ một cột DataFrame (tiện lợi hơn RDD).

        Ví dụ:
            >>> cms = CountMinSketchSpark.process_dataframe_column(
            ...     df, column="keyword", d=5, w=2048
            ... )

        Args:
            df: PySpark DataFrame.
            column: Tên cột chứa keywords.
            d: Depth cho CMS.
            w: Width cho CMS.

        Returns:
            CountMinSketch tổng hợp.
        """
        keywords_rdd = df.select(column).rdd.flatMap(lambda row: [row[0]])
        return CountMinSketchSpark.process_rdd(keywords_rdd, d=d, w=w)
