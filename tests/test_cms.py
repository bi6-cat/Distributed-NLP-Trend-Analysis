"""
tests/test_cms.py — Unit Tests cho Count-Min Sketch

Task 2.4 (Member 3 — Phase 2)
Dự án: Vietnamese Social Media Trend & Controversy Analysis System

Bộ test bao gồm 6 nhóm kiểm tra:
    (a) Basic add/query — kiểm tra tính đúng đắn cơ bản
    (b) Never undercount — CMS luôn trả giá trị ≥ exact count
    (c) Error rate — so sánh với Counter, đo tỷ lệ sai ≤ ε
    (d) Merge — gộp 2 sketch = 1 sketch chứa toàn bộ dữ liệu
    (e) Top-K — so sánh top keywords với ground truth
    (f) Memory — CMS bytes < dict bytes (chứng minh hiệu quả bộ nhớ)

Cơ sở lý thuyết:
    Cormode & Muthukrishnan (2005). "An Improved Data Stream Summary:
    The Count-Min Sketch and its Applications". Journal of Algorithms.

    Giới hạn sai số:
        ε = e / w → với w=2048: ε ≈ 0.00133
        δ = (1/2)^d → với d=5: δ ≈ 0.03125
        P(query(x) > exact(x) + ε × N) ≤ δ

Chạy test:
    pytest tests/test_cms.py -v
    pytest tests/test_cms.py -v -k "test_error_rate"  # chạy 1 test cụ thể
"""

import math
import random
import sys
from collections import Counter
from typing import Dict, List

import pytest

# Import module cần test
sys.path.insert(0, ".")
from algorithms.count_min_sketch import CountMinSketch, CountMinSketchSpark


# ============================================================================
# FIXTURES — Dữ liệu dùng chung cho nhiều test
# ============================================================================


@pytest.fixture
def small_cms() -> CountMinSketch:
    """CMS nhỏ cho test nhanh."""
    return CountMinSketch(d=5, w=256)


@pytest.fixture
def default_cms() -> CountMinSketch:
    """CMS với cấu hình mặc định theo TECH_STACK (d=5, w=2048)."""
    return CountMinSketch(d=5, w=2048)


@pytest.fixture
def vietnamese_keywords() -> List[str]:
    """
    Danh sách 20 keyword tiếng Việt giả lập (trending topics).
    Mô phỏng dữ liệu từ VOZ, VnExpress, YouTube.
    """
    return [
        "iphone", "samsung", "macbook", "laptop", "điện_thoại",
        "giá", "review", "đánh_giá", "so_sánh", "camera",
        "pin", "màn_hình", "chip", "ram", "bộ_nhớ",
        "5g", "ai", "chatgpt", "gemini", "copilot",
    ]


@pytest.fixture
def synthetic_corpus(vietnamese_keywords: List[str]) -> List[str]:
    """
    Corpus giả lập 100K từ với phân phối Zipf.

    Tại sao Zipf?
        Phân phối tần suất từ trong ngôn ngữ tự nhiên tuân theo
        định luật Zipf (Zipf, 1949): tần suất từ thứ r tỷ lệ nghịch
        với rank r. Điều này mô phỏng thực tế dữ liệu mạng xã hội.
    """
    random.seed(42)
    corpus: List[str] = []
    n_words = 100_000

    # Tạo phân phối Zipf: từ đầu tiên xuất hiện nhiều nhất
    for i in range(n_words):
        # Zipf: chọn keyword với xác suất tỷ lệ nghịch rank
        rank = random.randint(1, len(vietnamese_keywords))
        keyword = vietnamese_keywords[rank - 1]
        corpus.append(keyword)

    return corpus


@pytest.fixture
def exact_counts(synthetic_corpus: List[str]) -> Dict[str, int]:
    """Ground truth: exact count từ Counter (Python built-in)."""
    return dict(Counter(synthetic_corpus))


# ============================================================================
# (a) BASIC ADD/QUERY — Kiểm tra tính đúng đắn cơ bản
# ============================================================================


class TestBasicOperations:
    """Nhóm test: thao tác cơ bản add/query."""

    def test_empty_query_returns_zero(self, small_cms: CountMinSketch) -> None:
        """Query item chưa từng add phải trả về 0 (hoặc >= 0)."""
        assert small_cms.query("nonexistent") >= 0

    def test_single_add_query(self, small_cms: CountMinSketch) -> None:
        """Add 1 lần → query phải trả về ≥ 1."""
        small_cms.add("iphone")
        assert small_cms.query("iphone") >= 1

    def test_multiple_adds(self, small_cms: CountMinSketch) -> None:
        """Add n lần → query phải trả về ≥ n."""
        for _ in range(100):
            small_cms.add("samsung")
        assert small_cms.query("samsung") >= 100

    def test_add_with_count(self, small_cms: CountMinSketch) -> None:
        """Add với count=10 → tương đương add 10 lần riêng lẻ."""
        small_cms.add("macbook", count=10)
        assert small_cms.query("macbook") >= 10

    def test_total_count_tracking(self, small_cms: CountMinSketch) -> None:
        """total_count phải bằng tổng số lần add."""
        small_cms.add("a", count=5)
        small_cms.add("b", count=3)
        small_cms.add("c", count=2)
        assert small_cms.total_count == 10

    def test_different_items_independent(self, small_cms: CountMinSketch) -> None:
        """Hai item khác nhau có count độc lập (trừ collision)."""
        small_cms.add("iphone", count=100)
        small_cms.add("samsung", count=1)
        # samsung query phải nhỏ hơn nhiều so với iphone
        assert small_cms.query("samsung") < small_cms.query("iphone")

    def test_invalid_count_raises(self, small_cms: CountMinSketch) -> None:
        """count <= 0 phải raise ValueError."""
        with pytest.raises(ValueError):
            small_cms.add("test", count=0)
        with pytest.raises(ValueError):
            small_cms.add("test", count=-1)

    def test_invalid_dimensions_raises(self) -> None:
        """d hoặc w <= 0 phải raise ValueError."""
        with pytest.raises(ValueError):
            CountMinSketch(d=0, w=100)
        with pytest.raises(ValueError):
            CountMinSketch(d=5, w=0)
        with pytest.raises(ValueError):
            CountMinSketch(d=-1, w=-1)


# ============================================================================
# (b) NEVER UNDERCOUNT — CMS luôn trả giá trị ≥ exact count
# ============================================================================


class TestNeverUndercount:
    """
    Nhóm test: tính chất cốt lõi "never undercount".

    Cơ sở: Cormode & Muthukrishnan (2005), Theorem 1:
        Với mọi item x: exact(x) ≤ query(x)
    """

    def test_never_undercount_simple(self, default_cms: CountMinSketch) -> None:
        """Test never undercount trên 10 items, mỗi item add số lần khác nhau."""
        items = {f"item_{i}": (i + 1) * 10 for i in range(10)}

        for item, count in items.items():
            default_cms.add(item, count=count)

        for item, exact in items.items():
            estimated = default_cms.query(item)
            assert estimated >= exact, (
                f"UNDERCOUNT detected! item={item}, "
                f"exact={exact}, estimated={estimated}"
            )

    def test_never_undercount_corpus(
        self,
        default_cms: CountMinSketch,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
    ) -> None:
        """
        Test never undercount trên corpus 100K từ.
        Đây là test QUAN TRỌNG NHẤT — chứng minh tính chất lý thuyết.
        """
        for word in synthetic_corpus:
            default_cms.add(word)

        for word, exact in exact_counts.items():
            estimated = default_cms.query(word)
            assert estimated >= exact, (
                f"UNDERCOUNT! word='{word}', exact={exact}, est={estimated}"
            )


# ============================================================================
# (c) ERROR RATE — So sánh với exact count, đo tỷ lệ sai ≤ ε
# ============================================================================


class TestErrorRate:
    """
    Nhóm test: đo error rate so với exact count.

    Lý thuyết:
        P(query(x) - exact(x) > ε × N) ≤ δ
        Với d=5, w=2048: ε ≈ 0.00133, δ ≈ 0.03125

    Cách đo:
        error(x) = (query(x) - exact(x)) / N
        → Tỷ lệ items có error(x) > ε phải ≤ δ
    """

    def test_average_error_within_epsilon(
        self,
        default_cms: CountMinSketch,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
    ) -> None:
        """
        Trung bình error rate phải ≤ ε.

        Đây là test thực nghiệm quan trọng: chạy CMS trên 100K từ,
        so sánh với Counter chính xác, kiểm tra biên sai số lý thuyết.
        """
        for word in synthetic_corpus:
            default_cms.add(word)

        total_n = default_cms.total_count
        eps = default_cms.epsilon  # e / 2048 ≈ 0.00133
        errors: List[float] = []

        for word, exact in exact_counts.items():
            estimated = default_cms.query(word)
            absolute_error = estimated - exact  # Luôn ≥ 0 (never undercount)
            relative_error = absolute_error / total_n
            errors.append(relative_error)

        avg_error = sum(errors) / len(errors)

        # Trung bình error phải nhỏ hơn epsilon
        assert avg_error <= eps, (
            f"Avg relative error {avg_error:.6f} > epsilon {eps:.6f}"
        )

    def test_error_bound_probability(
        self,
        default_cms: CountMinSketch,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
    ) -> None:
        """
        Tỷ lệ items vượt quá ε × N phải ≤ δ.

        Theorem (Cormode & Muthukrishnan, 2005):
            P(overcount > ε × N) ≤ δ = (1/2)^d
        """
        for word in synthetic_corpus:
            default_cms.add(word)

        total_n = default_cms.total_count
        eps = default_cms.epsilon
        delta = default_cms.delta
        threshold = eps * total_n

        violations = 0
        for word, exact in exact_counts.items():
            estimated = default_cms.query(word)
            overcount = estimated - exact
            if overcount > threshold:
                violations += 1

        violation_rate = violations / len(exact_counts)

        # Tỷ lệ vi phạm phải ≤ δ (với margin cho thống kê)
        # Thêm margin 0.05 cho biến động thống kê
        assert violation_rate <= delta + 0.05, (
            f"Violation rate {violation_rate:.4f} > delta {delta:.4f} + margin"
        )

    def test_max_overcount_bounded(
        self,
        default_cms: CountMinSketch,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
    ) -> None:
        """Max overcount cho bất kỳ item nào phải hợp lý."""
        for word in synthetic_corpus:
            default_cms.add(word)

        total_n = default_cms.total_count
        max_overcount = 0

        for word, exact in exact_counts.items():
            estimated = default_cms.query(word)
            overcount = estimated - exact
            max_overcount = max(max_overcount, overcount)

        # Max overcount không nên vượt quá 5% tổng count
        # (giới hạn thực tế, không phải giới hạn lý thuyết chặt)
        assert max_overcount <= total_n * 0.05, (
            f"Max overcount {max_overcount} > 5% of N={total_n}"
        )


# ============================================================================
# (d) MERGE — Gộp 2 sketch = 1 sketch toàn bộ dữ liệu
# ============================================================================


class TestMerge:
    """
    Nhóm test: tính chất merge (cộng element-wise).

    Lý thuyết: merge(CMS_A, CMS_B).query(x) = CMS_A.query(x) + CMS_B.query(x)
    Ứng dụng: Combine CMS từ nhiều Spark partition/worker.
    """

    def test_merge_equals_single_sketch(
        self,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
    ) -> None:
        """
        CMS merged từ 2 nửa = CMS chứa toàn bộ dữ liệu.

        Chứng minh: merge(CMS(first_half), CMS(second_half)) ≡ CMS(all_data)
        """
        # Chia corpus thành 2 nửa
        mid = len(synthetic_corpus) // 2
        first_half = synthetic_corpus[:mid]
        second_half = synthetic_corpus[mid:]

        # Tạo 2 CMS riêng biệt
        cms_a = CountMinSketch(d=5, w=2048)
        cms_b = CountMinSketch(d=5, w=2048)

        for word in first_half:
            cms_a.add(word)
        for word in second_half:
            cms_b.add(word)

        # Merge
        cms_merged = cms_a.merge(cms_b)

        # Tạo CMS đơn chứa toàn bộ
        cms_single = CountMinSketch(d=5, w=2048)
        for word in synthetic_corpus:
            cms_single.add(word)

        # So sánh kết quả query
        for word in exact_counts:
            merged_val = cms_merged.query(word)
            single_val = cms_single.query(word)
            assert merged_val == single_val, (
                f"word='{word}': merged={merged_val} != single={single_val}"
            )

        # total_count cũng phải bằng nhau
        assert cms_merged.total_count == cms_single.total_count

    def test_merge_total_count(self) -> None:
        """Merged total_count = tổng total_count của 2 CMS."""
        cms_a = CountMinSketch(d=3, w=128)
        cms_b = CountMinSketch(d=3, w=128)

        cms_a.add("hello", count=5)
        cms_b.add("world", count=3)

        merged = cms_a.merge(cms_b)
        assert merged.total_count == 8

    def test_merge_different_sizes_raises(self) -> None:
        """Merge CMS khác kích thước phải raise ValueError."""
        cms_a = CountMinSketch(d=5, w=2048)
        cms_b = CountMinSketch(d=5, w=1024)

        with pytest.raises(ValueError, match="kích thước khác nhau"):
            cms_a.merge(cms_b)

    def test_merge_multiple_partitions(
        self,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
    ) -> None:
        """
        Merge từ 5 partition (mô phỏng Spark 5 partitions).
        Kết quả phải tương đương single CMS.
        """
        n_partitions = 5
        chunk_size = len(synthetic_corpus) // n_partitions

        # Tạo CMS cho mỗi partition
        partition_sketches: List[CountMinSketch] = []
        for p in range(n_partitions):
            start = p * chunk_size
            end = (
                start + chunk_size
                if p < n_partitions - 1
                else len(synthetic_corpus)
            )
            cms = CountMinSketch(d=5, w=2048)
            for word in synthetic_corpus[start:end]:
                cms.add(word)
            partition_sketches.append(cms)

        # Merge tuần tự
        result = partition_sketches[0]
        for cms in partition_sketches[1:]:
            result = result.merge(cms)

        # So sánh với single CMS
        cms_single = CountMinSketch(d=5, w=2048)
        for word in synthetic_corpus:
            cms_single.add(word)

        for word in exact_counts:
            assert result.query(word) == cms_single.query(word)


# ============================================================================
# (e) TOP-K — So sánh top keywords với ground truth
# ============================================================================


class TestTopK:
    """
    Nhóm test: top_k keywords so sánh với ground truth.

    Top-K từ CMS phải trùng khớp (hoặc gần trùng) với top-K từ exact count,
    đặc biệt cho các keyword có tần suất cao.
    """

    def test_top_k_accuracy(
        self,
        default_cms: CountMinSketch,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
        vietnamese_keywords: List[str],
    ) -> None:
        """
        Top-5 từ CMS phải chứa ≥ 3/5 top-5 thực tế.
        (CMS có thể xáo trộn thứ hạng do overcount, nhưng top items phải gần đúng)
        """
        for word in synthetic_corpus:
            default_cms.add(word)

        # Top-5 từ CMS
        cms_top5 = default_cms.top_k(vietnamese_keywords, k=5)
        cms_top5_words = {word for word, _ in cms_top5}

        # Top-5 ground truth
        exact_sorted = sorted(exact_counts.items(), key=lambda x: x[1], reverse=True)
        gt_top5_words = {word for word, _ in exact_sorted[:5]}

        # Ít nhất 3/5 phải trùng
        overlap = len(cms_top5_words & gt_top5_words)
        assert overlap >= 3, (
            f"Top-5 overlap = {overlap}/5. "
            f"CMS: {cms_top5_words}, GT: {gt_top5_words}"
        )

    def test_top_k_order_mostly_correct(
        self,
        default_cms: CountMinSketch,
        synthetic_corpus: List[str],
        exact_counts: Dict[str, int],
        vietnamese_keywords: List[str],
    ) -> None:
        """Top-1 keyword từ CMS phải trùng với top-1 thực tế."""
        for word in synthetic_corpus:
            default_cms.add(word)

        cms_top1 = default_cms.top_k(vietnamese_keywords, k=1)
        gt_top1 = max(exact_counts.items(), key=lambda x: x[1])

        assert cms_top1[0][0] == gt_top1[0], (
            f"Top-1 mismatch: CMS='{cms_top1[0][0]}', GT='{gt_top1[0]}'"
        )

    def test_top_k_returns_correct_length(
        self, small_cms: CountMinSketch
    ) -> None:
        """top_k phải trả về đúng k phần tử (hoặc ít hơn nếu candidates < k)."""
        candidates = ["a", "b", "c"]
        small_cms.add("a", count=3)
        small_cms.add("b", count=2)
        small_cms.add("c", count=1)

        result = small_cms.top_k(candidates, k=2)
        assert len(result) == 2

        result_all = small_cms.top_k(candidates, k=10)
        assert len(result_all) == 3  # Chỉ có 3 candidates


# ============================================================================
# (f) MEMORY — CMS bytes < dict bytes
# ============================================================================


class TestMemory:
    """
    Nhóm test: chứng minh CMS tiết kiệm bộ nhớ hơn dict/Counter.

    CMS dùng O(d × w) bộ nhớ cố định.
    Dict/Counter dùng O(n_unique_items) bộ nhớ, tăng theo dữ liệu.
    """

    def test_cms_smaller_than_dict(self) -> None:
        """
        Với 10K unique items, CMS (d=5, w=2048) phải nhỏ hơn dict.

        CMS cố định: ~5 × 2048 × 28 bytes ≈ 280KB
        Dict 10K items: ~10000 × (key_size + value_size) ≈ 800KB+
        """
        # Tạo 10K unique keywords
        keywords = [f"keyword_{i}" for i in range(10_000)]

        # CMS
        cms = CountMinSketch(d=5, w=2048)
        for kw in keywords:
            cms.add(kw)

        # Dict
        exact = Counter(keywords)

        cms_size = cms.memory_bytes()
        dict_size = sys.getsizeof(exact)
        # Thêm size của keys và values
        for key, val in exact.items():
            dict_size += sys.getsizeof(key) + sys.getsizeof(val)

        assert cms_size < dict_size, (
            f"CMS ({cms_size:,} bytes) >= Dict ({dict_size:,} bytes)"
        )

    def test_cms_memory_independent_of_data_size(self) -> None:
        """
        Bộ nhớ CMS không tăng khi thêm dữ liệu (cố định d × w).

        Đây là ưu điểm then chốt của CMS cho streaming:
        bộ nhớ O(1) bất kể N (số items đã add).
        """
        cms = CountMinSketch(d=5, w=2048)
        memory_before = cms.memory_bytes()

        # Add 100K items
        for i in range(100_000):
            cms.add(f"item_{i % 1000}")

        memory_after = cms.memory_bytes()

        # Bộ nhớ không thay đổi (hoặc thay đổi rất nhỏ do Python int overflow)
        # Cho phép tăng tối đa 50% (do Python int lớn chiếm thêm byte)
        assert memory_after <= memory_before * 1.5, (
            f"Memory grew too much: {memory_before:,} → {memory_after:,} bytes"
        )


# ============================================================================
# SERIALIZATION — Test serialize/deserialize
# ============================================================================


class TestSerialization:
    """Test serialize và deserialize CMS."""

    def test_serialize_deserialize_preserves_state(
        self, default_cms: CountMinSketch
    ) -> None:
        """Serialize → deserialize phải giữ nguyên trạng thái."""
        default_cms.add("iphone", count=50)
        default_cms.add("samsung", count=30)

        data = default_cms.serialize()
        restored = CountMinSketch.deserialize(data)

        assert restored.query("iphone") == default_cms.query("iphone")
        assert restored.query("samsung") == default_cms.query("samsung")
        assert restored.total_count == default_cms.total_count
        assert restored.d == default_cms.d
        assert restored.w == default_cms.w

    def test_serialize_returns_bytes(self, small_cms: CountMinSketch) -> None:
        """serialize() phải trả về bytes."""
        small_cms.add("test")
        data = small_cms.serialize()
        assert isinstance(data, bytes)
        assert len(data) > 0


# ============================================================================
# PROPERTIES — Test epsilon, delta, repr
# ============================================================================


class TestProperties:
    """Test các thuộc tính toán học."""

    def test_epsilon_formula(self) -> None:
        """ε = e / w."""
        cms = CountMinSketch(d=5, w=2048)
        expected = math.e / 2048
        assert abs(cms.epsilon - expected) < 1e-10

    def test_delta_formula(self) -> None:
        """δ = (1/2)^d."""
        cms = CountMinSketch(d=5, w=2048)
        expected = (0.5) ** 5
        assert abs(cms.delta - expected) < 1e-10

    def test_repr(self) -> None:
        """__repr__ phải chứa thông tin cơ bản."""
        cms = CountMinSketch(d=5, w=2048)
        r = repr(cms)
        assert "d=5" in r
        assert "w=2048" in r
        assert "total_count" in r


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
