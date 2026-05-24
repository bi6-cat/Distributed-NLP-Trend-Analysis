"""
algorithms — Cấu trúc dữ liệu xác suất & thuật toán streaming.

Module này chứa các thuật toán CS246 được triển khai cho dự án
Vietnamese Social Media Trend & Controversy Analysis:

    - CountMinSketch: Đếm tần suất keyword streaming (ε=e/w, δ=(1/2)^d)
    - CountMinSketchSpark: Wrapper tích hợp CMS với PySpark
    - MinHashDeduplicator: Khử duplicate bằng MinHash LSH (datasketch)
"""

# NOTE: CountMinSketch phụ thuộc vào `mmh3` (MurmurHash3).
# Dùng lazy import để tránh crash Spark executor khi package này bị load
# bởi các worker không dùng CMS (ví dụ: cleaning_job.py chỉ dùng MinHash).
# Import trực tiếp khi cần: from algorithms.count_min_sketch import CountMinSketch

__all__ = ["CountMinSketch", "CountMinSketchSpark", "MinHashDeduplicator"]
