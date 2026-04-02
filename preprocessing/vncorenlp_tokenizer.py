import os
from typing import Optional


class VnCoreNLPTokenizer:
    """
    Wrapper lazy-init cho VnCoreNLP.

    Không load JAR trong __init__ — chỉ load lần đầu khi tokenize() được gọi.
    Điều này an toàn khi dùng trong Spark mapPartitions:
      - Worker import class mà không cần JAR có mặt ngay lập tức
      - JAR chỉ cần tồn tại tại thời điểm gọi tokenize()
    """

    def __init__(self, jar_path: str = "vncorenlp/VnCoreNLP-1.1.1.jar"):
        # Dùng absolute path để tránh lỗi relative path trên cluster workers
        self.jar_path   = os.path.abspath(jar_path)
        self._annotator = None   # lazy — chưa load

    def _get_annotator(self):
        """Load JAR lần đầu khi cần (lazy init)."""
        if self._annotator is None:
            import vncorenlp
            self._annotator = vncorenlp.VnCoreNLP(
                self.jar_path,
                annotators="wseg",
                max_heap_size="-Xmx512m",
            )
            print(f"[VnCoreNLPTokenizer] Loaded JAR from {self.jar_path}")
        return self._annotator

    def tokenize(self, text: str) -> str:
        """
        Word segmentation cho tiếng Việt.
        "học máy rất tốt" → "học_máy rất tốt"
        """
        if not text or not text.strip():
            return ""
        try:
            annotator = self._get_annotator()
            sentences = annotator.tokenize(text)
            tokens = [token for sent in sentences for token in sent]
            return " ".join(tokens)
        except Exception as e:
            print(f"[VnCoreNLPTokenizer] ERROR: {e}, fallback underthesea")
            return self._fallback_tokenize(text)

    def _fallback_tokenize(self, text: str) -> str:
        try:
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
        except ImportError:
            return text

    def close(self):
        """Giải phóng JVM process — gọi khi worker shutdown."""
        if self._annotator is not None:
            try:
                self._annotator.close()
            except Exception:
                pass
            self._annotator = None