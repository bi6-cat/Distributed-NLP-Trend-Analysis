import os
import vncorenlp
from typing import Optional


class VnCoreNLPTokenizer:
    def __init__(self, jar_path: str = "vncorenlp/VnCoreNLP-1.1.1.jar"):
        self.jar_path = os.path.abspath(jar_path)
        self.annotator = vncorenlp.VnCoreNLP(
            self.jar_path,
            annotators="wseg",
            max_heap_size="-Xmx512m"
        )
        print(f"[VnCoreNLPTokenizer] Loaded JAR from {self.jar_path}")

    def tokenize(self, text: str) -> str:
        """
        Word segmentation cho tiếng Việt.
        "học máy rất tốt" → "học_máy rất tốt"
        """
        if not text or not text.strip():
            return ""
        try:
            sentences = self.annotator.tokenize(text)
            # Mỗi sentence là list các token, join lại
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
        self.annotator.close()