"""
Wrapper cho VnCoreNLP word segmentation.
VnCoreNLP yêu cầu Java 11+ và chạy như một HTTP server.

Cách khởi động VnCoreNLP server:
    cd vncorenlp/
    java -Xmx2g -jar VnCoreNLP-1.2.jar -port 9000 -annotators wseg

Member 4 — NLP Engineer
"""

import requests
from typing import Optional


class VnCoreNLPTokenizer:
    """
    Client gửi request đến VnCoreNLP server để tokenize.

    VnCoreNLP chạy trên mỗi worker node (Member 2 cài qua Ansible).
    """

    def __init__(self, host: str = "http://localhost", port: int = 9000):
        self.url = f"{host}:{port}"
        self._check_server()

    def tokenize(self, text: str) -> str:
        """
        Word segmentation cho 1 câu/đoạn.

        Args:
            text: Văn bản cần tokenize.

        Returns:
            Chuỗi đã tokenize, từ ghép nối bằng "_".
            Ví dụ: "học máy rất tốt" → "học_máy rất tốt"
        """
        if not text or not text.strip():
            return ""

        try:
            response = requests.post(
                f"{self.url}/handle",
                data=text.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=10,
            )
            response.raise_for_status()

            # VnCoreNLP trả về dạng JSON với annotated sentences
            result = response.json()
            tokens = []
            for sentence in result.get("sentences", []):
                for token in sentence:
                    form = token.get("form", "")
                    tokens.append(form)

            return " ".join(tokens)

        except requests.exceptions.ConnectionError:
            print("[VnCoreNLPTokenizer] ERROR: Cannot connect to VnCoreNLP server.")
            return self._fallback_tokenize(text)
        except requests.exceptions.Timeout:
            print("[VnCoreNLPTokenizer] ERROR: VnCoreNLP server timeout.")
            return self._fallback_tokenize(text)
        except Exception as e:
            print(f"[VnCoreNLPTokenizer] ERROR: {e}")
            return self._fallback_tokenize(text)

    def _fallback_tokenize(self, text: str) -> str:
        """Fallback: dùng underthesea nếu VnCoreNLP không khả dụng."""
        try:
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
        except ImportError:
            # Fallback cuối: trả về text gốc (chỉ split by space)
            return text

    def _check_server(self):
        """Kiểm tra VnCoreNLP server có đang chạy không."""
        try:
            response = requests.get(self.url, timeout=5)
            print(f"[VnCoreNLPTokenizer] Server OK at {self.url}")
        except requests.exceptions.ConnectionError:
            print(
                f"[VnCoreNLPTokenizer] WARNING: Server not running at {self.url}. "
                f"Will fallback to underthesea."
            )