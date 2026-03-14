"""
Text Preprocessing Pipeline cho Vietnamese Tech Trend Analysis.
Pipeline: Raw Text → Clean Text → Tokenized Text → Ready for NLP

Member 4 — NLP Engineer
"""

import re
import unicodedata
from typing import List, Optional

from preprocessing.slang_normalizer import SlangNormalizer
from preprocessing.vncorenlp_tokenizer import VnCoreNLPTokenizer


class TextPreprocessor:
    """
    Pipeline xử lý văn bản tiếng Việt.

    Các bước:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove emails
        5. Remove emojis
        6. Normalize unicode (NFC)
        7. Normalize whitespace
        8. Normalize teencode/slang → tiếng Việt chuẩn
        9. Remove special characters (giữ dấu tiếng Việt)
        10. Word segmentation (VnCoreNLP / underthesea)
        11. Remove stopwords
    """

    def __init__(
        self,
        slang_dict_path: str = "data/slang_dict.json",
        stopwords_path: str = "data/stopwords_vi.txt",
        use_vncorenlp: bool = True,
        vncorenlp_jar: str = "vncorenlp/VnCoreNLP-1.1.1.jar"
    ):
        # Slang normalizer
        self.slang_normalizer = SlangNormalizer(slang_dict_path)

        # Tokenizer
        self.use_vncorenlp = use_vncorenlp
        if use_vncorenlp:
            self.tokenizer = VnCoreNLPTokenizer(jar_path=vncorenlp_jar)
        else:
            # Fallback: underthesea (thuần Python, dễ cài hơn)
            from underthesea import word_tokenize
            self.word_tokenize = word_tokenize

        # Stopwords
        self.stopwords = self._load_stopwords(stopwords_path)

        # Regex patterns (compile 1 lần để tối ưu performance)
        self._html_pattern = re.compile(r"<[^>]+>")
        self._url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self._email_pattern = re.compile(r"\S+@\S+\.\S+")
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        # Giữ chữ cái tiếng Việt, số, khoảng trắng, dấu gạch dưới
        self._special_char_pattern = re.compile(
            r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
            r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
            flags=re.IGNORECASE,
        )
        self._whitespace_pattern = re.compile(r"\s+")


    def remove_html(self, text: str) -> str:
        """Xóa tất cả HTML tags."""
        return self._html_pattern.sub(" ", text)

    def remove_urls(self, text: str) -> str:
        """Xóa URLs."""
        return self._url_pattern.sub(" ", text)

    def remove_emails(self, text: str) -> str:
        """Xóa email addresses."""
        return self._email_pattern.sub(" ", text)
    
    def remove_mentions(self, text: str) -> str:
        """Xóa mentions."""
        return re.sub(r"@\w+", " ", text)
    
    def remove_emojis(self, text: str) -> str:
        """Xóa emojis."""
        return self._emoji_pattern.sub(" ", text)

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa Unicode về dạng NFC (tổ hợp)."""
        return unicodedata.normalize("NFC", text)

    def remove_special_characters(self, text: str) -> str:
        """Xóa ký tự đặc biệt, giữ chữ tiếng Việt và số."""
        return self._special_char_pattern.sub(" ", text)

    def normalize_whitespace(self, text: str) -> str:
        """Gộp nhiều khoảng trắng thành 1, strip đầu cuối."""
        return self._whitespace_pattern.sub(" ", text).strip()

    def lowercase(self, text: str) -> str:
        """Chuyển về chữ thường."""
        return text.lower()

    def clean(self, text: str) -> str:
        """
        Chạy toàn bộ bước cleaning (chưa tokenize).

        Returns:
            Chuỗi đã clean, sẵn sàng cho tokenization.
        """
        if not text or not isinstance(text, str):
            return ""

        text = self.lowercase(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_mentions(text)
        text = self.remove_emojis(text)
        text = self.normalize_unicode(text)
        text = self.slang_normalizer.normalize(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)

        return text


    def tokenize(self, text: str) -> str:
        """
        Word segmentation cho tiếng Việt.
        Ví dụ: "học máy" → "học_máy"

        Returns:
            Chuỗi đã tokenize (các từ ghép nối bằng _).
        """
        if not text:
            return ""

        if self.use_vncorenlp:
            return self.tokenizer.tokenize(text)
        else:
            return self.word_tokenize(text, format="text")


    def remove_stopwords(self, text: str) -> str:
        """Xóa stopwords tiếng Việt."""
        if not text:
            return ""
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords]
        return " ".join(filtered)


    def preprocess(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Chạy full pipeline: clean → tokenize → remove stopwords.

        Args:
            text: Văn bản gốc.
            remove_stopwords: Có xóa stopwords không (mặc định True).

        Returns:
            Văn bản đã xử lý hoàn chỉnh.
        """
        text = self.clean(text)

        if not text:
            return ""

        text = self.tokenize(text)

        if remove_stopwords:
            text = self.remove_stopwords(text)

        return text

    def preprocess_batch(self, texts: List[str], remove_stopwords: bool = True) -> List[str]:
        """
        Xử lý batch văn bản (dùng cho Spark mapPartitions).

        Args:
            texts: Danh sách văn bản gốc.
            remove_stopwords: Có xóa stopwords không.

        Returns:
            Danh sách văn bản đã xử lý.
        """
        return [self.preprocess(t, remove_stopwords) for t in texts]


    def _load_stopwords(self, path: str) -> set:
        """Load stopwords từ file text (mỗi dòng 1 từ)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                stopwords = set(line.strip() for line in f if line.strip())
            print(f"[TextPreprocessor] Loaded {len(stopwords)} stopwords from {path}")
            return stopwords
        except FileNotFoundError:
            print(f"[TextPreprocessor] WARNING: Stopwords file not found: {path}")
            return set()
    