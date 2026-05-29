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
        self.stopwords.update(self._build_custom_stopwords())
        self.topic_stopwords = set(self.stopwords)
        self.topic_stopwords.update(self._build_topic_only_stopwords())

        # Regex patterns (compile 1 lần để tối ưu performance)
        self._html_pattern = re.compile(r"<[^>]+>")
        self._url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self._email_pattern = re.compile(r"\S+@\S+\.\S+")
        self._forum_quote_pattern = re.compile(r"\b\w+\s+said.*?click to expand\s*", flags=re.IGNORECASE | re.DOTALL)
        self._source_noise_pattern = re.compile(
            r"\b("
            r"hóng|hong|ib|inb|inbox|pm|rep|cmt|comment|quote|thread|topic|sub|up|upp|"
            r"lol|lmao|wtf|vl|vkl|vcl|dm|đm|cmnr|kkk|haha|hehe|hihi|ahihi|uhi|uhm"
            r")\b",
            flags=re.IGNORECASE,
        )
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
    
    def remove_forum_quotes(self, text: str) -> str:
        """Xóa các đoạn trích dẫn (quote) rác của diễn đàn XenForo (như VOZ)."""
        return self._forum_quote_pattern.sub(" ", text)

    def remove_source_specific_noise(self, text: str) -> str:
        """Xóa các từ đệm/noise đặc thù forum-social không mang nội dung chủ đề."""
        return self._source_noise_pattern.sub(" ", text)
    
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
        text = self.remove_forum_quotes(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_mentions(text)
        text = self.remove_emojis(text)
        text = self.normalize_unicode(text)
        text = self.slang_normalizer.normalize(text)
        text = self.remove_source_specific_noise(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)

        return text

    def clean_html(self, text: str) -> str:
        """
        Chỉ xóa HTML tags và các đoạn quote thừa của forum, giữ nguyên casing và ký tự đặc biệt.
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = self.remove_forum_quotes(text)
        text = self.remove_html(text)
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

    def remove_topic_stopwords(self, text: str) -> str:
        """Xóa stopwords mạnh tay hơn cho topic modeling."""
        if not text:
            return ""
        words = text.split()
        filtered = [w for w in words if w not in self.topic_stopwords]
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

    def preprocess_for_topic(self, text: str) -> str:
        """
        Biến thể mạnh tay hơn cho topic modeling.

        Tách riêng khỏi segmented_text để sentiment không bị mất tín hiệu ngữ nghĩa.
        """
        text = self.clean(text)

        if not text:
            return ""

        text = self.tokenize(text)
        text = self.remove_topic_stopwords(text)
        return self._filter_topic_tokens(text)

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
                # Đồng bộ với pipeline clean() vốn đã lowercase text trước khi tokenize.
                stopwords = set(line.strip().lower() for line in f if line.strip())
            print(f"[TextPreprocessor] Loaded {len(stopwords)} stopwords from {path}")
            return stopwords
        except FileNotFoundError:
            print(f"[TextPreprocessor] WARNING: Stopwords file not found: {path}")
            return set()

    def _build_custom_stopwords(self) -> set:
        """
        Stopwords bổ sung cho dữ liệu forum/công nghệ.

        Bao gồm:
        - discourse/forum fillers thường phá topic quality
        - đơn vị công nghệ/spec tokens ít giá trị chủ đề
        """
        return {
            "thì", "là", "và", "các", "với", "của", "cho", "trong", "được",
            "có", "mà", "hơn", "mới", "đâu", "đến", "từ", "khi", "đi", "nên",
            "cũng", "như", "lại", "thấy", "người", "một", "phải", "trên", "ra",
            "luôn", "chưa", "hay", "nào", "gì", "tầm", "lúc", "vẫn",
            "này", "kia", "đó", "ấy", "mình", "nó", "bác", "con", "cái",
            "tôi", "bạn", "thím", "ông", "anh", "chị", "em",
            "rồi", "thôi", "sao", "thế", "vậy", "ừ", "uh", "ờ", "à",
            "ko", "k", "kh", "đc", "dc", "ae", "mn", "mng", "bro", "ad",
            "hóng", "hong", "ib", "inb", "inbox", "pm", "rep", "cmt", "sub",
            "up", "upp", "vl", "vkl", "vcl", "lol", "lmao", "wtf", "kkk",
            "haha", "hehe", "hihi", "ahihi", "khz", "mhz", "ghz", "ms", "mm", "cm", "m", "inch",
            "gb", "mb", "tb",
        }

    def _build_topic_only_stopwords(self) -> set:
        """Stopwords chỉ dùng cho topic modeling, không áp lên sentiment path."""
        return {
            "không", "nhưng", "đã", "để", "vào", "nếu", "đang", "đây", "sau",
            "còn", "bản", "sẽ", "quá", "rất", "thật", "to", "ai", "theo",
            "ngày", "có_thể", "nhé", "nữa", "fen", "tq", "tr", "views", "kb",
            "wick", "hê", "nhỉ", "bên", "chứ", "mấy", "chỉ", "vì", "họ",
            "những", "đẹp", "lên", "thêm", "khác", "giờ", "hết", "mỗi",
            "đúng", "for", "edited", "fan", "nan",
            # generic verbs / adjectives / discourse fillers that still dominate topics
            "mua", "dùng", "xài", "làm", "bị", "nhiều", "ngon", "chắc",
            "bảo", "nhìn", "cao", "tốt", "gần", "biết", "nay", "đấy",
            "cả", "lấy", "cứ", "lắm", "việc", "bằng", "nghe", "mở", "về", "cần",
            # residual noise from current corpus
            "attachments", "attachment", "views", "jpg", "jpeg", "png", "webp",
            "screenshot", "sent", "using", "from", "via", "vozfapp", "vozvnapp",
            "thenextvoz", "com", "www", "http", "https", "the", "with", "you",
            "is", "to", "on", "in", "ktc", "đt",
        }

    def _filter_topic_tokens(self, text: str) -> str:
        """Bỏ token rác thường phá chất lượng topic sau segmentation."""
        if not text:
            return ""

        filtered = []
        for token in text.split():
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            if re.fullmatch(r"[0-9]+[a-z_]*", token):
                continue
            if re.fullmatch(r"[a-z]{1,2}", token):
                continue
            filtered.append(token)
        return " ".join(filtered)
    
