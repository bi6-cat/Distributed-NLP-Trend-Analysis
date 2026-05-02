# Handoff M4 → M2: Hướng dẫn dùng Preprocessing Pipeline & Adapters

> **Từ:** Member 4 — NLP Engineer
> **Cho:** Member 2 — DevOps / Infra
> **Mục tiêu:** M2 tích hợp code M4 vào Spark cleaning job để tạo `tech_radar.stg_posts_core`

---

## Tổng quan vai trò

```
M4 cung cấp:                          M2 dùng để:
─────────────────────────────          ──────────────────────────────────────
preprocessing/text_cleaner.py    →     clean + segment text trong Spark job
schemas/voz_adapter.py           →     parse raw VOZ CSV → schema chuẩn
schemas/vatvo_adapter.py         →     parse raw VatVo CSV → schema chuẩn
schemas/vnexpress_adapter.py     →     parse raw VnExpress CSV → schema chuẩn
schemas/models.py                →     Pydantic validation (UniversalSocialPost)
```

M2 **không cần tự viết** text cleaning hay schema parsing — dùng nguyên code M4.

---

## 1. Cài đặt dependencies

```bash
# Trên tất cả Spark worker nodes
pip install underthesea pydantic datasketch clickhouse-connect

# Nếu dùng VnCoreNLP (cần Java 11)
# jar đặt tại: vncorenlp/VnCoreNLP-1.1.1.jar
```

> **Lưu ý:** Mặc định dùng `underthesea` (`use_vncorenlp=False`). Khi Java 11 sẵn sàng trên cluster, đổi sang `use_vncorenlp=True` để có word segmentation chính xác hơn.

---

## 2. TextPreprocessor — dùng trong Spark mapPartitions

### Import

```python
from preprocessing.text_cleaner import TextPreprocessor
```

### Khởi tạo (1 lần mỗi partition — lazy init)

```python
preprocessor = TextPreprocessor(
    slang_dict_path="data/slang_dict.json",   # hoặc HDFS path đã copy về /tmp
    stopwords_path="data/stopwords_vi.txt",
    use_vncorenlp=False,   # True khi Java 11 sẵn sàng trên workers
)
```

### Xử lý 1 bản ghi

```python
body      = preprocessor.clean(raw_text)       # → field `body` (HTML stripped, normalized)
segmented = preprocessor.preprocess(raw_text)  # → field `segmented_text` (full pipeline)
```

### Xử lý batch (hiệu quả hơn trong mapPartitions)

```python
bodies     = [row.content for row in batch]
seg_texts  = preprocessor.preprocess_batch(bodies)  # trả List[str]
```

### Phân biệt 2 output fields

| Method | Output field trong stg_posts_core | Mô tả |
|--------|-----------------------------------|-------|
| `clean(text)` | `body` | HTML stripped, lowercase, no emoji/URL |
| `preprocess(text)` | `segmented_text` | Thêm word segmentation + stopword removal |

---

## 3. Adapters — parse raw CSV từ M1

### 3.1 VOZ Adapter

```python
from schemas.voz_adapter import VozAdapter

adapter = VozAdapter()

# Từ file CSV
comments_df, posts_df = adapter.from_csv(
    comments_path="data/Data_NLP_DM/voz/comments.csv",
    posts_path="data/Data_NLP_DM/voz/posts.csv",
)

# Từ list dict (khi đọc từ HDFS JSON)
comments_df = adapter.comments_to_df(raw_list)
posts_df    = adapter.posts_to_df(raw_list)
```

**Output columns comments_df:**

| Column | Type | Ghi chú |
|--------|------|---------|
| `comment_id` | str | Dùng làm `post_id` |
| `user` | str | Dùng làm `author` |
| `comment` | str | Dùng làm `body` (raw) |
| `created_at` | int | Unix timestamp (giây) |
| `reaction_count` | int | Đã parse từ `"Ưng (3) \| Haha (1)"` → `4` |
| `id_post` | str | Dùng làm `parent_id` |

**Output columns posts_df:**

| Column | Type | Ghi chú |
|--------|------|---------|
| `id_post` | str | Dùng làm `post_id` |
| `author_name` | str | Dùng làm `author` |
| `title` | str | Dùng làm `body` và `title` |
| `created_at` | int | Unix timestamp (giây) |
| `view_count` | int\|None | Đã parse từ `"Views\n4,050"` → `4050` |
| `comment_count` | int\|None | Đã parse từ `"Replies\n195"` → `195` |

### 3.2 VatVo Adapter

```python
from schemas.vatvo_adapter import VatVoAdapter

adapter  = VatVoAdapter()
vatvo_df = adapter.from_csv("data/Data_NLP_DM/vatvo/articles.csv")
```

**Output columns:**

| Column | Type | Ghi chú |
|--------|------|---------|
| `post_id` | str | ID bài viết |
| `author` | str | Tên tác giả |
| `title` | str | Tiêu đề |
| `content` | str | Nội dung toàn văn → dùng làm `body` |
| `created_at` | int | Unix timestamp (giây) |
| `source` | str | `"vatvo"` |

### 3.3 VnExpress Adapter

```python
from schemas.vnexpress_adapter import VnExpressAdapter

adapter = VnExpressAdapter()
posts_df, comments_df = adapter.from_csv(
    posts_path="data/Data_NLP_DM/vnexpress/post_vnexpress.csv",
    comments_path="data/Data_NLP_DM/vnexpress/comment_vnexpress.csv",
)
```

**Output columns** (cả posts và comments đã được chuẩn hoá về cùng schema):

| Column | Type | Ghi chú |
|--------|------|---------|
| `post_id` | str | Đã prefix `"vne_"` để tránh trùng với VOZ |
| `source` | str | `"vnexpress"` |
| `author` | str | Username |
| `body` | str | Nội dung (đã strip sẵn) |
| `created_at` | int | Unix timestamp (giây, múi giờ ICT +7) |
| `parent_id` | str\|None | `None` nếu là bài post, `"vne_{id_post}"` nếu là comment |
| `reaction_count` | int | Đã parse từ `'{"Thích": 28}'` → `28` |

---

## 4. Map sang schema stg_posts_core

Sau khi dùng adapter, M2 map sang đúng các cột của `stg_posts_core`:

```python
from preprocessing.text_cleaner import TextPreprocessor

preprocessor = TextPreprocessor(
    slang_dict_path="data/slang_dict.json",
    stopwords_path="data/stopwords_vi.txt",
    use_vncorenlp=False,
)

# Ví dụ với VOZ comments
stg_df = pd.DataFrame({
    "post_id":        comments_df["comment_id"],
    "source":         "voz",
    "author":         comments_df["user"],
    "title":          None,
    "body":           comments_df["comment"].apply(preprocessor.clean),
    "segmented_text": preprocessor.preprocess_batch(comments_df["comment"].tolist()),
    "parent_id":      comments_df["id_post"].astype(str),
    "reaction_count": comments_df["reaction_count"].fillna(0).astype(int),
    "comment_count":  0,
    "view_count":     None,
    "created_at":     comments_df["created_at"].apply(
                          lambda ts: datetime.utcfromtimestamp(int(ts))
                      ),
    "crawled_at":     datetime.utcnow(),
})
```

---

## 5. Schema chuẩn stg_posts_core (để kiểm tra)

```sql
CREATE TABLE IF NOT EXISTS tech_radar.stg_posts_core (
    post_id         String,
    source          LowCardinality(String),   -- 'voz','vatvo','vnexpress','youtube'
    author          String,
    title           Nullable(String),
    body            String,                   -- clean() output
    segmented_text  String,                   -- preprocess() output
    parent_id       Nullable(String),
    reaction_count  Int32   DEFAULT 0,
    comment_count   Int32   DEFAULT 0,
    view_count      Nullable(Int32),
    created_at      DateTime,
    crawled_at      DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (source, created_at, post_id)
TTL created_at + INTERVAL 1 YEAR;
```

---

## 6. Tích hợp vào Spark cleaning_job.py

Pattern dùng trong `mapPartitions`:

```python
def process_partition(iterator):
    import sys, os
    sys.path.insert(0, "/path/to/project")   # hoặc dùng SparkFiles

    from preprocessing.text_cleaner import TextPreprocessor
    from schemas.voz_adapter import VozAdapter

    preprocessor = TextPreprocessor(
        slang_dict_path=os.environ["NLP_SLANG_DICT"],
        stopwords_path=os.environ["NLP_STOPWORDS"],
        use_vncorenlp=False,
    )

    for row in iterator:
        body      = preprocessor.clean(row.content or "")
        segmented = preprocessor.preprocess(row.content or "")
        yield (row.post_id, row.source, row.author, body, segmented, ...)
```

> **Quan trọng:** Khởi tạo `TextPreprocessor` **trong** `process_partition` (lazy init), không phải ngoài — tránh lỗi serialize object qua mạng Spark.

---

## 7. Tham chiếu nhanh

| Việc cần làm | File M4 cung cấp |
|---|---|
| Clean HTML / emoji / URL | `preprocessing/text_cleaner.py` → `TextPreprocessor.clean()` |
| Word segmentation + stopwords | `preprocessing/text_cleaner.py` → `TextPreprocessor.preprocess()` |
| Parse VOZ CSV | `schemas/voz_adapter.py` → `VozAdapter` |
| Parse VatVo CSV | `schemas/vatvo_adapter.py` → `VatVoAdapter` |
| Parse VnExpress CSV | `schemas/vnexpress_adapter.py` → `VnExpressAdapter` |
| Pydantic schema validation | `schemas/models.py` → `UniversalSocialPost` |
| Script giả lập local (không cần Spark) | `scripts/load_stg_posts_core.py` |

---

## 8. Chạy thử local (không cần Spark cluster)

Để verify pipeline trước khi deploy lên cluster:

```bash
# Dry-run: kiểm tra output mà không insert ClickHouse
python scripts/load_stg_posts_core.py --dry-run

# Insert vào ClickHouse local
python scripts/load_stg_posts_core.py \
    --host localhost --port 8123 \
    --db tech_radar --user default --password ""

# Bỏ qua LSH dedup (nếu chưa cài datasketch)
python scripts/load_stg_posts_core.py --no-dedup --dry-run
```

Nếu chạy được `--dry-run` thành công → pipeline M4 hoạt động đúng, M2 có thể port vào Spark job.

---

## 9. Liên hệ / Blockers

| Vấn đề | Cách xử lý |
|--------|------------|
| `use_vncorenlp=True` crash | Java 11 chưa cài → dùng `use_vncorenlp=False` tạm |
| `slang_dict.json` không tìm thấy | Đặt tại `data/slang_dict.json` hoặc set env `NLP_SLANG_DICT` |
| `stopwords_vi.txt` không tìm thấy | Đặt tại `data/stopwords_vi.txt` hoặc set env `NLP_STOPWORDS` |
| Adapter skip nhiều records | Kiểm tra log WARNING — thường do format time thay đổi trong CSV mới |
| `vatvo` bị reject bởi Pydantic | `VALID_SOURCES` trong `schemas/models.py` đã có `"vatvo"` — OK |
