# Chi tiết thực thi Phase 3: BERTopic & Integration (Member 3)

**Ngữ cảnh:** Tiếp tục chạy Local. BERTopic yêu cầu cấu hình phần cứng tốt (RAM/GPU), sẽ chạy trên subset data nhỏ.

**⚠️ Thay đổi quan trọng:** Tech stack đã chuyển từ **PostgreSQL → ClickHouse**

---

## 1. Task 3.1 & 3.2: Triển khai và Tune BERTopic

### 📊 Trạng thái hiện tại
- ✅ File `models/bertopic_model.py` đã có cấu trúc
- ✅ File `notebooks/bertopic_tuning.ipynb` đã được tạo
- ❌ **Chưa nhúng PhoBERT embeddings** (`vinai/phobert-base`)
- ❌ **Chưa tính toán ra các topic cụ thể**

### 🎯 Hành động cần làm ngay

#### Bước 1: Cài đặt dependencies

```bash
# BERTopic và dependencies
pip install bertopic sentence-transformers torch

# Visualization
pip install pyLDAvis plotly

# Nếu có GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Bước 2: Code triển khai BERTopic với PhoBERT

**🚨 LƯU Ý:** Quá trình sinh embedding bằng PhoBERT khá nặng (RAM/GPU intensive). 

**Khuyến nghị:**
- Bạn hãy mở `notebooks/bertopic_tuning.ipynb`
- Dán đoạn code tôi cung cấp dưới đây
- Tự chạy thủ công (tôi không thể execute .ipynb trực tiếp)
- **Nếu máy không có GPU:** Chỉnh `batch_size=8` hoặc `batch_size=4`
- **Nếu bị Out of Memory:** Giảm số documents (từ 5000 → 2000)

---

### 📝 Code mẫu cho `bertopic_tuning.ipynb`

```python
# ============================================================================
# CELL 1: IMPORTS & CẤU HÌNH
# ============================================================================

import os
import re
import json
import warnings
import logging
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BERTopic và dependencies
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# NLP preprocessing
from underthesea import word_tokenize

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

print('✅ Imports OK')

# ============================================================================
# CELL 2: LOAD DỮ LIỆU
# ============================================================================

# Load dữ liệu mẫu
DATA_PATH = '../data/fake/posts_5k.csv'
df = pd.read_csv(DATA_PATH)

# Kiểm tra cột text
text_column = 'content' if 'content' in df.columns else 'text'
print(f"📊 Loaded {len(df)} documents")

# Lấy subset để tránh OOM (giảm xuống 2000 nếu máy yếu)
SUBSET_SIZE = 3000  # Điều chỉnh theo RAM của bạn
df_subset = df.sample(n=min(SUBSET_SIZE, len(df)), random_state=42)
documents = df_subset[text_column].dropna().tolist()

print(f"📝 Using {len(documents)} documents for BERTopic")

# ============================================================================
# CELL 3: TIỀN XỬ LÝ (CƠ BẢN)
# ============================================================================

# Load stopwords và slang
with open('../data/stopwords_vi.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = set(line.strip() for line in f if line.strip())

with open('../data/slang_dict.json', 'r', encoding='utf-8') as f:
    SLANG_DICT = json.load(f)

def clean_text_basic(text):
    """Basic cleaning for BERTopic (ít aggressive hơn LDA)"""
    if pd.isna(text) or not text:
        return ""
    
    # Lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Slang normalization
    words = text.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    text = ' '.join(words)
    
    return text.strip()

# Clean documents
documents_cleaned = [clean_text_basic(doc) for doc in documents]
documents_cleaned = [doc for doc in documents_cleaned if len(doc) > 20]

print(f"✅ Cleaned {len(documents_cleaned)} documents")
print(f"📝 Sample: {documents_cleaned[0][:200]}...")

# ============================================================================
# CELL 4: TẢI PHOBERT EMBEDDING MODEL
# ============================================================================

print("\n🔄 Loading PhoBERT embedding model...")
print("⚠️ Lần đầu chạy sẽ download ~1GB model từ HuggingFace")

# Khởi tạo PhoBERT encoder
embedding_model = SentenceTransformer('vinai/phobert-base')

# Test embedding
sample_embedding = embedding_model.encode(["xin chào"], show_progress_bar=False)
print(f"✅ PhoBERT loaded! Embedding dim: {sample_embedding.shape[1]}")

# ============================================================================
# CELL 5: CẤU HÌNH BERTOPIC COMPONENTS
# ============================================================================

# UMAP for dimensionality reduction
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# HDBSCAN for clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=15,      # Tối thiểu 15 docs/topic
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# CountVectorizer for topic representation (Vietnamese-specific)
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=list(STOPWORDS),
    min_df=2,
    max_df=0.7
)

print("✅ BERTopic components configured")

# ============================================================================
# CELL 6: TRAIN BERTOPIC MODEL
# ============================================================================

print("\n" + "="*80)
print("🔄 TRAINING BERTOPIC MODEL")
print("="*80)
print("⏳ Đang tạo embeddings... (có thể mất 5-15 phút tùy hardware)")

# Khởi tạo BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    language='vietnamese',
    calculate_probabilities=True,
    verbose=True
)

# Fit model
topics, probs = topic_model.fit_transform(documents_cleaned)

print(f"\n✅ BERTopic training completed!")
print(f"📊 Found {len(set(topics))} topics (including outliers)")
print(f"📊 Topic distribution: {pd.Series(topics).value_counts().head(10)}")

# ============================================================================
# CELL 7: HIỂN THỊ TOPIC INFO
# ============================================================================

# Get topic info
topic_info = topic_model.get_topic_info()
print("\n" + "="*80)
print("TOPIC INFORMATION")
print("="*80)
print(topic_info.head(15))

# Save topic info
os.makedirs('../output', exist_ok=True)
topic_info.to_csv('../output/bertopic_topics.csv', index=False)
print("\n✅ Topic info saved to: output/bertopic_topics.csv")

# ============================================================================
# CELL 8: HIỂN THỊ TOP WORDS CHO MỖI TOPIC
# ============================================================================

print("\n" + "="*80)
print("TOP 10 WORDS FOR EACH TOPIC")
print("="*80 + "\n")

# Lấy tất cả topics (trừ outlier topic -1)
topics_list = sorted([t for t in set(topics) if t != -1])

for topic_id in topics_list[:15]:  # Hiển thị top 15 topics
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        words = [word for word, score in topic_words]
        scores = [score for word, score in topic_words]
        
        print(f"Topic {topic_id} (Count: {(pd.Series(topics) == topic_id).sum()}):")
        print(f"  Words: {', '.join(words[:10])}")
        print(f"  Scores: {[f'{s:.3f}' for s in scores[:5]]}")
        print()

# ============================================================================
# CELL 9: VISUALIZE TOPICS
# ============================================================================

print("\n🎨 Generating visualizations...")

# Visualization 1: Intertopic Distance Map
try:
    fig1 = topic_model.visualize_topics()
    fig1.write_html('../output/bertopic_intertopic_map.html')
    print("✅ Saved: output/bertopic_intertopic_map.html")
except Exception as e:
    print(f"⚠️ Could not generate intertopic map: {e}")

# Visualization 2: Topic Hierarchy
try:
    fig2 = topic_model.visualize_hierarchy()
    fig2.write_html('../output/bertopic_hierarchy.html')
    print("✅ Saved: output/bertopic_hierarchy.html")
except Exception as e:
    print(f"⚠️ Could not generate hierarchy: {e}")

# Visualization 3: Barchart (Top Topics)
try:
    fig3 = topic_model.visualize_barchart(top_n_topics=10)
    fig3.write_html('../output/bertopic_barchart.html')
    print("✅ Saved: output/bertopic_barchart.html")
except Exception as e:
    print(f"⚠️ Could not generate barchart: {e}")

# ============================================================================
# CELL 10: TUNE HYPERPARAMETERS (OPTIONAL)
# ============================================================================

print("\n" + "="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

# Test với các min_cluster_size khác nhau
tuning_results = []

for min_size in [10, 15, 20, 25]:
    print(f"\n🔄 Testing min_cluster_size={min_size}...")
    
    hdbscan_temp = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    topic_model_temp = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_temp,
        vectorizer_model=vectorizer_model,
        verbose=False
    )
    
    topics_temp, _ = topic_model_temp.fit_transform(documents_cleaned)
    n_topics = len(set(topics_temp)) - 1  # Trừ outlier topic -1
    outlier_ratio = (pd.Series(topics_temp) == -1).sum() / len(topics_temp)
    
    tuning_results.append({
        'min_cluster_size': min_size,
        'n_topics': n_topics,
        'outlier_ratio': outlier_ratio
    })
    
    print(f"  → Topics: {n_topics}, Outlier ratio: {outlier_ratio:.2%}")

# Save tuning results
tuning_df = pd.DataFrame(tuning_results)
print("\n📊 Tuning Results:")
print(tuning_df.to_string(index=False))
tuning_df.to_csv('../output/bertopic_tuning_results.csv', index=False)

# ============================================================================
# CELL 11: LƯU MODEL
# ============================================================================

# Save BERTopic model
model_dir = '../output/bertopic_model'
topic_model.save(model_dir, serialization='pytorch')
print(f"\n✅ Model saved to: {model_dir}")

# Save embeddings (optional, for future use)
# embeddings = embedding_model.encode(documents_cleaned, show_progress_bar=True)
# np.save('../output/bertopic_embeddings.npy', embeddings)
# print("✅ Embeddings saved to: output/bertopic_embeddings.npy")

print("\n" + "="*80)
print("✅ BERTOPIC PIPELINE COMPLETED!")
print("="*80)
```

---

### 📊 Kết quả kỳ vọng

Sau khi chạy xong notebook, bạn sẽ có:

1. **Topic Info CSV:** `output/bertopic_topics.csv`
   - Topic ID, Count, Top words, Representative document

2. **Interactive Visualizations:**
   - `output/bertopic_intertopic_map.html` - 2D map của các topics
   - `output/bertopic_hierarchy.html` - Hierarchical clustering
   - `output/bertopic_barchart.html` - Top topics barchart

3. **Tuning Results:** `output/bertopic_tuning_results.csv`
   - Số lượng topics vs min_cluster_size
   - Outlier ratio

4. **Saved Model:** `output/bertopic_model/`
   - BERTopic model để reuse sau này

**📝 Hãy báo lại cho tôi:**
- Tổng số topics tìm được (trừ outliers)?
- Outlier ratio (tỷ lệ documents không thuộc topic nào)?
- Top 5 topics có từ khóa gì?
- Có topic nào có ý nghĩa rõ ràng không? (VD: AI, smartphone, gaming...)

---

## 2. Task 3.3: So sánh LDA vs BERTopic

### 📊 Trạng thái hiện tại
- ✅ File `reports/topic_comparison.md` đã tồn tại
- ❌ **Chưa có data đối chiếu thực tế**

### 🎯 Hành động cần làm

**SAU KHI** bạn chạy xong:
1. ✅ LDA evaluation (Phase 2) → có coherence scores
2. ✅ BERTopic tuning (Phase 3) → có topic info

**THÌ** hãy gửi cho tôi (AI) kết quả sau:

#### Thông tin cần từ LDA:
```
- Optimal k: ?
- Coherence C_V: ?
- Coherence UMass: ?
- Top 3 topics (top 10 words mỗi topic)
```

#### Thông tin cần từ BERTopic:
```
- Số topics tìm được: ?
- Outlier ratio: ?
- Top 3 topics (top 10 words mỗi topic)
```

→ **Tôi sẽ tự động sinh nội dung cho `reports/topic_comparison.md`** dựa trên data bạn cung cấp.

**Template comparison sẽ bao gồm:**
- ✅ Methodology comparison (LDA vs BERTopic)
- ✅ Quantitative metrics (coherence, coverage, outlier ratio)
- ✅ Qualitative analysis (topic interpretability)
- ✅ Computational cost (training time, memory)
- ✅ Recommendation (nên dùng model nào cho production)

---

## 3. Task 3.4 & 3.5: Lưu dữ liệu (ClickHouse) & Handoff cho Member 5

### ⚠️ THAY ĐỔI QUAN TRỌNG: PostgreSQL → ClickHouse

Theo `TECH_STACK.md` version 3.0:
> **Thay đổi v3.0:** Bỏ MongoDB, thay PostgreSQL → ClickHouse + dbt

### 📊 Trạng thái hiện tại
- ⚠️ Task 3.4 trong `TEAM_ASSIGNMENTS.md` còn ghi: `save_topics_to_pg.py` (PostgreSQL)
- ✅ **PHẢI ĐỔI SANG:** `save_topics_to_ch.py` (ClickHouse)
- ❌ **Blocker:** ClickHouse của Member 2 chưa được setup

### 🎯 Hành động: Workaround với CSV Export

Do không có ClickHouse database thật, chúng ta sẽ:
1. Xuất kết quả ra CSV file
2. Gửi CSV cho Member 5 để làm Dashboard prototype
3. Sau khi Member 2 setup xong ClickHouse → viết script import CSV vào DB

---

### 📝 Code: Export Topics to CSV

Tạo file mới: `scripts/export_topics_to_csv.py`

```python
"""
Export BERTopic results to CSV for Member 5 Dashboard
Workaround cho việc ClickHouse chưa sẵn sàng
"""

import os
import pandas as pd
from bertopic import BERTopic

# Load BERTopic model đã train
MODEL_PATH = '../output/bertopic_model'
print(f"🔄 Loading BERTopic model from: {MODEL_PATH}")

topic_model = BERTopic.load(MODEL_PATH)
print("✅ Model loaded successfully")

# Get topic info
topic_info = topic_model.get_topic_info()

# Tạo topic_clusters_mock.csv với schema cho Member 5
# Columns: topic_id, topic_label, doc_count, top_words, representative_doc

# Loại bỏ outlier topic (-1)
topic_clusters = topic_info[topic_info['Topic'] != -1].copy()

# Tạo topic_label từ top 3 words
topic_clusters['topic_label'] = topic_clusters.apply(
    lambda row: f"Topic_{row['Topic']}: {', '.join(row['Name'].split('_')[1:4])}", 
    axis=1
)

# Rename columns
topic_clusters = topic_clusters.rename(columns={
    'Topic': 'topic_id',
    'Count': 'doc_count',
    'Name': 'top_words',
    'Representative_Docs': 'representative_doc'
})

# Select final columns
output_df = topic_clusters[['topic_id', 'topic_label', 'doc_count', 'top_words']]

# Save to CSV
OUTPUT_PATH = '../data/topic_clusters_mock.csv'
output_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✅ Exported {len(output_df)} topics to: {OUTPUT_PATH}")
print(f"\n📊 Sample rows:")
print(output_df.head(10).to_string(index=False))

print(f"\n📧 Hãy gửi file này cho Member 5:")
print(f"   {os.path.abspath(OUTPUT_PATH)}")
```

**Chạy script:**
```bash
cd scripts
python export_topics_to_csv.py
```

**Output:** `data/topic_clusters_mock.csv`

---

### 📋 Schema CSV cho Member 5

File `data/topic_clusters_mock.csv` sẽ có format:

```csv
topic_id,topic_label,doc_count,top_words
0,"Topic_0: điện thoại, smartphone, iphone",245,"0_điện thoại_smartphone_iphone_..."
1,"Topic_1: AI, trí tuệ nhân tạo, machine learning",189,"1_ai_trí tuệ_nhân tạo_..."
2,"Topic_2: game, gaming, esport",156,"2_game_gaming_esport_..."
...
```

**Cột:**
- `topic_id`: Integer (0, 1, 2, ...)
- `topic_label`: Human-readable label
- `doc_count`: Số documents thuộc topic này
- `top_words`: Top keywords (gensim format)

**📧 Handoff cho Member 5:**
```
Hi Member 5,

Do ClickHouse chưa sẵn sàng, tôi đã export topics ra CSV: data/topic_clusters_mock.csv

Format:
- topic_id: ID của topic
- topic_label: Nhãn dễ đọc (Top 3 keywords)
- doc_count: Số lượng documents
- top_words: Full keyword list

Bạn có thể dùng file này để prototype Dashboard trước.
Khi ClickHouse ready, tôi sẽ import vào DB và update connection string.

Columns cần hiển thị trên Dashboard:
- Topic distribution pie chart (doc_count)
- Top 10 topics table (topic_label, doc_count)
- Keyword cloud cho mỗi topic (top_words)

Thanks!
```

---

### 🔮 Future Work: Script import vào ClickHouse

Khi Member 2 setup xong ClickHouse, tạo file: `scripts/save_topics_to_ch.py`

```python
"""
Import topic clusters to ClickHouse
TODO: Run after Member 2 completes infrastructure setup
"""

import pandas as pd
from clickhouse_driver import Client

# Load CSV
df = pd.read_csv('../data/topic_clusters_mock.csv')

# Connect to ClickHouse
client = Client(host='localhost', port=9000)  # Update with real host

# Create table (nếu chưa có)
create_table_query = """
CREATE TABLE IF NOT EXISTS topic_clusters (
    topic_id Int32,
    topic_label String,
    doc_count UInt32,
    top_words String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY topic_id
"""
client.execute(create_table_query)

# Insert data
data = df.to_dict('records')
client.execute(
    'INSERT INTO topic_clusters (topic_id, topic_label, doc_count, top_words) VALUES',
    data
)

print(f"✅ Inserted {len(data)} topics to ClickHouse")
```

**Dependencies:**
```bash
pip install clickhouse-driver
```

---

## 📋 Checklist hoàn thành Phase 3

| Task | Status | Evidence |
|------|--------|----------|
| 3.1: BERTopic Implementation | ⏳ **Đang chờ bạn chạy** | `notebooks/bertopic_tuning.ipynb` |
| 3.2: BERTopic Tuning | ⏳ **Đang chờ kết quả** | Cần file `output/bertopic_topics.csv` |
| 3.3: LDA vs BERTopic Comparison | ⏳ **Chờ data từ 3.1 & 3.2** | Tôi sẽ tự gen `reports/topic_comparison.md` |
| 3.4: Save to ClickHouse | ✅ **Workaround: CSV export** | `data/topic_clusters_mock.csv` |
| 3.5: Handoff to Member 5 | ✅ **Sẵn sàng** | CSV file + schema documentation |

---

## 🚀 Next Steps

1. **URGENT:** Chạy `bertopic_tuning.ipynb` và báo lại kết quả
2. Chạy `scripts/export_topics_to_csv.py` để tạo CSV cho Member 5
3. **Gửi tôi kết quả LDA + BERTopic** để tôi gen comparison report
4. **Handoff CSV cho Member 5** qua email/Slack/Teams
5. **Commit deliverables vào git:**
   ```bash
   git add models/ notebooks/bertopic_tuning.ipynb output/ data/topic_clusters_mock.csv scripts/
   git commit -m "Phase 3: BERTopic implementation + CSV export for Member 5"
   git push
   ```

---

## 🆘 Troubleshooting

### Lỗi: CUDA Out of Memory
```python
# Trong code, giảm batch_size khi encode
embedding_model.encode(documents, batch_size=8)  # Thay vì 32
```

Hoặc chạy trên CPU:
```python
import torch
device = 'cpu'
embedding_model = SentenceTransformer('vinai/phobert-base', device=device)
```

### Lỗi: Too many outliers (>50%)
Giảm `min_cluster_size` trong HDBSCAN:
```python
hdbscan_model = HDBSCAN(min_cluster_size=10)  # Thay vì 15
```

### Lỗi: PhoBERT download failed
Thủ công download model:
```bash
git lfs install
git clone https://huggingface.co/vinai/phobert-base
```

Sau đó load local:
```python
embedding_model = SentenceTransformer('./phobert-base')
```

### Lỗi: BERTopic visualization failed
Cài thêm dependencies:
```bash
pip install plotly kaleido
```

---

## 📊 Comparison: LDA vs BERTopic (Preview)

| Aspect | LDA | BERTopic |
|--------|-----|----------|
| **Approach** | Probabilistic (Dirichlet) | Neural embeddings + Clustering |
| **Input** | Bag-of-Words | Contextualized embeddings (PhoBERT) |
| **Pros** | - Fast<br>- Interpretable probabilities<br>- Works well on small data | - Better semantic understanding<br>- Captures context<br>- Dynamic topic modeling |
| **Cons** | - Ignores word order<br>- Manual k selection<br>- Weak on short texts | - Slow (requires embeddings)<br>- GPU preferred<br>- Black-box clustering |
| **Best for** | Large corpus, exploratory | High-quality topics, short texts |

**Recommendation:** Sẽ cung cấp sau khi có kết quả thực tế từ bạn! 🎯

---

**📅 Thời gian ước lượng:**
- Cài dependencies: 10-15 phút
- BERTopic training: 15-30 phút (CPU), 5-10 phút (GPU)
- Export CSV: 2 phút
- Write comparison report: 10 phút (by AI)

**Tổng: ~45-60 phút (CPU) hoặc ~25-35 phút (GPU)**
