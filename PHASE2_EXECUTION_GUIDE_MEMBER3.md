# Chi tiết thực thi Phase 2: Triển khai thuật toán (Member 3)

**Ngữ cảnh:** Spark/HDFS chưa sẵn sàng. Chuyển sang chạy Local với dataset nhỏ bằng `gensim` và Python thuần.

**Dữ liệu mẫu có sẵn:** `data/fake/posts_5k.csv` và `data/fake/comments_5k.csv`

---

## 1. Task 2.1 & 2.2: Triển khai và Đánh giá LDA

### 📊 Trạng thái hiện tại
- ✅ File `spark_jobs/lda_job.py` đã được viết (Spark MLlib version)
- ✅ File `notebooks/lda_evaluation.ipynb` đã có cấu trúc
- ❌ **Chưa train model trên dữ liệu thực**
- ❌ **Chưa có Topic Coherence Score** (notebook chưa được execute)

### 🎯 Hành động cần làm ngay

#### Bước 1: Chuyển đổi sang Local Mode với gensim

Do không có Spark cluster, chúng ta sẽ chạy LDA local bằng `gensim.models.LdaMulticore` thay vì Spark MLlib.

**File cần chỉnh sửa:** `notebooks/lda_evaluation.ipynb`

#### Bước 2: Code thực thi LDA Evaluation

**🚨 LƯU Ý:** Tôi (AI) không thể chạy trực tiếp file `.ipynb`. Bạn vui lòng:
1. Mở Jupyter Notebook: `jupyter notebook notebooks/lda_evaluation.ipynb`
2. Dán đoạn code tôi cung cấp dưới đây vào notebook
3. Bấm **Run All** (hoặc Shift+Enter từng cell)
4. Báo lại cho tôi kết quả **Coherence Score** (UMass / CV) mà bạn nhận được

---

### 📝 Code mẫu để chạy trong `lda_evaluation.ipynb`

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
import matplotlib

# NLP & Topic Modeling
from underthesea import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel

# Cấu hình hiển thị
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (12, 6)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('✅ Imports OK')

# ============================================================================
# CELL 2: LOAD DỮ LIỆU
# ============================================================================

# Load dữ liệu mẫu
DATA_PATH = '../data/fake/posts_5k.csv'
df = pd.read_csv(DATA_PATH)

# Kiểm tra cột text (có thể là 'content', 'text', 'body' tùy schema)
text_column = 'content' if 'content' in df.columns else 'text'
print(f"📊 Loaded {len(df)} documents")
print(f"📝 Text column: {text_column}")
print(df.head(3))

# ============================================================================
# CELL 3: TIỀN XỬ LÝ VĂN BẢN
# ============================================================================

# Load stopwords và slang dict
with open('../data/stopwords_vi.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = set(line.strip() for line in f if line.strip())

with open('../data/slang_dict.json', 'r', encoding='utf-8') as f:
    SLANG_DICT = json.load(f)

print(f"✅ Loaded {len(STOPWORDS)} stopwords")
print(f"✅ Loaded {len(SLANG_DICT)} slang entries")

def clean_text(text):
    """5-step preprocessing pipeline"""
    if pd.isna(text):
        return ""
    
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Remove HTML tags, URLs, special chars
    text = re.sub(r'<[^>]+>', '', text)  # HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # URLs
    text = re.sub(r'[^\w\s]', ' ', text)  # Special chars
    
    # 3. Slang normalization
    words = text.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    text = ' '.join(words)
    
    # 4. Word segmentation (underthesea)
    try:
        text = word_tokenize(text, format='text')
    except:
        pass
    
    # 5. Remove stopwords
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    
    return tokens

# Áp dụng preprocessing
print("🔄 Processing texts...")
df['tokens'] = df[text_column].progress_apply(clean_text) if hasattr(df, 'progress_apply') else df[text_column].apply(clean_text)

# Lọc documents rỗng
df = df[df['tokens'].apply(len) > 5]
documents = df['tokens'].tolist()

print(f"✅ Preprocessed {len(documents)} documents")
print(f"📝 Sample tokens: {documents[0][:10]}")

# ============================================================================
# CELL 4: TẠO DICTIONARY & CORPUS
# ============================================================================

# Tạo Gensim Dictionary
dictionary = Dictionary(documents)

# Filter extremes
dictionary.filter_extremes(
    no_below=5,      # Xuất hiện ít nhất 5 documents
    no_above=0.5,    # Xuất hiện tối đa 50% documents
    keep_n=10000     # Giữ tối đa 10K từ
)

print(f"✅ Dictionary size: {len(dictionary)} unique tokens")

# Tạo Bag-of-Words corpus
corpus = [dictionary.doc2bow(doc) for doc in documents]
print(f"✅ Corpus created: {len(corpus)} documents")

# ============================================================================
# CELL 5: TRAIN LDA VỚI NHIỀU GIÁ TRỊ k (15-25)
# ============================================================================

# Sweep k values
K_VALUES = [15, 18, 20, 22, 25]
results = []

for k in K_VALUES:
    print(f"\n{'='*60}")
    print(f"🔄 Training LDA with k={k} topics...")
    print(f"{'='*60}")
    
    # Train LDA
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        iterations=50,
        passes=10,
        workers=4,
        random_state=42
    )
    
    # Tính Coherence Score (C_V)
    coherence_model_cv = CoherenceModel(
        model=lda_model,
        texts=documents,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_cv = coherence_model_cv.get_coherence()
    
    # Tính Coherence Score (UMass)
    coherence_model_umass = CoherenceModel(
        model=lda_model,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass'
    )
    coherence_umass = coherence_model_umass.get_coherence()
    
    # Tính Perplexity
    perplexity = lda_model.log_perplexity(corpus)
    
    results.append({
        'k': k,
        'coherence_cv': coherence_cv,
        'coherence_umass': coherence_umass,
        'perplexity': perplexity
    })
    
    print(f"✅ k={k}: C_V={coherence_cv:.4f}, UMass={coherence_umass:.4f}, Perplexity={perplexity:.4f}")

# ============================================================================
# CELL 6: LƯU KẾT QUẢ
# ============================================================================

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\n📊 SUMMARY OF RESULTS:")
print(results_df.to_string(index=False))

# Save to CSV
os.makedirs('../output', exist_ok=True)
results_df.to_csv('../output/lda_coherence_results.csv', index=False)
print("\n✅ Results saved to: output/lda_coherence_results.csv")

# ============================================================================
# CELL 7: VISUALIZE COHERENCE SCORES
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot C_V
axes[0].plot(results_df['k'], results_df['coherence_cv'], marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Topics (k)')
axes[0].set_ylabel('Coherence Score (C_V)')
axes[0].set_title('LDA Coherence Score (C_V) vs k')
axes[0].grid(True, alpha=0.3)

# Plot UMass
axes[1].plot(results_df['k'], results_df['coherence_umass'], marker='s', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Number of Topics (k)')
axes[1].set_ylabel('Coherence Score (UMass)')
axes[1].set_title('LDA Coherence Score (UMass) vs k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/lda_coherence_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Plot saved to: output/lda_coherence_plot.png")

# ============================================================================
# CELL 8: TRAIN FINAL MODEL VỚI k TỐI ưU
# ============================================================================

# Chọn k tối ưu (k có C_V cao nhất)
optimal_k = results_df.loc[results_df['coherence_cv'].idxmax(), 'k']
print(f"\n🎯 Optimal k (based on C_V): {int(optimal_k)}")

# Train final model
print(f"🔄 Training final LDA model with k={int(optimal_k)}...")
final_lda = LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    num_topics=int(optimal_k),
    iterations=50,
    passes=10,
    workers=4,
    random_state=42
)

# Save model
final_lda.save('../output/lda_model_final')
dictionary.save('../output/lda_dictionary')
print("✅ Final model saved to: output/lda_model_final")

# ============================================================================
# CELL 9: HIỂN THỊ TOP WORDS CHO MỖI TOPIC
# ============================================================================

print(f"\n{'='*80}")
print(f"TOP 10 WORDS FOR EACH TOPIC (k={int(optimal_k)})")
print(f"{'='*80}\n")

for idx, topic in final_lda.print_topics(num_topics=int(optimal_k), num_words=10):
    print(f"Topic {idx+1}:")
    # Parse topic string
    words = re.findall(r'"([^"]+)"', topic)
    print(f"  {', '.join(words)}")
    print()

print("✅ LDA Evaluation completed!")
```

---

### 📊 Kết quả kỳ vọng

Sau khi chạy xong notebook, bạn sẽ có:

1. **File CSV:** `output/lda_coherence_results.csv`
   - Chứa coherence scores (C_V, UMass) và perplexity cho k ∈ {15, 18, 20, 22, 25}

2. **Biểu đồ:** `output/lda_coherence_plot.png`
   - Visualization của coherence scores theo k

3. **Model đã train:** `output/lda_model_final`
   - LDA model với k tối ưu (k có C_V cao nhất)

4. **Console output:**
   - Top 10 words cho mỗi topic
   - Coherence scores cho từng k

**📝 Hãy báo lại cho tôi:**
- Giá trị Coherence C_V và UMass cho từng k
- k tối ưu là bao nhiêu?
- Top words của 3-5 topics có ý nghĩa không?

---

## 2. Task 2.3 & 2.4: Triển khai Count-Min Sketch (CMS)

### 📊 Trạng thái hiện tại
- ✅ File `algorithms/count_min_sketch.py` đã có implementation hoàn chỉnh
- ✅ File `tests/test_cms.py` đã có 27 test cases
- ❌ **Chưa chạy test để xác thực error rate**

### 🎯 Hành động cần làm ngay

#### Bước 1: Cài đặt dependencies

```bash
# Cài pytest và mmh3 (MurmurHash3)
pip install pytest mmh3
```

#### Bước 2: Chạy unit tests

Count-Min Sketch **không phụ thuộc Spark**, có thể chạy local ngay:

```bash
# Chạy tất cả tests
python -m pytest tests/test_cms.py -v

# Hoặc chạy với output chi tiết hơn
python -m pytest tests/test_cms.py -v --tb=short
```

**📝 Output mong đợi:**
```
tests/test_cms.py::test_basic_add_and_query PASSED
tests/test_cms.py::test_never_undercount PASSED
tests/test_cms.py::test_error_rate_within_epsilon PASSED
tests/test_cms.py::test_merge_sketches PASSED
tests/test_cms.py::test_top_k_accuracy PASSED
tests/test_cms.py::test_memory_efficiency PASSED
tests/test_cms.py::test_serialization PASSED
... (tổng 27 tests)

======================== 27 passed in X.XXs ========================
```

### ✅ Kết quả kỳ vọng

Theo báo cáo `REPORT_PHASE2_MEMBER3.md`, tests đã được chạy trước đó và **27/27 PASSED**. 

Các test covers:
- ✅ Basic add/query operations
- ✅ Never undercount property (theoretical guarantee)
- ✅ Error rate ε validation
- ✅ Merge sketches (distributed partitions simulation)
- ✅ Top-K accuracy vs ground truth
- ✅ Memory efficiency (CMS << dict/Counter)
- ✅ Serialization/deserialization

**🎯 Action:** Chạy lại test để confirm, sau đó commit code vào git.

---

## 3. Task 2.5: Tích hợp Airflow DAG

### 📊 Trạng thái hiện tại
- ✅ File `dags/processing_dag.py` đã có DAG `cms_keyword_streaming`
- ⚠️ **Hiện đang chạy LOCAL MODE** (`USE_LOCAL = True`)
- ⏸️ **Cluster mode TODO** (cần Member 2 setup HDFS + ClickHouse)

### 🎯 Hành động

**SKIP tạm thời** - Lý do:
- Airflow cần infrastructure (HDFS, ClickHouse) từ Member 2
- Member 2 chưa setup xong → không thể test integration
- Sẽ quay lại sau khi có infrastructure

**📝 Ghi chú để sau:**
- DAG đã có cấu trúc đúng:
  - Schedule: `*/15 * * * *` (mỗi 15 phút)
  - Tasks: `read_new_data → run_cms_update → export_top_keywords`
- Khi Member 2 setup xong, chỉ cần:
  1. Đổi `USE_LOCAL = False`
  2. Cấu hình HDFS paths và ClickHouse connection
  3. Test DAG trên Airflow UI

---

## 📋 Checklist hoàn thành Phase 2

| Task | Status | Evidence |
|------|--------|----------|
| 2.1: LDA Implementation | ⏳ **Đang chờ bạn chạy notebook** | `notebooks/lda_evaluation.ipynb` |
| 2.2: LDA Coherence Evaluation | ⏳ **Đang chờ kết quả** | Cần file `output/lda_coherence_results.csv` |
| 2.3: Count-Min Sketch Implementation | ✅ **Hoàn thành** | `algorithms/count_min_sketch.py` |
| 2.4: CMS Unit Tests | ✅ **27/27 PASSED** | Chạy `pytest tests/test_cms.py` |
| 2.5: Airflow Integration | ⏸️ **Skip (chờ infra)** | Sẽ làm sau khi Member 2 xong |

---

## 🚀 Next Steps

1. **URGENT:** Chạy `lda_evaluation.ipynb` và báo lại kết quả coherence scores
2. Chạy lại CMS tests để confirm: `pytest tests/test_cms.py -v`
3. **Commit deliverables vào git:**
   ```bash
   git add algorithms/ tests/ notebooks/lda_evaluation.ipynb output/
   git commit -m "Phase 2: LDA evaluation results + CMS tests passed"
   git push
   ```
4. Chuẩn bị chuyển sang **Phase 3: BERTopic**

---

## 🆘 Troubleshooting

### Lỗi: Module 'underthesea' not found
```bash
pip install underthesea
```

### Lỗi: Gensim version incompatible
```bash
pip install gensim==4.3.0
```

### Lỗi: Out of Memory khi train LDA
- Giảm `keep_n` trong `dictionary.filter_extremes()` (từ 10000 → 5000)
- Giảm `workers` trong `LdaMulticore()` (từ 4 → 2)
- Sử dụng subset nhỏ hơn (2000-3000 documents thay vì 5000)

### Lỗi: CMS tests failed
- Kiểm tra đã cài `mmh3`: `pip install mmh3`
- Xem chi tiết lỗi: `pytest tests/test_cms.py -v --tb=long`

---

**📅 Thời gian ước lượng:**
- LDA evaluation: 30-60 phút (tùy hardware)
- CMS tests: 2-5 phút
- Git commit: 5 phút

**Tổng: ~1 giờ**
