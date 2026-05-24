# 📚 THIẾT KẾ NOTEBOOK: train_bertopic_kaggle.ipynb

## 🎯 MỤC ĐÍCH

Train BERTopic model trên Kaggle GPU để tránh GPU local bị quá tải.

---

## 🏗️ KIẾN TRÚC NOTEBOOK

### Nguyên tắc thiết kế:

```
PROTOTYPE → FUNCTIONS → CLASS → PRODUCTION

Notebook này áp dụng cách tiếp cận:
1. Setup & Test nhỏ
2. Viết code từng bước
3. Refactor thành class
4. Train & Save
```

---

## 📋 CẤU TRÚC NOTEBOOK (9 CELLS)

### **CELL 1: 📦 Setup & Dependencies** (3-5 phút)
**Mục đích:** Cài packages và verify GPU

**Nội dung:**
- Install: bertopic, transformers, umap-learn, hdbscan, gensim
- Check GPU availability
- Print GPU info

**Output mong đợi:**
```
✅ Installation complete!
🔧 GPU available: True
   GPU name: Tesla T4
   GPU memory: 15.0 GB
```

---

### **CELL 2: 📊 Load & Explore Data** (30 giây)
**Mục đích:** Load data và explore

**Nội dung:**
- Read CSV từ Kaggle input
- Kiểm tra số lượng documents
- Print sample documents

**Output mong đợi:**
```
✅ Loaded 10,000 documents

Sample documents:
1. iphone 15 pro max camera tốt nhưng pin yếu...
2. samsung galaxy s23 ultra giá rẻ hơn...
3. xiaomi redmi note 12 pro đáng mua...
```

**NOTE:** Path data cần sửa tùy cách upload:
- Upload trực tiếp: `/kaggle/input/result.csv`
- Upload dataset: `/kaggle/input/[dataset-name]/result.csv`

---

### **CELL 3: 🧪 Test PhoBERT (Prototype)** (2-3 phút)
**Mục đích:** Test xem PhoBERT encode được không

**Nội dung:**
- Load PhoBERT
- Encode 10 documents thử
- Check embeddings shape

**Output mong đợi:**
```
Loading PhoBERT...
✅ Embeddings shape: (10, 768)
✅ PhoBERT works!
```

**Ý nghĩa:** Đây là bước PROTOTYPE - test nhanh trước khi viết class!

---

### **CELL 4: 🤖 Define BERTopic Class** (5 giây)
**Mục đích:** Define class để tái sử dụng

**Nội dung:**
- Class `VietnameseBERTopicModel`
- Methods: `__init__`, `fit`, `get_topic_info`, `calculate_coherence`, `save`
- Đầy đủ docstrings

**Output mong đợi:**
```
✅ BERTopic class defined!
```

**Ý nghĩa:** Đây là bước REFACTOR - tổ chức code thành class!

---

### **CELL 5: 🚀 Initialize Model** (2-3 phút)
**Mục đích:** Khởi tạo model với hyperparameters

**Nội dung:**
- Create model instance
- Config: n_neighbors=15, min_cluster_size=15, etc.
- Load PhoBERT, setup UMAP, HDBSCAN

**Output mong đợi:**
```
🚀 Using GPU: Tesla T4

[1/4] Loading vinai/phobert-base...
[2/4] Configuring UMAP...
[3/4] Configuring HDBSCAN...
[4/4] Building BERTopic pipeline...
✅ Model initialized!
```

---

### **CELL 6: 🏋️ Train Model** (20-40 phút)
**Mục đích:** Train BERTopic trên toàn bộ dataset

**Nội dung:**
- Call `model.fit(documents)`
- Track training time
- Display results

**Output mong đợi:**
```
TRAINING BERTOPIC
Documents: 10,000

[Encoding documents with PhoBERT...]
[Reducing dimensions with UMAP...]
[Clustering with HDBSCAN...]
[Extracting topics with c-TF-IDF...]

✅ TRAINING COMPLETED
Topics found: 12
Outliers: 150 (1.5%)

⏱️ Training time: 25.3 minutes
```

**NOTE:** Đây là cell lâu nhất!

---

### **CELL 7: 📊 Analyze Topics** (1 phút)
**Mục đích:** Xem topics đã train

**Nội dung:**
- Get topic info DataFrame
- Display top 5 topics
- Show top words per topic

**Output mong đợi:**
```
📊 TOPIC SUMMARY:

   Topic  Count                    Name
0     -1    150              -1_outlier
1      0    450  0_iphone_pro_max_camera
2      1    320     1_samsung_galaxy_note
3      2    280          2_xiaomi_redmi

🏆 TOP 5 TOPICS

Topic 0: 0_iphone_pro_max_camera
  Documents: 450
  Top words: iphone(0.425), pro(0.380), max(0.350), camera(0.312), apple(0.298)

Topic 1: 1_samsung_galaxy_note
  Documents: 320
  Top words: samsung(0.412), galaxy(0.365), note(0.340), s23(0.298), ultra(0.275)
```

---

### **CELL 8: 📈 Calculate Coherence** (10-20 phút) - OPTIONAL
**Mục đích:** Tính coherence score để đánh giá chất lượng

**Nội dung:**
- Call `model.calculate_coherence(documents)`
- Using Gensim CoherenceModel
- Display score

**Output mong đợi:**
```
📈 Calculating coherence score...

Calculating coherence (c_v)...
✅ Coherence C_V: 0.4123
```

**NOTE:** 
- Cell này có thể bỏ qua nếu muốn nhanh!
- Sử dụng `processes=1` để tránh lỗi multiprocessing trên Kaggle

---

### **CELL 9: 💾 Save Results** (1 phút)
**Mục đích:** Save model và results

**Nội dung:**
- Save topic info to CSV
- Save model files
- Save summary.pkl
- List output files

**Output mong đợi:**
```
✅ Saved: bertopic_topics.csv
✅ Saved: bertopic_model/
✅ Saved: summary.pkl

💾 ALL RESULTS SAVED!

📂 Files in /kaggle/working/output/:
total 125M
bertopic_topics.csv       12K
bertopic_model/          124M
summary.pkl               2K
```

**Files structure:**
```
/kaggle/working/output/
├── bertopic_topics.csv          # Topic summary
├── bertopic_model/              # Model files
│   ├── bertopic_model/
│   ├── config.pkl
│   └── topics.pkl
└── summary.pkl                  # Training summary
```

---

## 📊 WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────┐
│ CELL 1: Setup                                           │
│ Install packages → Check GPU                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 2: Load Data                                       │
│ Read CSV → Explore documents                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 3: Test PhoBERT (PROTOTYPE)                        │
│ Load PhoBERT → Encode 10 docs → Verify works           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 4: Define Class (REFACTOR)                         │
│ Class VietnameseBERTopicModel with all methods          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 5: Initialize Model                                │
│ Create instance → Load PhoBERT → Setup UMAP/HDBSCAN    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 6: Train Model ⏰ (20-40 phút)                     │
│ Encode all docs → Reduce dims → Cluster → Extract      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 7: Analyze Topics                                  │
│ Get topic info → Display top 5 → Show words            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 8: Calculate Coherence (OPTIONAL) ⏰ (10-20 phút) │
│ Gensim CoherenceModel → C_V score                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CELL 9: Save Results                                    │
│ Save CSV + Model + Summary → Download from Output tab  │
└─────────────────────────────────────────────────────────┘
```

---

## ⏱️ TIMELINE

```
00:00 - 00:05   Cell 1: Setup                    ☕
00:05 - 00:06   Cell 2: Load data                
00:06 - 00:09   Cell 3: Test PhoBERT             ☕
00:09 - 00:09   Cell 4: Define class             
00:09 - 00:12   Cell 5: Initialize model         ☕
00:12 - 00:52   Cell 6: Train model              ☕☕☕ (LONGEST!)
00:52 - 00:53   Cell 7: Analyze topics           
00:53 - 01:13   Cell 8: Coherence (optional)     ☕☕
01:13 - 01:14   Cell 9: Save results             

TOTAL: ~35-75 minutes (tùy có chạy Cell 8 không)
```

---

## 🎯 SO SÁNH VỚI LOCAL

| Aspect | Local | Kaggle |
|--------|-------|--------|
| **GPU** | RTX 3060 (6GB) | Tesla T4 (15GB) |
| **RAM** | 16GB | 30GB |
| **Speed** | ❌ Crash | ✅ Chạy mượt |
| **Cost** | Free | Free (30h/week) |
| **Setup** | Phức tạp | Đơn giản |
| **Output** | Local disk | Download sau |

**Kết luận:** Kaggle tốt hơn cho dataset lớn!

---

## 📝 DESIGN PRINCIPLES

### 1. Iterative Development
```
Prototype (Cell 3) → Refactor (Cell 4) → Production (Cell 6)
```

### 2. Modularity
Mỗi cell làm 1 việc rõ ràng, dễ debug

### 3. Progress Tracking
Print progress ở mỗi bước quan trọng

### 4. Error Handling
```python
# GPU check
if torch.cuda.is_available():
    print("✅ GPU OK")
else:
    print("⚠️  No GPU")
```

### 5. Clear Output
```
✅ Success messages
⚠️  Warnings
❌ Errors (if any)
```

---

## 🔧 CUSTOMIZATION

### Muốn train nhanh hơn?

**Cell 2 - Sample data:**
```python
documents = documents[:5000]  # Chỉ lấy 5K docs
```

**Cell 5 - Giảm params:**
```python
model = VietnameseBERTopicModel(
    min_cluster_size=10,  # Giảm từ 15
    n_components=3         # Giảm từ 5
)
```

**Cell 8 - Bỏ qua:**
Không chạy cell này!

---

### Muốn nhiều topics hơn?

**Cell 5 - Tăng sensitivity:**
```python
model = VietnameseBERTopicModel(
    min_cluster_size=10,   # Giảm
    n_neighbors=30         # Tăng
)
```

---

## 🚨 COMMON ERRORS & FIXES

### Error 1: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Fix:**
- Cell 2: Sample nhỏ hơn `documents[:5000]`
- Cell 5: Giảm `min_cluster_size`
- Settings: Restart notebook

---

### Error 2: File Not Found
```
FileNotFoundError: result.csv
```

**Fix:**
Cell 2 - Check path:
```python
!ls /kaggle/input/  # List files
# Sửa path cho đúng
```

---

### Error 3: No GPU
```
GPU available: False
```

**Fix:**
- Settings → Accelerator → GPU T4 x2
- Save → Restart notebook

---

## 📥 DOWNLOAD & USE

### Sau khi train xong:

1. **Download từ Kaggle:**
   - Output tab → Download All
   - Hoặc download từng file

2. **Copy vào local project:**
   ```bash
   mkdir output/bertopic_model_kaggle
   cp -r bertopic_model/* output/bertopic_model_kaggle/
   cp bertopic_topics.csv output/
   ```

3. **Load và sử dụng:**
   ```python
   from models.bertopic_model import VietnameseBERTopicModel
   
   model = VietnameseBERTopicModel.load("output/bertopic_model_kaggle/")
   new_topics = model.transform(["iphone 15 pro max"])
   ```

---

## ✅ CHECKLIST

Trước khi chạy:
- [ ] Đăng nhập Kaggle
- [ ] Bật GPU (Settings → GPU T4 x2)
- [ ] Upload data (result.csv)
- [ ] Import notebook hoặc copy code

Trong khi chạy:
- [ ] Cell 1: Packages installed OK
- [ ] Cell 2: Data loaded OK
- [ ] Cell 3: PhoBERT works
- [ ] Cell 4: Class defined
- [ ] Cell 5: Model initialized
- [ ] Cell 6: Training completed
- [ ] Cell 7: Topics look good
- [ ] Cell 8: Coherence calculated (optional)
- [ ] Cell 9: Results saved

Sau khi xong:
- [ ] Download all files
- [ ] Copy vào local project
- [ ] Test model loaded OK

---

## 📚 REFERENCES

### Notebook design inspired by:
- **Prototype → Refactor → Production** pattern
- **Kaggle best practices**
- **Jupyter notebook conventions**

### Libraries:
- BERTopic: https://maartengr.github.io/BERTopic/
- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- UMAP: https://umap-learn.readthedocs.io/
- HDBSCAN: https://hdbscan.readthedocs.io/

---

**Created:** 2026-04-04  
**Author:** Member 3 (ML Engineer)  
**Task:** 3.1 - BERTopic Implementation  
**Platform:** Kaggle GPU Training
