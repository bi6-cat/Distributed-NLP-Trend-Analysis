# Nghiên Cứu So Sánh BERTopic, LDA và TF-IDF trong Phân Tích Xu Hướng Mạng Xã Hội Tiếng Việt



---

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Bối Cảnh và Động Lực Nghiên Cứu](#2-bối-cảnh-và-động-lực-nghiên-cứu)
3. [Đặc Thù Ngôn Ngữ Tiếng Việt trong NLP](#3-đặc-thù-ngôn-ngữ-tiếng-việt-trong-nlp)
4. [Phương Pháp TF-IDF](#4-phương-pháp-tf-idf)
5. [Phương Pháp LDA (Latent Dirichlet Allocation)](#5-phương-pháp-lda)
6. [Phương Pháp BERTopic](#6-phương-pháp-bertopic)
7. [Kết Luận và Hướng Phát Triển](#8-kết-luận)
8. [Tài Liệu Tham Khảo](#9-tài-liệu-tham-khảo)

---

## 1. Tổng Quan

Phân tích xu hướng mạng xã hội (Social Media Trend Analysis) là một bài toán cốt lõi trong hệ thống lắng nghe xã hội (Social Listening). Đặc biệt với tiếng Việt — ngôn ngữ có hơn 97 triệu người dùng và lượng nội dung mạng xã hội khổng lồ — việc lựa chọn thuật toán Topic Modeling phù hợp có tác động trực tiếp đến chất lượng insight thu được.

Nghiên cứu này phân tích chuyên sâu ba phương pháp chính:

| Phương pháp | Loại | Độ phức tạp | Phù hợp với |
|---|---|---|---|
| **TF-IDF** | Statistical | Thấp | Keyword extraction, ranking |
| **LDA** | Probabilistic Generative | Trung bình | Topic discovery truyền thống |
| **BERTopic** | Neural + Clustering | Cao | Context-aware topic modeling |

### Câu Hỏi Nghiên Cứu

1. Phương pháp nào cho kết quả topic coherence cao nhất trên dữ liệu mạng xã hội tiếng Việt?
2. Làm thế nào để tối ưu hóa từng phương pháp cho đặc thù ngôn ngữ tiếng Việt?
3. Khi nào nên kết hợp các phương pháp thay vì dùng đơn lẻ?

---

## 2. Bối Cảnh và Động Lực Nghiên Cứu

### 2.1 Thị Trường Mạng Xã Hội Việt Nam

- **Facebook:** ~75 triệu người dùng tại Việt Nam (2024)
- **TikTok:** >40 triệu người dùng, tốc độ tăng trưởng nội dung +200%/năm
- **Zalo:** Nền tảng đặc thù, hơn 77 triệu tài khoản với văn phong riêng
- **Twitter/X, YouTube Comments:** Nguồn dữ liệu phong phú về tech, entertainment

### 2.2 Thách Thức Đặc Thù

```
Dữ liệu mạng xã hội tiếng Việt:
  ├── Ngôn ngữ không chuẩn (teen code, viết tắt, lỗi chính tả)
  ├── Code-switching (trộn Anh-Việt): "Cái này quá lit, không có gì bằng"
  ├── Emoji và emoticon tích hợp vào ngữ nghĩa
  ├── Dấu thanh bị bỏ (không dấu): "toi di an com" = "tôi đi ăn cơm"
  └── Phương ngữ: Nam/Bắc/Trung với từ vựng khác nhau
```

---

## 3. Đặc Thù Ngôn Ngữ Tiếng Việt trong NLP

### 3.1 Hệ Thống Thanh Điệu

Tiếng Việt có **6 thanh điệu** tạo ra từ hoàn toàn khác nhau từ cùng một chuỗi ký tự gốc:

```
ma → ma (ghost) | má (mother) | mà (but) | mả (grave) | mã (code) | mạ (rice seedling)
```

**Hệ quả trong NLP:**
- Bỏ dấu làm mất hoàn toàn ngữ nghĩa → cần restoration model trước khi xử lý
- Tokenization nhạy cảm với dấu → lỗi dấu = lỗi token

### 3.2 Vấn Đề Word Segmentation

Tiếng Việt là ngôn ngữ **phân tích tính** (analytic language), từ đa tiết được viết tách rời bằng dấu cách:

```
"học sinh" (student) ≠ "học" + "sinh" (study + birth/live)
"công ty" (company) ≠ "công" + "ty" (work + ?)
"bình thường" (normal) ≠ "bình" + "thường" (flat + often)
```

**Công cụ Word Segmentation được dùng phổ biến:**

| Tool | Độ chính xác (F1) | Tốc độ | Ghi chú |
|---|---|---|---|
| `underthesea` | ~97% | Trung bình | Python-native, phổ biến nhất |
| `VnCoreNLP` | ~97.5% | Nhanh | Java-based, cần JVM |
| `RDRsegmenter` | ~96% | Rất nhanh | Rule-based, nhẹ |
| `pyvi` | ~94% | Nhanh | Pure Python, dễ dùng |

### 3.3 Stopwords Tiếng Việt

Stopwords tiếng Việt có đặc điểm:
- Nhiều hư từ (function words): "và", "hoặc", "của", "trong", "với"...
- Các từ mang nghĩa trong ngữ cảnh nhất định nhưng cần loại ở context khác
- Cần custom stopword list theo domain (mạng xã hội vs. báo chí vs. pháp lý)

---

## 4. Phương Pháp TF-IDF

### 4.1 Lý Thuyết

**TF-IDF (Term Frequency - Inverse Document Frequency)** là phương pháp thống kê đo lường tầm quan trọng tương đối của một từ trong tài liệu so với toàn bộ corpus.

#### Công Thức

$$TF(t, d) = \frac{\text{Số lần từ } t \text{ xuất hiện trong tài liệu } d}{\text{Tổng số từ trong tài liệu } d}$$

$$IDF(t, D) = \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)$$

$$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$

#### Biến thể BM25 (Tốt hơn cho Short Text)

Dữ liệu mạng xã hội thường ngắn (tweet, comment), BM25 phù hợp hơn TF-IDF thuần túy:

$$BM25(t, d) = IDF(t) \cdot \frac{TF(t,d) \cdot (k_1 + 1)}{TF(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

Với `k1 ∈ [1.2, 2.0]` và `b = 0.75` cho văn bản mạng xã hội tiếng Việt.

### 4.2 Ứng Dụng trong Phân Tích Xu Hướng

TF-IDF không trực tiếp phát hiện topic, nhưng được dùng cho:

1. **Keyword Extraction per Time Window** — Tìm từ khóa xu hướng theo khung thời gian
2. **Feature Engineering cho LDA/BERTopic** — Làm đầu vào hoặc post-processing
3. **Topic Labeling** — Gán nhãn tự động cho cluster sau khi clustering

### 4.3 Điểm Mạnh và Hạn Chế

| Tiêu chí | TF-IDF |
|---|---|
| ✅ Tốc độ xử lý | Rất nhanh — xử lý hàng triệu docs/giây |
| ✅ Interpretability | Hoàn toàn minh bạch, dễ debug |
| ✅ Tài nguyên | CPU-only, memory thấp |
| ✅ Không cần training | Unsupervised, domain-agnostic |
| ❌ Semantic understanding | Không hiểu nghĩa, "điện thoại" ≠ "smartphone" |
| ❌ Context-blind | "ngân hàng" (river bank) = "ngân hàng" (financial bank) |
| ❌ Topic coherence | Không đảm bảo các keyword thuộc cùng một chủ đề |
| ❌ Short text | Hiệu suất giảm với văn bản rất ngắn |

---

## 5. Phương Pháp LDA

### 5.1 Lý Thuyết Generative Model

**LDA (Latent Dirichlet Allocation)** — Blei, Ng & Jordan (2003) — là mô hình sinh xác suất (probabilistic generative model). LDA giả định mỗi tài liệu là **hỗn hợp của các topic**, và mỗi topic là **phân phối trên từ vựng**.

#### Quá Trình Sinh (Generative Process)

```
Với mỗi topic k ∈ {1,...,K}:
    φ_k ~ Dirichlet(β)          # Phân phối từ của topic k

Với mỗi document d:
    θ_d ~ Dirichlet(α)          # Phân phối topic của document d
    Với mỗi từ w_n trong d:
        z_n ~ Multinomial(θ_d)  # Chọn topic
        w_n ~ Multinomial(φ_z_n) # Sinh từ từ topic đó
```

#### Phân Phối Dirichlet — Trực Giác

```
α nhỏ (< 1): Documents tập trung vào ít topic → topic chuyên biệt
α lớn (> 1): Documents trải đều trên nhiều topic → topic tổng quát

β nhỏ: Topics tập trung vào ít từ → topic rõ ràng, sắc nét
β lớn: Topics trải đều trên nhiều từ → topic mờ nhạt
```

**Khuyến nghị cho mạng xã hội tiếng Việt:**
- `α = 50/K` (heuristic Griffiths & Steyvers)
- `β = 0.01` — topic sắc nét hơn cho short text

### 5.2 Inference: Collapsed Gibbs Sampling

Trong thực tế, dùng **Collapsed Gibbs Sampling** để ước tính phân phối hậu nghiệm:

$$P(z_i = k \mid \mathbf{z}_{-i}, \mathbf{w}) \propto \frac{n_{d,k}^{-i} + \alpha}{\sum_{k'} n_{d,k'}^{-i} + K\alpha} \cdot \frac{n_{k,w_i}^{-i} + \beta}{\sum_{v} n_{k,v}^{-i} + V\beta}$$

Với `n_{d,k}` là số từ trong document `d` được gán cho topic `k`.

### 5.3 Thách Thức của LDA với Văn Bản Mạng Xã Hội

```
Vấn đề cốt lõi:

LDA giả định:  Văn bản DÀI với nhiều từ
Thực tế:       Comment Facebook trung bình chỉ 15-30 từ

→ Giải pháp: Aggregation Strategies
  ├── User-level: Gộp tất cả posts của 1 user thành 1 document
  ├── Thread-level: Gộp post + toàn bộ comments
  ├── Temporal: Gộp posts theo khung giờ/ngày
  └── Hashtag-level: Gộp posts có cùng hashtag
```

### 5.4 Điểm Mạnh và Hạn Chế

| Tiêu chí | LDA |
|---|---|
| ✅ Probabilistic framework | Có độ đo xác suất rõ ràng |
| ✅ Soft assignment | Một document có thể thuộc nhiều topics |
| ✅ Scalable | Có thể xử lý corpus hàng triệu documents |
| ✅ Mature ecosystem | Gensim, Mallet, Spark LDA |
| ❌ Bag-of-words | Mất thứ tự từ, mất context |
| ❌ Short text | Hiệu suất giảm mạnh |
| ❌ Hyperparameter tuning | Cần điều chỉnh K, α, β |
| ❌ Static model | Không tự cập nhật với data mới |

---

## 6. Phương Pháp BERTopic

### 6.1 Kiến Trúc Tổng Quan

**BERTopic** (Grootendorst, 2022) kết hợp ba thành phần chính:

```
Input Documents
      │
      ▼
┌─────────────────────────────────┐
│  1. SENTENCE TRANSFORMER        │
│     Tạo dense embeddings        │
│     (semantic representation)   │
└────────────────┬────────────────┘
                 │ 768-dim vectors
                 ▼
┌─────────────────────────────────┐
│  2. UMAP                        │
│     Giảm chiều xuống 5-50 dim   │
│     (giữ cấu trúc topo học)     │
└────────────────┬────────────────┘
                 │ 5-dim vectors
                 ▼
┌─────────────────────────────────┐
│  3. HDBSCAN                     │
│     Clustering dày đặc          │
│     (không cần biết K)          │
└────────────────┬────────────────┘
                 │ cluster labels
                 ▼
┌─────────────────────────────────┐
│  4. c-TF-IDF                    │
│     Topic Representation        │
│     (class-based TF-IDF)        │
└────────────────┬────────────────┘
                 │
                 ▼
           Topic Labels + Keywords
```

### 6.2 Class-based TF-IDF (c-TF-IDF)

BERTopic dùng c-TF-IDF để tạo topic representation sau khi clustering:

$$ctfidf_{t,c} = \frac{f_{t,c}}{|c|} \times \log\left(1 + \frac{A}{\sum_j f_{t,j}}\right)$$

Với:
- `f_{t,c}`: Tần suất từ `t` trong class (cluster) `c`
- `|c|`: Tổng số từ trong class `c`
- `A`: Tổng số từ trên tất cả classes

### 6.3 Lựa Chọn Embedding Model cho Tiếng Việt

| Model | Loại | Ghi chú |
|---|---|---|
| `vinai/phobert-base` | Monolingual | Tốt nhất cho tiếng Việt, 135M params |
| `vinai/phobert-large` | Monolingual | Chất lượng cao nhất, 370M params |
| `intfloat/multilingual-e5-large` | Multilingual | Hỗ trợ đa ngôn ngữ mạnh |
| `paraphrase-multilingual-mpnet-base-v2` | Multilingual | Tốt cho semantic similarity |
| `keepitreal/vietnamese-sbert` | Vietnamese | Fine-tuned cho sentence similarity |

### 6.4 Chiến Lược Xử Lý Outlier

Trong dữ liệu mạng xã hội, HDBSCAN thường tạo ra 20-40% outlier (topic = -1). Các chiến lược xử lý:

| Strategy | Mô tả | Khi nào dùng |
|---|---|---|
| `probabilities` | Gán về topic có xác suất cao nhất | Ưu tiên dùng trước |
| `c-tf-idf` | Dùng cosine similarity với topic vectors | Khi không có probs |
| `embeddings` | Dùng sentence embedding similarity | Khi cần precision cao |
| Giữ nguyên | Gán label "General" | Khi muốn biết docs ngoài topic |

### 6.5 Điểm Mạnh và Hạn Chế

| Tiêu chí | BERTopic |
|---|---|
| ✅ Semantic understanding | Hiểu nghĩa thực sự của văn bản |
| ✅ Short text | Hoạt động tốt với comment ngắn |
| ✅ Không cần chọn K | HDBSCAN tự xác định số clusters |
| ✅ Dynamic topics | Hỗ trợ temporal analysis tích hợp |
| ✅ Multilingual | PhoBERT được pre-trained trên tiếng Việt |
| ❌ Tài nguyên | Cần GPU, RAM lớn (minimum 8GB) |
| ❌ Tốc độ | Chậm hơn LDA 10-100x |
| ❌ Embedding cost | Đắt với corpus cực lớn (>10M docs) |
| ❌ Reproducibility | UMAP/HDBSCAN có tính stochastic |

---


## 7. Khi Nào Dùng Phương Pháp Nào

```
Use Case Matrix:

Yêu cầu                          | Khuyến nghị
─────────────────────────────────|──────────────────────────
Real-time alert (< 1 phút)       | TF-IDF với BM25
Keyword monitoring dashboard      | TF-IDF + temporal window
Topic discovery (corpus mới)      | BERTopic trước, LDA verify
Large corpus (>5M docs)           | LDA (online learning)
Short text (comment, tweet)       | BERTopic > LDA >> TF-IDF
Long articles/threads             | LDA ≈ BERTopic > TF-IDF
Multilingual content (Anh-Việt)  | BERTopic (multilingual-E5)
Interpretable cho stakeholder     | LDA > BERTopic >> TF-IDF
Production với resource hạn chế  | LDA hoặc TF-IDF
Research/accuracy là ưu tiên      | BERTopic
```

---


## 8. Kết Luận

### 8.1 Tóm Tắt Phát Hiện Chính

1. **BERTopic vượt trội** về chất lượng topic nhờ sử dụng PhoBERT captures semantic meaning của tiếng Việt.

2. **LDA là lựa chọn cân bằng** giữa quality và efficiency — phù hợp cho production system với resource hạn chế và cần interpretability.

3. **TF-IDF không thể thay thế** cho real-time applications — không có phương pháp nào cho tốc độ tương đương với độ trễ < 1ms.

4. **Word segmentation là bottleneck** quan trọng nhất — lỗi ở bước này lan truyền qua toàn bộ pipeline.


## 9. Tài Liệu Tham Khảo

### Papers Gốc

1. **Blei, D.M., Ng, A.Y., & Jordan, M.I.** (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022.

2. **Grootendorst, M.** (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

3. **Devlin, J., Chang, M.W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.

4. **Nguyen, D.Q., Nguyen, A.T.** (2020). PhoBERT: Pre-trained language models for Vietnamese. *EMNLP 2020 Findings*.

5. **Robertson, S., & Zaragoza, H.** (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4).

### Vietnamese NLP Resources

6. **Nguyen, T.L., et al.** (2019). VnCoreNLP: A Vietnamese Natural Language Processing Toolkit. *NAACL 2019 Demo*.

7. **Nguyen, P.T., et al.** (2023). Vietnamese Social Media Text Analysis: Challenges and Approaches. *VLSP Workshop 2023*.

### Công Cụ và Thư Viện

8. **underthesea** — Vietnamese NLP Library: https://github.com/undertheseanlp/underthesea

9. **BERTopic Documentation**: https://maartengr.github.io/BERTopic/

10. **PhoBERT Model Hub**: https://huggingface.co/vinai/phobert-base

11. **Gensim LDA Documentation**: https://radimrehurek.com/gensim/models/ldamodel.html
