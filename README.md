# Vietnamese Tech Trend & Controversy Radar
### Hệ thống phân tích xu hướng & dư luận mạng xã hội Việt Nam — Big Data + NLP trên HPC Cluster

> **Trạng thái:** 🟢 Phase 1 — Thiết kế & Thu thập dữ liệu

---

## Chúng ta đang xây dựng gì?

> **Một câu:** Hệ thống tự động thu thập hàng trăm nghìn bài đăng tiếng Việt update mỗi ngày, phân tích cảm xúc và chủ đề bằng AI, rồi hiển thị lên dashboard để bất kỳ ai cũng thấy được **"đang có chuyện gì hot trên mạng?"** và **"dư luận đang nghĩ gì?"**

**Sản phẩm cuối cùng trông như thế nào:**
- Một **web dashboard** mở lên là thấy top các chủ đề đang trending hôm nay trên VOZ, tinhte, VnExpress, YouTube
- Mỗi chủ đề có **biểu đồ cảm xúc** (tích cực / tiêu cực / trung lập) theo thời gian
- **Cảnh báo tự động** khi có "khủng hoảng dư luận" — ví dụ một sản phẩm công nghệ bị chỉ trích hoặc bàn tán ồ ạt trong 1 khoảng thời gian ngắn

---

## Định nghĩa "Thành công" (Definition of Done)

Dự án coi là **hoàn thành** khi đáp ứng đủ các tiêu chí sau:

| # | Tiêu chí
|---|---|
| 1 | Pipeline chạy **end-to-end tự động** mỗi ngày không cần can thiệp thủ công 
| 2 | Xử lý được **≥ 500K records** trên cluster phân tán 
| 3 | Sentiment model đạt **Macro F1 ≥ 0.75** trên tiếng Việt 
| 4 | Dashboard hiển thị **dữ liệu thật**, cập nhật hàng ngày 
| 5 | Có **benchmark** chứng minh hệ thống phân tán nhanh hơn 1 máy đơn 
| 6 | **Thuật toán CS246** được triển khai thực sự (không phải chỉ mô tả)

---

## Mỗi người đóng góp gì vào sản phẩm?

```
Member 1 (Data Engineer)   →  Thu thập bài đăng từ VOZ, VnExpress, YouTube

Member 2 (DevOps/Infra)    →  Dựng HPC cluster, Spark, HDFS bằng Ansible

Member 3 (ML Engineer)     →  Trả lời câu hỏi: "Người ta đang nói về chủ đề gì?"
                               LDA + BERTopic + Count-Min Sketch

Member 4 (NLP Engineer)    →  Trả lời câu hỏi: "Người ta đang cảm thấy thế nào?"
                               PhoBERT Sentiment + Crisis Detection

Member 5 (Full-stack)      →  Biến tất cả thành thứ người thường nhìn vào hiểu được
                               Trend Scoring + Streamlit Dashboard
```

---

## Kiến trúc hệ thống

```
[VOZ · VnExpress · YouTube]
          │  crawl 
          ▼
    MongoDB (raw JSON)
          │  export
          ▼
     HDFS (Parquet)
          │  Spark reads
          ▼
  Apache Spark 3.5 (PySpark)
   ├── LSH/MinHash  → loại bỏ trùng lặp
   ├── PhoBERT      → gán nhãn cảm xúc
   └── BERTopic/LDA → phân cụm chủ đề
          │  result
          ▼
     PostgreSQL (structured results)
          │  query
          ▼
   Streamlit Dashboard  ← user xem ở đây
```

> **Quản lý hạ tầng:** Ansible tự động cấu hình toàn bộ cluster
> **Lên lịch pipeline:** Apache Airflow chạy tự động

---

## Các thuật toán CS246 phải triển khai

Đây là **yêu cầu bắt buộc** của môn học — phải có code thực sự, không chỉ mô tả:

| Thuật toán | Dùng để làm gì trong dự án | Member phụ trách |
|---|---|---|
| **MinHash + LSH** | Loại bỏ bài đăng trùng lặp / gần giống nhau | M2 |
| **Count-Min Sketch** | Đếm tần suất keyword mà không dùng hết RAM | M3 |

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|---|---|
| [`TECH_STACK.md`](TECH_STACK.md) | Tech stack, kiến trúc từng layer, code mẫu, requirements |
| [`TEAM_ASSIGNMENTS.md`](TEAM_ASSIGNMENTS.md) | Phân công tasks chi tiết theo từng tuần cho 5 thành viên |

---

## Lộ trình 4 Phases

| Phase | Mục tiêu cần đạt được |
|---|---|
| **Phase 1** | Cluster chạy được, crawl được data, schema DB xong |
| **Phase 2** | LSH dedup chạy, LDA ra topics, PhoBERT F1 ≥ 0.75, TrendScore tính được |
| **Phase 3** | Full pipeline end-to-end, dashboard kết nối live DB, crisis detection hoạt động |
| **Phase 4** | Benchmark hoàn chỉnh, báo cáo từng người viết xong, demo live |

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/<org>/Distributed-NLP-Trend-Analysis.git
cd Distributed-NLP-Trend-Analysis

# 2. Đọc tài liệu theo thứ tự này:
#    README.md (file này)  →  TEAM_ASSIGNMENTS.md  →  TECH_STACK.md

# 3. Setup môi trường local để dev
docker-compose up -d          # Khởi động MongoDB + PostgreSQL local

conda create -n nlp-trend python=3.10
conda activate nlp-trend
pip install -r requirements.txt

# 4. Liên hệ Member 2 để được cấp quyền truy cập HPC cluster
```