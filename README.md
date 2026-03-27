# 📡 Vietnamese Tech Trend & Controversy Radar
### Hệ thống phân tích xu hướng & dư luận mạng xã hội Việt Nam — Big Data + NLP trên HPC Cluster

[![Status](https://img.shields.io/badge/Status-🟢_Phase_1:_Infrastructure_&_Ingestion-brightgreen)](#-lộ-trình-phát-triển-roadmap)
[![Tech](https://img.shields.io/badge/Stack-Spark_|_HDFS_|_ClickHouse_|_dbt-blue)](#-công-nghệ-cốt-lõi-tech-stack)

---

## 📌 Tổng Quan Dự Án
Hệ thống tự động thu thập hàng trăm nghìn tương tác mỗi ngày từ các cộng đồng công nghệ lớn nhất Việt Nam (VOZ, Tinhte, YouTube), xử lý phân tán để nhận diện xu hướng, phân tích cảm xúc và phát hiện sớm các cuộc khủng hoảng dư luận bằng AI.

---

## 💎 Giá Trị Thực Tiễn & Khả Năng Thực Tế
Dự án được thiết kế như một nền tảng **Social Listening** cấp doanh nghiệp với 5 năng lực cốt lõi:

*   **Phát Hiện Khủng Hoảng (Crisis Radar):** Tự động nhận diện các "spike" tiêu cực (ví dụ: lỗi sản phẩm bị chỉ trích ồ ạt) bằng thuật toán *Anomaly Detection* (Isolation Forest), giúp doanh nghiệp phản ứng trong "giờ vàng".
*   **Khử Nhiễu & Spam (Anti-Seeding):** Sử dụng thuật toán **MinHash + LSH** trên cụm Spark để gộp hàng vạn bình luận trùng lặp từ các nick ảo (seeding), trả lại con số thảo luận thật sự.
*   **Đọc Vị Cộng Đồng (Topic Modeling):** Tự động gom nhóm các bình luận bằng từ lóng (teencode) qua **LDA/BERTopic** để xác định chính xác tâm điểm thảo luận ("liệt cảm ứng", "sọc màn hình"...) mà không cần nhập từ khóa thủ công.
*   **Nhiệt Kế Cảm Xúc (Sentiment Scoring):** Chấm điểm thái độ (Khen/Chê/Trung lập) bằng mô hình **PhoBERT** đã được fine-tune chuyên sâu cho ngữ cảnh tiếng Việt.
*   **Chấm Điểm "Sức Nóng" (Trend Matrix):** Xếp hạng xu hướng dựa trên ma trận toán học gồm Vận tốc (Velocity), Gia tốc (Acceleration) và Mức độ lan tỏa (Engagement).

---

## 🏗️ Kiến Trúc Hệ Thống (Distributed Architecture)

Hệ thống vận hành theo kiến trúc 7 lớp phân tách trên cụm **VirtualBox/HPC Cluster**:

```text
[ Sources: VOZ, Tinhte, YouTube,... ]
             │
             ▼ (Crawl)
     [ HDFS: Raw JSONL ]
             │
             ▼ (PySpark Clean & Dedup LSH)
    [ HDFS: Staged Parquet ]
             │
             ▼ (Distributed NLP & Topic)
 [ ClickHouse: Staging Tables ]
             │
             ▼ (dbt Transformations)
  [ ClickHouse: Analysts Marts ]
             │
             ▼ (Real-time Query)
    [ Streamlit Dashboard ]

---------------------------------------------------
Quản lý (Airflow): Lên lịch & Điều phối toàn pipeline
Hạ tầng (Ansible): Tự động cấu hình Spark/HDFS Cluster
```

---

## ⚙️ Các Thuật Toán Cốt Lõi (CS246 Requirements)

Dự án triển khai thực tế 3 thuật toán trọng tâm của môn học:
1.  **MinHash + LSH:** Xử lý bài toán Near-Duplicate Detection trên hàng triệu bản ghi.
2.  **Count-Min Sketch:** Đếm tần suất keyword theo cửa sổ thời gian (Sliding Window) với bộ nhớ tối thiểu.
3.  **PageRank / HITS:** Xác định các "Influencers" và "Authorities" trong mạng lưới thảo luận (User Interaction Graph).

---

## 🛠️ Công Nghệ Cốt Lõi (Tech Stack)

| Layer | Technology |
|---|---|
| **Ingestion** | Python (Requests, BS4, Selenium, YouTube API), Airflow |
| **Storage** | **HDFS** (Raw & Staged files), **ClickHouse** (OLAP Data Warehouse) |
| **Processing** | **Apache Spark 3.5** (Distributed Computing) |
| **Analytics** | **dbt** (SQL modeling), Pydantic (Validation) |
| **NLP/ML** | **PhoBERT** (Sentiment), BERTopic/LDA (Topic), Isolation Forest (Anomaly) |
| **Visuals** | **Streamlit**, Plotly, NetworkX |
| **Infra** | **Ansible** (Configuration Management), Docker (Local Dev) |

---

## 👥 Thành Viên & Trách Nhiệm

*   **Member 1 (Data Engineer):** Ingestion Layer — Xây dựng crawlers & write trực tiếp HDFS.
*   **Member 2 (DevOps/Lead):** Infra Layer & Big Data Core — Quản lý Cluster (Ansible), Spark Cleaning, thuật toán LSH/MinHash.
*   **Member 3 (ML Engineer):** Analysis Layer — Topic Modeling (LDA/BERTopic) & Thuật toán Count-Min Sketch.
*   **Member 4 (NLP Engineer):** AI Layer — Sentiment Analysis (PhoBERT) & Crisis Detection (Isolation Forest).
*   **Member 5 (Full-stack):** Visualization Layer — dbt transformation, Trend Score logic & Dashboard UI.

---

## 🗺️ Lộ Trình Phát Triển (Roadmap)

1.  **Phase 1 (Tuần 1-2):** Dựng cụm Cluster (Spark/HDFS/ClickHouse), hoàn thiện Crawlers & Pydantic validation.
2.  **Phase 2 (Tuần 3-4):** Triển khai Spark Clean Job, LSH Dedup, Fine-tune PhoBERT & LDA prototype.
3.  **Phase 3 (Tuần 5-6):** Tích hợp end-to-end pipeline qua Airflow, dbt Marts & Dashboard live data.
4.  **Phase 4 (Tuần 7-8):** Benchmarking hiệu năng (Strong/Weak Scaling), hoàn thiện báo cáo & Demo.

---

## 🚀 Quick Start (Cho Developer)

Dự án đã được đóng gói sẵn các lệnh quản trị qua `Makefile`:

```bash
# 1. Clone dự án
git clone https://github.com/<org>/Distributed-NLP-Trend-Analysis.git

# 2. Khởi động môi trường Dev (ClickHouse + Airflow local)
make dev

# 3. Cài đặt thư viện Python (Mọi thành viên)
make install

# 4. Kiểm tra code (Lint & Test)
make lint
make test

# 5. Cài đặt cụm VMs qua Ansible (Dành cho Member 2)
cd ansible && ansible-playbook -i inventory/local_vms.ini playbooks/site.yml
```

---
> **Lưu Ý:** Xem chi tiết bàn giao tại [HANDOFF_LOG.md](HANDOFF_LOG.md) hoặc cấu hình chi tiết tại [TECH_STACK.md](TECH_STACK.md).