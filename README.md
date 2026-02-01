**Vietnamese Social Media Trend & Controversy Analysis System on HPC Cluster**

## 📝 Giới thiệu dự án
Dự án tập trung vào việc khai phá dữ liệu quy mô lớn (Big Data) từ các cộng đồng trực tuyến Việt Nam (Facebook, Voz, Tinh tế). Mục tiêu là xây dựng một hệ thống có khả năng phát hiện xu hướng (trending) và phân tích dư luận (sentiment) bằng cách tận dụng sức mạnh của tính toán song song trên cụm **HPC**.

> **Trạng thái:** 🟢 Giai đoạn Thiết kế & Thu thập dữ liệu (Phase 1).

## 🎯 Mục tiêu kỹ thuật (CS246-based)
Dự án được thiết kế để áp dụng và thực chứng các kỹ thuật:
* **Shingling & LSH**: Phát hiện và loại bỏ các nội dung trùng lặp (near-duplicates) từ hàng triệu bài đăng.
* **PageRank/HITS**: Xác định các cá nhân/nguồn tin có sức ảnh hưởng (Authority) trong mạng lưới thảo luận.
* **Streaming Algorithms**: Sử dụng *Count-Min Sketch* để đếm tần suất từ khóa theo thời gian thực mà không làm tràn bộ nhớ.
* **Distributed NLP**: Triển khai mô hình Sentiment Analysis (PhoBERT) song song trên các Worker nodes để tối ưu hiệu năng.

## 🏗 Kiến trúc hệ thống dự kiến
Hệ thống sẽ được vận hành trên cụm **HPC Semi-Lab** hiện có của nhóm:
* **Infrastructure**: Quản lý và cấu hình tự động bằng **Ansible**.
* **Computing**: **Apache Spark 3.5+** (PySpark) đóng vai trò lõi tính toán phân tán.
* **Storage**: **HDFS** cho dữ liệu thô và **MongoDB/PostgreSQL** cho dữ liệu đã qua xử lý.
* **Interface**: Dashboard tương tác xây dựng trên **Streamlit**.



## 📅 Lộ trình thực hiện (Timeline)
Hệ thống được triển khai theo tiến độ 15 tuần:
1. **Phase 1**: Thiết lập hạ tầng HPC bằng Ansible, viết script Crawler và thu thập Dataset thô.
2. **Phase 2**: Triển khai lõi thuật toán Big Data (LSH, MinHashing) trên Spark.
3. **Phase 3**: Tích hợp mô hình NLP và xử lý ngôn ngữ tiếng Việt quy mô lớn.
4. **Phase 4**: Hoàn thiện Dashboard, đánh giá hiệu năng (Benchmarking) và báo cáo cuối kỳ.