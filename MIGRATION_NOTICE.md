# 📢 THÔNG BÁO: CHUYỂN ĐỔI HẠ TẦNG SANG DOCKER COMPOSE

Xin chào cả Team,

Tài liệu này đánh dấu một sự thay đổi quan trọng về môi trường vận hành của dự án **Distributed NLP Trend Analysis**. Chúng ta đã chính thức **chuyển đổi hạ tầng Local từ Vagrant/VirtualBox + Ansible sang sử dụng triệt để Docker Compose**.

Việc gỡ bỏ kiến trúc máy ảo kiểu cũ (Vagrant) giúp hệ thống nhẹ hơn, tối ưu tài nguyên hơn (ngốn ít RAM cho hệ điều hành khách) và có thể dễ dàng chia sẻ, review source code trên bất kỳ hệ điều hành Linux/MacOS nào thay vì bó buộc trên Windows.

---

## 1. Tóm tắt những thay đổi cốt lõi

*   **Tạm biệt Vagrant & Ansible:** Team **KHÔNG** cần phải chạy file `Vagrantfile` hay thư mục `ansible/playbooks/` nữa. Các script thiết lập dài ngoẵng đã bị loại bỏ chức năng.
*   **Không còn dùng PowerShell (`*.ps1`):** Mọi công việc từ `start_cluster.ps1`, `clean_project.ps1` đến `run_pipeline.ps1` đều đã được thay thế thành Shell Script chuẩn (`.sh`) để tương tác thẳng với Docker.
*   **Thay đổi cách Submit Job:** Các script trong `/scripts` (như `spark_submit_cluster.sh`, `run_sentiment.sh`) trước kia gọi lệnh `vagrant ssh` để xuyên vào máy ảo, nay đã được update mã nguồn để dùng `docker exec` giao tiếp thẳng vào các Container.
*   **Toàn bộ hệ thống gói gọn trong `docker-compose.yml`:** HDFS (NameNode/DataNode), Spark (Master/Worker), Airflow và ClickHouse đều đã quy về 1 mạng lưới chung dễ bề kiểm soát.

---

## 2. Bảng đối chiếu Lệnh (Cũ -> Mới)

Anh em làm quen theo bảng ánh xạ lệnh dưới đây để thao tác không bị bỡ ngỡ:

| Tác Vụ | Cũ (Windows/Vagrant) | Mới (Linux/Docker) |
| :--- | :--- | :--- |
| **Bật Cụm Máy Chủ** | `.\start_cluster.ps1` | `docker-compose up -d` |
| **Khẩn Cấp: Xóa Sạch DB (Factory Reset)** | `.\clean_project.ps1` | `docker-compose down -v` (Bật lại bằng lệnh `up -d`) |
| **Chạy Pipeline Tự Động Toàn Diện** | `.\run_pipeline.ps1` | `./run_pipeline.sh` |
| **Lệnh Console HDFS** | `vagrant ssh master -c "/opt/hadoop/bin/hdfs..."` | `docker exec -it namenode hdfs dfs...` |
| **Chọc vào Data Warehouse** | Mở IP `192.168.56.14` | Mở IP `localhost:8123` hoặc `docker exec -it clickhouse ...` |

---

## 3. Thứ tự các bước Dành Cho Người Mới Tham Gia Test

Nếu có thành viên mới hoặc anh em muốn Test trên server này, tất cả vòng đời thao tác chỉ gói gọn trong vài dòng lệnh sau:

1. Đợi các dịch vụ lên sóng xanh: `docker-compose up -d`
2. Cấp quyền chạy script: `chmod +x run_pipeline.sh scripts/*.sh`
3. Gọi kịch bản chạy dữ liệu từ A-Z (Đã bao gồm Crawl -> Clean -> LDA Topic -> PhoBERT Sentiment -> dbt -> ClickHouse): 
   ```bash
   ./run_pipeline.sh
   ```
4. Bật giao diện UI xem báo cáo kết quả:
   ```bash
   streamlit run dashboard/app.py
   ```

---

## 4. Bảng Tra Cứu Dịch Vụ / Cổng (Ports Map)

Do đã chuyển sang Docker, các địa chỉ IP của máy ảo cũ (`192.168.56.x` từ Vagrant) **không còn hiệu lực**. Mọi truy cập UI và kết nối hiện tại đều trỏ về `localhost` trên máy chủ chứa Docker.

| Hệ thống / Dịch vụ | Giao diện Web / Cổng kết nối | Tài khoản mặc định | Mô tả chức năng |
| :--- | :--- | :--- | :--- |
| **Airflow Web UI** | [http://localhost:8081](http://localhost:8081) | `admin` / `admin` | Quản lý / Trigger luồng DAG (đổi sang 8081 để không trùng Spark) |
| **Spark Master UI**| [http://localhost:8080](http://localhost:8080) | Không có | Theo dõi tài nguyên Worker, xem Logs của Job đang chạy |
| **HDFS NameNode UI**| [http://localhost:9870](http://localhost:9870) | Không có | Quản lý file trên HDFS theo thời gian thực (Trình duyệt tệp) |
| **ClickHouse HTTP**| [http://localhost:8123/play](http://localhost:8123/play)| `default` / *(Trống)* | URL để test nhanh câu lệnh SQL (ClickHouse Playground) |
| **ClickHouse TCP** | `localhost:9001` | `default` / *(Trống)* | Port kết nối quản lý bằng DBeaver/DataGrip (Map từ 9000 nội bộ) |
| **Dashboard (Web)**| [http://localhost:8501](http://localhost:8501) | Chưa tích hợp | Trang báo cáo Streamlit cho end-user |

