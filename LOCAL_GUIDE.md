# 🚀 Hướng Dẫn Vận Hành Hệ Thống (Local Environment)

Tài liệu này hướng dẫn cách sử dụng Docker Compose để quản lý cụm HPC và chạy pipeline xử lý dữ liệu.

---

## 1. Các Script & Lệnh Chính

| Lệnh / Script | Công dụng | Tần suất dùng |
| :--- | :--- | :--- |
| \docker-compose up -d\ | Khởi động cluster (HDFS, Spark, ClickHouse, Airflow) | Mỗi khi bắt đầu làm việc |
| \docker-compose down\ | Tắt cluster | Khi dừng làm việc |
| \docker-compose down -v\ | Xóa sạch dữ liệu toàn bộ cluster (Factory Reset) | Khi muốn chạy setup lại từ đầu |
| \./run_pipeline.sh\ | Chạy toàn bộ luồng pipeline từ đầu đến cuối | Khi muốn xử lý dữ liệu mới |

> [!IMPORTANT]
> **Quy tắc vàng:**
> Lần đầu tiên chạy \docker-compose up -d\ có thể mất thời gian do tải image Docker. Các lần sau sẽ khởi động rất nhanh.

---

## 2. Quy Trình Chạy Pipeline Chuẩn

Để chạy dự án, hãy thực hiện theo thứ tự sau:

### Bước 1: Khởi động hệ thống
Mở Terminal tại thư mục project:
\\ash
docker-compose up -d
\
### Bước 2: Dọn dẹp dữ liệu cũ (Tùy chọn)
Nếu bạn muốn xóa sạch dữ liệu cũ để chạy lại bản mới nhất:
\\ash
docker-compose down -v
docker-compose up -d
\
### Bước 3: Thực thi Pipeline xử lý
Đảm bảo đã cấp quyền thực thi cho các file bash:
\\ash
chmod +x run_pipeline.sh scripts/*.sh
\Sau đó khởi chạy:
\\ash
./run_pipeline.sh
\
---

## 4. Quản lý Dữ liệu Thử nghiệm (Crawlers)

Dữ liệu thô dùng để chạy thử pipeline được lưu tại thư mục local: \crawlers/data/\. Mặc định đã có sẵn ít dữ liệu để chạy thử.

### Cách lấy dữ liệu mới:
1. **Kích hoạt môi trường ảo (venv):**
   \\ash
   source venv/bin/activate  # Hoặc .env\Scriptsctivate trên Windows
   \
2. **Chạy các bản Crawler:**
   * **VOZ:** \python crawlers/voz.py   * **VatVo:** \python crawlers/vatvo.py   * **VnExpress:** \python crawlers/vnexpress.py
Sau đó chạy \./run_pipeline.sh\ để đẩy vào hệ thống.

---

## 5. Truy Cập Các Giao Diện (Web UI)

*   **HDFS Web UI:** [http://localhost:9870](http://localhost:9870)
*   **Spark Master:** [http://localhost:8080](http://localhost:8080)
*   **Airflow UI:** [http://localhost:8081](http://localhost:8081) (\dmin\ / \dmin\)
*   **ClickHouse:** [http://localhost:8123](http://localhost:8123)
*   **Dashboard:** \streamlit run dashboard/app.py