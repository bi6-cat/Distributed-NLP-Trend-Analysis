# 🚀 Hướng Dẫn Vận Hành Hệ Thống (Local Environment)

Tài liệu này hướng dẫn cách sử dụng các script PowerShell để quản lý cụm máy ảo và chạy pipeline xử lý dữ liệu.

---

## 1. Các Script Chính

| Script | Công dụng | Tần suất dùng |
| :--- | :--- | :--- |
| `.\deploy_cluster.ps1` | Cài đặt toàn bộ Hadoop, Spark, ClickHouse, Airflow... | Chỉ chạy 1 lần khi mới clone dự án |
| `.\start_cluster.ps1` | Khởi động các máy ảo và bật dịch vụ (HDFS, Spark, CH) | Mỗi khi bắt đầu làm việc |
| `.\clean_project.ps1` | Xóa sạch dữ liệu trên HDFS, ClickHouse và file tạm local | Khi muốn chạy lại từ đầu (reset) |
| `.\run_pipeline.ps1` | Chạy toàn bộ luồng: Ingest -> Spark Cleaning -> ClickHouse | Khi muốn xử lý dữ liệu mới |

> [!IMPORTANT]
> **Quy tắc vàng:**
> * **Chỉ chạy `.\deploy_cluster.ps1` một lần duy nhất** khi bạn mới cài dự án vào máy. Quá trình này rất lâu vì nó tải và cài đặt hàng GB phần mềm.
> * **Từ lần thứ 2 trở đi**, bạn chỉ cần chạy `.\start_cluster.ps1` để bật máy ảo và các dịch vụ. Không bao giờ chạy lại `deploy` trừ khi bạn muốn cài lại toàn bộ hệ thống.

---

## 2. Quy Trình Chạy Pipeline Chuẩn

Để chạy dự án một cách an toàn và sạch sẽ nhất, hãy thực hiện theo thứ tự sau:

### Bước 1: Khởi động hệ thống
Mở PowerShell (quyền Admin nếu cần) tại thư mục project:
```powershell
.\start_cluster.ps1
```

### Bước 2: Dọn dẹp dữ liệu cũ (Tùy chọn)
Nếu bạn muốn xóa sạch dữ liệu cũ để chạy lại bản mới nhất:
```powershell
.\clean_project.ps1
```

### Bước 3: Thực thi Pipeline xử lý
Lệnh này sẽ tự động làm 3 việc: Upload dữ liệu thô -> Chạy Spark xử lý -> Đẩy vào ClickHouse.
```powershell
.\run_pipeline.ps1
```

---

## 3. Chạy Lẻ Từng Công Việc (Nâng cao)

Nếu pipeline bị lỗi ở một bước, bạn có thể chạy lại riêng bước đó:

*   **Chỉ Ingest (Đẩy file CSV từ local lên HDFS):**
    ```powershell
    python crawlers/upload_to_hdfs.py
    ```

*   **Chỉ chạy Spark Job (Làm sạch & Dedup):**
    ```powershell
    vagrant ssh master -c "bash /vagrant/scripts/spark_submit_cluster.sh"
    ```

*   **Chỉ nạp dữ liệu vào ClickHouse (HDFS -> ClickHouse):**
    ```powershell
    vagrant ssh master -c "bash /vagrant/scripts/ingest_hdfs_to_clickhouse.sh"
    ```

---

## 4. Quản lý Dữ liệu Thử nghiệm (Crawlers)

Dữ liệu thô dùng để chạy thử pipeline được lưu tại thư mục local: `crawlers/data/`. Mặc định đã có sẵn ít dữ liệu để chạy thử.

### Cách lấy dữ liệu mới:
Hãy copy dataset vào thư mục này, hoặc thực hiện các bước sau để cào mới dữ liệu trực tiếp từ các website:

1. **Kích hoạt môi trường ảo (venv):**
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Chạy các bản Crawler:**
   * **VOZ:** `python crawlers/voz.py`
   * **VatVo:** `python crawlers/vatvo.py`
   * **VnExpress:** `python crawlers/vnexpress.py`

Dữ liệu sau khi cào sẽ tự động lưu vào `crawlers/data/voz/comments.csv`, v.v. Sau đó bạn có thể chạy `.\run_pipeline.ps1` để xử lý đống dữ liệu mới này.


---

## 5. Kiểm tra Hệ thống (Web UI)

*   **HDFS Web UI:** [http://192.168.56.11:9870](http://192.168.56.11:9870) (Xem file trên kho)
*   **Spark Master UI:** [http://192.168.56.11:8080](http://192.168.56.11:8080) (Theo dõi job đang chạy)
*   **ClickHouse HTTP:** [http://192.168.56.14:8123](http://192.168.56.14:8123) (Cơ sở dữ liệu đích)

---

> **Lưu ý:** Luôn đảm bảo bạn đang ở trong môi trường ảo Python (`venv`) trước khi chạy các script liên quan đến crawler.
