# Server / Docker Infrastructure Guide: Distributed NLP Trend Analysis

Tài liệu này cung cấp thông tin chi tiết về cơ sở hạ tầng cụm (cluster) hiện tại để các thành viên trong nhóm có thể tham chiếu và kết nối. Chúng ta đã chia tay Vagrant/máy ảo local và chuyển toàn bộ hệ thống sang **Docker Compose**.

## 🖥️ Cấu trúc Container (Docker)

Hệ thống hiện tại chạy hoàn toàn trên Docker và có thể truy cập qua `localhost` nếu chạy trực tiếp trên máy, hoặc IP của Server nếu deploy từ xa. Các services chạy trên cùng chung 1 mạng lưới:

| Container / Dịch vụ | Tên Service (Docker) | Mô tả |
| :--- | :--- | :--- |
| **NameNode / Master** | `namenode` | Chạy HDFS NameNode và đóng vai trò gateway chính. |
| **DataNode 1 & 2** | `datanode1`, `datanode2` | Chạy HDFS DataNode lưu trữ phân tán. |
| **Spark Master** | `spark-master` | Điều phối tính toán Spark. |
| **Spark Workers** | `spark-worker-1`, `spark-worker-2` | Chạy tác vụ Spark. |
| **Airflow** | `airflow` | Đảm nhiệm lập lịch DAGs. |
| **ClickHouse** | `clickhouse-server` | OLAP Database cho Dashboard. |

> **Thao tác nhanh**: Để vào terminal của một container bất kỳ, dùng lệnh:
> `docker exec -it <tên_container> bash`

---

## 🐘 Apache Hadoop HDFS

Hệ thống lưu trữ phân tán dùng để lưu trữ dữ liệu lớn (Big Data). Dữ liệu được map thẳng vào Volume của Docker.

| Thành phần | URL / Connection String |
| :--- | :--- |
| **HDFS Web UI** | [http://localhost:9870](http://localhost:9870) |
| **Lệnh HDFS Client** | `docker exec -it namenode hdfs dfs -ls /` |

---

## 🎇 Apache Spark

Nền tảng tính toán phân tán cho xử lý NLP.
Bạn không cần SSH vào máy ảo nữa, chạy job trực tiếp bằng script trong thư mục root hoặc lệnh docker.

| Thành phần | URL / Connection String |
| :--- | :--- |
| **Spark Master Web UI** | [http://localhost:8080](http://localhost:8080) |
| **Spark Master Submit** | Dùng script `./scripts/spark_submit_cluster.sh` |

---

## ⏳ Apache Airflow

Hệ thống lập lịch và điều phối Data Pipeline.

| Thành phần | URL / Connection String |
| :--- | :--- |
| **Airflow Web UI** | [http://localhost:8081](http://localhost:8081) (Đổi sang 8081 tránh trùng Spark) |

> [!NOTE]
> Username / Password mặc định của Airflow là: `admin` / `admin`

---

## 📊 ClickHouse

Cơ sở dữ liệu OLAP dạng cột cho Analytics và Dashboard.
Để query, bạn có thể gọi thẳng client từ Docker.

| Thành phần | URL / Connection String |
| :--- | :--- |
| **ClickHouse HTTP (Playground)** | [http://localhost:8123/play](http://localhost:8123/play) |
| **Quản lý bằng DBeaver** | `localhost:9001` (Map từ port 9000 nội bộ) |
| **Lệnh Console SQL** | `docker exec -it clickhouse-server clickhouse-client` |

---

## 🚀 Thao Tác Cơ Bản Với Server Lifecycle

- **Mở Cluster**: `docker-compose up -d`
- **Tắt Cluster**: `docker-compose down`
- **Reset hạ tầng (Cẩn thận mất dữ liệu!)**: `docker-compose down -v`
- **Xem logs**: `docker-compose logs --tail=100 -f <tên_service>` (ví dụ: `docker-compose logs -f airflow`)
- **Chạy toàn bộ pipeline**: `./run_pipeline.sh`
- **Mở Dashboard**: `streamlit run dashboard/app.py`
