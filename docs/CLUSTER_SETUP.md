# Hướng dẫn thiết lập Cluster với Ansible (VirtualBox)

Tài liệu này hướng dẫn cách deploy hệ thống Big Data cho dự án **Vietnamese Tech Trend & Controversy Radar** trên cụm máy ảo VirtualBox. Kiến trúc linh hoạt, hỗ trợ **không giới hạn worker nodes**.

## 1. Yêu cầu trước khi bắt đầu

- Máy host (hoặc máy chạy Ansible) cần có Ansible (`pip install ansible`).
- Bạn đã tạo các máy ảo Ubuntu Server trên VirtualBox với card mạng Host-Only (ví dụ: dải IP `192.168.56.x`).
- SSH Key được thiết lập từ máy host sang tất cả các máy ảo. Người dùng mặc định cần có quyền `sudo` không cần password (passwordless sudo).

> [!TIP]
> **Thêm Workers**: Bạn có thể thêm bao nhiêu máy ảo (worker) tùy ý bằng cách thêm IP của chúng vào block `[workers]` trong file `inventory/hosts.ini`. Tự động hoá qua Ansible sẽ lo phần còn lại.

## 2. Kiến trúc tổng quan

* **Master Node** (`192.168.56.10`): HDFS NameNode, Spark Master, Airflow, dbt-core.
* **Worker Nodes** (`192.168.56.11-13...`): HDFS DataNodes, Spark Workers. (Scale không giới hạn).
* **Storage Node** (`192.168.56.20`): ClickHouse Server.

*(Tất cả IPs cấu hình trong `inventory/hosts.ini`, bạn có thể thay đổi để phù hợp với IP máy ảo của bạn).*

## 3. Thứ tự chạy Playbooks (Lộ trình triển khai)

Chuyển vào thư mục `ansible/` và chạy các kịch bản sau theo thứ tự:

### Bước 1: Chuẩn bị môi trường (Java & Conda)

Tất cả các nodes đều cần Java 11 (cho Hadoop/Spark) và Conda (cho Python Runtime của PySpark).

```bash
ansible-playbook playbooks/01_java.yml
ansible-playbook playbooks/02_conda.yml
```

### Bước 2: Dựng Big Data Core (HDFS & Spark)

> [!WARNING]
> Node Master sẽ tự động `hdfs namenode -format` trong quá trình cài đặt HDFS. Điều này sẽ reset toàn bộ dữ liệu hiện có trên HDFS (mới cài đặt thì không sao).

```bash
ansible-playbook playbooks/03_hdfs.yml
ansible-playbook playbooks/04_spark.yml
```

_Sau khi xong, thử truy cập:_
* HDFS UI: `http://192.168.56.10:9870`
* Spark UI: `http://192.168.56.10:8080`

### Bước 3: Dựng Data Warehouse (ClickHouse)

Cài đặt cơ sở dữ liệu xử lý cột ClickHouse trên Node Storage (`192.168.56.20`).
```bash
ansible-playbook playbooks/05_clickhouse.yml
```
Kiểm tra kết nối bằng cách SSH vào Storage node và gõ `clickhouse-client`.

### Bước 4: Công cụ Quản lý và Schedule (dbt & Airflow)

Dbt phục vụ việc transform dữ liệu, còn Airflow schedule các Spark Jobs định kỳ.

```bash
ansible-playbook playbooks/06_dbt.yml
ansible-playbook playbooks/07_airflow.yml
```

## 4. Kiểm tra sự sẵn sàng của hệ thống (Verification)

M2 phải đảm bảo hệ thống đã sẵn sàng cho M1 (Data Engineer - Ingestion) và M3 (ML Engineer).
Bạn có thể xác thực bằng cách ssh vào máy Master và chạy lệnh đếm số pi của Spark:

```bash
ssh zett@192.168.56.10
source /etc/profile.d/spark.sh
spark-submit --class org.apache.spark.examples.SparkPi \
    --master spark://192.168.56.10:7077 \
    $SPARK_HOME/examples/jars/spark-examples_2.12-3.5.1.jar 10
```

Kết quả in ra `Pi is roughly 3.14...` có nghĩa là Cluster hoạt động bình thường!
