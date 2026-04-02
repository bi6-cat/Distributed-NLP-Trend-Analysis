# Hướng dẫn thiết lập Cluster với Ansible (VirtualBox)

Tài liệu này hướng dẫn cách deploy hệ thống Big Data cho dự án **Vietnamese Tech Trend & Controversy Radar** trên cụm máy ảo VirtualBox. Kiến trúc linh hoạt, hỗ trợ **không giới hạn worker nodes**.

## 1. Yêu cầu trước khi bắt đầu (Pre-requisites)

Bạn cần chuẩn bị các bước sau trên máy tính thật (Host) và các máy ảo (VMs) trước khi chạy Ansible.

### 1.1 Cài đặt Ansible trên máy Host
Ansible chỉ cần được cài đặt trên máy tính dùng để điều khiển.
- **Windows (Khuyên dùng WSL2):** `sudo apt update && sudo apt install python3-pip -y && pip3 install ansible`
- **MacOS / Linux:** `pip3 install ansible`
- **Kiểm tra:** `ansible --version` để đảm bảo đã cài đặt thành công.

### 1.2 Cấu hình Network VirtualBox
Hệ thống sử dụng dải IP nội bộ `192.168.56.x` để các máy giao tiếp.
1. Vào **File** -> **Tools** -> **Network Manager**.
2. Tạo card **Host-only Network** (mặc định là `192.168.56.1`). **Tắt DHCP**.
3. Cấu hình mỗi máy ảo có 2 card mạng:
   - **Adapter 1:** NAT (để có internet).
   - **Adapter 2:** Host-only Adapter (chọn card vừa tạo ở bước 2).
4. Đảm bảo cấu hình IP tĩnh trong Ubuntu (Netplan) khớp với dải `.11`, `.12`, `.13`, `.14`.

### 1.3 Thiết lập SSH Key (Passwordless SSH)
Để Ansible tự động kết nối mà không hỏi mật khẩu:
1. **Tạo Key (trên máy Host):** `ssh-keygen -t rsa -b 4096` (nhấn Enter liên tục).
2. **Copy Key sang các máy ảo:**
   ```bash
   ssh-copy-id zett@192.168.56.11
   ssh-copy-id zett@192.168.56.12
   ssh-copy-id zett@192.168.56.13
   ssh-copy-id zett@192.168.56.14
   ```

### 1.4 Quyền Sudo không mật khẩu (Passwordless Sudo)
Trên **mỗi máy ảo**, chạy lệnh `sudo visudo` và thêm dòng sau vào cuối:
```text
zett ALL=(ALL) NOPASSWD:ALL
```
Điều này cho phép Ansible cài đặt phần mềm từ xa mà không bị hỏi mật khẩu sudo.

### 1.5 Lựa chọn: Điều khiển tự máy Master (Tùy chọn)
Nếu bạn muốn dùng chính máy Master (`192.168.56.11`) để điều khiển toàn cụm:
1. SSH vào Master: `ssh zett@192.168.56.11`
2. Cài Ansible trên Master: `sudo apt update && sudo apt install ansible -y`
3. Tạo Key trên Master: `ssh-keygen -t rsa -b 4096`
4. Copy Key từ Master sang các node khác (bao gồm chính nó):
   ```bash
   ssh-copy-id zett@192.168.56.11
   ssh-copy-id zett@192.168.56.12
   ssh-copy-id zett@192.168.56.13
   ssh-copy-id zett@192.168.56.14
   ```
5. Đưa folder dự án lên Master và chạy lệnh playbook từ đó.

> [!TIP]
> **Thêm Workers**: Bạn có thể thêm bao nhiêu máy ảo (worker) tùy ý bằng cách thêm IP của chúng vào block `[workers]` trong file `inventory/hosts.ini` (nhớ cập nhật cả tệp `Vagrantfile`). Tự động hoá qua Ansible sẽ lo phần còn lại.

## 2. Kiến trúc tổng quan

* **Master Node** (`192.168.56.11`): HDFS NameNode, Spark Master, Airflow, dbt-core.
* **Worker Nodes** (`192.168.56.12-13`): HDFS DataNodes, Spark Workers.
* **Storage Node** (`192.168.56.14`): ClickHouse Server.

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
* HDFS UI: `http://192.168.56.11:9870`
* Spark UI: `http://192.168.56.11:8080`

### Bước 3: Dựng Data Warehouse (ClickHouse)

Cài đặt cơ sở dữ liệu xử lý cột ClickHouse trên Node Storage (`192.168.56.14`).
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
ssh zett@192.168.56.11
source /etc/profile.d/spark.sh
spark-submit --class org.apache.spark.examples.SparkPi \
    --master spark://192.168.56.11:7077 \
    $SPARK_HOME/examples/jars/spark-examples_2.12-3.5.1.jar 10
```

Kết quả in ra `Pi is roughly 3.14...` có nghĩa là Cluster hoạt động bình thường!
