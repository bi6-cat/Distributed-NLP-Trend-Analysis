# Distributed NLP Trend Analysis — Hướng Dẫn Chạy Từ Đầu

> Tài liệu này hướng dẫn người mới hoàn toàn có thể chạy được hệ thống từ bước đầu tiên (cài phần mềm) đến bước cuối cùng (xem dữ liệu trong ClickHouse).

---

## Mục lục

1. [Yêu cầu phần cứng & phần mềm](#1-yêu-cầu)
2. [Lần đầu: Triển khai cluster](#2-triển-khai-cluster-lần-đầu)
3. [Lần sau: Bật lại cluster](#3-bật-lại-cluster-lần-sau)
4. [Upload dữ liệu lên HDFS](#4-upload-dữ-liệu-lên-hdfs)
5. [Chạy Spark Cleaning Job](#5-chạy-spark-cleaning-job)
6. [Nạp dữ liệu từ HDFS vào ClickHouse](#6-nạp-dữ-liệu-vào-clickhouse)
7. [Xác minh kết quả](#7-xác-minh-kết-quả)
8. [Truy cập các dashboard](#8-truy-cập-các-dashboard)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)

---

## 1. Yêu cầu

### Phần cứng (tối thiểu)

| Tài nguyên | Yêu cầu |
|-----------|---------|
| **RAM** | 16 GB (cluster chiếm 4 VM × 4GB) |
| **CPU** | 6 cores |
| **Disk** | 40 GB SSD trống |
| **OS** | Windows 10/11 (64-bit) |

### Phần mềm cần cài trên máy Windows

| Phần mềm | Link tải | Lưu ý |
|----------|----------|-------|
| **VirtualBox 7.x** | https://www.virtualbox.org/wiki/Downloads | Phần mềm tạo máy ảo |
| **Vagrant** | https://developer.hashicorp.com/vagrant/downloads | Tự động hóa VM |
| **Git for Windows** | https://git-scm.com/download/win | Kéo source code |
| **PowerShell 5.1+** | Có sẵn trên Windows | Chạy script |

> [!IMPORTANT]
> Sau khi cài VirtualBox và Vagrant, **khởi động lại máy tính** trước khi tiếp tục.

---

## 2. Triển khai Cluster (Lần đầu)

### Bước 1: Kéo source code

```powershell
git clone <repo-url>
cd Distributed-NLP-Trend-Analysis-1
```

### Bước 2: Chạy script deploy tự động

Mở **PowerShell** tại thư mục dự án rồi chạy:

```powershell
.\deploy_cluster.ps1
```

Script sẽ tự động làm các việc sau (mất khoảng **15–25 phút** lần đầu):

```
[1/3] vagrant up          → Tạo 4 máy ảo Ubuntu 22.04
[2/3] setup_master.sh     → Cài Ansible + SSH keyless
[3/3] run_playbooks.sh    → Cài Hadoop, Spark, ClickHouse, Conda, Airflow
```

Khi thấy dòng cuối `SYSTEM READY! INSTALLATION COMPLETE.` là thành công.

> [!NOTE]
> Nếu deploy bị lỗi ở giữa chừng, chạy lại `.\deploy_cluster.ps1` — script được thiết kế để chạy lại an toàn, sẽ bỏ qua những bước đã xong.

---

## 3. Bật lại Cluster (Lần sau)

Sau khi tắt máy tính/VM, bật lại bằng cùng một lệnh:

```powershell
.\deploy_cluster.ps1
```

Lần này chỉ mất **2–5 phút** vì Hadoop/Spark đã được cài sẵn (script sẽ `skipping` các bước đó).

### Tắt cluster khi không dùng

```powershell
vagrant halt
```

> [!WARNING]
> Không tắt VM trực tiếp qua VirtualBox GUI — có thể làm hỏng filesystem của VM.

---

## 4. Upload Dữ liệu lên HDFS

Dữ liệu raw từ crawlers (CSV files) cần được đưa lên HDFS trước khi Spark xử lý.

### SSH vào Master node

```powershell
vagrant ssh master
```

### Tạo thư mục trên HDFS

```bash
hdfs dfs -mkdir -p /user/zett/raw_data/voz
hdfs dfs -mkdir -p /user/zett/raw_data/vatvo
hdfs dfs -mkdir -p /user/zett/raw_data/vnexpress
hdfs dfs -mkdir -p /user/zett/ref
```

### Upload file dữ liệu

```bash
# File data nằm trong /vagrant/ (= thư mục dự án trên Windows được mount vào VM)
hdfs dfs -put /vagrant/data/Data_NLP_DM/voz/comments.csv         /user/zett/raw_data/voz/
hdfs dfs -put /vagrant/data/Data_NLP_DM/voz/posts.csv            /user/zett/raw_data/voz/
hdfs dfs -put /vagrant/data/Data_NLP_DM/vatvo/articles.csv       /user/zett/raw_data/vatvo/
hdfs dfs -put /vagrant/data/Data_NLP_DM/vnexpress/post_vnexpress.csv     /user/zett/raw_data/vnexpress/
hdfs dfs -put /vagrant/data/Data_NLP_DM/vnexpress/comment_vnexpress.csv  /user/zett/raw_data/vnexpress/

# Upload file tham chiếu NLP
hdfs dfs -put /vagrant/data/slang_dict.json    /user/zett/ref/
hdfs dfs -put /vagrant/data/stopwords_vi.txt   /user/zett/ref/
```

### Kiểm tra

```bash
hdfs dfs -ls /user/zett/raw_data/voz/
# Expected: comments.csv, posts.csv
```

---

## 5. Chạy Spark Cleaning Job

Job này đọc CSV từ HDFS, clean text, dedup bằng MinHash LSH, rồi ghi ra Parquet.

### Cách 1: Chạy qua Makefile (đơn giản nhất)

```powershell
# Từ máy Windows
make run-cleaning-pipeline
```

### Cách 2: Chạy thủ công trên Master node

```bash
# SSH vào master
vagrant ssh master

# Đóng gói module Python vào zip để Spark distribute lên executors
cd /vagrant
zip -r dist/nlp_trend.zip preprocessing/ schemas/ algorithms/

# Chạy spark-submit
spark-submit \
    --master spark://192.168.56.11:7077 \
    --num-executors 2 \
    --executor-cores 2 \
    --executor-memory 4g \
    --driver-memory 2g \
    --py-files /vagrant/dist/nlp_trend.zip \
    --conf spark.executorEnv.NLP_SLANG_DICT=hdfs:///user/zett/ref/slang_dict.json \
    --conf spark.executorEnv.NLP_STOPWORDS=hdfs:///user/zett/ref/stopwords_vi.txt \
    /vagrant/spark_jobs/cleaning_job.py
```

### Kết quả mong đợi

```
[Cleaning] Tổng bản ghi sau clean: 45,230
[Dedup] Bắt đầu MinHash LSH dedup (threshold=0.8, num_perm=128, k=5)...
[Dedup] Kết quả: 45,230 → 38,104 bản ghi (loại 7,126 duplicates, 15.8%)
[DONE] Đã ghi 38,104 bản ghi → hdfs://192.168.56.11:9000/user/zett/staged/stg_posts_core
```

> [!TIP]
> Muốn test nhanh mà bỏ qua bước dedup:
> ```bash
> /vagrant/spark_jobs/cleaning_job.py --no-dedup
> ```

---

## 6. Nạp Dữ liệu vào ClickHouse

Sau khi Spark job xong, nạp Parquet từ HDFS vào ClickHouse.

### Bước 1: Tạo bảng trong ClickHouse (chỉ làm lần đầu)

```bash
# SSH vào storage node (nơi cài ClickHouse)
vagrant ssh storage

clickhouse-client --query "
CREATE TABLE IF NOT EXISTS tech_radar.stg_posts_core (
    post_id         String,
    source          LowCardinality(String),
    author          String,
    title           Nullable(String),
    body            String,
    segmented_text  String,
    parent_id       Nullable(String),
    reaction_count  Int32 DEFAULT 0,
    comment_count   Int32 DEFAULT 0,
    view_count      Nullable(Int32),
    created_at      DateTime,
    crawled_at      DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (source, created_at, post_id)
TTL created_at + INTERVAL 1 YEAR;
"
```

### Bước 2: Chạy script nạp dữ liệu

```bash
# Từ master node
bash /vagrant/scripts/ingest_hdfs_to_clickhouse.sh
```

---

## 7. Xác minh Kết quả

### Kiểm tra Parquet đã ghi thành công

```bash
# Trên Master node
hdfs dfs -ls /user/zett/staged/stg_posts_core/
# Expected: các thư mục source=voz/, source=vatvo/, source=vnexpress/

hdfs dfs -du -s -h /user/zett/staged/stg_posts_core/
# Expected: vài chục MB
```

### Kiểm tra dữ liệu trong ClickHouse

```bash
# Trên Storage node
clickhouse-client --query "SELECT count() FROM tech_radar.stg_posts_core"
# Expected: số bản ghi > 0

clickhouse-client --query "
SELECT source, count() as cnt
FROM tech_radar.stg_posts_core
GROUP BY source ORDER BY cnt DESC
"
# Expected: voz, vatvo, vnexpress với số lượng khác nhau

# Kiểm tra không còn duplicate
clickhouse-client --query "
SELECT post_id, count() as cnt
FROM tech_radar.stg_posts_core
GROUP BY post_id HAVING cnt > 1
"
# Expected: 0 rows (không có duplicate)
```

---

## 8. Truy cập các Dashboard

| Dịch vụ | URL | Thông tin đăng nhập |
|---------|-----|---------------------|
| **HDFS NameNode UI** | http://192.168.56.11:9870 | Không cần đăng nhập |
| **Spark Master UI** | http://192.168.56.11:8080 | Không cần đăng nhập |
| **Airflow** | http://192.168.56.11:8081 | `admin` / `admin` |
| **ClickHouse HTTP** | http://192.168.56.14:8123 | `default` / (không có mật khẩu) |

> [!NOTE]
> Thông tin VM: Username `zett` / Password `0008`

---

## 9. Xử lý Lỗi Thường Gặp

### `storage` VM boot timeout

**Triệu chứng:** Deploy fail ở bước khởi động `storage` VM.

```powershell
# Tắt tất cả VM, thử lại
vagrant halt
.\deploy_cluster.ps1
```

Nếu vẫn lỗi, tăng timeout trong `Vagrantfile`:
```ruby
config.vm.boot_timeout = 600  # tăng từ 300 lên 600 giây
```

### Hadoop cứ tải lại mỗi lần deploy

**Nguyên nhân:** File archive chưa tải xong hoặc bị xóa khỏi VM.

```bash
# SSH vào từng node, kiểm tra
vagrant ssh master
ls -lh ~/hadoop-*.tar.gz
ls -lh ~/spark-*.tgz
```

### `datasketch` not found khi chạy Spark job

```bash
# Cài thủ công lên tất cả nodes
vagrant ssh master -c "ansible-playbook /vagrant/ansible/playbooks/install_nlp_deps.yml"
```

### HDFS 0 Live Datanodes

```bash
vagrant ssh master
# Kiểm tra
hdfs dfsadmin -report

# Restart DataNode trên worker
vagrant ssh worker1
/opt/hadoop/bin/hdfs --daemon stop datanode
/opt/hadoop/bin/hdfs --daemon start datanode
```

### Spark job lỗi `ModuleNotFoundError: No module named 'preprocessing'`

```bash
# Đảm bảo đã zip và truyền --py-files
cd /vagrant
zip -r dist/nlp_trend.zip preprocessing/ schemas/ algorithms/
# Sau đó thêm --py-files /vagrant/dist/nlp_trend.zip vào spark-submit
```

---

*Mọi thắc mắc ping trên Zalo nhóm hoặc tạo issue trên Git repo.*


---

##  1. Yêu cầu Hệ thống (Hardware)

Do cluster chạy đồng thời 4 máy ảo (VM), máy tính của bạn cần cấu hình tối thiểu:
*   **RAM**: Khuyến nghị 16GB. 
    *   *Master Node*: 4GB
    *   *Worker Nodes*: 4GB x 2
    *   *Storage Node*: 4GB
*   **CPU**: Tối thiểu 6 cores (Cluster chiếm 6 vCPUs).
*   **Disk**: Ổ cứng SSD trống ít nhất 40GB.

---

##  2. Công cụ cần cài đặt (Prerequisites)

Hãy cài đặt các công cụ sau trước khi bắt đầu:
1.  **VirtualBox**: Phần mềm tạo máy ảo.
2.  **Vagrant**: Công cụ tự động hóa khởi tạo máy ảo.
3.  **Git for Windows**: Để lấy source code và dùng Git Bash.
4.  **PowerShell**: Có sẵn trên Windows (Dùng bản 5.1 hoặc 7+).

---

##  3. Triển khai One-Click (Deployment)

Dự án đã được tối ưu để cài đặt tự động 100%. Bạn không cần cài Ansible trên máy Windows.

1.  Mở PowerShell tại thư mục dự án sau khi kéo từ git về.
2.  Chạy lệnh sau:
    ```powershell
    .\deploy_cluster.ps1
    ```
    Script này sẽ:
    - `vagrant up`: Tạo 4 node (Ubuntu 22.04).
    - Cài đặt Ansible & SSH tự động bên trong Master Node.
    - Chạy các Playbook để cài Hadoop, Spark, ClickHouse, Airflow.

---

##  4. Truy cập Dashboard (Dịch vụ)

Sau khi cài đặt xong, bạn có thể truy cập các giao diện quản lý từ trình duyệt host qua các IP sau:

| Dịch vụ | Địa chỉ (IP:Port) | Mô tả |
| :--- | :--- | :--- |
| **HDFS NameNode** | [http://192.168.56.11:9870](http://192.168.56.11:9870) | Quản lý file trên cụm Hadoop |
| **Spark Master** | [http://192.168.56.11:8080](http://192.168.56.11:8080) | Quản lý các job tính toán phân tán |
| **Airflow UI** | [http://192.168.56.11:8081](http://192.168.56.11:8081) | Lập lịch và quản lý Pipeline (DAGs) |
| **ClickHouse** | [http://192.168.56.14:8123](http://192.168.56.14:8123) | Port HTTP của Database lưu trữ |

> [!NOTE]
> Username/Password Airflow: `admin` / `admin`
> Username/Password Virtual Machine: `zett` / `0008`

---

##  5. Tắt máy và Khởi động lại (Stop & Start)

Khi không dùng đến nữa, bạn không nên tắt thẳng máy ảo qua VirtualBox. Hãy làm theo cách sau:

### Cách tắt cụm máy ảo (Halt)
Mở PowerShell tại thư mục dự án:
```powershell
vagrant halt
```
Tất cả 4 máy ảo sẽ được lưu lại trạng thái và tắt an toàn.

### Cách bật lại và chạy ứng dụng (Start)
Các ứng dụng Big Data (Hadoop, Spark, Airflow) không tự động chạy lúc boot máy tính để tiết kiệm tài nguyên. **Để tự động bật lại cụm máy và chạy các dịch vụ này**, bạn chỉ cần chạy lại file cài đặt ban đầu:
```powershell
.\deploy_cluster.ps1
```
Script đã được tối ưu để tính toán tự động bỏ qua (skip) các bước cài đặt và chỉ kích hoạt lại những ứng dụng chưa chạy. Bạn sẽ chỉ tốn khoảng 1-2 phút thay vì phải chờ nguyên quá trình tải về như lần đầu.

---

##  6. Cấu trúc Dự án quan trọng

*   `/ansible/roles/`: Chứa kịch bản cài đặt cho từng component.
*   `/ansible/inventory/hosts.ini`: Danh sách các node và IP tương ứng.
*   `/scripts/`: Các script bổ trợ khởi tạo môi trường ban đầu.
*   `Vagrantfile`: Cấu hình thông số phần cứng của các máy ảo.

---


---
*Chúc cả team làm việc hiệu quả! Mọi thắc mắc hãy ping trên zalo nhóm dự án.* 
