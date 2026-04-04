#  Distributed NLP Trend Analysis - Hub Hướng Dẫn Kỹ Thuật (Local)

Chào mừng các bạn đến với dự án Phân tích xu hướng NLP phân tán. Đây là tài liệu hướng dẫn nhanh để giúp các thành viên mới (mems) thiết lập môi trường cluster Big Data nhanh nhất trên máy cá nhân.

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
| **ClickHouse** | 192.168.56.14:8123 | Port HTTP của Database lưu trữ |

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
