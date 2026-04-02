# Cluster Infrastructure Overview: Distributed NLP Trend Analysis

Tài liệu này cung cấp thông tin chi tiết về cơ sở hạ tầng cụm (cluster) hiện tại để các thành viên trong nhóm có thể tham chiếu và kết nối.

## 🖥️ Thông tin các Node (Máy chủ)

Hệ thống hiện tại bao gồm các node trên dải mạng `192.168.56.x`:

| Vai trò | Địa chỉ IP | Mô tả |
| :--- | :--- | :--- |
| **Master** | `192.168.56.11` | Chạy NameNode (Hadoop), Spark Master và điều phối cụm. |
| **Workers 1** | `192.168.56.12` | Chạy DataNode (Hadoop) và Spark Worker. |
| **Workers 2** | `192.168.56.13` | Chạy DataNode (Hadoop) và Spark Worker. |
| **Storage** | `192.168.56.14` | Node chuyên dụng cho mục đích lưu trữ phụ trợ. |

> [!NOTE]
> Người dùng quản trị (Ansible): `zett` (có quyền sudo).

---

## 🐘 Apache Hadoop HDFS

Hệ thống lưu trữ phân tán dùng để lưu trữ dữ liệu lớn (Big Data).

- **Phiên bản**: `3.3.6`
- **Vị trí cài đặt**: `/opt/hadoop` (Symlink của `/opt/hadoop-3.3.6`)
- **Dữ liệu HDFS**: 
  - NameNode (Master): `/data/hdfs/namenode`
  - DataNode (Workers): `/data/hdfs/datanode`

| Thành phần | URL / Connection String | Cổng mặc định |
| :--- | :--- | :--- |
| **HDFS Web UI** | [http://192.168.56.11:9870](http://192.168.56.11:9870) | `9870` |
| **HDFS RPC Service** | `hdfs://192.168.56.11:9000` | `9000` |

---

## 🎇 Apache Spark

Nền tảng tính toán phân tán cho xử lý NLP.

- **Phiên bản**: `3.5.1` (Build với Hadoop 3)
- **Vị trí cài đặt**: `/opt/spark` (Symlink của `/opt/spark-3.5.1-bin-hadoop3`)
- **Tài nguyên Worker**: 2 Cores / 4GB RAM mỗi node.

| Thành phần | URL / Connection String | Cổng mặc định |
| :--- | :--- | :--- |
| **Spark Master Web UI** | [http://192.168.56.11:8080](http://192.168.56.11:8080) | `8080` |
| **Spark Master Submit** | `spark://192.168.56.11:7077` | `7077` |

---

## 🐍 Conda & Python Environment

Môi trường thực thi code Python và các thư viện NLP.

- **Cài đặt tại**: `/opt/miniconda`
- **Tên môi trường**: `nlp-trend`
- **Phiên bản Python**: `3.10`
- **Lệnh kích hoạt**:
  ```bash
  source /etc/profile.d/conda.sh
  conda activate nlp-trend
  ```

---

## 🛡️ Kết nối từ xa (Tailscale)

Nếu bạn không ở trong cùng dải mạng LAN `192.168.56.x`, hãy sử dụng **Tailscale**:
1. Cài đặt và đăng nhập vào Tailscale network chung của nhóm.
2. Thay thế IP `192.168.56.11` bằng IP Tailscale của máy Master (ví dụ: `100.x.y.z`).
3. Truy cập vào các đường dẫn Web UI và RPC như bình thường.
