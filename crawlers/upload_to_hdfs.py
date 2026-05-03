import os
import requests
import urllib.parse

# Cấu hình HDFS (WebHDFS)
HDFS_HOST = "192.168.56.11"
HDFS_PORT = 9870
HDFS_USER = "zett"
HDFS_BASE_DIR = "/user/zett/raw_data"
DATA_DIR = "crawlers/data" # Thư mục chứa file local của crawlers

def hdfs_mkdirs(path):
    url = f"http://{HDFS_HOST}:{HDFS_PORT}/webhdfs/v1{path}?op=MKDIRS&user.name={HDFS_USER}"
    response = requests.put(url)
    response.raise_for_status()
    print(f"[HDFS] Created directory: {path}")

def upload_file_to_hdfs(local_path, hdfs_path):
    # Bước 1: Gọi lệnh CREATE (chưa gửi file)
    url = f"http://{HDFS_HOST}:{HDFS_PORT}/webhdfs/v1{hdfs_path}?op=CREATE&user.name={HDFS_USER}&overwrite=true"
    response = requests.put(url, allow_redirects=False)
    
    if response.status_code != 307:
        print(f"❌ Error initiating upload for {local_path}: {response.text}")
        return False
        
    # Lấy URL từ header Location để thực sự upload file
    datanode_url = response.headers.get("Location")
    
    # Bước 2: Upload nội dung file vào Datanode
    with open(local_path, "rb") as f:
        upload_response = requests.put(datanode_url, data=f)
        
    if upload_response.status_code == 201:
        print(f"✅ Successfully uploaded: {local_path} -> {hdfs_path}")
        return True
    else:
        print(f"❌ Failed to upload {local_path}: {upload_response.text}")
        return False

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Thư mục '{DATA_DIR}' không tồn tại. Chưa có dữ liệu để upload.")
        return

    # Đảm bảo thư mục đích trên HDFS đã tồn tại
    try:
        hdfs_mkdirs(HDFS_BASE_DIR)
    except Exception as e:
        print(f"Không thể kết nối đến HDFS: {e}")
        return

    files_uploaded = 0
    # Quét tất cả các file trong thư mục data/
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv") or file.endswith(".json"):
                local_path = os.path.join(root, file)
                
                # Tạo đường dẫn tương đối trên HDFS
                rel_path = os.path.relpath(local_path, DATA_DIR)
                # Đổi dấu \ thành / cho chuẩn HDFS path
                hdfs_path = f"{HDFS_BASE_DIR}/{rel_path}".replace("\\", "/")
                
                # Tạo thư mục cha trên HDFS nếu cần
                parent_dir = "/".join(hdfs_path.split("/")[:-1])
                hdfs_mkdirs(parent_dir)
                
                upload_file_to_hdfs(local_path, hdfs_path)
                files_uploaded += 1
                
    print(f"\n🎉 Hoàn thành! Đã đẩy {files_uploaded} file lên HDFS tại {HDFS_BASE_DIR}")

if __name__ == "__main__":
    print("🚀 Bắt đầu quá trình Batch Ingestion lên HDFS...")
    main()
