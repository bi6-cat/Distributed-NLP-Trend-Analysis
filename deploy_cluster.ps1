# Script tự động hóa 100% quá trình cài đặt Cluster
# Cách chạy: Mở PowerShell tại thư mục dự án và chạy .\deploy_cluster.ps1

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "       BẮT ĐẦU TỰ ĐỘNG CÀI ĐẶT BIG DATA CLUSTER       " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Khởi động các máy ảo (Vagrant Up)..." -ForegroundColor Yellow
vagrant up

Write-Host "`n[2/3] Cài đặt Ansible và thiết lập SSH Passwordless trên Master..." -ForegroundColor Yellow
# Gửi thẳng một đoạn bash script dài vào trong máy master để thực thi nội bộ
$setupScript = @"
#!/bin/bash
set -e

echo '>>> Đang cập nhật gói cài đặt...'
sudo apt update -qq
echo '>>> Đang cài đặt Ansible và sshpass...'
sudo DEBIAN_FRONTEND=noninteractive apt install ansible sshpass -y -qq

echo '>>> Tạo SSH Key cho Master (nếu chưa có)...'
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

echo '>>> Chia sẻ khóa SSH cho tất cả các Node trong cụm...'
for IP in 192.168.56.11 192.168.56.12 192.168.56.13 192.168.56.14; do
    echo "Mở khóa cho $IP..."
    sshpass -p '0008' ssh-copy-id -o StrictHostKeyChecking=no zett@`$IP >/dev/null 2>&1
done
"@

# Lưu ý: Lệnh thực thi bash bên trong Master
vagrant ssh master -c $setupScript

Write-Host "`n[3/3] Chạy Ansible Playbooks tự động hóa (Hadoop, Spark, ClickHouse...)..." -ForegroundColor Yellow
$ansibleScript = @"
#!/bin/bash
set -e
cd /vagrant/ansible

echo '>>> [01] Cài đặt Java...'
ansible-playbook playbooks/01_java.yml

echo '>>> [02] Cài đặt Conda (Python)...'
ansible-playbook playbooks/02_conda.yml

echo '>>> [03] Cài đặt HDFS...'
ansible-playbook playbooks/03_hdfs.yml

echo '>>> [04] Cài đặt Spark...'
ansible-playbook playbooks/04_spark.yml

echo '>>> [05] Cài đặt ClickHouse...'
ansible-playbook playbooks/05_clickhouse.yml

echo '>>> [06] Cài đặt dbt...'
ansible-playbook playbooks/06_dbt.yml

echo '>>> [07] Cài đặt Airflow...'
ansible-playbook playbooks/07_airflow.yml

echo '>>> TẤT CẢ ĐÃ HOÀN TẤT THÀNH CÔNG!'
"@

vagrant ssh master -c $ansibleScript

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "       HỆ THỐNG ĐÃ SẴN SÀNG! HOÀN TẤT CÀI ĐẶT.        " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "- HDFS UI  : http://192.168.56.11:9870"
Write-Host "- Spark UI : http://192.168.56.11:8080"
Write-Host "- Airflow  : http://192.168.56.11:8080 (hoặc port bạn đã đặt trong Ansible)"
