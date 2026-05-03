<#
.SYNOPSIS
Khởi động nhanh các dịch vụ (Hadoop, Spark, ClickHouse, Airflow) trên cụm VM.
#>

# Ép kiểu encoding cho console để hiển thị tiếng Việt
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   DANG KHOI DONG CAC DICH VU TREN CLUSTER" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Kiểm tra xem máy ảo đã chạy chưa
Write-Host "[1/2] Kiem tra trang thai may ao..." -ForegroundColor Green
$vagrantStatus = vagrant status master
if ($vagrantStatus -match "not created" -or $vagrantStatus -match "poweroff" -or $vagrantStatus -match "aborted") {
    Write-Host "[CANH BAO] May ao master dang tat. Dang bat..." -ForegroundColor Yellow
    vagrant up master
}

Write-Host "[2/2] Kich hoat Hadoop, Spark, ClickHouse va Airflow..." -ForegroundColor Green

# Chạy lệnh ansible-playbook thông qua vagrant ssh
# Sử dụng dấu nháy đơn bên ngoài để tránh lỗi giải thích biến của PowerShell
vagrant ssh master -c 'cd /vagrant && ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/start_services.yml'

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "  KHOI DONG DICH VU HOAN TAT THANH CONG!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "- Spark UI:      http://192.168.56.11:8080"
    Write-Host "- Hadoop HDFS:   http://192.168.56.11:9870"
    Write-Host "- Airflow UI:    http://192.168.56.11:8081"
} else {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "  CO LOI XAY RA KHI KHOI DONG DICH VU!" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
}
