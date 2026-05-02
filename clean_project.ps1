# clean_project.ps1
# Dọn dẹp toàn bộ dữ liệu tạm, logs và database để chạy lại từ đầu

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       CLEANING PROJECT           " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 1. Dọn dẹp local
Write-Host "`n[1/4] Cleaning local files..." -ForegroundColor Yellow
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
Get-ChildItem -Path . -Include "__pycache__", "*.pyc", ".pytest_cache" -Recurse | Remove-Item -Recurse -Force
Write-Host "Cleaned local files."

# 2. Dọn dẹp HDFS (Raw & Staged)
Write-Host "`n[2/4] Cleaning HDFS..." -ForegroundColor Yellow
vagrant ssh master -c "export HADOOP_USER_NAME=zett; /opt/hadoop/bin/hdfs dfs -rm -r -f /user/zett/raw_data/* /user/zett/staged/*"
Write-Host "Cleaned HDFS."

# 3. Dọn dẹp ClickHouse
Write-Host "`n[3/4] Cleaning ClickHouse..." -ForegroundColor Yellow
vagrant ssh master -c "curl -s -d 'CREATE DATABASE IF NOT EXISTS tech_radar' http://192.168.56.14:8123/"
vagrant ssh master -c "curl -s -d 'TRUNCATE TABLE IF EXISTS tech_radar.stg_posts_core' http://192.168.56.14:8123/"
Write-Host "Cleaned ClickHouse."

# 4. Kiểm tra trạng thái máy ảo
Write-Host "`n[4/4] Checking Cluster Status..." -ForegroundColor Yellow
vagrant status
Write-Host "Cleaning completed!"

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "   PROJECT IS READY TO RUN AGAIN!      " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
