# run_pipeline.ps1
# PowerShell script thay thế cho 'make run-cleaning-pipeline' trên Windows

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   RUNNING DISTRIBUTED CLEANING PIPELINE" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Đang chạy Spark Cleaning Job trên cluster..." -ForegroundColor Yellow
vagrant ssh master -c "bash /vagrant/scripts/spark_submit_cluster.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Spark Cleaning Job thất bại!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[2/2] Đang nạp dữ liệu từ HDFS vào ClickHouse..." -ForegroundColor Yellow
vagrant ssh master -c "bash /vagrant/scripts/ingest_hdfs_to_clickhouse.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Nạp dữ liệu vào ClickHouse thất bại!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "             PIPELINE HOÀN TẤT!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
