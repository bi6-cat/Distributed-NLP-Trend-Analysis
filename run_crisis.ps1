# run_crisis.ps1
# Chạy Crisis Detector đọc thẳng từ ClickHouse → ghi stg_crisis_events

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "        CRISIS DETECTION PIPELINE" -ForegroundColor Cyan
Write-Host "================================================"

python -c @"
from scripts.crisis_detector import CrisisDetector, print_report
detector = CrisisDetector(min_conditions=2)
events = detector.run_from_clickhouse(
    host='192.168.56.14',
    database='tech_radar',
    user='default',
    password='',
    write_back=True,
)
print_report(events)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Crisis Detector that bai!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "   HOAN TAT! Ket qua tai:" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "- ClickHouse: tech_radar.stg_crisis_events"
Write-Host "- UI        : http://192.168.56.14:8123"
