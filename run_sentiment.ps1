# run_sentiment.ps1
# Chạy Spark Sentiment Job → ghi stg_posts_nlp vào ClickHouse

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "         SENTIMENT ANALYSIS PIPELINE" -ForegroundColor Cyan
Write-Host "================================================"

vagrant ssh master -c "sed -i 's/\r//' /vagrant/scripts/run_sentiment.sh && bash /vagrant/scripts/run_sentiment.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Spark Sentiment Job that bai!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "   HOAN TAT!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "- stg_posts_nlp : ClickHouse tech_radar.stg_posts_nlp"
Write-Host "- ClickHouse UI : http://192.168.56.14:8123"
