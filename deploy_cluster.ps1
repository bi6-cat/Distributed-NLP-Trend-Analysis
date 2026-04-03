# 100% Automated Cluster Installation Script
# Usage: Open PowerShell in the project directory and run .\deploy_cluster.ps1

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "       STARTING AUTOMATED BIG DATA CLUSTER DEPLOY      " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Starting Virtual Machines (Vagrant Up)..." -ForegroundColor Yellow
vagrant up
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start VMs. Please check VirtualBox installation." -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/3] Installing Ansible and setting up SSH Passwordless on Master..." -ForegroundColor Yellow
# Run setup_master.sh from the shared folder, ensuring LF line endings
vagrant ssh master -- -T "tr -d '\r' < /vagrant/scripts/setup_master.sh > /tmp/setup_master.sh && bash /tmp/setup_master.sh"

Write-Host "`n[3/3] Running Ansible Playbooks (Hadoop, Spark, ClickHouse...)..." -ForegroundColor Yellow
# Run run_playbooks.sh from the shared folder, ensuring LF line endings
vagrant ssh master -- -T "tr -d '\r' < /vagrant/scripts/run_playbooks.sh > /tmp/run_playbooks.sh && bash /tmp/run_playbooks.sh"

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "       SYSTEM READY! INSTALLATION COMPLETE.           " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Access links (once VMs are fully provisioned):"
Write-Host "- HDFS UI  : http://192.168.56.11:9870"
Write-Host "- Spark UI : http://192.168.56.11:8080"
Write-Host "- Airflow  : http://192.168.56.11:8081"
