# Vietnamese Tech Trend Radar - Makefile

.PHONY: dev dev-down test lint setup-local-vms

# Run local development environment (ClickHouse + Airflow)
dev:
	docker-compose -f docker-compose.dev.yml up -d

# Stop local development environment
dev-down:
	docker-compose -f docker-compose.dev.yml down

# Install python dependencies locally
install:
	pip install -r requirements.txt

# Run pytest
test:
	pytest tests/ -v

# Run linter
lint:
	flake8 . --max-line-length=120

# Run local Vagrant / Multipass setup (Placeholder)
setup-vms-multipass:
	multipass launch -n master -c 4 -m 16G -d 50G
	multipass launch -n worker1 -c 4 -m 16G -d 50G
	multipass launch -n worker2 -c 4 -m 16G -d 50G
	multipass launch -n storage -c 4 -m 16G -d 50G

# Automation pipeline for Member 2 to run end-to-end cleaning and ingestion
run-cleaning-pipeline:
	@echo "================================================"
	@echo "   RUNNING DISTRIBUTED CLEANING PIPELINE"
	@echo "================================================"
	@echo "[1/2] Đang chạy Spark Cleaning Job trên cluster..."
	vagrant ssh master -c "bash /vagrant/scripts/spark_submit_cluster.sh"
	@echo "\n[2/2] Đang nạp dữ liệu từ HDFS vào ClickHouse..."
	vagrant ssh master -c "bash /vagrant/scripts/ingest_hdfs_to_clickhouse.sh"
	@echo "\n================================================"
	@echo "             PIPELINE HOÀN TẤT!"
	@echo "================================================"
