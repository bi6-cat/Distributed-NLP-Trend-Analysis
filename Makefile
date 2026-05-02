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

# Detect if running inside the master VM or on the host machine
# Hostname in VM is 'master'. On Windows host, it's usually the PC name.
CONTEXT := $(shell hostname)

ifeq ($(CONTEXT), master)
	RUN_ON_MASTER = bash
else
	RUN_ON_MASTER = vagrant ssh master -c
endif

# Automation pipeline for Member 2 to run end-to-end cleaning and ingestion
run-cleaning-pipeline:
	@echo "================================================"
	@echo "   RUNNING DISTRIBUTED CLEANING PIPELINE"
	@echo "================================================"
	@echo "[1/2] Đang chạy Spark Cleaning Job trên cluster..."
	@$(RUN_ON_MASTER) "/vagrant/scripts/spark_submit_cluster.sh"
	@echo ""
	@echo "[2/2] Đang nạp dữ liệu từ HDFS vào ClickHouse..."
	@$(RUN_ON_MASTER) "/vagrant/scripts/ingest_hdfs_to_clickhouse.sh"
	@echo ""
	@echo "================================================"
	@echo "             PIPELINE HOÀN TẤT!"
	@echo "================================================"
