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
