#!/bin/bash
set -e
cd /vagrant/ansible

# Explicitly set Ansible paths to bypass Vagrant permission issues
export ANSIBLE_CONFIG=/vagrant/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=/vagrant/ansible/roles
export ANSIBLE_INVENTORY=/vagrant/ansible/inventory/hosts.ini
export ANSIBLE_HOST_KEY_CHECKING=False

echo '>>> [01] Installing Java...'
ansible-playbook playbooks/01_java.yml

echo '>>> [02] Installing Conda (Python)...'
ansible-playbook playbooks/02_conda.yml

echo '>>> [03] Installing HDFS...'
ansible-playbook playbooks/03_hdfs.yml

echo '>>> [04] Installing Spark...'
ansible-playbook playbooks/04_spark.yml

echo '>>> [05] Installing ClickHouse...'
ansible-playbook playbooks/05_clickhouse.yml

echo '>>> [06] Installing dbt...'
ansible-playbook playbooks/06_dbt.yml

echo '>>> [07] Installing Airflow...'
ansible-playbook playbooks/07_airflow.yml

echo '>>> ALL TASKS COMPLETED SUCCESSFULLY!'
