#!/bin/bash
# run_pipeline.sh
# Bản thay thế Linux/Docker gọi tuần tự các module của 5 Members

echo -e "\e[36m================================================\e[0m"
echo -e "\e[36m   RUNNING DISTRIBUTED CLEANING PIPELINE (DOCKER) \e[0m"
echo -e "\e[36m================================================\e[0m"

# Khởi tạo DB nếu chưa có
docker exec -i clickhouse clickhouse-client --multiquery < warehouse/clickhouse/init_schema.sql || true

# Gọi lần lượt từng script của các Member
bash scripts/run_m1.sh && \
bash scripts/run_m2.sh && \
bash scripts/run_m3.sh && \
bash scripts/run_m4.sh && \
bash scripts/run_m5.sh

if [ $? -eq 0 ]; then
    echo -e "\n\e[32m================================================\e[0m"
    echo -e "\e[32m             PIPELINE SUCCESS!                \e[0m"
    echo -e "\e[32m================================================\e[0m"
else
    echo -e "\n\e[31m[ERROR] PIPELINE BỊ DỪNG DO CÓ LỖI TỪ THÀNH VIÊN!\e[0m"
fi