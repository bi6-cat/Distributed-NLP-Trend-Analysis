#!/bin/bash
# scripts/ingest_hdfs_to_clickhouse.sh
# Kịch bản nạp dữ liệu tự động từ HDFS sang ClickHouse
# Đã convert để tương thích môi trường docker

CLICKHOUSE_HOST=${1:-"clickhouse"}
CLICKHOUSE_PORT=${2:-"8123"}
DB_TABLE=${3:-"tech_radar.stg_posts_core"}
HDFS_PATH=${4:-"hdfs://namenode:9000/user/zett/staged/stg_posts_core/**/*.parquet"}

echo "--------------------------------------------------------"
echo "[INFO] Bắt đầu nạp dữ liệu từ HDFS vào ClickHouse"
echo "  - ClickHouse : http://$CLICKHOUSE_HOST:$CLICKHOUSE_PORT"
echo "  - Table      : $DB_TABLE"
echo "  - HDFS Data  : $HDFS_PATH"
echo "--------------------------------------------------------"

# Gửi từ airflow-scheduler host tới Clickhouse do ở ngoài docker ko gõ trực tiếp IP máy ảo Clickhouse được
docker exec airflow-scheduler bash -c "
# Xóa dữ liệu cũ trong bảng mục tiêu để tránh trùng lặp
curl -s -d 'TRUNCATE TABLE IF EXISTS $DB_TABLE' 'http://$CLICKHOUSE_HOST:$CLICKHOUSE_PORT/'

# Tạo câu lệnh SQL
QUERY=\"INSERT INTO $DB_TABLE SELECT * FROM hdfs('$HDFS_PATH', 'Parquet')\"

# Gửi HTTP POST request tới ClickHouse
curl -s -w '\nHTTP_STATUS:%{http_code}' -d \"\$QUERY\" 'http://$CLICKHOUSE_HOST:$CLICKHOUSE_PORT/' > /tmp/ch_output.txt

STATUS=\$(cat /tmp/ch_output.txt | grep 'HTTP_STATUS:' | cut -d':' -f2)
BODY=\$(cat /tmp/ch_output.txt | grep -v 'HTTP_STATUS:')

if [ \"\$STATUS\" -eq 200 ]; then
    echo '[SUCCESS] Dữ liệu đã được nạp thành công vào bảng $DB_TABLE!'
    COUNT=\$(curl -s -d 'SELECT count() FROM $DB_TABLE' 'http://$CLICKHOUSE_HOST:$CLICKHOUSE_PORT/')
    echo '[INFO] Tổng số bản ghi hiện tại trong bảng: \$COUNT'
else
    echo '[ERROR] Lỗi khi nạp dữ liệu! HTTP Status Code: \$STATUS'
    echo \"\$BODY\"
    exit 1
fi
"
