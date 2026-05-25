# HDFS to ClickHouse to dbt Pipeline

Tài liệu này mô tả phần nối từ Spark Silver trên HDFS sang ClickHouse staging, sau đó chạy dbt transformation cho warehouse.

## Mục tiêu

Luồng này dùng ClickHouse chủ động pull Parquet từ HDFS bằng `hdfs()`. Airflow chịu trách nhiệm chờ `_SUCCESS`, chạy SQL ingestion, kiểm tra core partition, rồi trigger dbt.

Entry points:

- Schema ClickHouse: `warehouse/clickhouse/init_schema.sql`
- SQL ingestion tham khảo: `warehouse/clickhouse/ingest_silver_to_staging.sql`
- Airflow DAG: `dags/silver_to_clickhouse_dag.py`
- dbt project: `warehouse/dbt_project`

## HDFS Silver Contract

Mỗi Spark job phải ghi Parquet Snappy theo logical date và tạo file `_SUCCESS` khi output hoàn tất.

| Dataset | HDFS path | ClickHouse table |
|---|---|---|
| Core posts | `/data/silver/posts_core/date={YYYY-MM-DD}/` | `tech_radar.stg_posts_core` |
| Sentiment | `/data/silver/posts_nlp/date={YYYY-MM-DD}/` | `tech_radar.stg_posts_nlp` |
| Post topics | `/data/silver/post_topics/date={YYYY-MM-DD}/` | `tech_radar.stg_post_topics` |
| Topic lookup | `/data/silver/topics/date={YYYY-MM-DD}/` | `tech_radar.stg_topics` |
| Keyword frequency | `/data/silver/keyword_freq/date={YYYY-MM-DD}/` | `tech_radar.stg_keyword_freq` |
| Crisis events | `/data/silver/crisis_events/date={YYYY-MM-DD}/` | `tech_radar.stg_crisis_events` |

Ví dụ marker cần có:

```bash
hdfs dfs -test -e hdfs://namenode:9000/data/silver/posts_core/date=2026-05-25/_SUCCESS
```

## ClickHouse Setup

Khởi tạo database và staging tables:

```bash
clickhouse-client --multiquery < warehouse/clickhouse/init_schema.sql
```

Các bảng idempotency:

- `stg_posts_core`: `MergeTree`; DAG xóa dữ liệu theo `created_at` của logical date trước khi insert.
- `stg_keyword_freq`: `MergeTree`; DAG xóa theo `window_start` trước khi insert.
- `stg_crisis_events`: `MergeTree`; DAG xóa theo `detected_at` trước khi insert.
- `stg_posts_nlp`, `stg_post_topics`, `stg_topics`: `ReplacingMergeTree`; rerun sẽ insert bản mới hơn và dbt đọc bằng `FINAL` ở các model cần latest view.

## Airflow DAG

DAG chính:

```text
silver_to_clickhouse_daily
```

Luồng task:

```text
wait _SUCCESS markers
    -> load ClickHouse staging tables
    -> assert stg_posts_core has rows for logical date
    -> dbt run
    -> dbt test
```

Các nhánh phụ như NLP, topics, keyword frequency, crisis events dùng sensor `soft_fail=True`. Nhánh core posts là bắt buộc vì dbt marts phụ thuộc vào core data.

Biến môi trường cần cấu hình trên Airflow worker:

| Variable | Ví dụ | Ghi chú |
|---|---|---|
| `HDFS_BASE` | `hdfs://namenode:9000` | Prefix HDFS dùng trong `hdfs()` và sensor |
| `CLICKHOUSE_HTTP_URL` | `http://clickhouse:8123/` | HTTP endpoint của ClickHouse |
| `CLICKHOUSE_USER` | `default` | User ClickHouse |
| `CLICKHOUSE_PASSWORD` | empty hoặc secret | Password ClickHouse |
| `DBT_PROJECT_DIR` | `/path/to/warehouse/dbt_project` | Có default theo repo root |
| `DBT_BIN` | `dbt` hoặc `/path/to/venv/bin/dbt` | Dùng khi Airflow chạy dbt trong virtualenv riêng |

## Manual Ingestion

File `warehouse/clickhouse/ingest_silver_to_staging.sql` là runbook SQL có Jinja placeholders. Khi chạy ngoài Airflow, cần render các tham số:

- `ds`: logical date, ví dụ `2026-05-25`
- `params.next_ds`: ngày kế tiếp, ví dụ `2026-05-26`
- `params.hdfs_base`: ví dụ `hdfs://namenode:9000`

Sau khi render, chạy qua ClickHouse:

```bash
clickhouse-client --multiquery < rendered_ingest.sql
```

## dbt Transformation

DAG chạy:

```bash
dbt run --select staging intermediate marts
dbt test --select staging intermediate marts
```

Các layer chính:

- `staging`: view mỏng trên ClickHouse staging tables.
- `intermediate`: join core, sentiment, topic assignment và tính enrichment theo giờ.
- `marts`: bảng phục vụ dashboard như `fct_topic_activity`, `dim_topics`, `fct_crisis_events`.

## Quick Validation

Kiểm tra HDFS marker:

```bash
hdfs dfs -ls hdfs://namenode:9000/data/silver/posts_core/date=2026-05-25/_SUCCESS
```

Kiểm tra ClickHouse staging:

```sql
SELECT count()
FROM tech_radar.stg_posts_core
WHERE created_at >= toDateTime('2026-05-25 00:00:00')
  AND created_at <  toDateTime('2026-05-26 00:00:00');
```

Kiểm tra dbt marts:

```sql
SELECT count() FROM tech_radar.fct_topic_activity;
SELECT count() FROM tech_radar.dim_topics;
SELECT count() FROM tech_radar.fct_crisis_events;
```

Kiểm tra mutation backlog sau rerun:

```sql
SELECT database, table, mutation_id, is_done, latest_fail_reason
FROM system.mutations
WHERE database = 'tech_radar'
ORDER BY create_time DESC;
```

## Lưu ý vận hành

- Không chạy nhiều DAG run cùng lúc cho cùng một logical date; DAG đã đặt `max_active_runs=1`.
- Spark phải tạo `_SUCCESS` sau cùng để tránh ClickHouse đọc partition chưa ghi xong.
- Backfill nhiều ngày sẽ tạo nhiều mutation do `ALTER TABLE ... DELETE`; nên theo dõi `system.mutations`.
- Nếu đổi tên HDFS folder ở Spark, cập nhật đồng thời trong `ingest_silver_to_staging.sql` và `silver_to_clickhouse_dag.py`.
- Nếu dashboard cần dữ liệu crisis, đảm bảo Spark M4 ghi đúng `/data/silver/crisis_events/date={YYYY-MM-DD}/`.
