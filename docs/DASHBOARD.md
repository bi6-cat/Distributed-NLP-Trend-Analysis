# Tech Trend & Controversy Radar Dashboard

Dashboard Next.js dùng để đọc dữ liệu trend, sentiment và crisis từ ClickHouse marts của pipeline trong repo.

## Yêu Cầu

- Node.js 18.17+ hoặc Node.js 20 LTS.
- npm đi kèm Node.js.
- Docker Desktop nếu muốn chạy đầy đủ data stack local.
- ClickHouse có sẵn dữ liệu marts do pipeline/dbt tạo ra.

Kiểm tra nhanh:

```bash
node -v
npm -v
```

## Cài Đặt Sau Khi Clone

```bash
cd dashboard
npm ci
```

Các thư viện chính đã được khai báo trong `package.json`:

- `next`, `react`, `react-dom`: framework dashboard.
- `@clickhouse/client`: kết nối ClickHouse.
- `recharts`: biểu đồ.
- `lucide-react`: icon UI.
- `tailwindcss`, `tailwind-merge`, `tailwindcss-animate`: styling.
- `@base-ui/react`, `class-variance-authority`, `clsx`: UI utilities.

## Cấu Hình Môi Trường

Tạo file `.env` trong thư mục `dashboard` từ file mẫu:

```bash
cp .env.example .env
```

Trên Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Nội dung mặc định:

```env
CLICKHOUSE_PROTOCOL=http
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=root
CLICKHOUSE_PASSWORD=root
CLICKHOUSE_DATABASE=tech_radar
```

Nếu ClickHouse chạy trên máy/server khác, sửa `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_DATABASE` cho đúng môi trường đó.

Không commit file `.env` thật vì có thể chứa thông tin kết nối riêng.

## Chuẩn Bị Dữ Liệu

Dashboard cần các bảng/marts sau trong ClickHouse:

- `tech_radar.dbt_fct_topic_activity`
- `tech_radar.dbt_dim_topics`
- `tech_radar.dbt_fct_crisis_events`
- `tech_radar.dbt_int_posts_enriched`

Nếu chạy local bằng Docker Compose, từ thư mục root repo:

```bash
docker-compose up -d
```

Sau đó chạy pipeline theo hướng dẫn trong `LOCAL_GUIDE.md` để tạo dữ liệu vào ClickHouse. Tóm tắt flow local:

1. Mở Airflow UI: <http://localhost:8081>
2. Đăng nhập: `admin/admin`
3. Unpause DAG `full_processing_pipeline`
4. Trigger DAG và chờ các task ingest/dbt hoàn tất

Kiểm tra nhanh ClickHouse:

```sql
SELECT count() FROM tech_radar.dbt_fct_topic_activity;
SELECT count() FROM tech_radar.dbt_dim_topics;
SELECT count() FROM tech_radar.dbt_fct_crisis_events;
SELECT count() FROM tech_radar.dbt_int_posts_enriched;
```

Nếu các bảng tồn tại nhưng số dòng bằng `0`, dashboard vẫn chạy nhưng widget sẽ hiển thị dữ liệu trống hoặc số liệu bằng `0`.

## Chạy Development

Trong thư mục `dashboard`:

```bash
npm run dev
```

Mở trình duyệt:

```text
http://localhost:3000
```

Nếu cổng `3000` đang bận:

```bash
npm run dev -- --port 3001
```

Sau đó mở:

```text
http://localhost:3001
```

## Health Check

Kiểm tra dashboard có kết nối được ClickHouse không:

```text
http://localhost:3000/api/health/clickhouse
```

Kết quả mong đợi:

```json
{
  "status": "ok",
  "clickhouse": {
    "ok": 1,
    "database": "tech_radar",
    "tables": 14
  }
}
```