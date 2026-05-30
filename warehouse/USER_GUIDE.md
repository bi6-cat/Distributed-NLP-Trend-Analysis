docker-compose down -v
docker-compose up -d

docker exec -i clickhouse-server clickhouse-client -n < init_schema.sql

docker exec -it clickhouse-server clickhouse-client
USE tech_radar;
SHOW TABLES;

exit;
 
cd warehouse/dbt_project

dbt debug
dbt compile

dbt run --full-refresh
dbt build --full-refresh

dbt test