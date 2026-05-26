import "server-only";
import { createClient } from "@clickhouse/client";

type ClickHouseClient = ReturnType<typeof createClient>;

const globalForClickhouse = globalThis as typeof globalThis & {
  __clickhouseClient?: ClickHouseClient;
};

function getEnv(name: string, fallback?: string): string {
  const value = process.env[name] ?? fallback;
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function buildClickHouseUrl(): string {
  const explicitUrl = process.env.CLICKHOUSE_URL;
  if (explicitUrl) return explicitUrl;

  const protocol = process.env.CLICKHOUSE_PROTOCOL ?? "http";
  const host = getEnv("CLICKHOUSE_HOST").trim();
  if (host.startsWith("http://") || host.startsWith("https://")) {
    // Support users pasting full URL into CLICKHOUSE_HOST.
    return host;
  }
  const port = getEnv("CLICKHOUSE_PORT", protocol === "https" ? "8443" : "8123");
  return `${protocol}://${host}:${port}`;
}

function createClickHouseClient(): ClickHouseClient {
  return createClient({
    url: buildClickHouseUrl(),
    username: getEnv("CLICKHOUSE_USER"),
    password: getEnv("CLICKHOUSE_PASSWORD", ""),
    database: getEnv("CLICKHOUSE_DATABASE"),
    request_timeout: 30000,
  });
}

export const clickhouse = globalForClickhouse.__clickhouseClient ?? createClickHouseClient();

if (process.env.NODE_ENV !== "production") {
  globalForClickhouse.__clickhouseClient = clickhouse;
}

export async function queryClickhouse<T>(
  query: string,
  params: Record<string, unknown> = {},
): Promise<T[]> {
  try {
    const resultSet = await clickhouse.query({
      query,
      format: "JSONEachRow",
      query_params: params,
    });
    return await resultSet.json<T>();
  } catch (error) {
    console.error("ClickHouse query failed", { error, query });
    throw new Error("Failed to fetch data from ClickHouse");
  }
}
