import { NextResponse } from "next/server";
import { queryClickhouse } from "@/lib/clickhouse";

export const dynamic = "force-dynamic";

type HealthRow = {
  ok: number;
  database: string;
  tables: number;
};

export async function GET() {
  try {
    const rows = await queryClickhouse<HealthRow>(`
      SELECT
        1 AS ok,
        currentDatabase() AS database,
        (
          SELECT count()
          FROM system.tables
          WHERE database = currentDatabase()
        ) AS tables
    `);

    return NextResponse.json({
      status: "ok",
      clickhouse: rows[0] ?? null,
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        message: error instanceof Error ? error.message : "Unknown ClickHouse error",
      },
      { status: 500 },
    );
  }
}
