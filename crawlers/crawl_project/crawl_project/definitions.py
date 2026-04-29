import os
import subprocess
import sys
from pathlib import Path

from dagster import Definitions, In, Nothing, OpExecutionContext, graph, op, multiprocess_executor, schedule


CRAWLERS_DIR = Path(__file__).resolve().parents[2]


def _run_crawler_script(context: OpExecutionContext, script_name: str) -> None:
    script_path = CRAWLERS_DIR / script_name

    if not script_path.exists():
        context.log.error(f"Crawler script not found: {script_path}")
        return

    context.log.info(f"Start crawler: {script_path}")

    try:
        completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(CRAWLERS_DIR),
        env={**os.environ.copy(), "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
)
    except Exception as e:
        context.log.error(f"Failed to start crawler subprocess: {e}")
        return

    # log output
    if completed.stdout:
        context.log.info(f"[{script_name} STDOUT]\n{completed.stdout}")

    if completed.stderr:
        context.log.warning(f"[{script_name} STDERR]\n{completed.stderr}")

    # IMPORTANT FIX: không raise nữa
    if completed.returncode != 0:
        context.log.error(
            f"Crawler {script_name} exited with code {completed.returncode} (NOT FAILING DAGSTER)"
        )
        return

    context.log.info(f"Crawler {script_name} finished successfully")


# ===== OPS =====

# @op(ins={"start": In(Nothing)})
# def crawl_vatvo(context: OpExecutionContext) -> None:
#     _run_crawler_script(context, "vatvo.py")


@op(ins={"start": In(Nothing)})
def crawl_vnexpress(context: OpExecutionContext) -> None:
    _run_crawler_script(context, "vnexpress.py")


@op(ins={"start": In(Nothing)})
def crawl_voz(context: OpExecutionContext) -> None:
    _run_crawler_script(context, "voz.py")


@op
def start_parallel_crawl():
    return None


@op(ins={"vatvo": In(Nothing), "vnexpress": In(Nothing), "voz": In(Nothing)})
def finish_parallel_crawl():
    return None


# ===== GRAPH =====

@graph
def crawl_all_graph():
    start = start_parallel_crawl()

    # vatvo_done = crawl_vatvo(start)
    vnexpress_done = crawl_vnexpress(start)
    voz_done = crawl_voz(start)

    finish_parallel_crawl(
        # vatvo=vatvo_done,
        vnexpress=vnexpress_done,
        voz=voz_done,
    )


crawl_all_job = crawl_all_graph.to_job(
    name="crawl_all_job",
    executor_def=multiprocess_executor,
)


@schedule(job=crawl_all_job, cron_schedule="0 2 * * *", execution_timezone="Asia/Ho_Chi_Minh")
def crawl_all_daily_schedule(_context):
    return {}


defs = Definitions(
    jobs=[crawl_all_job],
    schedules=[crawl_all_daily_schedule],
)