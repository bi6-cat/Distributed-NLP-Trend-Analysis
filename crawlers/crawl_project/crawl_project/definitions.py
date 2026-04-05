import os
import subprocess
import sys
from pathlib import Path

from dagster import Definitions, In, Nothing, OpExecutionContext, graph, multiprocess_executor, op, schedule


CRAWLERS_DIR = Path(__file__).resolve().parents[2]


def _run_crawler_script(context: OpExecutionContext, script_name: str) -> None:
    script_path = CRAWLERS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Crawler script not found: {script_path}")

    context.log.info(f"Start crawler: {script_path}")

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(CRAWLERS_DIR),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )

    if completed.stdout:
        context.log.info(completed.stdout)
    if completed.stderr:
        context.log.warning(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(f"Crawler failed ({script_name}) with exit code {completed.returncode}")


@op(ins={"start": In(Nothing)})
def crawl_vatvo(context: OpExecutionContext, start) -> None:
    del start
    _run_crawler_script(context, "vatvo.py")


@op(ins={"start": In(Nothing)})
def crawl_vnexpress(context: OpExecutionContext, start) -> None:
    del start
    _run_crawler_script(context, "vnexpress.py")


@op(ins={"start": In(Nothing)})
def crawl_voz(context: OpExecutionContext, start) -> None:
    del start
    _run_crawler_script(context, "voz.py")


@op
def start_parallel_crawl() -> None:
    # Fan-out signal op to make parallel execution explicit in graph.
    return None


@op
def finish_parallel_crawl(vatvo: None, vnexpress: None, voz: None) -> None:
    # Join op to mark that all crawler branches completed.
    return None


@graph
def crawl_all_graph() -> None:
    start = start_parallel_crawl()
    vatvo_done = crawl_vatvo(start)
    vnexpress_done = crawl_vnexpress(start)
    voz_done = crawl_voz(start)
    finish_parallel_crawl(vatvo_done, vnexpress_done, voz_done)


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
