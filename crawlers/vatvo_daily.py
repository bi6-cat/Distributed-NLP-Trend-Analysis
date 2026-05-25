import json
import csv
import os
import tempfile
import time
import re
import hashlib
import argparse
from datetime import datetime
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options

from remote_storage import RemoteStorage


# ==============================
# CONFIG
# ==============================
URLS = [
    "https://vatvostudio.vn/category/tin-tuc-moi-nhat/",
    "https://vatvostudio.vn/category/artificial-intelligence/",
    "https://vatvostudio.vn/category/vat-vo-danh-gia/",
    "https://vatvostudio.vn/category/goc-nhin/",
    "https://vatvostudio.vn/category/xem-xong-mua/",
    "https://vatvostudio.vn/category/tips-and-tricks/"
]

STORAGE = None
STORAGE_BACKEND = os.getenv("CRAWLER_STORAGE", "remote").strip().lower()
VATVO_DIR = "vatvo"
LOCAL_DATA_DIR = os.getenv(
    "CRAWLER_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)

# File dùng chung với crawl full
CRAWLED_FILE = "crawled_links.json"
CSV_FILE = "articles.csv"
CRAWL_DATE = os.getenv("CRAWLER_OUTPUT_DATE", datetime.now().strftime("%Y-%m-%d"))

# File riêng cho daily crawl
DAILY_RUNS_FILE = "daily_runs.json"

MAX_RETRY = 5
DELAY = 2
BATCH_SIZE = 5

# Daily không nên crawl quá sâu
MAX_DAILY_PAGES_PER_BASE = 5
MAX_RUNTIME_SECONDS = int(float(os.getenv("CRAWLER_MAX_RUNTIME_SECONDS", "10800")))

# Bo qua cac link dau cua page 1 vi thuong la bai noi bat/trung lap.
SKIP_FIRST_PAGE_LINKS = 4


def is_runtime_expired(started_at, max_runtime_seconds):
    return (
        max_runtime_seconds > 0
        and time.monotonic() - started_at >= max_runtime_seconds
    )


def dated_filename(filename, date_text=CRAWL_DATE):
    name, ext = os.path.splitext(filename)
    return f"{name}_{date_text}{ext}"


def daily_csv_file():
    return dated_filename(CSV_FILE)


def daily_runs_file():
    return dated_filename(DAILY_RUNS_FILE)


# ==============================
# SELENIUM
# ==============================
def create_driver():
    profile_dir = tempfile.TemporaryDirectory(prefix="vatvo-daily-chrome-")

    options = Options()
    options.page_load_strategy = "eager"

    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-data-dir={profile_dir.name}")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)

    return driver, profile_dir


def close_driver(driver, profile_dir):
    try:
        if driver is not None:
            driver.quit()
    except Exception:
        pass

    try:
        if profile_dir is not None:
            profile_dir.cleanup()
    except Exception:
        pass


def restart_driver(driver_state):
    close_driver(driver_state.get("driver"), driver_state.get("profile_dir"))
    driver, profile_dir = create_driver()
    driver_state["driver"] = driver
    driver_state["profile_dir"] = profile_dir


def is_bad_html(html):
    if not html:
        return True

    bad_markers = [
        "One moment, please...",
        "ERR_CONNECTION_TIMED_OUT",
        "This site can't be reached",
        "DNS_PROBE",
        "Aw, Snap!",
    ]

    return any(marker in html for marker in bad_markers)


def load_rendered_html(driver_state, url):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            driver_state["driver"].get(url)
            time.sleep(3)

            html = driver_state["driver"].page_source

            if is_bad_html(html):
                print(f"[BAD HTML] {attempt}/{MAX_RETRY} -> {url}")
                time.sleep(DELAY)
                continue

            return html

        except WebDriverException as exc:
            message = str(exc).lower()

            if any(keyword in message for keyword in [
                "invalid session id",
                "chrome not reachable",
                "session deleted",
                "disconnected",
            ]):
                print(f"[RESTART DRIVER] {attempt}/{MAX_RETRY} -> {url}")
                restart_driver(driver_state)
                time.sleep(DELAY)
                continue

            if any(keyword in message for keyword in [
                "timeout",
                "page load timeout",
                "err_connection_timed_out",
                "connection timed out",
            ]):
                print(f"[TIMEOUT] {attempt}/{MAX_RETRY} -> {url}")
                try:
                    driver_state["driver"].execute_script("window.stop();")
                    html = driver_state["driver"].page_source

                    if not is_bad_html(html):
                        return html

                except Exception:
                    pass

                time.sleep(DELAY)
                continue

            print(f"[DRIVER ERROR] {attempt}/{MAX_RETRY} -> {url} -> {exc}")
            time.sleep(DELAY)

    return None


# ==============================
# STORAGE
# ==============================
class LocalStorage:
    def __init__(self, base_dir=LOCAL_DATA_DIR):
        self.base_dir = os.path.abspath(base_dir)

    def path(self, *parts):
        return os.path.join(
            self.base_dir,
            *[str(part).strip("/\\") for part in parts],
        )

    def exists(self, path):
        return os.path.exists(path)

    def mkdir_parent(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def read_json(self, path, default):
        if not self.exists(path):
            return default

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def write_json(self, path, data):
        self.mkdir_parent(path)
        tmp_path = f"{path}.tmp"

        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)

    def init_csv(self, path, columns, encoding="utf-8"):
        if self.exists(path):
            return

        self.mkdir_parent(path)
        with open(path, "w", encoding=encoding, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

    def append_csv(self, path, records, encoding="utf-8"):
        if not records:
            return

        self.mkdir_parent(path)
        write_header = not self.exists(path) or os.path.getsize(path) == 0

        if write_header:
            fieldnames = list(records[0].keys())
        else:
            with open(path, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.reader(file)
                fieldnames = next(reader, None) or list(records[0].keys())

        output_encoding = "utf-8-sig" if write_header else encoding
        with open(path, "a", encoding=output_encoding, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")

            if write_header:
                writer.writeheader()

            for record in records:
                writer.writerow({key: record.get(key) for key in fieldnames})

    def read_csv_column_as_str_set(self, path, column):
        if not self.exists(path) or os.path.getsize(path) == 0:
            return set()

        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or column not in reader.fieldnames:
                return set()

            values = set()
            for row in reader:
                value = row.get(column)
                if value is not None and str(value).strip():
                    values.add(str(value).strip())

        return values

    def close(self):
        pass


def create_storage(backend):
    backend = (backend or "remote").strip().lower()

    if backend in {"remote", "server", "hdfs"}:
        print("[STORAGE] remote HDFS")
        return RemoteStorage()

    if backend in {"local", "file", "filesystem"}:
        print(f"[STORAGE] local filesystem: {LOCAL_DATA_DIR}")
        return LocalStorage()

    raise ValueError(
        f"Unsupported storage backend: {backend}. Use remote or local."
    )


def storage_path(filename):
    return STORAGE.path(VATVO_DIR, filename)


def load_json(filename, default=None):
    if default is None:
        default = {}
    return STORAGE.read_json(storage_path(filename), default)


def save_json(filename, data):
    STORAGE.write_json(storage_path(filename), data)


def init_csv():
    STORAGE.init_csv(
        storage_path(daily_csv_file()),
        ["post_id", "article", "author", "time", "content"],
    )


def load_existing_ids():
    existing_ids = set()
    for filename in [CSV_FILE, daily_csv_file()]:
        existing_ids.update(
            STORAGE.read_csv_column_as_str_set(
                storage_path(filename),
                "post_id",
            )
        )
    return existing_ids


def append_articles(batch):
    if not batch:
        return

    STORAGE.append_csv(storage_path(daily_csv_file()), batch)
    print(f"[SAVE CSV] {daily_csv_file()} | {len(batch)} articles")


# ==============================
# HELPERS
# ==============================
def normalize_url(href):
    href = href.split("#", 1)[0].rstrip("/") + "/"
    return href


def extract_article_links(soup):
    links = []
    seen = set()

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href:
            continue

        parsed = urlparse(href)

        if parsed.netloc not in {"vatvostudio.vn", "www.vatvostudio.vn"}:
            continue

        path = parsed.path.strip("/")
        if not path:
            continue

        if path.startswith(("category/", "author/", "tag/", "page/")):
            continue

        # Bài viết Vatvo thường có dạng:
        # https://vatvostudio.vn/slug-bai-viet/
        # Nếu path còn chứa "/" thì bỏ qua.
        if "/" in path:
            continue

        normalized = normalize_url(href)

        if normalized in seen:
            continue

        seen.add(normalized)
        links.append(normalized)

    return links


def get_page_url(base_url, page):
    if page == 1:
        return base_url
    return f"{base_url}page/{page}/"


def build_known_links(crawled_data):
    known_links = set()

    for base_url, pages in crawled_data.items():
        if not isinstance(pages, dict):
            continue

        for page, links in pages.items():
            if not isinstance(links, list):
                continue

            for link in links:
                known_links.add(normalize_url(link))

    return known_links


def build_category_known_links(crawled_data, base_url):
    known_links = set()
    pages = crawled_data.get(base_url, {})

    if not isinstance(pages, dict):
        return known_links

    for links in pages.values():
        if not isinstance(links, list):
            continue

        for link in links:
            known_links.add(normalize_url(link))

    return known_links


def ensure_crawled_slot(crawled_data, base_url, page):
    if base_url not in crawled_data:
        crawled_data[base_url] = {}

    page_key = str(page)

    if page_key not in crawled_data[base_url]:
        crawled_data[base_url][page_key] = []

    return page_key


# ==============================
# CRAWL PAGE
# ==============================
def crawl_page(driver_state, base_url, page):
    page_url = get_page_url(base_url, page)

    html = load_rendered_html(driver_state, page_url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    links = extract_article_links(soup)

    print(f"[PAGE] {base_url} | page={page} | links={len(links)}")

    return links


# ==============================
# CRAWL ARTICLE
# ==============================
def parse_article(driver_state, url):
    html = load_rendered_html(driver_state, url)
    if html is None:
        print(f"[FAIL ARTICLE] {url}")
        return None, "load_failed"

    soup = BeautifulSoup(html, "html.parser")

    body = soup.select_one("body")
    post_id = None

    if body and body.get("class"):
        m = re.search(r"postid-(\d+)", " ".join(body["class"]))
        if m:
            post_id = m.group(1)

    if post_id is None:
        print(f"[NO POST ID] {url}")
        return None, "no_post_id"

    title_el = soup.select_one("h1.s-title.fw-headline")
    title = title_el.get_text(strip=True) if title_el else ""

    author_el = soup.select_one("div.meta-el a.meta-author-url.meta-author")
    author = author_el.get_text(strip=True) if author_el else ""

    time_tag = soup.select_one("time.updated-date")
    time_el = time_tag.get("datetime") if time_tag else None

    content_el = soup.select_one("div.entry-content")
    content = content_el.get_text(" ", strip=True) if content_el else ""

    article = {
        "post_id": post_id,
        "article": title,
        "author": author,
        "time": time_el,
        "content": content,
    }

    return article, "ok"


# ==============================
# DAILY CRAWL CATEGORY
# ==============================
def crawl_daily_category(
    driver_state,
    base_url,
    crawled_data,
    known_links,
    existing_ids,
    started_at,
    max_runtime_seconds,
):
    print(f"\n=== DAILY CATEGORY START: {base_url} ===")

    new_articles = []
    new_links = []

    stop_reason = None
    stop_link = None
    category_known_links = build_category_known_links(crawled_data, base_url)

    for page in range(1, MAX_DAILY_PAGES_PER_BASE + 1):
        if is_runtime_expired(started_at, max_runtime_seconds):
            stop_reason = "runtime_limit"
            print(f"[STOP RUNTIME LIMIT] {base_url}")
            break

        links = crawl_page(driver_state, base_url, page)

        if links is None:
            stop_reason = "page_load_failed"
            print(f"[STOP] page load failed -> {base_url} page {page}")
            break

        if not links:
            stop_reason = "no_links"
            print(f"[STOP] no links -> {base_url} page {page}")
            break

        page_key = ensure_crawled_slot(crawled_data, base_url, page)

        links_to_crawl = links
        if page == 1 and SKIP_FIRST_PAGE_LINKS > 0:
            skipped_links = links[:SKIP_FIRST_PAGE_LINKS]
            links_to_crawl = links[SKIP_FIRST_PAGE_LINKS:]
            print(
                f"[SKIP PAGE 1 LINKS] {base_url} | "
                f"skip={len(skipped_links)} | crawl={len(links_to_crawl)}"
            )

        for link in links_to_crawl:
            if is_runtime_expired(started_at, max_runtime_seconds):
                if new_articles:
                    append_articles(new_articles)
                    new_articles = []
                save_json(CRAWLED_FILE, crawled_data)
                stop_reason = "runtime_limit"
                print(f"[STOP RUNTIME LIMIT] {base_url}")
                return {
                    "base_url": base_url,
                    "new_count": len(new_links),
                    "new_links": new_links,
                    "stop_reason": stop_reason,
                    "stop_link": stop_link,
                }

            link = normalize_url(link)

            # Điều kiện dừng nhanh nhất:
            # link đã xuất hiện trong checkpoint cũ của chính category hiện tại.
            if link in category_known_links:
                stop_reason = "old_link_met"
                stop_link = link
                print(f"[STOP OLD LINK] {link}")
                return {
                    "base_url": base_url,
                    "new_count": len(new_articles),
                    "new_links": new_links,
                    "stop_reason": stop_reason,
                    "stop_link": stop_link,
                }

            if link in known_links:
                print(f"[SKIP GLOBAL OLD LINK] {link}")
                continue

            article, status = parse_article(driver_state, link)

            if status != "ok" or article is None:
                print(f"[SKIP ARTICLE] {link} | status={status}")
                continue

            post_id = article["post_id"]

            # Điều kiện dừng chắc chắn hơn:
            # link có thể chưa nằm trong crawled_links,
            # nhưng post_id đã tồn tại trong CSV.
            if post_id in existing_ids:
                stop_reason = "old_post_id_met"
                stop_link = link
                print(f"[STOP OLD POST ID] {link} | post_id={post_id}")
                return {
                    "base_url": base_url,
                    "new_count": len(new_articles),
                    "new_links": new_links,
                    "stop_reason": stop_reason,
                    "stop_link": stop_link,
                }

            new_articles.append(article)
            new_links.append(link)

            existing_ids.add(post_id)
            known_links.add(link)
            category_known_links.add(link)

            if link not in crawled_data[base_url][page_key]:
                crawled_data[base_url][page_key].append(link)

            print(f"[NEW] {link}")

            if len(new_articles) >= BATCH_SIZE:
                append_articles(new_articles)
                new_articles = []
                save_json(CRAWLED_FILE, crawled_data)

        # Lưu sau mỗi page để tránh mất checkpoint nếu lỗi giữa chừng.
        if new_articles:
            append_articles(new_articles)
            new_articles = []

        save_json(CRAWLED_FILE, crawled_data)

    if stop_reason is None:
        stop_reason = "max_daily_pages_reached"

    return {
        "base_url": base_url,
        "new_count": len(new_links),
        "new_links": new_links,
        "stop_reason": stop_reason,
        "stop_link": stop_link,
    }


# ==============================
# DAILY RUN LOG
# ==============================
def save_daily_run_log(run_result):
    daily_runs = load_json(daily_runs_file(), [])

    if not isinstance(daily_runs, list):
        daily_runs = []

    daily_runs.append(run_result)

    save_json(daily_runs_file(), daily_runs)


# ==============================
# MAIN
# ==============================
def main(storage_backend=STORAGE_BACKEND, max_runtime_seconds=MAX_RUNTIME_SECONDS):
    global STORAGE

    STORAGE = create_storage(storage_backend)
    init_csv()

    crawled_data = load_json(CRAWLED_FILE, {})
    existing_ids = load_existing_ids()
    known_links = build_known_links(crawled_data)

    driver, profile_dir = create_driver()
    driver_state = {
        "driver": driver,
        "profile_dir": profile_dir,
    }

    run_started_at = datetime.now().isoformat()

    run_result = {
        "run_started_at": run_started_at,
        "run_finished_at": None,
        "total_new": 0,
        "categories": [],
        "stop_reason": None,
    }

    started_at = time.monotonic()

    try:
        for base_url in URLS:
            if is_runtime_expired(started_at, max_runtime_seconds):
                run_result["stop_reason"] = "runtime_limit"
                print("[STOP RUNTIME LIMIT] daily run")
                break

            result = crawl_daily_category(
                driver_state=driver_state,
                base_url=base_url,
                crawled_data=crawled_data,
                known_links=known_links,
                existing_ids=existing_ids,
                started_at=started_at,
                max_runtime_seconds=max_runtime_seconds,
            )

            run_result["categories"].append(result)
            run_result["total_new"] += result["new_count"]

            # Lưu checkpoint sau từng category.
            save_json(CRAWLED_FILE, crawled_data)

            if result.get("stop_reason") == "runtime_limit":
                run_result["stop_reason"] = "runtime_limit"
                break

    finally:
        run_result["run_finished_at"] = datetime.now().isoformat()

        save_json(CRAWLED_FILE, crawled_data)
        save_daily_run_log(run_result)

        close_driver(
            driver_state.get("driver"),
            driver_state.get("profile_dir"),
        )

        STORAGE.close()

    print("\n=== DAILY DONE ===")
    print(json.dumps(run_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        choices=["remote", "local"],
        default=STORAGE_BACKEND if STORAGE_BACKEND in {"remote", "local"} else "remote",
        help="Backend lưu dữ liệu. Mặc định remote/HDFS để chạy trên server.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=MAX_RUNTIME_SECONDS,
        help="Số giây tối đa cho một lần chạy daily. Mặc định 10800 giây tương đương 3 giờ. Đặt 0 để tắt.",
    )
    args = parser.parse_args()

    main(
        storage_backend=args.storage,
        max_runtime_seconds=args.max_runtime_seconds,
    )
