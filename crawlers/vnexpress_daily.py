import json
import csv
import os
import tempfile
import argparse
from datetime import datetime, timedelta
import re
import time

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from remote_storage import RemoteStorage


# ========== CONFIG ==========
STORAGE = None
STORAGE_BACKEND = os.getenv("CRAWLER_STORAGE", "remote").strip().lower()
VNEXPRESS_DIR = "vnexpress"
LOCAL_DATA_DIR = os.getenv(
    "CRAWLER_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)

BASE_URLS = [
    "https://vnexpress.net/khoa-hoc-cong-nghe/thiet-bi",
    "https://vnexpress.net/khoa-hoc-cong-nghe/ai",
    "https://vnexpress.net/khoa-hoc-cong-nghe/vu-tru",
    "https://vnexpress.net/khoa-hoc-cong-nghe/chuyen-doi-so",
]

CHECKPOINT_PATH = "vnexpress_checkpoint.json"
POST_CSV_PATH = "post_vnexpress.csv"
COMMENT_CSV_PATH = "comment_vnexpress.csv"
CRAWL_DATE = os.getenv("CRAWLER_OUTPUT_DATE", datetime.now().strftime("%Y-%m-%d"))

REQUEST_TIMEOUT = 30
SLEEP_AFTER_OPEN_POST = float(os.getenv("VNEXPRESS_SLEEP_AFTER_OPEN_POST", "3"))
SLEEP_BETWEEN_POSTS = float(os.getenv("VNEXPRESS_SLEEP_BETWEEN_POSTS", "1.5"))
SLEEP_BETWEEN_PAGES = float(os.getenv("VNEXPRESS_SLEEP_BETWEEN_PAGES", "2"))

MAX_CLICK_ROUNDS = int(os.getenv("VNEXPRESS_MAX_CLICK_ROUNDS", "300"))

# Chống trường hợp checkpoint rỗng hoặc báo đổi layout làm không gặp bài cũ.
# Daily thường chỉ cần 5-20 page là đủ. Nếu muốn chạy lần đầu full thì tăng rất lớn.
MAX_PAGES_PER_BASE = int(os.getenv("VNEXPRESS_MAX_PAGES_PER_BASE", "30"))
MAX_RUNTIME_SECONDS = int(float(os.getenv("CRAWLER_MAX_RUNTIME_SECONDS", "10800")))

CSV_FLUSH_POST_BATCH = int(os.getenv("VNEXPRESS_CSV_FLUSH_POST_BATCH", "5"))


def is_runtime_expired(started_at, max_runtime_seconds):
    return (
        max_runtime_seconds > 0
        and time.monotonic() - started_at >= max_runtime_seconds
    )


def dated_filename(filename, date_text=CRAWL_DATE):
    name, ext = os.path.splitext(filename)
    return f"{name}_{date_text}{ext}"


def daily_post_csv_path():
    return dated_filename(POST_CSV_PATH)


def daily_comment_csv_path():
    return dated_filename(COMMENT_CSV_PATH)


# ========== LOCAL STORAGE ==========
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


# ========== CHECKPOINT HELPERS ==========
def load_checkpoint(path):
    return STORAGE.read_json(
        STORAGE.path(VNEXPRESS_DIR, path),
        {
            "bases": {},
            "processed_posts": [],
            "daily_runs": [],
            "updated_at": None,
        },
    )


def save_checkpoint(path, checkpoint: dict):
    checkpoint["updated_at"] = datetime.now().isoformat()
    STORAGE.write_json(STORAGE.path(VNEXPRESS_DIR, path), checkpoint)


def append_unique(lst, value):
    if value not in lst:
        lst.append(value)


def ensure_base_state(checkpoint: dict, base_url: str):
    bases = checkpoint.setdefault("bases", {})

    if base_url not in bases:
        bases[base_url] = {
            "next_page": 1,
            "visited_pages": [],
            "queued_links": [],
            "crawled_links": [],
            "current_page_url": None,
            "finished": False,
        }

    base_state = bases[base_url]

    # Đảm bảo các field cũ luôn tồn tại cho crawl toàn bộ
    base_state.setdefault("next_page", 1)
    base_state.setdefault("visited_pages", [])
    base_state.setdefault("queued_links", [])
    base_state.setdefault("crawled_links", [])
    base_state.setdefault("current_page_url", None)
    base_state.setdefault("finished", False)

    # Thêm vùng riêng cho daily crawl
    base_state.setdefault("daily", {
        "last_run_at": None,
        "last_pages_checked": [],
        "last_new_links": [],
        "last_stop_reason": None,
        "last_old_link_met": None,
        "last_error": None,
    })

    checkpoint.setdefault("processed_posts", [])
    checkpoint.setdefault("updated_at", None)

    return base_state


def mark_post_processed(checkpoint, base_state, processed_posts_set, link):
    append_unique(checkpoint["processed_posts"], link)
    append_unique(base_state["crawled_links"], link)
    processed_posts_set.add(link)


# ========== SELENIUM HELPERS ==========
def create_driver():
    profile_dir = tempfile.TemporaryDirectory(prefix="vnexpress-chrome-")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-data-dir={profile_dir.name}")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")

    # Tối ưu nhẹ cho server.
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--window-size=1366,768")

    return webdriver.Chrome(options=options), profile_dir


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


def is_driver_session_lost(exc: Exception):
    text = str(exc).lower()
    return any(
        keyword in text
        for keyword in [
            "connection refused",
            "connection reset by peer",
            "connectionreseterror",
            "connection aborted",
            "connectionabortederror",
            "failed to establish a new connection",
            "max retries exceeded",
            "invalid session id",
            "chrome not reachable",
            "session deleted",
            "disconnected",
            "winerror 10054",
            "winerror 10061",
            "/session/",
        ]
    )


def restart_driver(driver_state: dict):
    close_driver(driver_state.get("driver"), driver_state.get("profile_dir"))
    driver_state["driver"], driver_state["profile_dir"] = create_driver()


def safe_parse_post_and_comments(driver_state: dict, link: str):
    for attempt in range(2):
        try:
            return parse_post_and_comments_from_link(driver_state["driver"], link)
        except Exception as exc:
            if attempt == 0 and is_driver_session_lost(exc):
                print("  -> Selenium mất session, khởi động lại driver và thử lại")
                restart_driver(driver_state)
                continue
            raise


# ========== PARSING HELPERS ==========
def extract_post_id(link: str):
    m = re.search(r"(\d+)(?:\.html)?/?$", link.strip())
    return int(m.group(1)) if m else None


def extract_user_id(href: str):
    if not href:
        return None

    m = re.search(r"(\d+)(?:\.html)?/?$", href.strip())
    return int(m.group(1)) if m else None


def parse_comment_time(raw_time: str, now: datetime = None):
    if not raw_time:
        return None

    now = now or datetime.now()
    text = " ".join(raw_time.split()).lower()

    rel_h = re.search(r"(\d+)\s*h\s*trước", text)
    if rel_h:
        dt = now - timedelta(hours=int(rel_h.group(1)))
        return dt.strftime("%H:%M %d/%m/%Y")

    rel_min = re.search(r"(\d+)\s*phút\s*trước", text)
    if rel_min:
        dt = now - timedelta(minutes=int(rel_min.group(1)))
        return dt.strftime("%H:%M %d/%m/%Y")

    abs_1 = re.search(
        r"(\d{1,2}):(\d{2})\s+(\d{1,2})/(\d{1,2})(?:/(\d{4}))?",
        text,
    )
    if abs_1:
        hour, minute, day, month, year = abs_1.groups()
        y = int(year) if year else now.year
        dt = datetime(y, int(month), int(day), int(hour), int(minute))
        return dt.strftime("%H:%M %d/%m/%Y")

    abs_2 = re.search(
        r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\s*,?\s*(\d{1,2}):(\d{2})",
        text,
    )
    if abs_2:
        day, month, year, hour, minute = abs_2.groups()
        y = int(year) if year else now.year
        dt = datetime(y, int(month), int(day), int(hour), int(minute))
        return dt.strftime("%H:%M %d/%m/%Y")

    return raw_time.strip()


def extract_comment_content(comment_node):
    content_node = (
        comment_node.select_one("p.full_content")
        or comment_node.select_one("p.content_more")
        or comment_node.select_one("p.content")
    )

    if not content_node:
        return None

    cloned = BeautifulSoup(str(content_node), "html.parser")
    name_tag = cloned.select_one("span.txt-name")
    if name_tag:
        name_tag.decompose()

    return cloned.get_text(" ", strip=True)


def extract_reaction_detail(comment_node):
    detail = {}

    reaction_items = comment_node.select("div.reactions-detail div.item")
    for item in reaction_items:
        img = item.select_one("span.icons img")
        strong = item.select_one("strong")

        react_name = img.get("alt", "").strip() if img else ""
        count_text = strong.get_text(" ", strip=True) if strong else ""

        m = re.search(r"\d+", count_text)
        if react_name and m:
            detail[react_name] = detail.get(react_name, 0) + int(m.group())

    return detail


def expand_all_comments(driver):
    selectors = [
        "a.view_all_reply",
        "a.txt_666",
        "p.count-reply",
        "div.view_more_coment.width_common.mb10",
    ]

    for _ in range(MAX_CLICK_ROUNDS):
        buttons = []
        for css in selectors:
            buttons.extend(driver.find_elements(By.CSS_SELECTOR, css))

        if not buttons:
            break

        clicked = 0
        for btn in buttons:
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});",
                    btn,
                )
                time.sleep(0.15)
                driver.execute_script("arguments[0].click();", btn)
                clicked += 1
                time.sleep(0.25)
            except Exception:
                continue

        if clicked == 0:
            break


def parse_post_and_comments_from_link(driver, link: str):
    driver.get(link)
    time.sleep(SLEEP_AFTER_OPEN_POST)

    expand_all_comments(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    post_id = extract_post_id(link)

    post_time_node = soup.select_one("span.date")
    post_time = post_time_node.get_text(" ", strip=True) if post_time_node else None

    title_node = soup.select_one("h1.title-detail")
    title = title_node.get_text(" ", strip=True) if title_node else None

    description_node = soup.select_one("p.description")
    description = description_node.get_text(" ", strip=True) if description_node else None

    content = "\n".join(
        p.get_text(" ", strip=True)
        for p in soup.select("p.Normal")
    )

    post_record = {
        "id_post": post_id,
        "link_post": link,
        "post_title": title,
        "post_time": post_time,
        "post_description": description,
        "post_content": content,
        "crawled_at": datetime.now().isoformat(),
    }

    comment_records = []

    for node in soup.select("div.content-comment"):
        nickname = node.select_one("a.nickname")
        user_name = nickname.get_text(" ", strip=True) if nickname else None
        user_href = nickname.get("href", "") if nickname else ""
        user_id = extract_user_id(user_href)

        comment_content = extract_comment_content(node)

        time_node = node.select_one("span.time-com") or node.select_one("span.time")
        raw_time = time_node.get_text(" ", strip=True) if time_node else ""
        comment_time = parse_comment_time(raw_time)

        reaction_detail = extract_reaction_detail(node)

        comment_records.append(
            {
                "id_post": post_id,
                "link_post": link,
                "user_id": user_id,
                "user_name": user_name,
                "comment_content": comment_content,
                "comment_time": comment_time,
                "reaction_detail": json.dumps(reaction_detail, ensure_ascii=False),
                "crawled_at": datetime.now().isoformat(),
            }
        )

    return post_record, comment_records


# ========== PAGE LINK EXTRACT ==========
def build_page_url(base_url: str, page: int):
    # VnExpress page 1 có thể dùng base trực tiếp hoặc -p1/.
    # Dùng -p1/ để đồng nhất với code cũ.
    return f"{base_url}-p{page}/"


def fetch_article_links_from_page(base_url: str, page: int):
    page_url = build_page_url(base_url, page)
    print(f"[PAGE] {page_url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(page_url, timeout=REQUEST_TIMEOUT, headers=headers)

    if r.status_code != 200:
        return page_url, [], f"status_{r.status_code}"

    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    seen = set()

    # Selector chính đang dùng trong code cũ.
    for a in soup.select("h2.title-news a"):
        href = a.get("href")
        if href and href not in seen:
            links.append(href)
            seen.add(href)

    # Fallback nếu layout thay đổi.
    if not links:
        for a in soup.select("h3.title-news a, article.item-news a"):
            href = a.get("href")
            if href and href.startswith("https://vnexpress.net/") and href not in seen:
                links.append(href)
                seen.add(href)

    return page_url, links, None


# ========== CSV HELPERS ==========
def append_records_to_csv(path, records: list):
    if not records:
        return

    STORAGE.append_csv(
        STORAGE.path(VNEXPRESS_DIR, dated_filename(path)),
        records,
        encoding="utf-8-sig",
    )


def flush_crawled_batch(
    post_batch,
    comment_batch,
    link_batch,
    base_state,
    checkpoint,
    processed_posts,
):
    if not link_batch:
        return

    append_records_to_csv(POST_CSV_PATH, post_batch)
    append_records_to_csv(COMMENT_CSV_PATH, comment_batch)

    for link in link_batch:
        mark_post_processed(checkpoint, base_state, processed_posts, link)

    save_checkpoint(CHECKPOINT_PATH, checkpoint)

    post_batch.clear()
    comment_batch.clear()
    link_batch.clear()


# ========== DAILY CRAWL CORE ==========
def crawl_base_daily(base_url, checkpoint, driver_state, started_at, max_runtime_seconds):
    """
    Daily incremental crawl.

    Logic:
    - Luôn bắt đầu từ page = 1.
    - Crawl link từ trên xuống dưới.
    - Nếu gặp link đã có trong processed_posts hoặc crawled_links thì dừng base hiện tại.
    - Không cập nhật next_page, visited_pages, queued_links, finished của full crawl.
    - Chỉ ghi thông tin daily vào base_state["daily"].
    """

    base_state = ensure_base_state(checkpoint, base_url)

    # Đảm bảo checkpoint cũ có đủ field cần thiết
    checkpoint.setdefault("processed_posts", [])
    base_state.setdefault("crawled_links", [])

    # Namespace riêng cho daily, không phá field của full crawl
    base_state.setdefault("daily", {
        "last_run_at": None,
        "last_pages_checked": [],
        "last_new_links": [],
        "last_stop_reason": None,
        "last_old_link_met": None,
        "last_error": None,
    })

    processed_posts = set(checkpoint.get("processed_posts", []))
    base_crawled = set(base_state.get("crawled_links", []))

    page = 1
    stop_base = False

    pages_checked = []
    new_links_this_run = []

    stop_reason = None
    old_link_met = None
    last_error = None

    post_batch = []
    comment_batch = []
    link_batch = []

    print(f"\n=== DAILY START BASE: {base_url} ===")

    while (
        not stop_base
        and page <= MAX_PAGES_PER_BASE
        and not is_runtime_expired(started_at, max_runtime_seconds)
    ):
        page_url, article_links, page_error = fetch_article_links_from_page(base_url, page)

        pages_checked.append(page)

        if page_error:
            stop_reason = page_error
            print(f"  -> lỗi page: {page_error}, dừng base này")
            break

        if not article_links:
            stop_reason = "no_links_on_page"
            print("  -> page không có link bài viết, dừng base này")
            break

        print(f"  -> tìm thấy {len(article_links)} link trên page {page}")

        for link_post in article_links:
            if is_runtime_expired(started_at, max_runtime_seconds):
                stop_base = True
                stop_reason = "runtime_limit"
                print("  -> đạt giới hạn thời gian chạy, dừng base sau khi flush")
                break

            # Daily incremental:
            # gặp bài cũ là dừng luôn base, vì các bài phía dưới thường cũ hơn.
            if link_post in processed_posts or link_post in base_crawled:
                stop_base = True
                stop_reason = "met_old_link"
                old_link_met = link_post

                print(f"  -> gặp bài cũ, dừng base: {link_post}")
                break

            print(f"[NEW POST] {link_post}")

            try:
                post_record, comment_records = safe_parse_post_and_comments(
                    driver_state,
                    link_post,
                )

                post_batch.append(post_record)
                comment_batch.extend(comment_records)
                link_batch.append(link_post)
                new_links_this_run.append(link_post)

                print(f"  -> crawl xong post + {len(comment_records)} comments")

                if len(post_batch) >= CSV_FLUSH_POST_BATCH:
                    flush_crawled_batch(
                        post_batch,
                        comment_batch,
                        link_batch,
                        base_state,
                        checkpoint,
                        processed_posts,
                    )

                    # Sau khi flush, cập nhật lại set để tránh trùng trong cùng lần chạy
                    base_crawled = set(base_state.get("crawled_links", []))

                time.sleep(SLEEP_BETWEEN_POSTS)

            except Exception as exc:
                # Lưu các bài đã crawl thành công trước khi lỗi
                flush_crawled_batch(
                    post_batch,
                    comment_batch,
                    link_batch,
                    base_state,
                    checkpoint,
                    processed_posts,
                )

                last_error = str(exc)
                stop_reason = "post_error"
                stop_base = True

                print(f"  -> lỗi post, dừng base để lần sau chạy lại từ page 1: {exc}")
                break

        # Flush cuối mỗi page
        flush_crawled_batch(
            post_batch,
            comment_batch,
            link_batch,
            base_state,
            checkpoint,
            processed_posts,
        )

        # Cập nhật lại set sau khi flush cuối page
        base_crawled = set(base_state.get("crawled_links", []))

        if stop_base:
            break

        page += 1
        time.sleep(SLEEP_BETWEEN_PAGES)

    if is_runtime_expired(started_at, max_runtime_seconds) and not stop_reason:
        stop_reason = "runtime_limit"

    if page > MAX_PAGES_PER_BASE and not stop_reason:
        stop_reason = "max_pages_reached"

    # Ghi riêng trạng thái daily, không ghi vào next_page / visited_pages / queued_links
    daily_state = base_state.setdefault("daily", {})

    daily_state["last_run_at"] = datetime.now().isoformat()
    daily_state["last_pages_checked"] = pages_checked
    daily_state["last_new_links"] = new_links_this_run
    daily_state["last_stop_reason"] = stop_reason
    daily_state["last_old_link_met"] = old_link_met
    daily_state["last_error"] = last_error

    save_checkpoint(CHECKPOINT_PATH, checkpoint)

    print(
        f"=== DAILY END BASE: {base_url} | "
        f"new={len(new_links_this_run)} | "
        f"pages={pages_checked} | "
        f"reason={stop_reason} ==="
    )

    return {
        "base_url": base_url,
        "new_count": len(new_links_this_run),
        "pages_checked": pages_checked,
        "stop_reason": stop_reason,
        "old_link_met": old_link_met,
        "error": last_error,
    }


def run_daily(storage_backend=STORAGE_BACKEND, max_runtime_seconds=MAX_RUNTIME_SECONDS):
    global STORAGE

    STORAGE = create_storage(storage_backend)

    checkpoint = load_checkpoint(CHECKPOINT_PATH)

    checkpoint.setdefault("bases", {})
    checkpoint.setdefault("processed_posts", [])
    checkpoint.setdefault("daily_runs", [])

    for base_url in BASE_URLS:
        ensure_base_state(checkpoint, base_url)

    save_checkpoint(CHECKPOINT_PATH, checkpoint)

    driver_state = {
        "driver": None,
        "profile_dir": None,
    }

    run_started_at = datetime.now().isoformat()
    run_summary = {
        "started_at": run_started_at,
        "ended_at": None,
        "bases": [],
        "stop_reason": None,
    }
    started_at = time.monotonic()

    try:
        driver_state["driver"], driver_state["profile_dir"] = create_driver()

        for base_url in BASE_URLS:
            if is_runtime_expired(started_at, max_runtime_seconds):
                run_summary["stop_reason"] = "runtime_limit"
                print("[STOP RUNTIME LIMIT] daily run")
                break

            result = crawl_base_daily(
                base_url,
                checkpoint,
                driver_state,
                started_at,
                max_runtime_seconds,
            )
            run_summary["bases"].append(result)

            if result.get("stop_reason") == "runtime_limit":
                run_summary["stop_reason"] = "runtime_limit"
                break

    finally:
        close_driver(driver_state.get("driver"), driver_state.get("profile_dir"))

        run_summary["ended_at"] = datetime.now().isoformat()
        checkpoint.setdefault("daily_runs", []).append(run_summary)

        # Chỉ giữ lịch sử 30 lần chạy gần nhất cho checkpoint nhẹ.
        checkpoint["daily_runs"] = checkpoint["daily_runs"][-30:]

        save_checkpoint(CHECKPOINT_PATH, checkpoint)
        STORAGE.close()

    print("\nDONE DAILY CRAWL")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Post CSV: {daily_post_csv_path()}")
    print(f"Comment CSV: {daily_comment_csv_path()}")


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

    run_daily(
        storage_backend=args.storage,
        max_runtime_seconds=args.max_runtime_seconds,
    )
