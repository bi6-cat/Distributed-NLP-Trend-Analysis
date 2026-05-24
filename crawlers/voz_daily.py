import time
import json
import logging
import random
import argparse
import csv
import os
import re
import tempfile
from typing import List, Tuple, Dict, Any, Optional, Set

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from remote_storage import RemoteStorage

# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://voz.vn/"
VOZ_DIR = "voz"
LOCAL_DATA_DIR = os.getenv(
    "CRAWLER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "server_data", "raw_data"),
)

CHECKPOINT_FILE = "checkpoint.json"
POST_FILE = "posts.csv"
COMMENT_FILE = "comments.csv"

STORAGE = None
STORAGE_BACKEND = os.getenv("CRAWLER_STORAGE", "remote").strip().lower()

# Delay mặc định. Có thể override bằng command line.
DELAY_MIN = 4
DELAY_MAX = 9

MAX_RETRY = 3
BACKOFF_BASE = 30

LONG_SLEEP_EVERY_THREADS = 50
LONG_SLEEP_MIN = 120
LONG_SLEEP_MAX = 300

HEADLESS = True

# Daily chỉ quét một số page đầu của mỗi forum.
# Khi gặp id_post cũ thì dừng forum hiện tại và chuyển forum tiếp theo.
MAX_FORUM_PAGES_DAILY = 5
MAX_RUNTIME_SECONDS = int(float(os.getenv("CRAWLER_MAX_RUNTIME_SECONDS", "10800")))

Link_FORUM = [
    "https://voz.vn/s/review-san-pham.103/",
    "https://voz.vn/f/tu-van-cau-hinh.70/",
    "https://voz.vn/f/overclocking-cooling-modding.6/",
    "https://voz.vn/f/amd.25/",
    "https://voz.vn/f/intel.24/",
    "https://voz.vn/f/gpu-man-hinh.8/",
    "https://voz.vn/f/phan-cung-chung.9/",
    "https://voz.vn/f/thiet-bi-ngoai-vi-phu-kien-mang.30/",
    "https://voz.vn/f/server-nas-render-farm.83/",
    "https://voz.vn/f/small-form-factor-pc.61/",
    "https://voz.vn/f/hackintosh.62/",
    "https://voz.vn/f/may-tinh-xach-tay.47/",
    "https://voz.vn/f/phan-mem.13/",
    "https://voz.vn/f/app-di-dong.21/",
    "https://voz.vn/f/pc-gaming.11/",
    "https://voz.vn/f/console-gaming.22/",
    "https://voz.vn/f/android.32/",
    "https://voz.vn/f/apple.36/",
    "https://voz.vn/f/multimedia.31/",
    "https://voz.vn/f/do-dien-tu-thiet-bi-gia-dung.10/",
    "https://voz.vn/f/chup-anh-quay-phim.75/",
    "https://voz.vn/f/goc-chien-luoc.101/",
    "https://voz.vn/f/may-tinh-de-ban.68/",
    "https://voz.vn/f/may-tinh-xach-tay.72/",
    "https://voz.vn/f/dien-thoai-di-dong.76/",
]


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class RuntimeLimitReached(Exception):
    pass


def ensure_runtime_available(started_at: float, max_runtime_seconds: int) -> None:
    if max_runtime_seconds > 0 and time.monotonic() - started_at >= max_runtime_seconds:
        raise RuntimeLimitReached(
            f"Reached runtime limit: {max_runtime_seconds} seconds"
        )


# =====================================================
# LOCAL STORAGE
# =====================================================

class LocalStorage:
    """
    Storage local giống cấu trúc file crawl full:
    path/read_json/write_json/append_csv/close.
    """

    def __init__(self, base_dir: str = LOCAL_DATA_DIR):
        self.base_dir = os.path.abspath(base_dir)

    def path(self, *parts: str) -> str:
        return os.path.join(
            self.base_dir,
            *[str(part).strip("/\\") for part in parts],
        )

    def read_json(self, path: str, default: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(path):
            return default

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, path: str, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)

    def append_csv(self, path: str, records: List[Dict[str, Any]]) -> None:
        if not records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0

        # Giữ ổn định schema theo file đã tồn tại.
        if not write_header:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                fieldnames = next(reader, None)
        else:
            fieldnames = None

        if not fieldnames:
            fieldnames = list(records[0].keys())

        normalized_records = []
        for record in records:
            normalized_records.append({key: record.get(key) for key in fieldnames})

        encoding = "utf-8-sig" if write_header else "utf-8"

        with open(path, "a", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

            if write_header:
                writer.writeheader()

            writer.writerows(normalized_records)

    def read_csv_column_as_str_set(self, path: str, column: str) -> Set[str]:
        return read_csv_column_as_str_set(path, column)

    def close(self) -> None:
        pass


def create_storage(backend: str):
    backend = (backend or "remote").strip().lower()

    if backend in {"remote", "server", "hdfs"}:
        logger.info("Using remote HDFS storage")
        return RemoteStorage()

    if backend in {"local", "file", "filesystem"}:
        logger.info("Using local filesystem storage")
        return LocalStorage()

    raise ValueError(
        f"Unsupported storage backend: {backend}. Use remote or local."
    )


# =====================================================
# DRIVER
# =====================================================

def create_driver() -> Tuple[webdriver.Chrome, tempfile.TemporaryDirectory]:
    profile_dir = tempfile.TemporaryDirectory(prefix="voz-daily-chrome-")

    options = Options()
    options.page_load_strategy = "eager"

    if HEADLESS:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1366,768")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-sync")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--mute-audio")

    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--blink-settings=imagesEnabled=false")

    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    options.add_argument(f"--user-data-dir={profile_dir.name}")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(45)
    driver.implicitly_wait(0)

    # Chặn tài nguyên nặng để giảm tải server và tăng tốc.
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setBlockedURLs", {
            "urls": [
                "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg", "*.ico",
                "*.woff", "*.woff2", "*.ttf", "*.otf",
                "*.mp4", "*.webm",
            ]
        })
    except Exception as e:
        logger.warning(f"Cannot block resources: {e}")

    return driver, profile_dir


# =====================================================
# CHECKPOINT DAILY
# =====================================================

"""
Checkpoint daily chỉ dùng để resume thread đang crawl dở.
Không dùng checkpoint daily để quyết định bài cũ.
Bài cũ được check bằng id_post trong posts.csv.

{
  "type": "daily",
  "last_run": "2026-05-22 22:30:00",
  "daily_threads": {
    "https://voz.vn/t/example.123/": {
      "id_post": "123",
      "done": true,
      "last_page": 3,
      "total_pages": 3
    }
  },
  "daily_runs": []
}
"""

def default_daily_checkpoint() -> Dict[str, Any]:
    return {
        "forums": {},
        "threads": {},
        "last_run": None,
        "daily_threads": {},
        "daily_runs": []
    }


def load_daily_checkpoint() -> Dict[str, Any]:
    checkpoint = STORAGE.read_json(
        STORAGE.path(VOZ_DIR, CHECKPOINT_FILE),
        default_daily_checkpoint(),
    )
    checkpoint.setdefault("forums", {})
    checkpoint.setdefault("threads", {})
    checkpoint.setdefault("daily_threads", {})
    checkpoint.setdefault("daily_runs", [])
    checkpoint.setdefault("last_run", None)
    return checkpoint


def save_daily_checkpoint(cp: Dict[str, Any]) -> None:
    cp["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STORAGE.write_json(
        STORAGE.path(VOZ_DIR, CHECKPOINT_FILE),
        cp
    )


# =====================================================
# SAFE LOAD PAGE
# =====================================================

def random_sleep() -> None:
    sleep_time = random.uniform(DELAY_MIN, DELAY_MAX)
    logger.info(f"😴 Sleep {sleep_time:.2f}s")
    time.sleep(sleep_time)


def load_page_once(driver: webdriver.Chrome, url: str) -> BeautifulSoup:
    logger.info(f"🌐 LOAD: {url}")
    driver.get(url)
    random_sleep()
    return BeautifulSoup(driver.page_source, "html.parser")


def load_page(driver: webdriver.Chrome, url: str) -> BeautifulSoup:
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            return load_page_once(driver, url)
        except Exception as e:
            last_error = e
            wait_time = BACKOFF_BASE * attempt + random.uniform(5, 15)

            logger.warning(
                f"⚠️ LOAD FAILED attempt={attempt}/{MAX_RETRY} "
                f"url={url} error={e}"
            )
            logger.info(f"⏳ Backoff {wait_time:.2f}s")
            time.sleep(wait_time)

    raise RuntimeError(f"Cannot load page after {MAX_RETRY} retries: {url}") from last_error


# =====================================================
# PAGINATION / URL
# =====================================================

def get_last_page(soup: BeautifulSoup) -> int:
    pages = soup.select("li.pageNav-page")

    if not pages:
        return 1

    nums = []
    for page in pages:
        try:
            nums.append(int(page.text.strip()))
        except Exception:
            pass

    return max(nums) if nums else 1


def build_page_url(base: str, p: int) -> str:
    if p == 1:
        return base

    return f"{base.rstrip('/')}/page-{p}"


def normalize_thread_url(url: str) -> str:
    """
    Chuẩn hóa URL thread về dạng không có /page-x.
    """
    if not url:
        return url

    url = url.split("#")[0].split("?")[0]
    url = re.sub(r"/page-\d+/?$", "/", url.rstrip("/"))
    return url if url.endswith("/") else f"{url}/"


def extract_post_id_from_url(url: str) -> Optional[str]:
    """
    Ví dụ:
    https://voz.vn/t/hoi-may-anh-sony-mirrorless-2020.4390/
    => 4390
    """
    if not url:
        return None

    clean_url = normalize_thread_url(url).rstrip("/")
    match = re.search(r"\.(\d+)$", clean_url)

    if not match:
        return None

    return match.group(1)


# =====================================================
# GET THREADS
# =====================================================

def get_threads(soup: BeautifulSoup, skip_sticky: bool = True) -> List[str]:
    links = []

    # Ưu tiên duyệt theo structItem để có thể bỏ sticky.
    rows = soup.select("div.structItem--thread")

    if rows:
        for row in rows:
            row_classes = row.get("class", [])
            class_text = " ".join(row_classes).lower()

            if skip_sticky and ("sticky" in class_text or "is-sticky" in class_text):
                continue

            a = None
            for candidate in row.select("div.structItem-title a[href]"):
                href = candidate.get("href", "")
                if "/t/" in href:
                    a = candidate
                    break

            if not a:
                continue

            href = a.get("href")
            if not href or "/t/" not in href:
                continue

            if href.startswith("/"):
                href = BASE_URL.rstrip("/") + href

            links.append(normalize_thread_url(href))
    else:
        for a in soup.select("div.structItem-title a"):
            href = a.get("href")

            if not href or "/t/" not in href:
                continue

            if href.startswith("/"):
                href = BASE_URL.rstrip("/") + href

            links.append(normalize_thread_url(href))

    # Giữ thứ tự, loại duplicate.
    seen = set()
    unique_links = []

    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


# =====================================================
# OLD POST IDS FROM POST FILES
# =====================================================

def read_csv_column_as_str_set(path: str, column: str) -> Set[str]:
    ids = set()

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ids

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                return ids

            target_col = None
            for col in reader.fieldnames:
                if col and col.strip().lower() == column.strip().lower():
                    target_col = col
                    break

            if not target_col:
                logger.warning(f"CSV has no {column} column: {path}")
                return ids

            for row in reader:
                value = row.get(target_col)
                if value is None:
                    continue

                value = str(value).strip()
                if value and value.lower() not in {"none", "nan", "null"}:
                    ids.add(value)

    except Exception as e:
        logger.warning(f"Cannot read id_post from {path}: {e}")

    return ids


def load_existing_post_ids() -> Set[str]:
    """
    Đọc id_post từ posts.csv.
    Đây là nguồn chính để daily biết bài nào đã có.
    """
    path = STORAGE.path(VOZ_DIR, POST_FILE)
    known_ids = STORAGE.read_csv_column_as_str_set(path, "id_post")
    logger.info(f"Loaded old ids from {path}: {len(known_ids)}")

    logger.info(f"Total existing id_post: {len(known_ids)}")
    return known_ids


# =====================================================
# REACTION POPUP
# =====================================================

def close_overlay_if_any(driver: webdriver.Chrome) -> None:
    try:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(random.uniform(0.2, 0.5))
    except Exception:
        pass


def extract_reactions_from_message(
    driver: webdriver.Chrome,
    message_element,
    timeout: int = 5
) -> Optional[str]:
    try:
        element = message_element.find_element(
            By.CSS_SELECTOR,
            "a.reactionsBar-link"
        )

        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});",
            element
        )
        time.sleep(random.uniform(0.3, 0.8))

        driver.execute_script("arguments[0].click();", element)

        popup = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".overlay-container")
            )
        )

        react_comments = popup.find_elements(
            By.CSS_SELECTOR,
            "span.reaction-text.js-reactionText"
        )

        reactions = "|".join(
            [r.text.strip() for r in react_comments if r.text.strip()]
        )

        close_overlay_if_any(driver)
        return reactions if reactions else None

    except Exception:
        close_overlay_if_any(driver)
        return None


# =====================================================
# PARSE POST + COMMENT
# =====================================================

def safe_text(obj) -> Optional[str]:
    try:
        return obj.text.strip()
    except Exception:
        return None


def parse_posts(
    soup: BeautifulSoup,
    driver: webdriver.Chrome,
    url: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    comments = []

    names = soup.select('span[itemprop="name"]')

    try:
        category = names[1].text.strip()
    except Exception:
        category = None

    try:
        subcategory = names[2].text.strip()
    except Exception:
        subcategory = None

    # Ưu tiên lấy id_post từ URL thread vì ổn định hơn selector.
    id_post = extract_post_id_from_url(url)

    if not id_post:
        try:
            id_post = soup.select('a.u-concealed')[1].get('href').split('.')[-1].split('/')[0]
        except Exception:
            id_post = None

    title = safe_text(soup.select_one('h1.p-title-value'))

    try:
        time_post = soup.select_one('time.u-dt').get('title')
    except Exception:
        time_post = None

    try:
        id_author = soup.select_one('a.username').get('data-user-id')
    except Exception:
        id_author = None

    author_name = safe_text(soup.select_one('a.username'))
    replies_post = safe_text(soup.select_one('dl.pairs.pairs--justified.count--replies'))
    views_post = safe_text(soup.select_one('dl.pairs.pairs--justified.count--views'))

    post = {
        "id_post": id_post,
        "title": title,
        "time_post": time_post,
        "replies_post": replies_post,
        "views_post": views_post,
        "id_author": id_author,
        "author_name": author_name,
        "category": category,
        "subcategory": subcategory,
        "url": normalize_thread_url(url),
        "crawl_source": "daily",
        "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    contents = soup.select('div.message-inner')
    contents_driver = driver.find_elements(By.CSS_SELECTOR, "div.message-inner")

    for idx, c in enumerate(contents):
        username_el = c.select_one('a.username')

        try:
            id_user = username_el.get('data-user-id')
        except Exception:
            id_user = None

        user = safe_text(username_el)

        try:
            time_comment = c.select_one('time').get('title')
        except Exception:
            time_comment = None

        comment = safe_text(c.select_one('div.message-content'))

        reactions = None
        if idx < len(contents_driver):
            reactions = extract_reactions_from_message(
                driver=driver,
                message_element=contents_driver[idx]
            )

        comments.append({
            "id_post": id_post,
            "id_user": id_user,
            "user": user,
            "time": time_comment,
            "comment": comment,
            "url": url,
            "reactions": reactions,
            "crawl_source": "daily",
            "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return post, comments


# =====================================================
# SAVE DATA
# =====================================================

def save_daily_data(posts: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> None:
    if posts:
        STORAGE.append_csv(
            STORAGE.path(VOZ_DIR, POST_FILE),
            posts
        )

    if comments:
        STORAGE.append_csv(
            STORAGE.path(VOZ_DIR, COMMENT_FILE),
            comments
        )


# =====================================================
# CRAWL THREAD DAILY
# =====================================================

def crawl_thread_daily(
    driver: webdriver.Chrome,
    thread_url: str,
    checkpoint: Dict[str, Any],
    existing_post_ids: Set[str],
    started_at: float,
    max_runtime_seconds: int,
) -> bool:
    """
    Crawl 1 thread mới hoặc resume thread daily đang crawl dở.

    Return True nếu crawl thành công.
    Return False nếu lỗi.
    """
    thread_url = normalize_thread_url(thread_url)
    id_post = extract_post_id_from_url(thread_url)

    if not id_post:
        logger.warning(f"Cannot extract id_post from thread URL: {thread_url}")
        return False

    state = checkpoint["daily_threads"].get(thread_url, {})

    if state.get("done") is True:
        logger.info(f"SKIP DAILY THREAD DONE: {thread_url}")
        return True

    logger.info(f"DAILY THREAD: {thread_url} | id_post={id_post}")

    try:
        ensure_runtime_available(started_at, max_runtime_seconds)
        soup = load_page(driver, thread_url)
        last_page = get_last_page(soup)

        last_crawled = int(state.get("last_page", 0) or 0)

        # Nếu đang resume từ giữa thì không lưu lại post row nữa.
        should_save_post = last_crawled == 0

        for p in range(last_crawled + 1, last_page + 1):
            ensure_runtime_available(started_at, max_runtime_seconds)
            page_url = build_page_url(thread_url, p)

            logger.info(f"DAILY THREAD PAGE {p}/{last_page}: {page_url}")

            soup = load_page(driver, page_url)
            post, comments = parse_posts(soup, driver, page_url)

            parsed_id = str(post.get("id_post") or id_post)
            post["id_post"] = parsed_id

            for c in comments:
                c["id_post"] = parsed_id

            # Tránh duplicate post trong posts.csv:
            # chỉ lưu post ở page đầu tiên của thread.
            posts_to_save = [post] if should_save_post and p == 1 else []
            save_daily_data(posts_to_save, comments)

            checkpoint["daily_threads"][thread_url] = {
                "id_post": parsed_id,
                "done": False,
                "last_page": p,
                "total_pages": last_page,
            }
            save_daily_checkpoint(checkpoint)

            logger.info(
                f"SAVED DAILY PAGE {p}/{last_page} | "
                f"comments={len(comments)}"
            )

        checkpoint["daily_threads"][thread_url] = {
            "id_post": id_post,
            "done": True,
            "last_page": last_page,
            "total_pages": last_page,
        }
        save_daily_checkpoint(checkpoint)

        existing_post_ids.add(str(id_post))

        logger.info(f"DONE DAILY THREAD: {thread_url}")
        return True

    except Exception as e:
        logger.error(f"ERROR DAILY THREAD {thread_url}: {e}")

        checkpoint["daily_threads"].setdefault(thread_url, {
            "id_post": id_post,
            "done": False,
            "last_page": 0,
        })
        save_daily_checkpoint(checkpoint)

        return False


# =====================================================
# RESUME INCOMPLETE DAILY THREADS
# =====================================================

def resume_incomplete_daily_threads(
    driver: webdriver.Chrome,
    checkpoint: Dict[str, Any],
    existing_post_ids: Set[str],
    started_at: float,
    max_runtime_seconds: int,
) -> int:
    threads = checkpoint.get("daily_threads", {})

    incomplete_threads = [
        thread_url
        for thread_url, state in threads.items()
        if state.get("done") is not True
    ]

    if not incomplete_threads:
        return 0

    logger.info(f"Resume incomplete daily threads: {len(incomplete_threads)}")

    count = 0
    for thread_url in incomplete_threads:
        ensure_runtime_available(started_at, max_runtime_seconds)
        ok = crawl_thread_daily(
            driver=driver,
            thread_url=thread_url,
            checkpoint=checkpoint,
            existing_post_ids=existing_post_ids,
            started_at=started_at,
            max_runtime_seconds=max_runtime_seconds,
        )
        if ok:
            count += 1

    return count


# =====================================================
# CRAWL FORUM DAILY
# =====================================================

def crawl_forum_daily(
    driver: webdriver.Chrome,
    forum_url: str,
    checkpoint: Dict[str, Any],
    existing_post_ids: Set[str],
    max_forum_pages: int,
    thread_counter: Dict[str, int],
    skip_sticky: bool,
    started_at: float,
    max_runtime_seconds: int,
) -> int:
    """
    Daily logic:
    - Crawl forum từ page 1 xuống.
    - Lấy thread theo thứ tự hiển thị.
    - Nếu id_post đã tồn tại trong posts.csv => dừng forum hiện tại.
    - Nếu là bài mới => crawl toàn bộ thread.
    """
    logger.info("=" * 80)
    logger.info(f"DAILY FORUM: {forum_url}")

    new_threads = 0

    for p in range(1, max_forum_pages + 1):
        ensure_runtime_available(started_at, max_runtime_seconds)
        page_url = build_page_url(forum_url, p)
        logger.info(f"DAILY FORUM PAGE {p}/{max_forum_pages}: {page_url}")

        try:
            soup = load_page(driver, page_url)
        except Exception as e:
            logger.error(f"ERROR DAILY FORUM PAGE {page_url}: {e}")
            break

        threads = get_threads(soup, skip_sticky=skip_sticky)
        logger.info(f"THREADS FOUND: {len(threads)}")

        if not threads:
            logger.info(f"No thread found, stop forum: {forum_url}")
            break

        for thread_url in threads:
            ensure_runtime_available(started_at, max_runtime_seconds)
            id_post = extract_post_id_from_url(thread_url)

            if not id_post:
                logger.warning(f"Skip thread without id_post: {thread_url}")
                continue

            if str(id_post) in existing_post_ids:
                logger.info(
                    f"MEET OLD id_post={id_post}. "
                    f"Stop forum and move to next forum: {forum_url}"
                )
                return new_threads

            ok = crawl_thread_daily(
                driver=driver,
                thread_url=thread_url,
                checkpoint=checkpoint,
                existing_post_ids=existing_post_ids,
                started_at=started_at,
                max_runtime_seconds=max_runtime_seconds,
            )

            if ok:
                new_threads += 1
                thread_counter["count"] += 1

            if (
                LONG_SLEEP_EVERY_THREADS > 0
                and thread_counter["count"] > 0
                and thread_counter["count"] % LONG_SLEEP_EVERY_THREADS == 0
            ):
                sleep_time = random.uniform(LONG_SLEEP_MIN, LONG_SLEEP_MAX)
                logger.info(
                    f"LONG SLEEP after "
                    f"{thread_counter['count']} daily threads: {sleep_time:.2f}s"
                )
                time.sleep(sleep_time)

    return new_threads


# =====================================================
# MAIN RUN DAILY
# =====================================================

def run_daily(
    max_forum_pages: int,
    delay_min: int,
    delay_max: int,
    headless: bool,
    long_sleep_every: int,
    long_sleep_min: int,
    long_sleep_max: int,
    selected_forums: Optional[List[str]],
    skip_sticky: bool,
    storage_backend: str,
    max_runtime_seconds: int,
) -> None:
    global STORAGE
    global DELAY_MIN, DELAY_MAX, HEADLESS
    global LONG_SLEEP_EVERY_THREADS, LONG_SLEEP_MIN, LONG_SLEEP_MAX

    DELAY_MIN = delay_min
    DELAY_MAX = delay_max
    HEADLESS = headless

    LONG_SLEEP_EVERY_THREADS = long_sleep_every
    LONG_SLEEP_MIN = long_sleep_min
    LONG_SLEEP_MAX = long_sleep_max

    STORAGE = create_storage(storage_backend)

    forums = selected_forums if selected_forums else Link_FORUM

    logger.info("=" * 80)
    logger.info("START DAILY CRAWL")
    logger.info(f"CHECKPOINT: {CHECKPOINT_FILE}")
    logger.info(f"POST FILE: {POST_FILE}")
    logger.info(f"COMMENT FILE: {COMMENT_FILE}")
    logger.info(f"DELAY: {DELAY_MIN}-{DELAY_MAX}s")
    logger.info(f"HEADLESS: {HEADLESS}")
    logger.info(f"MAX FORUM PAGES DAILY: {max_forum_pages}")
    logger.info(f"SKIP STICKY: {skip_sticky}")
    logger.info(f"MAX RUNTIME: {max_runtime_seconds}s")
    logger.info(f"STORAGE BACKEND: {storage_backend}")
    if storage_backend.strip().lower() in {"local", "file", "filesystem"}:
        logger.info(f"LOCAL DATA DIR: {LOCAL_DATA_DIR}")
    logger.info("=" * 80)

    checkpoint = load_daily_checkpoint()
    existing_post_ids = load_existing_post_ids()

    logger.info(f"FORUM COUNT: {len(forums)}")
    for forum_url in forums:
        logger.info(f"   - {forum_url}")

    if not forums:
        logger.warning("No forum selected for this daily run.")
        return

    driver, profile_dir = create_driver()
    thread_counter = {"count": 0}

    total_new_threads = 0
    resumed_threads = 0

    run_record = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_forum_pages": max_forum_pages,
        "forums": forums,
        "new_threads": 0,
        "resumed_threads": 0,
        "stop_reason": None,
    }

    started_at = time.monotonic()

    try:
        resumed_threads = resume_incomplete_daily_threads(
            driver=driver,
            checkpoint=checkpoint,
            existing_post_ids=existing_post_ids,
            started_at=started_at,
            max_runtime_seconds=max_runtime_seconds,
        )

        for forum_url in forums:
            ensure_runtime_available(started_at, max_runtime_seconds)
            try:
                count = crawl_forum_daily(
                    driver=driver,
                    forum_url=forum_url,
                    checkpoint=checkpoint,
                    existing_post_ids=existing_post_ids,
                    max_forum_pages=max_forum_pages,
                    thread_counter=thread_counter,
                    skip_sticky=skip_sticky,
                    started_at=started_at,
                    max_runtime_seconds=max_runtime_seconds,
                )
                total_new_threads += count
                logger.info(f"DONE DAILY FORUM: {forum_url} | new_threads={count}")

            except RuntimeLimitReached:
                raise
            except Exception as e:
                logger.error(f"ERROR DAILY FORUM {forum_url}: {e}")

    except RuntimeLimitReached as e:
        run_record["stop_reason"] = "runtime_limit"
        logger.warning(f"{e}. Stop daily crawl safely.")

    finally:
        run_record["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run_record["new_threads"] = total_new_threads
        run_record["resumed_threads"] = resumed_threads

        checkpoint.setdefault("daily_runs", []).append(run_record)
        # Giữ checkpoint không quá phình.
        checkpoint["daily_runs"] = checkpoint["daily_runs"][-30:]
        save_daily_checkpoint(checkpoint)

        try:
            driver.quit()
        except Exception:
            pass

        try:
            profile_dir.cleanup()
        except Exception:
            pass

        try:
            STORAGE.close()
        except Exception:
            pass

    logger.info("=" * 80)
    logger.info(
        f"FINISHED DAILY CRAWL | "
        f"new_threads={total_new_threads} | resumed_threads={resumed_threads}"
    )
    logger.info("=" * 80)


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-forum-pages",
        type=int,
        default=MAX_FORUM_PAGES_DAILY,
        help="Daily crawl tối đa bao nhiêu page đầu của mỗi forum."
    )

    parser.add_argument(
        "--delay-min",
        type=int,
        default=4,
        help="Delay nhỏ nhất giữa các lần load page."
    )

    parser.add_argument(
        "--delay-max",
        type=int,
        default=9,
        help="Delay lớn nhất giữa các lần load page."
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Chạy Chrome ở chế độ headless. Mặc định bật."
    )

    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Tắt headless để debug bằng giao diện."
    )

    parser.add_argument(
        "--long-sleep-every",
        type=int,
        default=50,
        help="Sau bao nhiêu thread daily thì nghỉ dài. Đặt 0 để tắt."
    )

    parser.add_argument(
        "--long-sleep-min",
        type=int,
        default=120,
        help="Thời gian nghỉ dài nhỏ nhất, đơn vị giây."
    )

    parser.add_argument(
        "--long-sleep-max",
        type=int,
        default=300,
        help="Thời gian nghỉ dài lớn nhất, đơn vị giây."
    )

    parser.add_argument(
        "--forum",
        action="append",
        default=None,
        help="Chỉ crawl một hoặc nhiều forum cụ thể. Có thể truyền nhiều lần."
    )

    parser.add_argument(
        "--include-sticky",
        action="store_true",
        help="Không bỏ qua sticky thread. Mặc định daily sẽ bỏ sticky để tránh gặp bài cũ quá sớm."
    )

    parser.add_argument(
        "--storage",
        choices=["remote", "local"],
        default=STORAGE_BACKEND if STORAGE_BACKEND in {"remote", "local"} else "remote",
        help="Backend lưu dữ liệu. Mặc định remote/HDFS để chạy trên server."
    )

    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=MAX_RUNTIME_SECONDS,
        help="Số giây tối đa cho một lần chạy daily. Mặc định 10800 giây tương đương 3 giờ. Đặt 0 để tắt."
    )

    args = parser.parse_args()

    run_daily(
        max_forum_pages=args.max_forum_pages,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        headless=args.headless,
        long_sleep_every=args.long_sleep_every,
        long_sleep_min=args.long_sleep_min,
        long_sleep_max=args.long_sleep_max,
        selected_forums=args.forum,
        skip_sticky=not args.include_sticky,
        storage_backend=args.storage,
        max_runtime_seconds=args.max_runtime_seconds,
    )
