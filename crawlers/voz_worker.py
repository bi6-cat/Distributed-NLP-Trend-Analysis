import time
import json
import logging
import random
import argparse
import csv
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional

from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from curl_cffi import requests as cffi_requests

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://voz.vn/"
VOZ_DIR = "voz"
LOCAL_DATA_DIR = os.getenv(
    "CRAWLER_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)

# Mỗi worker sẽ tự đổi tên file theo worker_id
CHECKPOINT_FILE = None
POST_FILE = None
COMMENT_FILE = None

STORAGE = None

# Delay mặc định. Có thể override bằng command line.
DELAY_MIN = 4
DELAY_MAX = 9

MAX_RETRY = 3
BACKOFF_BASE = 30

LONG_SLEEP_EVERY_THREADS = 50
LONG_SLEEP_MIN = 120
LONG_SLEEP_MAX = 300

HEADLESS = False
HIDE_WINDOW = False
CHROME_PROFILE_DIR = os.getenv(
    "VOZ_CHROME_PROFILE_DIR",
    os.path.join(LOCAL_DATA_DIR, ".chrome", "voz_worker"),
)
ALLOW_RESOURCE_BLOCKING = os.getenv(
    "VOZ_BLOCK_HEAVY_RESOURCES",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
CLOUDFLARE_WAIT_SECONDS = int(os.getenv("VOZ_CLOUDFLARE_WAIT_SECONDS", "180"))


Link_FORUM = [
    # "https://voz.vn/s/review-san-pham.103/",
    # "https://voz.vn/f/tu-van-cau-hinh.70/",
    "https://voz.vn/f/dien-thoai-di-dong.76/",
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


# =====================================================
# LOCAL STORAGE
# =====================================================

class LocalStorage:
    """
    Storage local có cùng interface tối thiểu với RemoteStorage:
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

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def append_csv(self, path: str, records: List[Dict[str, Any]]) -> None:
        if not records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        fieldnames = list(records[0].keys())

        encoding = "utf-8-sig" if write_header else "utf-8"

        with open(path, "a", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerows(records)

    def close(self) -> None:
        pass


# =====================================================
# WORKER SPLIT
# =====================================================

def get_worker_forums(
    links: List[str],
    worker_id: int,
    links_per_worker: int
) -> List[str]:
    """
    Chia link theo block liên tiếp.

    Ví dụ links_per_worker = 5:
    worker 0: link 0-4
    worker 1: link 5-9
    worker 2: link 10-14
    """
    start = worker_id * links_per_worker
    end = start + links_per_worker
    return links[start:end]


def get_worker_profile_dir(profile_root: str, worker_id: int) -> str:
    return os.path.join(os.path.abspath(profile_root), f"worker_{worker_id}")


# =====================================================
# DRIVER (undetected-chromedriver)
# =====================================================

def create_driver() -> Tuple[uc.Chrome, Optional[tempfile.TemporaryDirectory]]:
    """
    Dùng undetected_chromedriver để bypass Cloudflare và bot detection.

    Profile lưu cookie/session giữa các lần chạy; nếu Cloudflare xuất hiện,
    xác minh thủ công một lần trong cửa sổ Chrome — clearance sẽ được giữ lại.
    """
    profile_dir = None
    user_data_dir = os.path.abspath(CHROME_PROFILE_DIR)
    os.makedirs(user_data_dir, exist_ok=True)

    options = uc.ChromeOptions()
    options.page_load_strategy = "eager"

    if HIDE_WINDOW and not HEADLESS:
        # Đẩy cửa sổ ra ngoài màn hình, ít bị detect hơn headless
        options.add_argument("--window-position=-32000,-32000")
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

    if ALLOW_RESOURCE_BLOCKING:
        options.add_argument("--blink-settings=imagesEnabled=false")

    driver = uc.Chrome(
        options=options,
        user_data_dir=user_data_dir,
        headless=HEADLESS,
        version_main=148,
    )
    driver.set_page_load_timeout(45)
    driver.implicitly_wait(0)

    if ALLOW_RESOURCE_BLOCKING:
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setBlockedURLs", {
                "urls": [
                    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp",
                    "*.svg", "*.ico", "*.woff", "*.woff2", "*.ttf",
                    "*.otf", "*.mp4", "*.webm",
                ]
            })
        except Exception as e:
            logger.warning(f"Cannot block resources: {e}")

    return driver, profile_dir


# =====================================================
# CURL_CFFI SESSION
# =====================================================

def _sync_cookies_to_cffi(driver: uc.Chrome, session: cffi_requests.Session) -> None:
    """Đồng bộ cookie từ UC driver sang curl_cffi session."""
    for cookie in driver.get_cookies():
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain", ""),
        )


def create_cffi_session(driver: uc.Chrome) -> cffi_requests.Session:
    """
    Tạo curl_cffi session giả lập TLS fingerprint Chrome.
    Cookie được đồng bộ từ driver để dùng chung clearance Cloudflare.
    """
    session = cffi_requests.Session(impersonate="chrome120")
    _sync_cookies_to_cffi(driver, session)
    return session


def is_cloudflare_challenge_cffi(soup: BeautifulSoup, status_code: int) -> bool:
    if status_code in (403, 429, 503):
        return True

    page_text = soup.get_text(" ", strip=True).lower()
    text_signals = [
        "just a moment",
        "checking your browser",
        "verify you are human",
        "cf-challenge",
        "cdn-cgi/challenge-platform",
    ]
    return (
        any(signal in page_text for signal in text_signals)
        or soup.select_one('input[name="cf-turnstile-response"]') is not None
        or soup.select_one(".cf-browser-verification") is not None
    )


def fetch_page_cffi(
    session: cffi_requests.Session,
    url: str,
) -> Optional[BeautifulSoup]:
    """
    Fetch trang bằng curl_cffi (nhanh, không cần browser).
    Trả về None nếu bị Cloudflare block hoặc lỗi.
    """
    try:
        resp = session.get(url, timeout=30, allow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")

        if is_cloudflare_challenge_cffi(soup, resp.status_code):
            logger.debug(f"curl_cffi bị Cloudflare block: {url}")
            return None

        return soup
    except Exception as e:
        logger.debug(f"curl_cffi lỗi ({url}): {e}")
        return None


# =====================================================
# CHECKPOINT
# =====================================================

"""
Cấu trúc checkpoint riêng từng worker:

{
  "worker_id": 0,
  "forums": {
    forum_url: {
      "done_pages": [page_url...]
    }
  },
  "threads": {
    thread_url: {
      "done": true/false,
      "last_page": int
    }
  }
}
"""

def default_checkpoint(worker_id: int) -> Dict[str, Any]:
    return {
        "worker_id": worker_id,
        "forums": {},
        "threads": {}
    }


def load_checkpoint(worker_id: int) -> Dict[str, Any]:
    return STORAGE.read_json(
        STORAGE.path(VOZ_DIR, CHECKPOINT_FILE),
        default_checkpoint(worker_id),
    )


def save_checkpoint(cp: Dict[str, Any]) -> None:
    STORAGE.write_json(
        STORAGE.path(VOZ_DIR, CHECKPOINT_FILE),
        cp
    )


# =====================================================
# SAFE LOAD PAGE (UC driver)
# =====================================================

def random_sleep() -> None:
    sleep_time = random.uniform(DELAY_MIN, DELAY_MAX)
    logger.info(f"Sleep {sleep_time:.2f}s")
    time.sleep(sleep_time)


def is_cloudflare_challenge(driver: uc.Chrome, soup: BeautifulSoup) -> bool:
    title = (driver.title or "").lower()
    current_url = (driver.current_url or "").lower()
    page_text = soup.get_text(" ", strip=True).lower()

    title_signals = [
        "just a moment",
        "checking your browser",
        "verify you are human",
        "cf-challenge",
        "cloudflare",
    ]
    text_signals = [
        "just a moment",
        "checking your browser",
        "verify you are human",
        "cf-challenge",
        "cdn-cgi/challenge-platform",
    ]

    return (
        "cdn-cgi/challenge-platform" in current_url
        or any(signal in title for signal in title_signals)
        or any(signal in page_text for signal in text_signals)
        or soup.select_one('input[name="cf-turnstile-response"]') is not None
        or soup.select_one(".cf-browser-verification") is not None
    )


def wait_for_cloudflare_clearance(
    driver: uc.Chrome,
    url: str,
    timeout: int = CLOUDFLARE_WAIT_SECONDS,
) -> BeautifulSoup:
    if HEADLESS:
        raise RuntimeError(
            "Cloudflare challenge detected in headless mode. "
            "Run without --headless once and complete the browser check manually."
        )

    logger.warning(
        "Cloudflare challenge detected. Complete the check in the opened Chrome "
        f"window within {timeout}s: {url}"
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        if not is_cloudflare_challenge(driver, soup):
            logger.info("Cloudflare check cleared, continue crawling.")
            random_sleep()
            return soup

    raise RuntimeError(
        f"Cloudflare challenge was not cleared within {timeout}s: {url}"
    )


def load_page_once(driver: uc.Chrome, url: str) -> BeautifulSoup:
    logger.info(f"LOAD (UC): {url}")
    driver.get(url)
    random_sleep()
    soup = BeautifulSoup(driver.page_source, "html.parser")

    if is_cloudflare_challenge(driver, soup):
        soup = wait_for_cloudflare_clearance(driver, url)

    return soup


def load_page(driver: uc.Chrome, url: str) -> BeautifulSoup:
    """Load page có retry + backoff."""
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            return load_page_once(driver, url)
        except Exception as e:
            last_error = e
            wait_time = BACKOFF_BASE * attempt + random.uniform(5, 15)

            logger.warning(
                f"LOAD FAILED attempt={attempt}/{MAX_RETRY} "
                f"url={url} error={e}"
            )
            logger.info(f"Backoff {wait_time:.2f}s")
            time.sleep(wait_time)

    raise RuntimeError(f"Cannot load page after {MAX_RETRY} retries: {url}") from last_error


def load_forum_page(
    driver: uc.Chrome,
    url: str,
    cffi_session: Optional[cffi_requests.Session],
) -> BeautifulSoup:
    """
    Thử fetch bằng curl_cffi trước (nhanh hơn, không tốn browser resource).
    Nếu bị block hoặc lỗi, fall back sang UC driver và sync lại cookies.
    """
    if cffi_session is not None:
        soup = fetch_page_cffi(cffi_session, url)
        if soup is not None:
            logger.info(f"LOAD (cffi): {url}")
            random_sleep()
            return soup

        logger.info(f"curl_cffi failed, fallback UC driver: {url}")

    soup = load_page(driver, url)

    if cffi_session is not None:
        _sync_cookies_to_cffi(driver, cffi_session)

    return soup


# =====================================================
# PAGINATION
# =====================================================

def get_last_page(soup: BeautifulSoup) -> int:
    pages = soup.select("li.pageNav-page")

    if not pages:
        return 1

    try:
        return int(pages[-1].text.strip())
    except Exception:
        return 1


def build_page_url(base: str, p: int) -> str:
    if p == 1:
        return base

    return f"{base.rstrip('/')}/page-{p}"


# =====================================================
# GET THREADS
# =====================================================

def get_threads(soup: BeautifulSoup) -> List[str]:
    links = []

    for a in soup.select("div.structItem-title a"):
        href = a.get("href")

        if not href or "/t/" not in href:
            continue

        if href.startswith("/"):
            href = BASE_URL.rstrip("/") + href

        links.append(href)

    # Giữ thứ tự, loại duplicate
    seen = set()
    unique_links = []

    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


# =====================================================
# REACTION POPUP
# =====================================================

def close_overlay_if_any(driver: uc.Chrome) -> None:
    try:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(random.uniform(0.2, 0.5))
    except Exception:
        pass


def extract_reactions_from_message(
    driver: uc.Chrome,
    message_element,
    timeout: int = 5
) -> Optional[str]:
    """
    Click reaction popup để lấy reaction text.

    Nếu không có reaction hoặc lỗi thì trả về None.
    """
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


def safe_attr(obj, attr: str) -> Optional[str]:
    try:
        return obj.get(attr)
    except Exception:
        return None


def parse_posts(
    soup: BeautifulSoup,
    driver: uc.Chrome,
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
        "url": url
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
            "reactions": reactions
        })

    return post, comments


# =====================================================
# SAVE DATA
# =====================================================

def save_data(posts: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> None:
    """Mỗi worker ghi file riêng nên không bị conflict với worker khác."""
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
# CRAWL THREAD (UC driver — cần JS để lấy reactions)
# =====================================================

def crawl_thread(
    driver: uc.Chrome,
    thread_url: str,
    checkpoint: Dict[str, Any]
) -> bool:
    """
    Return True nếu crawl thread mới hoặc tiếp tục crawl.
    Return False nếu skip vì đã done.
    """

    if checkpoint["threads"].get(thread_url, {}).get("done"):
        logger.info(f"SKIP THREAD DONE: {thread_url}")
        return False

    logger.info(f"THREAD: {thread_url}")

    soup = load_page(driver, thread_url)
    last_page = get_last_page(soup)

    last_crawled = checkpoint["threads"].get(
        thread_url,
        {}
    ).get("last_page", 0)

    for p in range(last_crawled + 1, last_page + 1):
        page_url = build_page_url(thread_url, p)

        logger.info(f"THREAD PAGE {p}/{last_page}: {page_url}")

        soup = load_page(driver, page_url)
        post, cmt = parse_posts(soup, driver, page_url)

        # Lưu ngay sau từng page
        save_data([post], cmt)

        checkpoint["threads"][thread_url] = {
            "done": False,
            "last_page": p
        }
        save_checkpoint(checkpoint)

        logger.info(
            f"SAVED THREAD PAGE {p}/{last_page} | "
            f"comments={len(cmt)}"
        )

    checkpoint["threads"][thread_url] = {
        "done": True,
        "last_page": last_page
    }
    save_checkpoint(checkpoint)

    logger.info(f"DONE THREAD: {thread_url}")

    return True


# =====================================================
# CRAWL FORUM (dùng curl_cffi cho listing pages)
# =====================================================

def crawl_forum(
    driver: uc.Chrome,
    forum_url: str,
    checkpoint: Dict[str, Any],
    thread_counter: Dict[str, int],
    cffi_session: Optional[cffi_requests.Session] = None,
) -> None:

    logger.info(f"FORUM: {forum_url}")

    soup = load_forum_page(driver, forum_url, cffi_session)
    last_page = get_last_page(soup)

    done_pages = checkpoint["forums"].get(
        forum_url,
        {}
    ).get("done_pages", [])

    for p in range(1, last_page + 1):
        page_url = build_page_url(forum_url, p)

        if page_url in done_pages:
            logger.info(f"SKIP FORUM PAGE DONE: {page_url}")
            continue

        logger.info(f"FORUM PAGE {p}/{last_page}: {page_url}")

        soup = load_forum_page(driver, page_url, cffi_session)
        threads = get_threads(soup)

        logger.info(f"THREADS FOUND: {len(threads)}")

        for thread in threads:
            try:
                crawled = crawl_thread(driver, thread, checkpoint)

                if crawled:
                    thread_counter["count"] += 1

                if (
                    LONG_SLEEP_EVERY_THREADS > 0
                    and thread_counter["count"] > 0
                    and thread_counter["count"] % LONG_SLEEP_EVERY_THREADS == 0
                ):
                    sleep_time = random.uniform(LONG_SLEEP_MIN, LONG_SLEEP_MAX)
                    logger.info(
                        f"LONG SLEEP after "
                        f"{thread_counter['count']} threads: {sleep_time:.2f}s"
                    )
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"ERROR THREAD {thread}: {e}")

        checkpoint["forums"].setdefault(
            forum_url,
            {"done_pages": []}
        )

        if page_url not in checkpoint["forums"][forum_url]["done_pages"]:
            checkpoint["forums"][forum_url]["done_pages"].append(page_url)

        save_checkpoint(checkpoint)

        logger.info(f"DONE FORUM PAGE: {page_url}")


# =====================================================
# MAIN RUN
# =====================================================

def run(
    worker_id: int,
    links_per_worker: int,
    delay_min: int,
    delay_max: int,
    headless: bool,
    hide_window: bool,
    long_sleep_every: int,
    long_sleep_min: int,
    long_sleep_max: int,
    chrome_profile_dir: str,
    cloudflare_wait_seconds: int,
    block_heavy_resources: bool,
) -> None:

    global STORAGE
    global CHECKPOINT_FILE, POST_FILE, COMMENT_FILE
    global DELAY_MIN, DELAY_MAX, HEADLESS, HIDE_WINDOW
    global CHROME_PROFILE_DIR, CLOUDFLARE_WAIT_SECONDS, ALLOW_RESOURCE_BLOCKING
    global LONG_SLEEP_EVERY_THREADS, LONG_SLEEP_MIN, LONG_SLEEP_MAX

    DELAY_MIN = delay_min
    DELAY_MAX = delay_max
    HEADLESS = headless
    HIDE_WINDOW = hide_window
    CHROME_PROFILE_DIR = get_worker_profile_dir(chrome_profile_dir, worker_id)
    CLOUDFLARE_WAIT_SECONDS = cloudflare_wait_seconds
    ALLOW_RESOURCE_BLOCKING = block_heavy_resources

    LONG_SLEEP_EVERY_THREADS = long_sleep_every
    LONG_SLEEP_MIN = long_sleep_min
    LONG_SLEEP_MAX = long_sleep_max

    CHECKPOINT_FILE = f"checkpoint_worker_{worker_id}.json"
    POST_FILE = f"posts_worker_{worker_id}.csv"
    COMMENT_FILE = f"comments_worker_{worker_id}.csv"

    my_forums = get_worker_forums(
        links=Link_FORUM,
        worker_id=worker_id,
        links_per_worker=links_per_worker
    )

    logger.info("=" * 80)
    logger.info(f"START WORKER {worker_id}")
    logger.info(f"CHECKPOINT: {CHECKPOINT_FILE}")
    logger.info(f"POST FILE: {POST_FILE}")
    logger.info(f"COMMENT FILE: {COMMENT_FILE}")
    logger.info(f"DELAY: {DELAY_MIN}-{DELAY_MAX}s")
    logger.info(f"HEADLESS: {HEADLESS}")
    logger.info(f"CHROME PROFILE: {os.path.abspath(CHROME_PROFILE_DIR)}")
    logger.info(f"CLOUDFLARE WAIT: {CLOUDFLARE_WAIT_SECONDS}s")
    logger.info(f"BLOCK HEAVY RESOURCES: {ALLOW_RESOURCE_BLOCKING}")
    logger.info(f"LINKS PER WORKER: {links_per_worker}")
    logger.info(f"FORUM COUNT: {len(my_forums)}")
    logger.info(f"LOCAL DATA DIR: {LOCAL_DATA_DIR}")

    for f in my_forums:
        logger.info(f"   - {f}")

    logger.info("=" * 80)

    if not my_forums:
        logger.warning(f"Worker {worker_id} không có forum nào để crawl.")
        return

    STORAGE = LocalStorage()
    driver, profile_dir = create_driver()

    # Warm-up: mở BASE_URL để lấy cookie hợp lệ trước khi tạo cffi session
    logger.info(f"Warm-up: {BASE_URL}")
    try:
        load_page_once(driver, BASE_URL)
    except Exception as e:
        logger.warning(f"Warm-up failed: {e}")

    cffi_session = create_cffi_session(driver)
    logger.info("curl_cffi session created (impersonate=chrome120)")

    checkpoint = load_checkpoint(worker_id)
    thread_counter = {"count": 0}

    try:
        for forum in my_forums:
            try:
                crawl_forum(
                    driver=driver,
                    forum_url=forum,
                    checkpoint=checkpoint,
                    thread_counter=thread_counter,
                    cffi_session=cffi_session,
                )
            except Exception as e:
                logger.error(f"ERROR FORUM {forum}: {e}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass

        try:
            if profile_dir is not None:
                profile_dir.cleanup()
        except Exception:
            pass

        try:
            STORAGE.close()
        except Exception:
            pass

    logger.info(f"FINISHED WORKER {worker_id}")


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="ID của worker. Ví dụ: 0, 1, 2, 3..."
    )

    parser.add_argument(
        "--links-per-worker",
        type=int,
        default=5,
        help="Mỗi worker crawl bao nhiêu forum link liên tiếp."
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
        help="Chạy Chrome ở chế độ headless."
    )

    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Ẩn cửa sổ Chrome khỏi màn hình (đẩy ra ngoài viewport, ít bị Cloudflare detect hơn --headless)."
    )

    parser.add_argument(
        "--chrome-profile-dir",
        default=CHROME_PROFILE_DIR,
        help="Thu muc Chrome profile de giu cookie/session hop le giua cac lan chay."
    )

    parser.add_argument(
        "--cloudflare-wait-seconds",
        type=int,
        default=CLOUDFLARE_WAIT_SECONDS,
        help="So giay cho ban xac minh Cloudflare thu cong khi chay non-headless."
    )

    parser.add_argument(
        "--block-heavy-resources",
        action="store_true",
        default=ALLOW_RESOURCE_BLOCKING,
        help="Chan anh/font/video de tang toc. Nen tat neu hay gap Cloudflare."
    )

    parser.add_argument(
        "--long-sleep-every",
        type=int,
        default=50,
        help="Sau bao nhiêu thread thì nghỉ dài. Đặt 0 để tắt."
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

    args = parser.parse_args()

    run(
        worker_id=args.worker_id,
        links_per_worker=args.links_per_worker,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        headless=args.headless,
        hide_window=args.hide_window,
        long_sleep_every=args.long_sleep_every,
        long_sleep_min=args.long_sleep_min,
        long_sleep_max=args.long_sleep_max,
        chrome_profile_dir=args.chrome_profile_dir,
        cloudflare_wait_seconds=args.cloudflare_wait_seconds,
        block_heavy_resources=args.block_heavy_resources,
    )
