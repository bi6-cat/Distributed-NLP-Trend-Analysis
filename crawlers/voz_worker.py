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

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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


# =====================================================
# DRIVER
# =====================================================

# def create_driver() -> Tuple[webdriver.Chrome, tempfile.TemporaryDirectory]:
#     profile_dir = tempfile.TemporaryDirectory(prefix="voz-chrome-")

#     options = Options()

#     if HEADLESS:
#         options.add_argument("--headless=new")

#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument(f"--user-data-dir={profile_dir.name}")
#     options.add_argument("--no-first-run")
#     options.add_argument("--no-default-browser-check")

#     # Một số option giúp Selenium ổn định hơn khi chạy nhiều worker
#     options.add_argument("--disable-gpu")
#     options.add_argument("--disable-extensions")
#     options.add_argument("--disable-notifications")
#     options.add_argument("--disable-popup-blocking")

#     # Nếu chạy trên server Linux không có GUI thì mở 2 dòng này
#     # options.add_argument("--no-sandbox")
#     # options.add_argument("--disable-dev-shm-usage")

#     driver = webdriver.Chrome(options=options)
#     driver.set_page_load_timeout(60)

#     return driver, profile_dir

def create_driver():
    profile_dir = tempfile.TemporaryDirectory(prefix="voz-chrome-")

    options = Options()
    options.page_load_strategy = "eager"

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

    # Chặn tài nguyên nặng
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setBlockedURLs", {
            "urls": [
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.gif",
                "*.webp",
                "*.svg",
                "*.ico",
                "*.woff",
                "*.woff2",
                "*.ttf",
                "*.otf",
                "*.mp4",
                "*.webm",
            ]
        })
    except Exception as e:
        logger.warning(f"Cannot block resources: {e}")

    return driver, profile_dir
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
# SAFE LOAD PAGE
# =====================================================

def random_sleep() -> None:
    sleep_time = random.uniform(DELAY_MIN, DELAY_MAX)
    logger.info(f"Sleep {sleep_time:.2f}s")
    time.sleep(sleep_time)


def load_page_once(driver: webdriver.Chrome, url: str) -> BeautifulSoup:
    logger.info(f"LOAD: {url}")
    driver.get(url)
    random_sleep()
    return BeautifulSoup(driver.page_source, "html.parser")


def load_page(driver: webdriver.Chrome, url: str) -> BeautifulSoup:
    """
    Load page có retry + backoff để tránh retry quá dồn dập.
    """
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
    """
    Mỗi worker ghi file riêng nên không bị conflict với worker khác.
    """
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
# CRAWL THREAD
# =====================================================

def crawl_thread(
    driver: webdriver.Chrome,
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
# CRAWL FORUM
# =====================================================

def crawl_forum(
    driver: webdriver.Chrome,
    forum_url: str,
    checkpoint: Dict[str, Any],
    thread_counter: Dict[str, int]
) -> None:

    logger.info(f"FORUM: {forum_url}")

    soup = load_page(driver, forum_url)
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

        soup = load_page(driver, page_url)
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
    long_sleep_every: int,
    long_sleep_min: int,
    long_sleep_max: int,
) -> None:

    global STORAGE
    global CHECKPOINT_FILE, POST_FILE, COMMENT_FILE
    global DELAY_MIN, DELAY_MAX, HEADLESS
    global LONG_SLEEP_EVERY_THREADS, LONG_SLEEP_MIN, LONG_SLEEP_MAX

    DELAY_MIN = delay_min
    DELAY_MAX = delay_max
    HEADLESS = headless

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
    checkpoint = load_checkpoint(worker_id)

    thread_counter = {"count": 0}

    try:
        for forum in my_forums:
            try:
                crawl_forum(
                    driver=driver,
                    forum_url=forum,
                    checkpoint=checkpoint,
                    thread_counter=thread_counter
                )
            except Exception as e:
                logger.error(f"ERROR FORUM {forum}: {e}")

    finally:
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
        long_sleep_every=args.long_sleep_every,
        long_sleep_min=args.long_sleep_min,
        long_sleep_max=args.long_sleep_max,
    )
