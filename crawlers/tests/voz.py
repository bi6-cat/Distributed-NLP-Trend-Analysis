import time
import json
import logging
import random
import os
import tempfile
import requests
import pandas as pd
import hashlib
import io
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import quote, urlparse, urlunparse

# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://voz.vn/"

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
              "https://voz.vn/f/small-form-factor-pc.61/"
              "https://voz.vn/f/hackintosh.62/",
              "https://voz.vn/f/may-tinh-xach-tay.47/",
              "https://voz.vn/f/phan-mem.13/",
              "https://voz.vn/f/app-di-dong.21/,"
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
              "https://voz.vn/f/dien-thoai-di-dong.76/"]

HDFS_HOST = "192.168.56.11"
HDFS_PORT = 9870
HDFS_USER = "zett"
HDFS_BASE_DIR = "/user/zett/voz"

HDFS_HOST = os.getenv("HDFS_HOST", HDFS_HOST)
HDFS_PORT = int(os.getenv("HDFS_PORT", str(HDFS_PORT)))
HDFS_USER = os.getenv("HDFS_USER", HDFS_USER)
HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", HDFS_BASE_DIR)
HDFS_HOSTS = [h.strip() for h in os.getenv("HDFS_HOSTS", "").split(",") if h.strip()]

if HDFS_HOSTS and HDFS_HOST not in HDFS_HOSTS:
    HDFS_HOSTS.insert(0, HDFS_HOST)
elif not HDFS_HOSTS:
    HDFS_HOSTS = [HDFS_HOST]

DATANODE_HOST_MAP = {
    "master": "192.168.56.11",
    "worker1": "192.168.56.12",
    "worker2": "192.168.56.13",
    "storage": "192.168.56.14",
}

for pair in os.getenv("HDFS_DATANODE_MAP", "").split(","):
    pair = pair.strip()
    if "=" in pair:
        key, value = pair.split("=", 1)
        DATANODE_HOST_MAP[key.strip()] = value.strip()


def _resolve_redirect_url(location: str) -> str:
    parsed = urlparse(location)
    mapped_ip = DATANODE_HOST_MAP.get(parsed.hostname)
    if mapped_ip:
        new_netloc = f"{mapped_ip}:{parsed.port}" if parsed.port else mapped_ip
        location = urlunparse(parsed._replace(netloc=new_netloc))
    return location


CHECKPOINT_FILE = f"{HDFS_BASE_DIR}/checkpoint.json"
POST_FILE = f"{HDFS_BASE_DIR}/posts.csv"
COMMENT_FILE = f"{HDFS_BASE_DIR}/comments.csv"

DELAY_MIN = 2
DELAY_MAX = 5

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# DRIVER
# =====================================================

def create_driver():
    profile_dir = tempfile.TemporaryDirectory(prefix="voz-chrome-")
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-data-dir={profile_dir.name}")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    return webdriver.Chrome(options=options), profile_dir


def load_checkpoint():
    content = hdfs_open(CHECKPOINT_FILE)
    if content:
        return json.loads(content)
    return {"forums": {}, "threads": {}}


def save_checkpoint(cp):
    hdfs_write(CHECKPOINT_FILE, json.dumps(cp, indent=2, ensure_ascii=False), overwrite=True)


# =====================================================
# HDFS (WEBHDFS)
# =====================================================
def hdfs_url(path, op, extra_params=None):
    safe_path = quote(path if path.startswith("/") else f"/{path}", safe="/")
    params = {"op": op, "user.name": HDFS_USER}
    if extra_params:
        params.update(extra_params)
    return f"http://{HDFS_HOST}:{HDFS_PORT}/webhdfs/v1{safe_path}", params


def select_reachable_hdfs_host():
    global HDFS_HOST

    last_error = None
    for host in HDFS_HOSTS:
        try:
            test_url = f"http://{host}:{HDFS_PORT}/webhdfs/v1/?op=GETHOMEDIRECTORY&user.name={HDFS_USER}"
            response = requests.get(test_url, timeout=10)
            if response.status_code in (200, 401, 403):
                HDFS_HOST = host
                print(f"[HDFS] Using WebHDFS host: {HDFS_HOST}:{HDFS_PORT}")
                return
            last_error = f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)

    raise RuntimeError(
        "[HDFS CONNECT ERROR] Cannot connect to any WebHDFS host. "
        f"Tried: {', '.join(HDFS_HOSTS)} on port {HDFS_PORT}. "
        "Set HDFS_HOST or HDFS_HOSTS env to reachable address (LAN or Tailscale) and retry. "
        f"Last error: {last_error}"
    )


def hdfs_request(method, path, op, extra_params=None, allow_redirects=True, data=None):
    url, params = hdfs_url(path, op, extra_params)
    try:
        response = requests.request(
            method,
            url,
            params=params,
            allow_redirects=False,
            timeout=10,
            data=data,
        )

        if allow_redirects and response.is_redirect:
            location = _resolve_redirect_url(response.headers["Location"])
            response = requests.request(method, location, timeout=10, data=data)

        return response
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "[HDFS CONNECT ERROR] Cannot connect to WebHDFS "
            f"http://{HDFS_HOST}:{HDFS_PORT}. "
            "Set HDFS_HOST/HDFS_PORT env to reachable address (LAN or Tailscale) and retry. "
            f"Original error: {exc}"
        ) from exc


def hdfs_exists(path):
    response = hdfs_request("GET", path, "GETFILESTATUS")
    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return False


def hdfs_mkdirs(path):
    response = hdfs_request("PUT", path, "MKDIRS")
    response.raise_for_status()


def hdfs_open(path):
    response = hdfs_request("GET", path, "OPEN")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def hdfs_write(path, data, overwrite=True):
    parent = os.path.dirname(path).replace("\\", "/")
    if parent:
        hdfs_mkdirs(parent)

    response = hdfs_request(
        "PUT",
        path,
        "CREATE",
        {"overwrite": str(overwrite).lower()},
        allow_redirects=False,
    )
    response.raise_for_status()

    location = response.headers.get("Location")
    if not location:
        raise RuntimeError(f"[HDFS] Missing redirect Location for CREATE: {path}")

    location = _resolve_redirect_url(location)
    response_2 = requests.put(location, data=data.encode("utf-8"), timeout=10)
    response_2.raise_for_status()


def hdfs_append(path, data):
    if not hdfs_exists(path):
        hdfs_write(path, data, overwrite=True)
        return

    response = hdfs_request("POST", path, "APPEND", allow_redirects=False)
    response.raise_for_status()

    location = response.headers.get("Location")
    if not location:
        raise RuntimeError(f"[HDFS] Missing redirect Location for APPEND: {path}")

    location = _resolve_redirect_url(location)
    response_2 = requests.post(location, data=data.encode("utf-8"), timeout=10)
    response_2.raise_for_status()


def load_json(file):
    content = hdfs_open(file)
    if content:
        return json.loads(content)
    return {}


def save_json(file, data):
    hdfs_write(file, json.dumps(data, indent=2, ensure_ascii=False), overwrite=True)


def init_csv():
    if not hdfs_exists(POST_FILE):
        header = pd.DataFrame(columns=["id_post", "title", "time_post", "replies_post", "views_post", "id_author", "author_name", "category", "subcategory"])
        hdfs_write(POST_FILE, header.to_csv(index=False), overwrite=True)
    if not hdfs_exists(COMMENT_FILE):
        header = pd.DataFrame(columns=["id_post", "id_user", "user", "time", "comment", "url", "reactions"])
        hdfs_write(COMMENT_FILE, header.to_csv(index=False), overwrite=True)


def append_csv(path, df):
    hdfs_append(path, df.to_csv(index=False, header=False))

# =====================================================
# CHECKPOINT STRUCTURE
# =====================================================
"""
{
  "forums": {
    forum_url: {
        "done_pages": [page_url...]
    }
  },
  "threads": {
    thread_url: {
        "done": True/False,
        "last_page": int
    }
  }
}
"""

# =====================================================
# LOAD PAGE
# =====================================================

def load_page(driver, url):
    logger.info(f"🌐 LOAD: {url}")
    driver.get(url)
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    return BeautifulSoup(driver.page_source, "html.parser")

# =====================================================
# PAGINATION
# =====================================================

def get_last_page(soup):
    pages = soup.select("li.pageNav-page")
    if not pages:
        return 1
    try:
        return int(pages[-1].text)
    except:
        return 1

def build_page_url(base, p):
    return base if p == 1 else f"{base}page-{p}"

# =====================================================
# GET THREADS
# =====================================================

def get_threads(soup):
    links = []
    for a in soup.select("div.structItem-title a"):
        href = a.get("href")
        if not href or "/t/" not in href:
            continue
        if href.startswith("/"):
            href = BASE_URL.rstrip("/") + href
        links.append(href)
    return list(set(links))

# =====================================================
# PARSE POST + COMMENT
# =====================================================

def parse_posts(soup, driver, url):

    comments = []

    try:
        category = soup.select('span[itemprop="name"]')[1].text.strip()
    except:
        category = None
    try:
        subcategory = soup.select('span[itemprop="name"]')[2].text.strip()
    except:
        subcategory = None

    try:
        id_post = soup.select('a.u-concealed')[1].get('href').split('.')[-1].split('/')[0]
    except:
        id_post = None

    try:
        title = soup.select_one('h1.p-title-value').text.strip()
    except:
        title = None

    try:
        time_post = soup.select_one('time.u-dt').text.strip()
    except:
        time_post = None

    try:
        id_author = soup.select_one('a.username').get('data-user-id')
    except:
        id_author = None
    
    try:
        author_name = soup.select_one('a.username').text.strip()
    except:
        author_name = None
    
    try:
        replies_post = soup.select_one('dl.pairs.pairs--justified.count--replies').text.strip()
    except:
        replies_post = None

    try:
        views_post = soup.select_one('dl.pairs.pairs--justified.count--views').text.strip()
    except:
        views_post = None

    post = {
        "id_post": id_post,
        "title": title,
        "time_post": time_post,
        "replies_post": replies_post,
        "views_post": views_post,
        "id_author": id_author,
        "author_name": author_name,
        "category": category,
        "subcategory": subcategory
    }

    contents = soup.select('div.message-inner')
    contents_driver = driver.find_elements(By.CSS_SELECTOR, "div.message-inner")

    for idx, c in enumerate(contents):

        try:
            id_user = c.select_one('a.username').get('data-user-id')
        except:
            id_user = None

        try:
            user = c.select_one('a.username').text.strip()
        except:
            user = None

        try:
            time_comment = c.select_one('time').get('title')
        except:
            time_comment = None

        try:
            comment = c.select_one('div.message-content').text.strip()
        except:
            comment = None

        # =============================
        # 🔥 CLICK REACTIONS (PHẦN BẠN YÊU CẦU)
        # =============================
        reactions = None
        try:
            element = contents_driver[idx].find_element(
                By.CSS_SELECTOR, "a.reactionsBar-link"
            )

            # scroll + click
            driver.execute_script("arguments[0].scrollIntoView();", element)
            driver.execute_script("arguments[0].click();", element)

            # wait popup xuất hiện
            popup = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".overlay-container"))
            )
            react_comments = popup.find_elements(By.CSS_SELECTOR, "span.reaction-text.js-reactionText")
            reactions = "".join([r.text for r in react_comments])

            

        except Exception as e:
            pass

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
# SAVE
# =====================================================

def save_data(posts, comments):

    if posts:
        df = pd.DataFrame(posts)
        append_csv(POST_FILE, df)

    if comments:
        df = pd.DataFrame(comments)
        append_csv(COMMENT_FILE, df)

# =====================================================
# CRAWL THREAD
# =====================================================

def crawl_thread(driver, thread_url, checkpoint):

    # skip nếu done
    if checkpoint["threads"].get(thread_url, {}).get("done"):
        logger.info(f"⏭️ SKIP THREAD (DONE): {thread_url}")
        return

    logger.info(f"🧵 THREAD: {thread_url}")

    soup = load_page(driver, thread_url)
    last_page = get_last_page(soup)

    last_crawled = checkpoint["threads"].get(thread_url, {}).get("last_page", 0)

    posts = []
    comments = []

    for p in range(last_crawled + 1, last_page + 1):

        url = build_page_url(thread_url, p)

        logger.info(f"   📄 THREAD PAGE {p}/{last_page}")

        soup = load_page(driver, url)

        post, cmt = parse_posts(soup, driver, url)

        posts.append(post)
        comments.extend(cmt)

        # save checkpoint từng page
        checkpoint["threads"][thread_url] = {
            "done": False,
            "last_page": p
        }
        save_checkpoint(checkpoint)

    # DONE THREAD
    checkpoint["threads"][thread_url] = {
        "done": True,
        "last_page": last_page
    }
    save_checkpoint(checkpoint)

    save_data(posts, comments)

    logger.info(f"✅ DONE THREAD: {thread_url}")

# =====================================================
# CRAWL FORUM
# =====================================================

def crawl_forum(driver, forum_url, checkpoint):

    logger.info(f"📂 FORUM: {forum_url}")

    soup = load_page(driver, forum_url)
    last_page = get_last_page(soup)

    done_pages = checkpoint["forums"].get(forum_url, {}).get("done_pages", [])

    for p in range(1, last_page + 1):

        page_url = build_page_url(forum_url, p)

        if page_url in done_pages:
            logger.info(f"⏭️ SKIP PAGE: {page_url}")
            continue

        logger.info(f"📄 FORUM PAGE {p}/{last_page}")

        soup = load_page(driver, page_url)

        threads = get_threads(soup)

        logger.info(f"   🔗 THREADS FOUND: {len(threads)}")

        for thread in threads:
            crawl_thread(driver, thread, checkpoint)

        # mark page done
        checkpoint["forums"].setdefault(forum_url, {"done_pages": []})
        checkpoint["forums"][forum_url]["done_pages"].append(page_url)

        save_checkpoint(checkpoint)

        logger.info(f"✅ DONE PAGE: {page_url}")

# =====================================================
# MAIN
# =====================================================

def run():

    logger.info("🚀 START PIPELINE")

    select_reachable_hdfs_host()
    init_csv()

    driver, profile_dir = create_driver()
    checkpoint = load_checkpoint()

    try:
        for forum in Link_FORUM:
            try:
                crawl_forum(driver, forum, checkpoint)
            except Exception as e:
                logger.error(f"❌ ERROR FORUM {forum}: {e}")
    finally:
        driver.quit()
        profile_dir.cleanup()

    logger.info("🎯 FINISHED")

# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    run()