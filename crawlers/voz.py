import time
import json
import logging
import random
import os
import tempfile
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(SCRIPT_DIR / "data")))
VOZ_DIR = DATA_DIR / "voz"

CHECKPOINT_FILE = VOZ_DIR / "checkpoint.json"

POST_FILE = VOZ_DIR / "posts.csv"
COMMENT_FILE = VOZ_DIR / "comments.csv"

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
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-data-dir={profile_dir.name}")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    return webdriver.Chrome(options=options), profile_dir

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

def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return {"forums": {}, "threads": {}}
    with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(cp):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)

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
        POST_FILE.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(posts)
        df.to_csv(POST_FILE, mode="a", index=False, header=not POST_FILE.exists())

    if comments:
        COMMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(comments)
        df.to_csv(COMMENT_FILE, mode="a", index=False, header=not COMMENT_FILE.exists())

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

    print("\n===========================================")
    print("🚀 Bắt đầu tự động đẩy dữ liệu lên HDFS...")
    print("===========================================")
    try:
        from upload_to_hdfs import main as upload_main
        upload_main()
    except Exception as e:
        print(f"❌ Lỗi khi tự động tải dữ liệu lên HDFS: {e}")