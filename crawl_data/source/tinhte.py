import time
import json
import logging
import random
import re
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# ======================================
# CONFIG
# ======================================

BASE_URL = "https://tinhte.vn/"
OUTPUT_FILE = "tinhte_raw.csv"
CHECKPOINT_FILE = "tinhte_checkpoint.json"

MAX_THREADS_PER_RUN = 5
PAGE_DELAY = 3


# ======================================
# LOGGING
# ======================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ======================================
# DRIVER
# ======================================

def create_driver():

    options = Options()
    options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    return driver


# ======================================
# UTILS
# ======================================

def random_sleep(a=2.5, b=5.5):

    time.sleep(round(random.uniform(a, b), 2))


def normalize_time(raw_time: str, now=None):

    if not raw_time:
        return ""

    raw_time = raw_time.strip().lower()
    now = now or datetime.now()

    try:
        parsed = datetime.strptime(raw_time, "%d/%m/%Y %H:%M")
        return parsed.strftime("%d/%m/%Y %H:%M")
    except:
        pass

    if "hôm qua" in raw_time:
        dt = now - timedelta(days=1)
        return dt.strftime("%d/%m/%Y %H:%M")

    match = re.search(r"(\d+)\s*(phút|giờ|ngày|tuần|tháng)", raw_time)

    if match:

        value = int(match.group(1))
        unit = match.group(2)

        delta_map = {
            "phút": timedelta(minutes=value),
            "giờ": timedelta(hours=value),
            "ngày": timedelta(days=value),
            "tuần": timedelta(weeks=value),
            "tháng": timedelta(days=value * 30)
        }

        dt = now - delta_map[unit]

        return dt.strftime("%d/%m/%Y %H:%M")

    return now.strftime("%d/%m/%Y %H:%M")


def get_post_id(url: str):

    path = urlparse(url).path.rstrip('/')

    match = re.search(r'\.(\d+)$', path)

    return match.group(1) if match else None


# ======================================
# CHECKPOINT
# ======================================

def load_checkpoint():

    if not Path(CHECKPOINT_FILE).exists():
        return {}

    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)


def save_checkpoint(data):

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


# ======================================
# LOAD PAGE
# ======================================

def load_page(driver, url):

    driver.get(url)

    random_sleep()

    soup = BeautifulSoup(driver.page_source, "html.parser")

    return soup


# ======================================
# GET THREADS
# ======================================

def get_latest_threads(driver):

    logger.info("Collect latest threads")

    soup = load_page(driver, BASE_URL)

    links = soup.select("li.jsx-4010778416")

    thread_urls = []

    for l in links:

        a = l.select_one("a")

        if a:

            href = a.get("href")

            if href and "thread" in href:

                thread_urls.append(href)

    logger.info(f"Found {len(thread_urls)} threads")

    return thread_urls[:MAX_THREADS_PER_RUN]


# ======================================
# THREAD PAGE COUNT
# ======================================

def get_last_page(driver, thread_url):

    soup = load_page(driver, thread_url)

    pages = soup.select("li.pageNav-page")

    if not pages:
        return 1

    return int(pages[-1].text)


# ======================================
# PARSE POSTS
# ======================================

def parse_posts(soup, page_url):

    posts = soup.select("article.message")

    data = []

    for p in posts:

        try:
            author = p.select_one(".username").text.strip()
        except:
            author = None

        try:
            content = p.select_one(".bbWrapper").text.strip()
        except:
            content = None

        try:
            time_raw = p.select_one("time").text
            time_post = normalize_time(time_raw)
        except:
            time_post = None

        data.append(
            {
                "author": author,
                "content": content,
                "time": time_post,
                "url": page_url
            }
        )

    return data


# ======================================
# CRAWL PAGE
# ======================================

def crawl_page(driver, url):

    logger.info(f"Crawl page: {url}")

    soup = load_page(driver, url)

    posts = parse_posts(soup, url)

    return posts


# ======================================
# SAVE BATCH
# ======================================

def save_batch(df):

    write_header = not Path(OUTPUT_FILE).exists()

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=write_header
    )

    logger.info(f"Saved batch {len(df)} rows")


# ======================================
# DEPTH THREAD CRAWLER
# ======================================

def crawl_thread_depth(driver, thread_url, checkpoint):

    logger.info(f"Crawl thread: {thread_url}")

    last_page_crawled = checkpoint.get(thread_url, 0)

    last_page = get_last_page(driver, thread_url)

    logger.info(f"Last page crawled: {last_page_crawled}")
    logger.info(f"Thread last page: {last_page}")

    for page in range(last_page_crawled + 1, last_page + 1):

        page_url = f"{thread_url}page-{page}"

        posts = crawl_page(driver, page_url)

        df = pd.DataFrame(posts)

        yield df

        checkpoint[thread_url] = page


# ======================================
# PIPELINE
# ======================================

def run_pipeline():

    logger.info("Start crawler")

    driver = create_driver()

    checkpoint = load_checkpoint()

    threads = get_latest_threads(driver)

    logger.info(f"Threads to crawl: {len(threads)}")

    for thread in threads:

        try:

            for batch in crawl_thread_depth(driver, thread, checkpoint):

                save_batch(batch)

        except Exception as e:

            logger.error(e)

    save_checkpoint(checkpoint)

    driver.quit()

    logger.info("Crawler finished")


# ======================================
# ENTRY
# ======================================

if __name__ == "__main__":

    run_pipeline()