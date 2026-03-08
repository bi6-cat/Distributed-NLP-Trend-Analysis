import time
import json
import logging
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium import webdriver


import undetected_chromedriver as uc


# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://voz.vn/"
CHECKPOINT_FILE = "voz_checkpoint.json"
OUTPUT_FILE = "voz_raw.csv"

MAX_THREADS_PER_RUN = 5
PAGE_DELAY = 3


# =====================================================
# LOG
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

    options = Options()

    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)

    return driver


# =====================================================
# CHECKPOINT
# =====================================================

def load_checkpoint():

    if not Path(CHECKPOINT_FILE).exists():
        return {}

    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)


def save_checkpoint(data):

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


# =====================================================
# PAGE LOADER
# =====================================================

def load_page(driver, url):

    driver.get(url)

    time.sleep(PAGE_DELAY)

    return BeautifulSoup(driver.page_source, "html.parser")


# =====================================================
# GET THREADS FROM HOME
# =====================================================

def get_latest_threads(driver):

    soup = load_page(driver, BASE_URL)

    threads = soup.select("div.node-body")

    results = []

    for t in threads:

        try:
            href = t.select("a")[0]["href"]
            results.append(BASE_URL + href)
        except:
            continue

    return results[:MAX_THREADS_PER_RUN]


# =====================================================
# GET THREAD PAGES
# =====================================================

def get_thread_last_page(driver, thread_url):

    soup = load_page(driver, thread_url)

    pages = soup.select("li.pageNav-page")

    if len(pages) == 0:
        return 1

    return int(pages[-1].text)


# =====================================================
# PARSE POSTS
# =====================================================

def parse_posts(soup, url):

    posts = soup.select("article.message")

    data = []

    for p in posts:

        try:
            author = p.select_one(".message-name").text.strip()
        except:
            author = None

        try:
            content = p.select_one(".bbWrapper").text.strip()
        except:
            content = None

        try:
            time_post = p.select_one("time")["datetime"]
        except:
            time_post = None

        data.append(
            {
                "author": author,
                "content": content,
                "time": time_post,
                "url": url
            }
        )

    return data


# =====================================================
# CRAWL PAGE
# =====================================================

def crawl_page(driver, url):

    soup = load_page(driver, url)

    posts = parse_posts(soup, url)

    return posts


# =====================================================
# SAVE BATCH
# =====================================================

def save_batch(df):

    write_header = not Path(OUTPUT_FILE).exists()

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=write_header
    )

    logger.info(f"Saved batch {len(df)} rows")


# =====================================================
# DEPTH CRAWLER
# =====================================================

def crawl_thread_depth(driver, thread_url, checkpoint):

    logger.info(f"Crawling thread: {thread_url}")

    last_page_crawled = checkpoint.get(thread_url, 0)

    last_page = get_thread_last_page(driver, thread_url)

    logger.info(f"Last page crawled: {last_page_crawled}")
    logger.info(f"Thread last page: {last_page}")

    for page in range(last_page_crawled + 1, last_page + 1):

        page_url = f"{thread_url}page-{page}"

        logger.info(f"Crawling page {page}")

        posts = crawl_page(driver, page_url)

        df = pd.DataFrame(posts)

        yield df

        checkpoint[thread_url] = page


# =====================================================
# MAIN PIPELINE
# =====================================================

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


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":

    run_pipeline()