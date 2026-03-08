import time
import json
import logging
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# =====================================
# CONFIG
# =====================================

BASE_URL = "https://vatvostudio.vn/category/tin-tuc-moi-nhat/"
OUTPUT_FILE = "vatvo_raw.csv"
CHECKPOINT_FILE = "vatvo_checkpoint.json"

MAX_ARTICLES_PER_RUN = 10
REQUEST_DELAY = 2


# =====================================
# LOG
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================
# DRIVER
# =====================================

def create_driver():

    options = Options()
    options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    return driver


# =====================================
# CHECKPOINT
# =====================================

def load_checkpoint():

    if not Path(CHECKPOINT_FILE).exists():
        return {"crawled_links": []}

    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)


def save_checkpoint(data):

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


# =====================================
# GET ARTICLE LINKS
# =====================================

def get_latest_links():

    logger.info("Collect article links")

    links = []

    page = 1

    while len(links) < MAX_ARTICLES_PER_RUN:

        url = f"{BASE_URL}page/{page}/"

        r = requests.get(url)

        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "html.parser")

        articles = soup.select("h3.post__title a")

        if not articles:
            break

        for a in articles:

            links.append(a["href"])

            if len(links) >= MAX_ARTICLES_PER_RUN:
                break

        page += 1

    logger.info(f"Collected {len(links)} links")

    return links


# =====================================
# PARSE ARTICLE
# =====================================

def parse_article(driver, url):

    logger.info(f"Crawl article: {url}")

    driver.get(url)

    time.sleep(REQUEST_DELAY)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    article = soup.find("article")

    # post id
    post_id = None
    if soup.body and soup.body.get("class"):
        m = re.search(r'postid-(\d+)', ' '.join(soup.body["class"]))
        if m:
            post_id = m.group(1)

    # title
    title_el = soup.select_one("h1.entry-title")
    title = title_el.get_text(strip=True) if title_el else None

    # author
    author_el = soup.select_one('div.entry-meta a[rel="author"]')
    author = author_el.get_text(strip=True) if author_el else None

    # time
    time_el = soup.select_one("div.entry-meta time")

    time_post = None

    if time_el and time_el.has_attr("datetime"):

        dt = datetime.fromisoformat(time_el["datetime"])

        time_post = dt.strftime("%d/%m/%Y %H:%M")

    # content
    content_el = soup.select_one("div.entry-content")

    content = content_el.get_text(" ", strip=True) if content_el else None

    return {
        "id": post_id,
        "title": title,
        "author": author,
        "time": time_post,
        "content": content,
        "url": url
    }


# =====================================
# SAVE BATCH
# =====================================

def save_batch(df):

    write_header = not Path(OUTPUT_FILE).exists()

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=write_header
    )

    logger.info(f"Saved batch {len(df)} rows")


# =====================================
# PIPELINE
# =====================================

def run_pipeline():

    logger.info("Start crawler")

    checkpoint = load_checkpoint()

    crawled_links = set(checkpoint["crawled_links"])

    links = get_latest_links()

    new_links = [l for l in links if l not in crawled_links]

    logger.info(f"New articles: {len(new_links)}")

    driver = create_driver()

    batch = []

    for link in new_links:

        try:

            data = parse_article(driver, link)

            batch.append(data)

            crawled_links.add(link)

        except Exception as e:

            logger.error(e)

    driver.quit()

    if batch:

        df = pd.DataFrame(batch)

        save_batch(df)

    checkpoint["crawled_links"] = list(crawled_links)

    save_checkpoint(checkpoint)

    logger.info("Crawler finished")


# =====================================
# ENTRY
# =====================================

if __name__ == "__main__":

    run_pipeline()