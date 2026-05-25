import json
import tempfile
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pandas as pd
import hashlib
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
VATVO_DIR = "vatvo"

PROGRESS_FILE = "progress.json"
CRAWLED_FILE = "crawled_links.json"
CSV_FILE = "articles.csv"

MAX_RETRY = 5
DELAY = 2
TIMEOUT = 10
BATCH_SIZE = 1

def ensure_data_dir():
    return None


def create_driver():
    profile_dir = tempfile.TemporaryDirectory(prefix="vatvo-chrome-")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.page_load_strategy = "eager"
    options.add_argument("--disable-gpu")
    options.add_argument("--headless=new")
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

    if profile_dir is not None:
        try:
            profile_dir.cleanup()
        except Exception:
            pass


def restart_driver(driver_state):
    close_driver(driver_state.get("driver"), driver_state.get("profile_dir"))
    driver_state["driver"], driver_state["profile_dir"] = create_driver()


def load_rendered_html(driver_state, url):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            driver_state["driver"].get(url)
            time.sleep(3)
            html = driver_state["driver"].page_source
            if any(
                marker in html
                for marker in [
                    "One moment, please...",
                    "ERR_CONNECTION_TIMED_OUT",
                    "This site can't be reached",
                    "DNS_PROBE",
                    "Aw, Snap!",
                ]
            ):
                print(f"[CHALLENGE] {attempt}/{MAX_RETRY} -> {url}")
                time.sleep(DELAY)
                continue
            return html
        except WebDriverException as exc:
            message = str(exc).lower()

            if any(
                keyword in message
                for keyword in [
                    "err_connection_timed_out",
                    "connection timed out",
                    "net::err_connection_timed_out",
                    "page load timeout",
                    "timeout",
                ]
            ):
                print(f"[URL TIMEOUT] {attempt}/{MAX_RETRY} -> {url} -> {exc}")
                try:
                    driver_state["driver"].execute_script("window.stop();")
                except Exception:
                    pass

                time.sleep(1)
                html = driver_state["driver"].page_source
                if html and not any(
                    marker in html
                    for marker in [
                        "One moment, please...",
                        "ERR_CONNECTION_TIMED_OUT",
                        "This site can't be reached",
                        "DNS_PROBE",
                        "Aw, Snap!",
                    ]
                ):
                    return html

                time.sleep(DELAY)
                continue

            if any(
                keyword in message
                for keyword in [
                    "invalid session id",
                    "chrome not reachable",
                    "session deleted",
                    "disconnected",
                ]
            ):
                print(f"[DRIVER ERROR] {attempt}/{MAX_RETRY} -> {url} -> {exc}")
                restart_driver(driver_state)
                time.sleep(DELAY)
                continue

            print(f"[DRIVER ERROR] {attempt}/{MAX_RETRY} -> {url} -> {exc}")
            time.sleep(DELAY)

    return None

# ==============================
# UTILS JSON
# ==============================
def load_json(file):
    return STORAGE.read_json(STORAGE.path(VATVO_DIR, file), {})

def save_json(file, data):
    STORAGE.write_json(STORAGE.path(VATVO_DIR, file), data)

# ==============================
# CSV (PANDAS)
# ==============================
def init_csv():
    STORAGE.init_csv(
        STORAGE.path(VATVO_DIR, CSV_FILE),
        ["post_id", "article", "author", "time", "content"],
    )

def load_existing_ids():
    return STORAGE.read_csv_column_as_str_set(
        STORAGE.path(VATVO_DIR, CSV_FILE),
        "post_id",
    )

def save_batch(batch):
    if not batch:
        return

    STORAGE.append_csv(STORAGE.path(VATVO_DIR, CSV_FILE), batch)
    print(f"[BATCH SAVED] {len(batch)} articles")

# ==============================
# REQUEST
# ==============================
def fetch(url):
    raise NotImplementedError("requests-based fetch is disabled for Vatvo because the site returns an anti-bot challenge page")

# ==============================
# HELPERS
# ==============================
def generate_post_id(url):
    return hashlib.md5(url.encode()).hexdigest()

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

        if "/" in path:
            continue

        normalized = href.split("#", 1)[0].rstrip("/") + "/"
        if normalized in seen:
            continue

        seen.add(normalized)
        links.append(normalized)

    return links

# ==============================
# CRAWL ARTICLE
# ==============================
from bs4 import BeautifulSoup
from datetime import datetime
import re

def crawl_article(url, existing_ids):
    html = load_rendered_html(crawl_article.driver_state, url)
    if html is None:
        print(f"[FAIL ARTICLE] {url}")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # ==============================
    # 1. Lấy post_id từ body class
    # ==============================
    body = soup.select_one("body")
    post_id = None

    if body and body.get("class"):
        m = re.search(r'postid-(\d+)', ' '.join(body["class"]))
        if m:
            post_id = m.group(1)

    if post_id is None:
        print(f"[NO ID] {url}")
        return None

    if post_id in existing_ids:
        print(f"[SKIP EXIST] {url}")
        return None

    # ==============================
    # 2. Title
    # ==============================
    title_el = soup.select_one("h1.s-title.fw-headline")
    title = title_el.get_text(strip=True) if title_el else ""

    # ==============================
    # 3. Author
    # ==============================
    author_el = soup.select_one('div.meta-el a.meta-author-url.meta-author')
    author = author_el.get_text(strip=True) if author_el else ""

    # ==============================
    # 4. Time
    # ==============================
    time_tag = soup.select_one('time.updated-date')
    time_el = time_tag.get('datetime') if time_tag else None


    # ==============================
    # 5. Content
    # ==============================
    content_el = soup.select_one("div.entry-content")
    content = content_el.get_text(" ", strip=True) if content_el else ""

    print(f"[CRAWLED] {url}")

    return {
        "post_id": post_id,
        "article": title,
        "author": author,
        "time": time_el,
        "content": content
    }

# ==============================
# CRAWL PAGE
# ==============================
def crawl_page(base_url, page):
    url = base_url if page == 1 else f"{base_url}page/{page}/"

    html = load_rendered_html(crawl_page.driver_state, url)
    if html is None:
        return False, None

    soup = BeautifulSoup(html, "html.parser")
    links = extract_article_links(soup)

    print(f"[PAGE] {page} -> {len(links)} links")

    if not links:
        print(f"[DONE-NO-LINKS] {base_url} page {page}")
        return True, []

    return False, links

# ==============================
# MAIN CRAWL CATEGORY
# ==============================
def crawl_category(base_url, progress, crawled_data, existing_ids):
    if base_url not in progress:
        progress[base_url] = {"page": 1, "done": False}

    if base_url not in crawled_data:
        crawled_data[base_url] = {}

    page = progress[base_url]["page"]
    page_attempts = 0

    while True:
        print(f"\n=== {base_url} | PAGE {page} ===")

        should_stop, links = crawl_page(base_url, page)

        if links is None:
            page_attempts += 1
            if page_attempts >= MAX_RETRY:
                print(f"[FAIL PAGE LOAD] {base_url} page {page}")
                break
            print(f"[RETRY PAGE] {base_url} page {page}")
            time.sleep(DELAY)
            continue

        page_attempts = 0

        if should_stop:
            progress[base_url]["done"] = True
            save_json(PROGRESS_FILE, progress)
            print(f"[DONE CATEGORY] {base_url}")
            break

        if str(page) not in crawled_data[base_url]:
            crawled_data[base_url][str(page)] = []

        crawled_links = set(crawled_data[base_url][str(page)])

        batch = []

        # ==============================
        # CRAWL ARTICLES
        # ==============================
        for link in links:
            if link in crawled_links:
                continue

            result = crawl_article(link, existing_ids)

            if result:
                batch.append(result)
                crawled_data[base_url][str(page)].append(link)
                existing_ids.add(result["post_id"])

            # batch save
            if len(batch) >= BATCH_SIZE:
                save_batch(batch)
                batch = []
                save_json(CRAWLED_FILE, crawled_data)

        # save remaining
        save_batch(batch)
        save_json(CRAWLED_FILE, crawled_data)

        # next page
        page += 1
        progress[base_url]["page"] = page
        save_json(PROGRESS_FILE, progress)

# ==============================
# MAIN
# ==============================
def main():
    global STORAGE

    STORAGE = RemoteStorage()
    init_csv()

    progress = load_json(PROGRESS_FILE)
    crawled_data = load_json(CRAWLED_FILE)
    existing_ids = load_existing_ids()

    driver, profile_dir = create_driver()
    crawl_page.driver_state = {"driver": driver, "profile_dir": profile_dir}
    crawl_article.driver_state = crawl_page.driver_state

    try:
        for url in URLS:
            crawl_category(url, progress, crawled_data, existing_ids)
    finally:
        close_driver(crawl_page.driver_state.get("driver"), crawl_page.driver_state.get("profile_dir"))
        STORAGE.close()

    print("\nDONE ALL")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
    print("Data and checkpoint were written directly to HDFS.")
