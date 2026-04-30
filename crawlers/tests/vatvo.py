import requests
from bs4 import BeautifulSoup
import time
import json
import os
import pandas as pd
import hashlib
import io
import re
from urllib.parse import quote, urlparse, urlunparse

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

HDFS_HOST = "192.168.56.11"
HDFS_PORT = 9870
HDFS_USER = "zett"
HDFS_BASE_DIR = "/user/zett/vatvo"

# Allow overriding from environment (useful when connecting via Tailscale).
HDFS_HOST = os.getenv("HDFS_HOST", HDFS_HOST)
HDFS_PORT = int(os.getenv("HDFS_PORT", str(HDFS_PORT)))
HDFS_USER = os.getenv("HDFS_USER", HDFS_USER)
HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", HDFS_BASE_DIR)
HDFS_HOSTS = [h.strip() for h in os.getenv("HDFS_HOSTS", "").split(",") if h.strip()]

if HDFS_HOSTS and HDFS_HOST not in HDFS_HOSTS:
    HDFS_HOSTS.insert(0, HDFS_HOST)
elif not HDFS_HOSTS:
    HDFS_HOSTS = [HDFS_HOST]

# Hostname -> IP mapping so redirect URLs from the NameNode
# (which contain DataNode hostnames like "worker1") can be resolved
# from machines outside the cluster.
DATANODE_HOST_MAP = {
    "master":  "192.168.56.11",
    "worker1": "192.168.56.12",
    "worker2": "192.168.56.13",
    "storage": "192.168.56.14",
}
# Allow overriding via env, e.g. HDFS_DATANODE_MAP="worker1=100.x.y.z,worker2=100.a.b.c"
for pair in os.getenv("HDFS_DATANODE_MAP", "").split(","):
    pair = pair.strip()
    if "=" in pair:
        k, v = pair.split("=", 1)
        DATANODE_HOST_MAP[k.strip()] = v.strip()


def _resolve_redirect_url(location: str) -> str:
    """Rewrite DataNode hostname in a WebHDFS redirect URL to its IP."""
    parsed = urlparse(location)
    mapped_ip = DATANODE_HOST_MAP.get(parsed.hostname)
    if mapped_ip:
        # Replace hostname, keep port
        new_netloc = f"{mapped_ip}:{parsed.port}" if parsed.port else mapped_ip
        location = urlunparse(parsed._replace(netloc=new_netloc))
    return location

PROGRESS_FILE = f"{HDFS_BASE_DIR}/progress.json"
CRAWLED_FILE = f"{HDFS_BASE_DIR}/crawled_links.json"
CSV_FILE = f"{HDFS_BASE_DIR}/articles.csv"

MAX_RETRY = 5
DELAY = 2
TIMEOUT = 10
BATCH_SIZE = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ==============================
# HDFS (WEBHDFS)
# ==============================
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
            r = requests.get(test_url, timeout=TIMEOUT)
            if r.status_code in (200, 401, 403):
                HDFS_HOST = host
                print(f"[HDFS] Using WebHDFS host: {HDFS_HOST}:{HDFS_PORT}")
                return
            last_error = f"HTTP {r.status_code}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)

    raise RuntimeError(
        "[HDFS CONNECT ERROR] Cannot connect to any WebHDFS host. "
        f"Tried: {', '.join(HDFS_HOSTS)} on port {HDFS_PORT}. "
        "Set HDFS_HOST or HDFS_HOSTS env to reachable address (LAN or Tailscale) and retry. "
        f"Last error: {last_error}"
    )


def hdfs_request(method, path, op, extra_params=None, allow_redirects=True, data=None):
    url, params = hdfs_url(path, op, extra_params)
    try:
        r = requests.request(
            method,
            url,
            params=params,
            allow_redirects=False,        # always intercept redirects
            timeout=TIMEOUT,
            data=data,
        )

        # If the caller wants redirects followed AND the NameNode issued one,
        # rewrite the DataNode hostname to its IP and follow manually.
        if allow_redirects and r.is_redirect:
            location = _resolve_redirect_url(r.headers["Location"])
            r = requests.request(method, location, timeout=TIMEOUT, data=data)

        return r
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "[HDFS CONNECT ERROR] Cannot connect to WebHDFS "
            f"http://{HDFS_HOST}:{HDFS_PORT}. "
            "Set HDFS_HOST/HDFS_PORT env to reachable address (LAN or Tailscale) and retry. "
            f"Original error: {e}"
        ) from e


def hdfs_exists(path):
    r = hdfs_request("GET", path, "GETFILESTATUS")
    if r.status_code == 200:
        return True
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return False


def hdfs_mkdirs(path):
    r = hdfs_request("PUT", path, "MKDIRS")
    r.raise_for_status()


def hdfs_open(path):
    r = hdfs_request("GET", path, "OPEN")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.text


def hdfs_write(path, data, overwrite=True):
    parent = os.path.dirname(path).replace("\\", "/")
    if parent:
        hdfs_mkdirs(parent)

    r = hdfs_request(
        "PUT",
        path,
        "CREATE",
        {"overwrite": str(overwrite).lower()},
        allow_redirects=False,
    )
    r.raise_for_status()

    location = r.headers.get("Location")
    if not location:
        raise RuntimeError(f"[HDFS] Missing redirect Location for CREATE: {path}")

    location = _resolve_redirect_url(location)
    r2 = requests.put(location, data=data.encode("utf-8"), timeout=TIMEOUT)
    r2.raise_for_status()


def hdfs_append(path, data):
    if not hdfs_exists(path):
        hdfs_write(path, data, overwrite=True)
        return

    r = hdfs_request("POST", path, "APPEND", allow_redirects=False)
    r.raise_for_status()

    location = r.headers.get("Location")
    if not location:
        raise RuntimeError(f"[HDFS] Missing redirect Location for APPEND: {path}")

    location = _resolve_redirect_url(location)
    r2 = requests.post(location, data=data.encode("utf-8"), timeout=TIMEOUT)
    r2.raise_for_status()

# ==============================
# UTILS JSON
# ==============================
def load_json(file):
    content = hdfs_open(file)
    if content:
        return json.loads(content)
    return {}


def save_json(file, data):
    hdfs_write(file, json.dumps(data, indent=2, ensure_ascii=False), overwrite=True)

# ==============================
# CSV (PANDAS)
# ==============================
def init_csv():
    if not hdfs_exists(CSV_FILE):
        header = pd.DataFrame(columns=["post_id", "article", "author", "time", "content"])
        hdfs_write(CSV_FILE, header.to_csv(index=False), overwrite=True)


def load_existing_ids():
    content = hdfs_open(CSV_FILE)
    if not content:
        return set()

    df = pd.read_csv(io.StringIO(content))
    if "post_id" not in df.columns:
        return set()
    return set(df["post_id"].astype(str))


def save_batch(batch):
    if not batch:
        return

    df = pd.DataFrame(batch)
    hdfs_append(CSV_FILE, df.to_csv(index=False, header=False))
    print(f"[BATCH SAVED] {len(batch)} articles")

# ==============================
# REQUEST
# ==============================
def fetch(url):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            return r

        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] {attempt}/{MAX_RETRY} -> {url}")
            time.sleep(DELAY)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {url} -> {e}")
            return None

    print(f"[FAILED] {url}")
    return None

# ==============================
# HELPERS
# ==============================
def generate_post_id(url):
    return hashlib.md5(url.encode()).hexdigest()

# ==============================
# CRAWL ARTICLE
# ==============================

def crawl_article(url, existing_ids):
    r = fetch(url)
    if r is None or r.status_code != 200:
        print(f"[FAIL ARTICLE] {url}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

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
    title_el = soup.select_one("h1.entry-title")
    title = title_el.get_text(strip=True) if title_el else ""

    # ==============================
    # 3. Author
    # ==============================
    author_el = soup.select_one('div.entry-meta a[rel="author"]')
    author = author_el.get_text(strip=True) if author_el else ""

    # ==============================
    # 4. Time
    # ==============================
    time_tag = soup.select_one('time.time.published')
    time_el = time_tag.get('title') if time_tag else None


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
    url = f"{base_url}page/{page}/"

    r = fetch(url)

    if r is None:
        return False, []

    if r.status_code == 404:
        print(f"[DONE-404] {base_url} page {page}")
        return True, []

    if r.status_code != 200:
        print(f"[SKIP] {url} status {r.status_code}")
        return False, []

    soup = BeautifulSoup(r.text, "html.parser")
    articles = soup.select("h3.post__title a")

    links = [a["href"] for a in articles]

    print(f"[PAGE] {page} -> {len(links)} links")

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

    while True:
        print(f"\n=== {base_url} | PAGE {page} ===")

        should_stop, links = crawl_page(base_url, page)

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
    try:
        select_reachable_hdfs_host()
        init_csv()

        progress = load_json(PROGRESS_FILE)
        crawled_data = load_json(CRAWLED_FILE)
        existing_ids = load_existing_ids()

        for url in URLS:
            crawl_category(url, progress, crawled_data, existing_ids)

        print("\nDONE ALL")
    except RuntimeError as e:
        print(str(e))
        print(
            "[HINT] Example (PowerShell):\n"
            "$env:HDFS_HOST='100.x.y.z'\n"
            "# or multiple candidates (LAN first, then Tailscale)\n"
            "$env:HDFS_HOSTS='192.168.56.11,100.x.y.z'\n"
            "$env:HDFS_PORT='9870'\n"
            "python crawlers/vatvo.py"
        )
        raise SystemExit(1)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()