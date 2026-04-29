import json
import os
import tempfile
from datetime import datetime, timedelta
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
from urllib.parse import quote, urlparse, urlunparse

# ========== CONFIG ==========
HDFS_HOST = "192.168.56.11"
HDFS_PORT = 9870
HDFS_USER = "zett"
HDFS_BASE_DIR = "/user/zett/vnexpress"

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
            response = requests.get(test_url, timeout=REQUEST_TIMEOUT)
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
            timeout=REQUEST_TIMEOUT,
            data=data,
        )

        if allow_redirects and response.is_redirect:
            location = _resolve_redirect_url(response.headers["Location"])
            response = requests.request(method, location, timeout=REQUEST_TIMEOUT, data=data)

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
    response_2 = requests.put(location, data=data.encode("utf-8"), timeout=REQUEST_TIMEOUT)
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
    response_2 = requests.post(location, data=data.encode("utf-8"), timeout=REQUEST_TIMEOUT)
    response_2.raise_for_status()


def load_json(path: str, default=None):
    content = hdfs_open(path)
    if content:
        return json.loads(content)
    return default if default is not None else {}


def save_json(path: str, data):
    hdfs_write(path, json.dumps(data, indent=2, ensure_ascii=False), overwrite=True)


def append_csv(path: str, df: pd.DataFrame):
    if not hdfs_exists(path):
        hdfs_write(path, df.to_csv(index=False), overwrite=True)
        return

    hdfs_append(path, df.to_csv(index=False, header=False))

BASE_URLS = [
    "https://vnexpress.net/khoa-hoc-cong-nghe/thiet-bi",
    "https://vnexpress.net/khoa-hoc-cong-nghe/ai",
    "https://vnexpress.net/khoa-hoc-cong-nghe/vu-tru",
    "https://vnexpress.net/khoa-hoc-cong-nghe/chuyen-doi-so"
]

CHECKPOINT_PATH = f"{HDFS_BASE_DIR}/vnexpress_checkpoint.json"
POST_CSV_PATH = f"{HDFS_BASE_DIR}/post_vnexpress.csv"
COMMENT_CSV_PATH = f"{HDFS_BASE_DIR}/comment_vnexpress.csv"

REQUEST_TIMEOUT = 30
SLEEP_AFTER_OPEN_POST = 3
MAX_CLICK_ROUNDS = 300

# ========== CHECKPOINT HELPERS ==========
def load_checkpoint(path: str):
    return load_json(
        path,
        default={"bases": {}, "processed_posts": [], "updated_at": None},
    )

def save_checkpoint(path: str, checkpoint: dict):
    checkpoint["updated_at"] = datetime.now().isoformat()
    save_json(path, checkpoint)

def append_unique(lst, value):
    if value not in lst:
        lst.append(value)

def create_driver():
    profile_dir = tempfile.TemporaryDirectory(prefix="vnexpress-chrome-")
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-data-dir={profile_dir.name}")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    return webdriver.Chrome(options=options), profile_dir

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

def is_driver_session_lost(exc: Exception):
    text = str(exc).lower()
    return any(
        keyword in text
        for keyword in [
            "connection refused",
            "connection reset by peer",
            "connectionreseterror",
            "connection aborted",
            "connectionabortederror",
            "failed to establish a new connection",
            "max retries exceeded",
            "invalid session id",
            "chrome not reachable",
            "session deleted",
            "disconnected",
            "winerror 10054",
            "winerror 10061",
            "/session/",
        ]
    )

def restart_driver(driver_state: dict):
    close_driver(driver_state.get("driver"), driver_state.get("profile_dir"))
    driver_state["driver"], driver_state["profile_dir"] = create_driver()

def safe_parse_post_and_comments(driver_state: dict, link: str):
    for attempt in range(2):
        try:
            return parse_post_and_comments_from_link(driver_state["driver"], link)
        except Exception as exc:
            if attempt == 0 and is_driver_session_lost(exc):
                print("  -> driver bị mất session, khởi động lại và thử lại link này")
                restart_driver(driver_state)
                continue
            raise

def ensure_base_state(checkpoint: dict, base_url: str):
    bases = checkpoint.setdefault("bases", {})
    if base_url not in bases:
        bases[base_url] = {
            "next_page": 1,
            "visited_pages": [],
            "queued_links": [],      # hàng chờ hiện tại (page đang chạy)
            "crawled_links": [],
            "current_page_url": None,
            "finished": False,
        }
    return bases[base_url]

# ========== PARSING HELPERS ==========
def extract_post_id(link: str):
    m = re.search(r"(\d+)(?:\.html)?/?$", link.strip())
    return int(m.group(1)) if m else None

def extract_user_id(href: str):
    if not href:
        return None
    m = re.search(r"(\d+)(?:\.html)?/?$", href.strip())
    return int(m.group(1)) if m else None

def parse_comment_time(raw_time: str, now: datetime = None):
    """
    - Nếu dạng tuyệt đối: HH:MM dd/m (hoặc có năm) => parse trực tiếp
    - Nếu dạng tương đối: Xh trước => lấy now - X giờ
    """
    if not raw_time:
        return None

    now = now or datetime.now()
    text = " ".join(raw_time.split()).lower()

    rel_h = re.search(r"(\d+)\s*h\s*trước", text)
    if rel_h:
        dt = now - timedelta(hours=int(rel_h.group(1)))
        return dt.strftime("%H:%M %d/%m/%Y")

    abs_1 = re.search(r"(\d{1,2}):(\d{2})\s+(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", text)
    if abs_1:
        hour, minute, day, month, year = abs_1.groups()
        y = int(year) if year else now.year
        dt = datetime(y, int(month), int(day), int(hour), int(minute))
        return dt.strftime("%H:%M %d/%m/%Y")

    abs_2 = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\s*,?\s*(\d{1,2}):(\d{2})", text)
    if abs_2:
        day, month, year, hour, minute = abs_2.groups()
        y = int(year) if year else now.year
        dt = datetime(y, int(month), int(day), int(hour), int(minute))
        return dt.strftime("%H:%M %d/%m/%Y")

    return raw_time.strip()

def extract_comment_content(comment_node):
    content_node = comment_node.select_one("p.full_content") or comment_node.select_one("p.content_more") or comment_node.select_one("p.content")
    if not content_node:
        return None

    cloned = BeautifulSoup(str(content_node), "html.parser")
    name_tag = cloned.select_one("span.txt-name")
    if name_tag:
        name_tag.decompose()
    return cloned.get_text(" ", strip=True)

def extract_reaction_detail(comment_node):
    """Lấy react theo yêu cầu: tên từ alt trong span.icons img, số lượng từ strong."""
    detail = {}
    reaction_items = comment_node.select("div.reactions-detail div.item")

    for item in reaction_items:
        img = item.select_one("span.icons img")
        strong = item.select_one("strong")
        react_name = img.get("alt", "").strip() if img else ""
        count_text = strong.get_text(" ", strip=True) if strong else ""
        m = re.search(r"\d+", count_text)
        if react_name and m:
            detail[react_name] = detail.get(react_name, 0) + int(m.group())

    return detail

def expand_all_comments(driver):
    selectors = [
        "a.view_all_reply",
        "a.txt_666",
        "p.count-reply",
        "div.view_more_coment.width_common.mb10",
    ]

    for _ in range(MAX_CLICK_ROUNDS):
        clicked = 0
        buttons = []
        for css in selectors:
            buttons.extend(driver.find_elements(By.CSS_SELECTOR, css))

        if not buttons:
            break

        for btn in buttons:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                time.sleep(0.15)
                driver.execute_script("arguments[0].click();", btn)
                clicked += 1
                time.sleep(0.25)
            except Exception:
                continue

        if clicked == 0:
            break

def parse_post_and_comments_from_link(driver, link: str):
    driver.get(link)
    time.sleep(SLEEP_AFTER_OPEN_POST)
    expand_all_comments(driver)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    post_id = extract_post_id(link)
    post_time_node = soup.select_one("span.date")
    post_time = post_time_node.get_text(" ", strip=True) if post_time_node else None
    content = "\n".join([p.get_text(" ", strip=True) for p in soup.select("p.Normal")])

    post_record = {
        "id_post": post_id,
        "link_post": link,
        "post_time": post_time,
        "post_content": content,
    }

    comment_nodes = soup.select("div.content-comment")
    comment_records = []

    for node in comment_nodes:
        nickname = node.select_one("a.nickname")
        user_name = nickname.get_text(" ", strip=True) if nickname else None
        user_href = nickname.get("href", "") if nickname else ""
        user_id = extract_user_id(user_href)

        comment_content = extract_comment_content(node)

        time_node = node.select_one("span.time-com") or node.select_one("span.time")
        raw_time = time_node.get_text(" ", strip=True) if time_node else ""
        comment_time = parse_comment_time(raw_time)

        reaction_detail = extract_reaction_detail(node)

        comment_records.append({
            "id_post": post_id,
            "user_id": user_id,
            "user_name": user_name,
            "comment_content": comment_content,
            "comment_time": comment_time,
            "reaction_detail": json.dumps(reaction_detail, ensure_ascii=False),
        })

    return post_record, comment_records

def append_records_to_csv(path: str, records: list):
    if not records:
        return
    df = pd.DataFrame(records)
    append_csv(path, df)

# ========== PAGE-BY-PAGE RUN ==========
select_reachable_hdfs_host()

checkpoint = load_checkpoint(CHECKPOINT_PATH)
for base_url in BASE_URLS:
    ensure_base_state(checkpoint, base_url)
save_checkpoint(CHECKPOINT_PATH, checkpoint)

driver_state = {}
driver_state["driver"], driver_state["profile_dir"] = create_driver()

try:
    processed_posts = set(checkpoint.get("processed_posts", []))

    for base_url in BASE_URLS:
        base_state = ensure_base_state(checkpoint, base_url)

        if base_state.get("finished", False):
            print(f"[SKIP BASE FINISHED] {base_url}")
            continue

        print(f"\n=== START BASE: {base_url} ===")

        # Chạy theo page: lấy links của page đó -> crawl hết queue page đó -> mới sang page tiếp
        while True:
            # Nếu queue page hiện tại rỗng thì mới fetch page mới
            if not base_state.get("queued_links", []):
                page = int(base_state.get("next_page", 1))
                page_url = f"{base_url}-p{page}/"
                print(f"[PAGE] {page_url}")

                try:
                    r = requests.get(page_url, timeout=REQUEST_TIMEOUT)
                except Exception as e:
                    print(f"  -> request lỗi ở base này: {e}")
                    break

                if r.status_code != 200:
                    # Không raise để không ảnh hưởng pipeline các base khác
                    print(f"  -> status={r.status_code}, kết thúc base này")
                    base_state["finished"] = True
                    save_checkpoint(CHECKPOINT_PATH, checkpoint)
                    break

                page_soup = BeautifulSoup(r.text, "html.parser")
                article_links = [a.get("href") for a in page_soup.select("h2.title-news a") if a.get("href")]

                # Đánh dấu page đã ghé
                append_unique(base_state["visited_pages"], page)
                base_state["next_page"] = page + 1
                base_state["current_page_url"] = page_url

                if not article_links:
                    # Hết links mới của base này -> chỉ kết thúc base này, pipeline vẫn chạy base khác
                    print("  -> không còn links mới, đánh dấu finished cho base này")
                    base_state["finished"] = True
                    base_state["queued_links"] = []
                    save_checkpoint(CHECKPOINT_PATH, checkpoint)
                    break

                crawled_set = set(base_state.get("crawled_links", []))
                processed_set = set(processed_posts)
                queue = []
                for lk in article_links:
                    if lk not in crawled_set and lk not in processed_set:
                        queue.append(lk)

                base_state["queued_links"] = queue
                save_checkpoint(CHECKPOINT_PATH, checkpoint)

                print(f"  -> links page: {len(article_links)}, vào queue crawl: {len(queue)}")

                # Nếu page này không có link mới để crawl thì chuyển ngay page tiếp
                if not base_state["queued_links"]:
                    print("  -> page này không có link mới, chuyển page tiếp")
                    continue

            # Crawl hết queue của page hiện tại trước khi chuyển page
            while base_state.get("queued_links", []):
                link_post = base_state["queued_links"].pop(0)
                save_checkpoint(CHECKPOINT_PATH, checkpoint)

                if link_post in processed_posts or link_post in base_state.get("crawled_links", []):
                    continue

                print(f"[POST] {link_post}")
                try:
                    post_record, comment_records = safe_parse_post_and_comments(driver_state, link_post)
                    append_records_to_csv(POST_CSV_PATH, [post_record])
                    append_records_to_csv(COMMENT_CSV_PATH, comment_records)

                    append_unique(base_state["crawled_links"], link_post)
                    append_unique(checkpoint["processed_posts"], link_post)
                    processed_posts.add(link_post)

                    save_checkpoint(CHECKPOINT_PATH, checkpoint)
                    print(f"  -> lưu xong post + {len(comment_records)} comments")

                except Exception as e:
                    # Không raise để pipeline vẫn tiếp tục; đưa link lỗi về cuối queue để thử lại lần chạy sau
                    base_state["queued_links"].append(link_post)
                    save_checkpoint(CHECKPOINT_PATH, checkpoint)
                    print(f"  -> lỗi post, giữ lại queue cho lần chạy sau: {e}")
                    # Dừng base hiện tại để tránh lặp lỗi liên tục, nhưng không làm hỏng pipeline
                    break

            # Nếu còn queue sau lỗi thì chuyển base_url tiếp theo, pipeline vẫn chạy
            if base_state.get("queued_links", []):
                print("  -> còn link lỗi trong queue, tạm dừng base này và chuyển base khác")
                break

        save_checkpoint(CHECKPOINT_PATH, checkpoint)
        print(f"=== END BASE: {base_url} ===")

finally:
    close_driver(driver_state.get("driver"), driver_state.get("profile_dir"))

print("DONE")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Post CSV: {POST_CSV_PATH}")
print(f"Comment CSV: {COMMENT_CSV_PATH}")