import pandas as pd
from bs4 import BeautifulSoup
import requests
import html5lib
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse
import random
def normalize_time(raw_time: str, now=None) -> str:
    """
    Chuyển time Tinhte về format DD/MM/YYYY HH:MM
    """
    if not raw_time:
        return ""

    raw_time = raw_time.strip().lower()
    now = now or datetime.now()

    # 1. Định dạng chuẩn đã có (dd/mm/yyyy hh:mm)
    try:
        parsed = datetime.strptime(raw_time, "%d/%m/%Y %H:%M")
        return parsed.strftime("%d/%m/%Y %H:%M")
    except ValueError:
        pass

    # 2. Hôm qua
    if "hôm qua" in raw_time:
        dt = now - timedelta(days=1)
        return dt.strftime("%d/%m/%Y %H:%M")

    # 3. Dạng số + đơn vị (22 phút, 13 giờ, 2 ngày…)
    match = re.search(r"(\d+)\s*(phút|giờ|ngày|tuần|tháng)", raw_time)
    if match:
        value = int(match.group(1))
        unit = match.group(2)

        delta_map = {
            "phút": timedelta(minutes=value),
            "giờ": timedelta(hours=value),
            "ngày": timedelta(days=value),
            "tuần": timedelta(weeks=value),
            "tháng": timedelta(days=value * 30),  # gần đúng
        }

        dt = now - delta_map[unit]
        return dt.strftime("%d/%m/%Y %H:%M")

    # 4. Fallback – trả về time hiện tại
    return now.strftime("%d/%m/%Y %H:%M")
 
def get_post_id(url: str) -> str | None:
    path = urlparse(url).path.rstrip('/')
    match = re.search(r'\.(\d+)$', path)
    return match.group(1) if match else None

def random_sleep(min_s=2.5, max_s=5.5):
    t = round(random.uniform(min_s, max_s), 2)
    time.sleep(t)

URL_original = 'https://tinhte.vn/'

# Đường dẫn đến chromedriver (bạn tải tại https://chromedriver.chromium.org/)
options = Options()
# options.add_argument("--headless")  # Xóa dòng này nếu muốn thấy trình duyệt
driver = webdriver.Chrome(options=options)

# Truy cập Wikipedia
driver.get(URL_original)
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
soup = BeautifulSoup(driver.page_source, 'html.parser')

link = soup.select('li.jsx-4010778416')

link_list = [i.select_one('a').get('href') for i in link if i.select_one('a') is not None]

i = "https://tinhte.vn/thread/trai-nghiem-robot-hut-bui-lau-nha-ecovacs-deebot-t90-pro-omni-con-lan-dai-hon-ai-yiko-xai-da-ghe.4097300/"
# Đường dẫn đến chromedriver (bạn tải tại https://chromedriver.chromium.org/)
options = Options()
options.add_argument("--headless")  # Xóa dòng này nếu muốn thấy trình duyệt
driver = webdriver.Chrome(options=options)
driver.get(i)
time.sleep(2)
soup = BeautifulSoup(driver.page_source, 'html.parser')