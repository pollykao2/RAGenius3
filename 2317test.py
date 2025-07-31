import requests
from dotenv import load_dotenv
import os

# 載入 .env
load_dotenv()

# 抓取環境變數
api_key = os.getenv("GOOGLE_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")

# 確認是否載入成功
if not api_key or not cse_id:
    raise ValueError("❌ 沒有正確讀取到 API_KEY 或 CSE_ID")

params = {
    "q": "鴻海",
    "key": api_key,
    "cx": cse_id,
    "num": 1
}

r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)

print("🔗 URL:", r.url)
print("✅ Status:", r.status_code)
print("📦 回傳 JSON:", r.json())

data = r.json()
items = data.get("items", [])

for item in items:
    print("📌 標題:", item.get("title"))
    print("🔗 連結:", item.get("link"))
    print("📄 摘要:", item.get("snippet"))
    print("=" * 50)

