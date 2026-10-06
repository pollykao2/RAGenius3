# config.py
import os
from datetime import datetime

RECENT_DAYS = 3
MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# 股票代號與中文名稱對照表
STOCK_MAP = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2303.TW": "聯電",
    "2382.TW": "廣達",
    "2454.TW": "聯發科",
    "3231.TW": "緯創",
    "0050.TW": "元大台灣50",
    "0056.TW": "元大高股息"
}