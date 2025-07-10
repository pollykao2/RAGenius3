from dotenv import load_dotenv
import os

load_dotenv()

import time 
import torch
import matplotlib.pyplot as plt

import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from openai import OpenAI
import yfinance as yf
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import string
from nltk.tokenize import TreebankWordTokenizer
import ta  

# ------------------------
#  extract_news_content
def extract_news_content(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = ' '.join(paragraphs)
        return content.strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# ------------------------
# 初始化 tokenizer
tokenizer = TreebankWordTokenizer()

# ------------------------
# 初始化 Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("news-index")

# ------------------------
# 初始化 BERT 向量化模型
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# ------------------------
# Google News RSS (搜尋鴻海)
start_time = time.perf_counter() 
rss_url = "https://news.google.com/rss/search?q=鴻海"
feed = feedparser.parse(rss_url)

news_list = []
for entry in feed.entries[:10]:
    news_list.append("[Google] " + entry.title + " - " + entry.link)

# ------------------------
# BM25 前置處理
tokenized_corpus = []
for i, entry in enumerate(feed.entries[:10]):
    content = extract_news_content(entry.link)
    full_text = entry.title + " " + content[:1000]
    tokens = tokenizer.tokenize(full_text.lower())
for i, entry in enumerate(feed.entries[:10]):
    content = extract_news_content(entry.link)
    full_text = entry.title + " " + content[:1000]
    tokens = tokenizer.tokenize(full_text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokenized_corpus.append(tokens)

bm25 = BM25Okapi(tokenized_corpus)
print(f"[計時] Google News & BM25 準備耗時: {time.perf_counter() - start_time:.2f} 秒") 

# ------------------------
# 把新聞轉向量並 upsert 到 Pinecone，多新聞內文爬取
start_time = time.perf_counter()
for i, entry in enumerate(feed.entries[:10]):
    content = extract_news_content(entry.link)
    if not content:
        content = ""
    full_text = entry.title + " " + content[:1000]  # 最多取 1000 字
    vector = model.encode(full_text).tolist()
    index.upsert([{
        "id": f"news-{i}",
        "values": vector,
        "metadata": {"text": full_text, "url": entry.link}
    }])
print(f"[計時] 向量 encode + Pinecone 上傳耗時: {time.perf_counter() - start_time:.2f} 秒") 
    # print(f" 已上傳到 Pinecone: {text}")

# ------------------------
# 使用 BM25 過濾後再用向量查詢
start_time = time.perf_counter()
query = "鴻海AI發展對市場影響"
query_tokens = tokenizer.tokenize(query.lower())
query_tokens = [t for t in query_tokens if t not in string.punctuation]

bm25_scores = bm25.get_scores(query_tokens)
sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

print("\n=== BM25 排序結果 (前5名) ===")
# for idx in sorted_indices[:5]:
#     print(f"Score: {bm25_scores[idx]:.4f}, Text: {news_list[idx]}")

filtered_news = [news_list[i] for i in sorted_indices[:5]]

# ------------------------
# Pinecone 語意查詢
query_vector = model.encode(query).tolist()
result = index.query(vector=query_vector, top_k=3, include_metadata=True)

matches = result.get("matches", [])
print(f"[計時] Pinecone 檢索耗時: {time.perf_counter() - start_time:.2f} 秒")  

prompt_content = "以下是與問題最相關的新聞，請分析它們對鴻海股價的影響：\n\n"
for match in matches:
    prompt_content += match["metadata"]["text"] + "\n"

# ------------------------
# Yahoo Finance & 技術指標 (用 ta)
stock_symbol = "2317.TW"
stock_name = "鴻海"
industry_pe = 14
growth_rate = 5
discount_rate = 8
interest_rate = 1.875
rate_position = "低利環境" if interest_rate < 2 else "正常或偏高"

ticker = yf.Ticker(stock_symbol)
info = ticker.info
eps = info.get("trailingEps", 0)
pe_ratio = info.get("trailingPE", 0)
roe = 18
moat = "穩定" if roe > 15 else "弱"
fcf = 2800
intrinsic_value = fcf * (1 + growth_rate/100) / ((discount_rate/100) - (growth_rate/100))
current_price = ticker.history(period="1d")["Close"].iloc[-1]
margin = (intrinsic_value - current_price) / current_price * 100

df = ticker.history(period="6mo")
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
rsi = df["RSI"].iloc[-1]
rsi_signal = "超賣(可能反彈)" if rsi < 30 else ("超買(可能拉回)" if rsi > 70 else "中性")

df["MACD"] = ta.trend.macd(df["Close"])
df["MACD_signal"] = ta.trend.macd_signal(df["Close"])
macd = df["MACD"].iloc[-1]
macd_signal = df["MACD_signal"].iloc[-1]
macd_signal_text = "黃金交叉(偏多)" if macd > macd_signal else "死亡交叉(偏空)"

# ------------------------
# Hugging Face 情緒分析
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
sentiments = []
for news in news_list:
    result = sentiment_pipeline(news)[0]
    sentiments.append(f"{news}\n情緒: {result['label']} (信心: {result['score']:.2f})")

# ------------------------
# GPT 分析
# GPT 分析
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt_content = f"""
你是專業股市分析師，從以下五面向：財報穩健度、產業週期、估值水位、技術指標、新聞情緒，
結合葛拉漢安全邊際理論、巴菲特護城河、彼得林區選股方法，以及總體經濟利率折現理論，
針對 {stock_name} 的詳細投資數據與技術指標進行分析，並嚴格依據數據做專業推論。

以下是計算結果：
- EPS: {eps:.2f}，本益比(PE): {pe_ratio:.2f}，產業平均PE: {industry_pe:.2f}
- 使用 DCF 模型 (FCF={fcf}億, g={growth_rate}%, r={discount_rate}%) 估值約為 {intrinsic_value:.2f}
- 現價: {current_price:.2f}，估值差距約 {margin:.1f}%
- ROE: 過去5年平均 {roe:.2f}% → 護城河: {moat}
- RSI: {rsi:.2f} → {rsi_signal}；MACD: {macd:.2f}/{macd_signal:.2f} → {macd_signal_text}
- 當前利率: {interest_rate:.2f}% ，處於 {rate_position}。
- 最新新聞情緒平均偏向: 正向/負向/中性


同時以下為最新新聞摘要與情緒：
"""

for match in matches:
    prompt_content += match["metadata"]["text"] + "\n"

prompt_content += """
你需要將上述數據與新聞完整融合，從「財報穩健度、產業週期、估值水位、技術指標、新聞情緒」
五大面向進行專業判讀，並結合：
- 葛拉漢安全邊際理論（低估高安全邊際）
- 巴菲特護城河（ROE穩健、獨佔優勢）
- 彼得林區選股（生活化洞察與成長性）
- 利率折現與通膨預期（宏觀經濟角度）


另外補充一些市場常識，你可以在推論時一併考慮：
- 若聯準會(Fed)降息，通常被視為對股票市場偏多的利多。
- 若CPI持續上漲，顯示通膨壓力增加，通常對股市偏空。
- 若企業持續投入AI與自動化，代表未來成長潛力，偏利多。
- 技術面若RSI過低、MACD黃金交叉，短期可能反彈；若RSI過高或MACD死亡交叉，短期可能回檔。

此外，**請務必預測明日的股價方向與預估幅度**（例如：「預估明日小幅上漲約1%」或「預期下跌1~2%」），
最後以1-2句話給出簡短的投資建議（如「可逢低分批佈局」或「建議短期觀望」），
禁止只泛泛而談，必須具體輸出推論與預測數據。

請用繁體中文回答。
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt_content}
    ]
)

print("\n=== GPT 回傳分析 ===")
print(response.choices[0].message.content)

# 顯示現價、近10日收盤價與走勢圖
# print(f"\n=== {stock_name}({stock_symbol}) 最新收盤價 ===")
# print(f"現價: {current_price:.2f}")

# print(f"\n=== {stock_name} 近10日收盤價 ===")
# print(df[["Close"]].tail(10))

# plt.figure(figsize=(10,5))
# plt.plot(df.index, df["Close"])
# plt.title(f"{stock_name}({stock_symbol}) 近6個月股價走勢")
# plt.xlabel("日期")
# plt.ylabel("收盤價")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()