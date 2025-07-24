from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

os.environ.pop("OPENAI_API_KEY", None)

# 強制讀取正確的 .env 檔案
load_dotenv(dotenv_path="C:/Users/user/Desktop/RAGenius3/.env")


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

from datetime import datetime, timedelta
import time




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

@app.route("/api/stock")
def stock_data():
    start_time = time.perf_counter()
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
    # Hugging Face 情緒分析
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    # ------------------------
    # Google News RSS (搜尋鴻海)
    start_time = time.perf_counter() 
    # Google News RSS (搜尋鴻海)
    rss_url = "https://news.google.com/rss/search?q=鴻海"
    feed = feedparser.parse(rss_url)

    # 篩選近30日的新聞
    cutoff_date = datetime.now() - timedelta(days=30)
    filtered_entries = []
    news_list = []

    for entry in feed.entries:
        if hasattr(entry, "published_parsed"):
            published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            if published >= cutoff_date:
                filtered_entries.append(entry)
                news_list.append("[Google] " + entry.title + " - " + entry.link)

# 最多只取10則
filtered_entries = filtered_entries[:10]

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
            full_text = entry.title + " " + content[:1000]
            vector = model.encode(full_text).tolist()

            # 執行情緒分析（用 full_text 而不是 title）
            sentiment_result = sentiment_pipeline(full_text)[0]
            sentiment_label = sentiment_result['label']      # POSITIVE or NEGATIVE
            sentiment_score = round(sentiment_result['score'], 3)

            # 上傳到 Pinecone，加入情緒標籤
            index.upsert([{
                "id": f"news-{i}",
                "values": vector,
                "metadata": {
                    "text": full_text,
                    "url": entry.link,
                    "sentiment": sentiment_label,
                    "sentiment_score": sentiment_score
                }
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

    # === 新增這段：統計情緒傾向 ===
    positive_count = sum(1 for m in matches if m['metadata'].get('sentiment') == 'POSITIVE')
    negative_count = sum(1 for m in matches if m['metadata'].get('sentiment') == 'NEGATIVE')

    if positive_count > negative_count:
        avg_sentiment_summary = "正向"
    elif negative_count > positive_count:
        avg_sentiment_summary = "負向"
    else:
        avg_sentiment_summary = "中性"



    prompt_content = "以下是與問題最相關的新聞，請分析它們對鴻海股價的影響：\n\n"
    for match in matches:
        sentiment = match["metadata"].get("sentiment", "未知")
        score = match["metadata"].get("sentiment_score", 0.0)
        prompt_content += f"【情緒：{sentiment}，信心：{score:.2f}】\n"
        prompt_content += match["metadata"]["text"] + "\n\n"

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
    **你的新聞必須是近30日的**，
    結合葛拉漢安全邊際理論、巴菲特護城河、彼得林區選股方法，以及總體經濟利率折現理論，
    針對 {stock_name} 的詳細投資數據與技術指標進行分析，並嚴格依據數據做專業推論。
     **邏輯推論** 與 **具體預測**

    以下是計算結果：
    - EPS: {eps:.2f}，本益比(PE): {pe_ratio:.2f}，產業平均PE: {industry_pe:.2f}
    - 使用 DCF 模型 (FCF={fcf}億, g={growth_rate}%, r={discount_rate}%) 估值約為 {intrinsic_value:.2f}
    - 現價: {current_price:.2f}，估值差距約 {margin:.1f}%
    - ROE: 過去5年平均 {roe:.2f}% → 護城河: {moat}
    - RSI: {rsi:.2f} → {rsi_signal}；MACD: {macd:.2f}/{macd_signal:.2f} → {macd_signal_text}
    - 當前利率: {interest_rate:.2f}% ，處於 {rate_position}。
    - 最新新聞情緒平均偏向: 正向/負向/中性


    📰【新聞情緒】
    - 本日新聞情緒平均：{avg_sentiment_summary}
    - 重點新聞摘要如下：
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
    - 最新新聞情緒平均偏向: {avg_sentiment_summary}

    你需要明確的指出
    1.明確新聞來源
    在提示中標明新聞平台／標題／時間，讓模型知道依據的是哪些資訊。

    2.聚焦重點元素
    引導模型先「摘要新聞要點」（事件、數據、消息），再在此基礎上推測心理。

    3.框架化分析維度
    可以從「情緒（Fear/Greed）」「預期（預期上漲/下跌）」「動機（逢低布局/獲利了結）」等面向拆解。

    4.產出格式引導
    明確要求「用條列」或「分段」回覆，有助於結構化結果。

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




    # === Flask API ===

    return jsonify({
        "chart": {
            "dates": df.index.strftime("%m/%d").tolist()[-10:],
            "prices": df["Close"].round(2).tolist()[-10:]
        },
        "gpt": response.choices[0].message.content,
        "news": [
            {
                "title": match["metadata"]["text"][:30] + "...",
                "sentiment": match["metadata"]["sentiment"],
                "score": round(match["metadata"]["sentiment_score"], 2)
            }
            for match in matches
        ]
    })



if __name__ == "__main__":
    app.run(port=5000)
