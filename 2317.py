from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import time
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
import numpy as np
import pandas as pd
from newsapi import NewsApiClient
from ta.momentum import RSIIndicator
from ta.trend import MACD
from urllib.parse import urlencode


# 初始化 Flask 應用與環境變數
app = Flask(__name__)
CORS(app)
os.environ.pop("OPENAI_API_KEY", None)
load_dotenv(dotenv_path="C:/Users/user/Desktop/RAGenius3/.env")


def extract_news_content(url: str) -> str:
    """
    透過 requests + BeautifulSoup 抓取網頁內所有 <p> 段落文字，
    並根據 meta time 標籤或 <time> 判斷是否近30天，超過返回空字符串。
    """
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 收集段落文字
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = ' '.join(paragraphs).strip()

        # 抓發布時間
        published_str = None
        time_tag = soup.find('time')
        if time_tag and time_tag.has_attr('datetime'):
            published_str = time_tag['datetime']
        elif soup.find('meta', attrs={"property": "article:published_time"}):
            published_str = soup.find('meta', attrs={"property": "article:published_time"})['content']
        elif soup.find('meta', attrs={"name": "pubdate"}):
            published_str = soup.find('meta', attrs={"name": "pubdate"})['content']

        if published_str:
            try:
                dt = datetime.fromisoformat(published_str.replace('Z', '+00:00')).astimezone()
                if dt < datetime.now() - timedelta(days=10):
                    return "", None
                return content, dt
            except:
                pass
        return content, None
    except Exception as e:
        print(f"❌ 無法抓取新聞 {url}: {e}")
        return ""


def search_google_news(query: str, num_results: int = 20):
    """
    使用 Google Programmable Search API 進行關鍵字搜尋，
    不含 dateRestrict 與 after:，由 extract_news_content 處理日期過濾。
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id  = os.getenv("GOOGLE_CSE_ID")

    url= "https://www.googleapis.com/customsearch/v1"

    per_page = 10
    total_pages = (num_results + per_page - 1) // per_page  # 向上取整數

    params_list = []
    for i in range(total_pages):
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": per_page,
            "start": i * per_page + 1,
            "hl": "zh-TW",
            "gl": "tw"
        }
        params_list.append(params)

        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        
        all_items = []

    for i, params in enumerate(params_list):
        encoded = urlencode(params, safe="")
        full_url = f"{url}?{encoded}"
        print(f"🔗 最終 URL [{i+1}]:", full_url)

        response = requests.get(full_url, headers=headers, timeout=10)

        print("🔍 Google 回傳狀態碼：", response.status_code)
        print("🧾 Google 回傳內容（前500字）:", response.text[:500])

        try:
            response.raise_for_status()
        except Exception as e:
            print(f"⚠️ 狀態錯誤：{e}")
            continue

        results = response.json()
        if "error" in results:
            print("❌ Google API 回傳錯誤：", results["error"].get("message"))
            continue

        items = results.get("items", [])
        all_items.extend(items)

    if not all_items:
        print("⚠️ 所有分頁都沒抓到 items")
        return []

    print(f"✅ 成功取得 {len(all_items)} 筆搜尋結果")



    return [
        {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet", "")
        }
        for item in all_items
    ]

def get_stock_metrics(stock_symbol: str, newsapi_key: str) -> dict:
    """
    動態從 Yahoo Finance 抓取 EPS、PE、DCF、ROE、RSI、MACD、利率等，
    並從 NewsAPI 抓取近 30 天新聞做情緒分析。
    """
    ticker = yf.Ticker(stock_symbol)
    info   = ticker.info

    # EPS、PE
    eps      = info.get("trailingEps", np.nan)
    pe_ratio = info.get("trailingPE", np.nan)

    # 產業平均 PE
    peers = info.get("industryPeers", [])
    peers_pe = []
    for peer in peers:
        try:
            p = yf.Ticker(peer).info.get("trailingPE")
            if p and p>0:
                peers_pe.append(p)
        except:
            continue
    industry_pe = float(np.mean(peers_pe)) if peers_pe else np.nan

    # DCF 估值（Free Cash Flow）
    row_label = "Free Cash Flow"
    try:
        if row_label in ticker.cashflow.index:
            cf_series = ticker.cashflow.loc[row_label]
        else:
            cf_series = pd.Series(dtype=float)

        cf = cf_series / 1e8
        fcf = cf.iloc[0] if len(cf) else np.nan
        growth_rate = ((cf.iloc[0]/cf.iloc[1] - 1)*100) if len(cf) >= 2 and cf.iloc[1] != 0 else np.nan
    except Exception as e:
        print(f"❌ 取得 FCF/Growth Rate 錯誤：{e}")
        fcf = np.nan
        growth_rate = np.nan

    # 折現率
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 10
    except:
        tnx = 3.5  # fallback 假設利率

    discount_rate = tnx / 100 + 0.05
    try:
        intrinsic_value = fcf * (1 + growth_rate/100) / (discount_rate - growth_rate/100)
    except:
        intrinsic_value = np.nan


    # 即時價格 & 估值差距
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    margin = (current_price - intrinsic_value) / intrinsic_value * 100

    # ROE & 護城河
    if "Net Income" in ticker.financials.index:
        netinc = ticker.financials.loc["Net Income"]
    else:
        netinc = pd.Series(dtype=float)

    if "Total Stockholder Equity" in ticker.balance_sheet.index:
        equity = ticker.balance_sheet.loc["Total Stockholder Equity"]
    else:
        equity = pd.Series(dtype=float)

    roe_series = (netinc / equity * 100).dropna()
    roe = float(roe_series.head(5).mean()) if len(roe_series) else np.nan
    moat = "強" if roe>15 else "中" if roe>5 else "弱"

    # RSI & MACD
    hist60 = ticker.history(period="60d")["Close"]
    rsi = float(RSIIndicator(hist60, window=14).rsi().iloc[-1])
    rsi_signal = "超買" if rsi>70 else "超賣" if rsi<30 else "中性"
    macd_ind = MACD(hist60)
    macd = float(macd_ind.macd().iloc[-1])
    macd_signal = float(macd_ind.macd_signal().iloc[-1])
    macd_signal_text = "買入" if macd>macd_signal else "賣出"

    # 利率位置
    interest_rate = tnx
    rate_position = "偏高" if interest_rate>3 else "偏低" if interest_rate<1 else "持平"

    # 新聞情緒（NewsAPI）
    newsapi = NewsApiClient(api_key=newsapi_key)
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=30)
    articles = newsapi.get_everything(
        q=stock_symbol,
        from_param=from_date.isoformat(),
        to=to_date.isoformat(),
        language="zh",
        sort_by="relevancy",
        page_size=15
    ).get("articles", [])
    headlines = [a.get("title","") for a in articles]
    sa_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = sa_pipe(headlines)
    scores = [1 if s["label"]=="POSITIVE" else -1 for s in sentiments]
    avg_score = sum(scores)/len(scores) if scores else 0
    avg_sent = "正向" if avg_score>0 else "負向" if avg_score<0 else "中性"

    return {
        "eps": eps,
        "pe_ratio": pe_ratio,
        "industry_pe": industry_pe,
        "fcf": fcf,
        "growth_rate": growth_rate,
        "discount_rate": discount_rate*100,
        "intrinsic_value": intrinsic_value,
        "current_price": current_price,
        "margin": margin,
        "roe": roe,
        "moat": moat,
        "rsi": rsi,
        "rsi_signal": rsi_signal,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_signal_text": macd_signal_text,
        "interest_rate": interest_rate,
        "rate_position": rate_position,
        "avg_sent": avg_sent,
        "headlines": headlines
    }

@app.route("/api/stock")
def stock_data():
    tokenizer = TreebankWordTokenizer()

    # Pinecone 與 Embedding & Sentiment Pipeline
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("news-index")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"

    )

    # 1) 搜新聞
    print("✅ GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
    print("✅ GOOGLE_CSE_ID:", os.getenv("GOOGLE_CSE_ID"))
    results = search_google_news("鴻海", num_results=10)
    filtered = []
    for it in results:
        title = it.get('title', '')
        link = it.get('link', '')
        print(f"🔗 嘗試抓新聞：{title} | {link}")

        content, pub_dt = extract_news_content(link)
        if content:
            sentiment = sentiment_pipe(title)[0]
            filtered.append({
                'title': title,
                'link': link,
                'content': content,
                'sentiment': sentiment['label'],
                'score': round(sentiment['score'], 3),
                'published': pub_dt
            })

    filtered.sort(key=lambda x: x['published'] or datetime.min, reverse=True)
    filtered = filtered[:10]

    # 2) BM25 前處理
    corpus = []
    for e in filtered:
        txt = e['title'] + ' ' + e['content'][:1000]
        toks = [t for t in tokenizer.tokenize(txt.lower()) if t not in string.punctuation]
        corpus.append(toks)

    # 3) 空 corpus guard
    if not corpus or all(len(doc)==0 for doc in corpus):
        matches = []
        avg_sent = "中性"
    else:
        # 4) BM25
        bm25 = BM25Okapi(corpus)
        # 5) 上傳向量
        for i,e in enumerate(filtered):
            text = e['title'] + ' ' + e['content'][:1000]
            vec  = model.encode(text).tolist()
            short_text = text[:500]  # 或保守點用 300
            s = sentiment_pipe(short_text)[0]
            
            index.upsert([{
                'id': f'news-{i}',
                'values': vec,
                'metadata': {
                'text': text,
                'url': e['link'],
                'sentiment': s['label'],
                'sentiment_score': round(s['score'],3)
                }
            }])
        # 6) BM25 取 top5，並做向量檢索 top3
        query   = "鴻海AI發展對市場影響"
        qtokens = [t for t in tokenizer.tokenize(query.lower())
                    if t not in string.punctuation]
        bm_scores = bm25.get_scores(qtokens)
        top_idx   = sorted(range(len(bm_scores)),
                        key=lambda i: bm_scores[i],
                        reverse=True)[:5]

        qvec    = model.encode(query).tolist()
        res     = index.query(vector=qvec, top_k=3, include_metadata=True)
        matches = res.get('matches', [])

        # 7) 平均情緒
        pos = sum(1 for m in matches
                if m['metadata']['sentiment'] == 'POSITIVE')
        neg = sum(1 for m in matches
                if m['metadata']['sentiment'] == 'NEGATIVE')
        avg_sent = '正向' if pos > neg else '負向' if neg > pos else '中性'

    # 財務與技術指標
    stock_symbol = "2317.TW"
    stock_name   = "鴻海"
    metrics = get_stock_metrics(stock_symbol, os.getenv("NEWSAPI_KEY"))

    eps               = metrics['eps']; pe_ratio          = metrics['pe_ratio']; industry_pe       = metrics['industry_pe']
    fcf               = metrics['fcf']; growth_rate       = metrics['growth_rate']; discount_rate     = metrics['discount_rate']
    intrinsic_value   = metrics['intrinsic_value']; current_price     = metrics['current_price']
    margin            = metrics['margin']; roe               = metrics['roe']; moat              = metrics['moat']
    rsi               = metrics['rsi']; rsi_signal        = metrics['rsi_signal']
    macd              = metrics['macd']; macd_signal       = metrics['macd_signal']; macd_signal_text  = metrics['macd_signal_text']
    interest_rate     = metrics['interest_rate']; rate_position     = metrics['rate_position']
    avg_sent_fin      = metrics['avg_sent']; headlines         = metrics['headlines']

    # 組 prompt 並呼叫 GPT
    prompt_content = f"""
        你是專業股市分析師，從以下五面向：財報穩健度、產業週期、估值水位、技術指標、新聞情緒，
        **你的新聞必須是近30日的**，
        結合葛拉漢安全邊際理論、巴菲特護城河、彼得林區選股方法，以及總體經濟利率折現理論，


        針對 {stock_name} 的詳細投資數據與技術指標進行分析，並嚴格依據數據做專業推論。
        **邏輯推論** 與 **具體預測**

        以下是計算結果：
        - EPS: {eps:.2f}，本益比(PE): {pe_ratio:.2f}，產業平均PE: {industry_pe:.2f}
        - 使用 DCF 模型 (FCF={fcf:.2f}億, g={growth_rate:.1f}%, r={discount_rate:.2f}%) 估值約為 {intrinsic_value:.2f}
        - 現價: {current_price:.2f}，估值差距約 {margin:.1f}%
        - ROE: 過去5年平均 {roe:.2f}% → 護城河: {moat}
        - RSI: {rsi:.2f} → {rsi_signal}；MACD: {macd:.2f}/{macd_signal:.2f} → {macd_signal_text}
        - 當前利率: {interest_rate:.2f}%，處於 {rate_position}
        - 最新新聞情緒平均偏向: {avg_sent}

        📰【新聞情緒】
        - 本日新聞情緒平均：{avg_sent}
        - 重點新聞摘要如下：
        1. {headlines[0]}
        2. {headlines[1]}
        3. …
        """
    for m in matches:
        prompt_content += m['metadata']['text'] + "\n"

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
        - 最新新聞情緒平均偏向: {avg_sent}
        

        你需要明確指出：
        1. 明確新聞來源（平台／標題／時間）。
        2. 聚焦重點元素（摘要新聞要點，再推測心理）。
        3. 框架化分析維度（情緒、預期、動機）。
        4. 產出格式引導（條列或分段）。
        5. **所有新聞皆來自最近10日，並以近3日為主。**


        此外，**請務必預測明日的股價方向與預估幅度**，並以1–2句話給出投資建議。

        請用繁體中文回答。
        """

    # GPT 呼叫
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": prompt_content}],
    )

    ticker = yf.Ticker("2317.TW")

    chart_df = ticker.history(period="10d")[["Close"]].reset_index()
    chart_df["Date"] = chart_df["Date"].dt.strftime("%Y-%m-%d")  # 時間格式轉為字串
    chart_data = chart_df.to_dict(orient="records")

    return jsonify({
        'news': [{'title': e['title'], 'link': e['link'], 'sentiment': e['sentiment'], 'score': e['score']} for e in filtered],
        'sentiment_summary': avg_sent,
        'gpt': response.choices[0].message.content,
        'chart_data': chart_data 
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
