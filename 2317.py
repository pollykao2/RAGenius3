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
from typing import Optional, Tuple

RECENT_DAYS = 3

# 初始化 Flask 應用與環境變數
app = Flask(__name__)
CORS(app)
os.environ.pop("OPENAI_API_KEY", None)
load_dotenv(dotenv_path="C:/Users/user/Desktop/RAGenius3/.env")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)


def extract_news_content(url: str) -> Tuple[str, Optional[datetime]]:
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = ' '.join(paragraphs).strip()

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
                if dt < datetime.now() - timedelta(days=RECENT_DAYS):
                    return "", None
                return content, dt
            except:
                pass
        return content, None
    except Exception as e:
        print(f"❌ 無法抓取新聞 {url}: {e}")
        return "", None



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
            "gl": "tw",
            "dateRestrict": "d3" 

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

# 1) 新增：通用的新聞搜尋（個股/大盤都可用）
def gather_news(query_list, num_results=10):
    all_news = []
    for q in query_list:
        # Google CSE
        g_items = search_google_news(q, num_results=min(10, num_results))
        for it in g_items:
            content, pub_dt = extract_news_content(it["link"])
            if content:
                s = sentiment_pipe(it["title"])[0]
                all_news.append({
                    "source": "GoogleCSE",
                    "query": q,
                    "title": it["title"],
                    "link": it["link"],
                    "published": pub_dt,
                    "sentiment": s["label"],
                    "score": round(s["score"], 3),
                    "content": content
                })
    # NewsAPI（大盤用關鍵字、語言：中文）
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        newsapi = NewsApiClient(api_key=newsapi_key)
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=RECENT_DAYS)
        for q in query_list:
            arts = newsapi.get_everything(
                q=q,
                from_param=from_date.isoformat(),
                to=to_date.isoformat(),
                language="zh",
                sort_by="publishedAt",
                page_size=10
            ).get("articles", [])
            for a in arts:
                title = a.get("title","")
                url = a.get("url","")
                content, pub_dt = extract_news_content(url)
                if content:
                    s = sentiment_pipe(title)[0]
                    all_news.append({
                        "source": "NewsAPI",
                        "query": q,
                        "title": title,
                        "link": url,
                        "published": pub_dt,
                        "sentiment": s["label"],
                        "score": round(s["score"], 3),
                        "content": content
                    })

    # 近10天 & 依日期新到舊
    all_news = [n for n in all_news if n.get("published") is None or n["published"] >= datetime.now()-timedelta(days=RECENT_DAYS)]
    all_news.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
    return all_news[:num_results]

# 2) 新增：通用指數技術面
def get_index_metrics(symbol: str):
    t = yf.Ticker(symbol)
    hist = t.history(period="6mo")["Close"].dropna()
    if len(hist) < 30:
        return {}
    rsi = float(RSIIndicator(hist, window=14).rsi().iloc[-1])
    macd_ind = MACD(hist)
    macd = float(macd_ind.macd().iloc[-1])
    macd_sig = float(macd_ind.macd_signal().iloc[-1])
    trend = "偏多" if macd > macd_sig else "偏空"
    return {
        "symbol": symbol,
        "last": float(hist.iloc[-1]),
        "rsi": rsi,
        "rsi_note": "超買" if rsi>70 else "超賣" if rsi<30 else "中性",
        "macd": macd, "macd_signal": macd_sig, "macd_note": "買入" if macd>macd_sig else "賣出",
        "trend": trend,
        "chart": {
            "dates": hist.index.strftime("%Y-%m-%d").tolist()[-30:],
            "prices": [round(x,2) for x in hist.tolist()[-30:]]
        }
    }


def get_stock_metrics(stock_symbol: str, newsapi_key: str) -> dict:
    """
    動態從 Yahoo Finance 抓取 EPS、PE、DCF、ROE、RSI、MACD、利率等，
    並從 NewsAPI 抓取近 10 天新聞做情緒分析。
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
    from_date = to_date - timedelta(days=RECENT_DAYS)
    articles = newsapi.get_everything(
        q=stock_symbol,
        from_param=from_date.isoformat(),
        to=to_date.isoformat(),
        language="zh",
        sort_by="relevancy",
        page_size=15
    ).get("articles", [])
    headlines = [a.get("title","") for a in articles]
    sentiments = sentiment_pipe(headlines)
    scores = [1 if s["label"] in ["1", "POSITIVE", "正向"] else -1 for s in sentiments]
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

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("news-index")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    # === 個股：鴻海 ===
    foxconn_news = gather_news(["鴻海", "富士康 2317", "鴻海 AI"])
    print("✅ 鴻海新聞筆數：", len(foxconn_news))
    # === 大盤（台股/美股） ===
    market_news = gather_news(["台股 大盤", "加權指數", "台灣股市", "美股", "NASDAQ", "S&P 500"], num_results=15)
    print("✅ 市場新聞筆數：", len(market_news))
    # === 上傳 BM25 + 向量（針對個股新聞）===
    filtered = foxconn_news[:10]
    corpus = []
    for e in filtered:
        txt = e['title'] + ' ' + e['content'][:1000]
        toks = [t for t in tokenizer.tokenize(txt.lower()) if t not in string.punctuation]
        corpus.append(toks)

    matches = []
    avg_sent = "中性"
    if corpus and not all(len(d)==0 for d in corpus):
        bm25 = BM25Okapi(corpus)
        for i,e in enumerate(filtered):
            text = e['title'] + ' ' + e['content'][:1000]
            vec  = model.encode(text).tolist()
            s = sentiment_pipe(text[:500])[0]
            metadata = {
                'text': text,
                'url': e['link'],
                'sentiment': s['label'],
                'sentiment_score': round(s['score'], 3),
            }
            if e.get('published'):
                metadata['published'] = e['published'].isoformat()

            index.upsert([{
                'id': f'news-{i}',
                'values': vec,
                'metadata': metadata,
            }])

        query = "鴻海 AI 產能與接單對台股影響"
        qvec = model.encode(query).tolist()
        res = index.query(vector=qvec, top_k=3, include_metadata=True)
        matches = res.get('matches', [])
        pos = sum(1 for m in matches if m['metadata']['sentiment']=='POSITIVE')
        neg = sum(1 for m in matches if m['metadata']['sentiment']=='NEGATIVE')
        avg_sent = '正向' if pos>neg else '負向' if neg>pos else '中性'

    # === 個股財務與技術 ===
    stock_name = "鴻海"
    metrics = get_stock_metrics("2317.TW", os.getenv("NEWSAPI_KEY"))

    # === 大盤指數（台股 + 美股概況）===
    idx_twii = get_index_metrics("^TWII")
    idx_spx  = get_index_metrics("^GSPC")
    idx_ixic = get_index_metrics("^IXIC")

    # === GPT：把「大盤 + 個股」一起分析 ===
    market_headlines = [n["title"] for n in market_news[:5]]
    fox_headlines = metrics["headlines"][:2] if metrics.get("headlines") else []

    prompt_content = f"""
    你是專業股市分析師，從以下五面向：財報穩健度、產業週期、估值水位、技術指標、新聞情緒，
    **你的新聞必須是近10日的**，
    結合葛拉漢安全邊際理論、巴菲特護城河、彼得林區選股方法，以及總體經濟利率折現理論，
    針對 {stock_name} 的詳細投資數據與技術指標進行分析，並嚴格依據數據做專業推論。
    **邏輯推論** 與 **具體預測**

    同時請從「大盤」與「個股」兩層做判讀，時間以近10日為主、近3日權重最高。

    [大盤]
    - 台股(^TWII) 收盤：約 {idx_twii.get('last')}, RSI {idx_twii.get('rsi'):.1f}（{idx_twii.get('rsi_note')}）, MACD:{idx_twii.get('macd'):.2f}/{idx_twii.get('macd_signal'):.2f}（{idx_twii.get('macd_note')}）, 趨勢：{idx_twii.get('trend')}
    - S&P500(^GSPC) RSI {idx_spx.get('rsi'):.1f}（{idx_spx.get('rsi_note')}）；NASDAQ(^IXIC) RSI {idx_ixic.get('rsi'):.1f}（{idx_ixic.get('rsi_note')}）
    - 大盤重點新聞（前5）：
    - """ + "\n  - ".join(market_headlines if market_headlines else ["無"]) + f"""

    [個股：{stock_name} 2317.TW]
    - EPS:{metrics['eps']}、PE:{metrics['pe_ratio']}、同業PE均值:{metrics['industry_pe']}
    - DCF估值:{metrics['intrinsic_value']:.2f}，現價:{metrics['current_price']:.2f}，折溢價：{metrics['margin']:.1f}%
    - ROE:{metrics['roe']:.1f}%（護城河：{metrics['moat']}）
    - RSI:{metrics['rsi']:.1f}（{metrics['rsi_signal']}），MACD:{metrics['macd']:.2f}/{metrics['macd_signal']:.2f}（{metrics['macd_signal_text']}）
    - 近30日個股新聞情緒：{metrics['avg_sent']}；重點標題：""" + "；".join(fox_headlines if fox_headlines else ["無"]) + f"""

    [近似新聞片段（RAG）重點]
    """ + "\n".join([m['metadata']['text'][:400] for m in matches]) + """

    你需要將上述數據與新聞完整融合，從「財報穩健度、產業週期、估值水位、技術指標、新聞情緒」進行論證。

    你需要明確指出：
    1. 明確新聞來源（平台／標題／時間）。
    2. 聚焦重點元素（摘要新聞要點，再推測市場/法人心理）。
    3. 框架化分析維度（情緒、預期、動機）。
    4. 產出格式引導（條列或分段）。
    5. **所有新聞以近3日為主，越新權重越高，越舊參考性越低。**

    **即使估值偏高，但若新聞情緒明顯偏正向，也應考慮短期樂觀心理可能帶來的上漲動能。**
    此外，**請務必預測明日的股價方向與預估幅度**（上漲/下跌/平盤 + 百分比），並以 1–2 句話給出投資建議。

    請用繁體中文回答。
    """


    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": prompt_content}],
    )

    # === 統一回傳的 chart 結構（個股）===
    ticker = yf.Ticker("2317.TW")
    stock_hist = ticker.history(period="10d")["Close"].reset_index()
    stock_chart = {
        "dates": stock_hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": [round(x,2) for x in stock_hist["Close"].tolist()]
    }

    return jsonify({
        "gpt": response.choices[0].message.content,
        "sentiment_summary": avg_sent,
        "stock_chart": stock_chart,
        "index_twii": idx_twii,
        "index_spx": idx_spx,
        "index_ixic": idx_ixic,
        "news": {
            "foxconn": [
                {"title": n["title"], "link": n["link"], "sentiment": n["sentiment"], "score": n["score"], "published": n["published"].isoformat() if n.get("published") else None}
                for n in foxconn_news[:10]
            ],
            "market": [
                {"title": n["title"], "link": n["link"], "sentiment": n["sentiment"], "score": n["score"], "query": n["query"], "published": n["published"].isoformat() if n.get("published") else None}
                for n in market_news[:15]
            ]
        }
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
