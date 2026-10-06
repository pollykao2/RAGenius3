# services/data_fetch.py
import os
import requests
import string
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urlencode
from newsapi import NewsApiClient
from ta.momentum import RSIIndicator
from ta.trend import MACD
from typing import Optional, Tuple
from config import RECENT_DAYS
from services.models import sentiment_pipe

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
    except:
        return "", None

def search_google_news(query: str, num_results: int = 10):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id  = os.getenv("GOOGLE_CSE_ID")
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        "key": api_key, "cx": cse_id, "q": query,
        "num": min(10, num_results), "start": 1,
        "hl": "zh-TW", "gl": "tw", "dateRestrict": "d3"
    }
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        results = response.json()
        return [{"title": item.get("title"), "link": item.get("link")} for item in results.get("items", [])]
    except:
        return []

def gather_news(query_list, num_results=10):
    all_news = []
    for q in query_list:
        g_items = search_google_news(q, num_results=num_results)
        for it in g_items:
            content, pub_dt = extract_news_content(it["link"])
            if content:
                s = sentiment_pipe(it["title"])[0]
                all_news.append({
                    "source": "GoogleCSE", "query": q, "title": it["title"], "link": it["link"],
                    "published": pub_dt, "sentiment": s["label"], "score": round(s["score"], 3), "content": content
                })
                
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key and len(all_news) < num_results:
        newsapi = NewsApiClient(api_key=newsapi_key)
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=RECENT_DAYS)
        for q in query_list:
            try:
                arts = newsapi.get_everything(q=q, from_param=from_date.isoformat(), to=to_date.isoformat(), language="zh", page_size=5).get("articles", [])
                for a in arts:
                    content, pub_dt = extract_news_content(a.get("url", ""))
                    if content:
                        s = sentiment_pipe(a.get("title", ""))[0]
                        all_news.append({
                            "source": "NewsAPI", "query": q, "title": a.get("title", ""), "link": a.get("url", ""),
                            "published": pub_dt, "sentiment": s["label"], "score": round(s["score"], 3), "content": content
                        })
            except:
                continue

    all_news = [n for n in all_news if n.get("published") is None or n["published"] >= datetime.now()-timedelta(days=RECENT_DAYS)]
    all_news.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
    return all_news[:num_results]

def get_index_metrics(symbol: str):
    t = yf.Ticker(symbol)
    hist = t.history(period="6mo")["Close"].dropna()
    if len(hist) < 30: return {}
    rsi = float(RSIIndicator(hist, window=14).rsi().iloc[-1])
    macd_ind = MACD(hist)
    macd = float(macd_ind.macd().iloc[-1])
    macd_sig = float(macd_ind.macd_signal().iloc[-1])
    return {
        "symbol": symbol, "last": float(hist.iloc[-1]), "rsi": rsi,
        "rsi_note": "超買" if rsi>70 else "超賣" if rsi<30 else "中性",
        "macd": macd, "macd_signal": macd_sig, "macd_note": "買入" if macd>macd_sig else "賣出",
        "trend": "偏多" if macd > macd_sig else "偏空"
    }

# services/data_fetch.py

def get_stock_metrics(stock_symbol: str) -> dict:
    ticker = yf.Ticker(stock_symbol)
    
    # 使用 try-catch 安全擷取 info，避免非個股標的噴錯
    try:
        info = ticker.info
        if not info:
            info = {}
    except:
        info = {}
        
    eps = info.get("trailingEps", np.nan)
    pe_ratio = info.get("trailingPE", np.nan)

    # 💡 【核心安全修正 1】：安全擷取現金流量表，若為 ETF 則優雅跳過不崩潰
    fcf = np.nan
    growth_rate = 5.0
    try:
        if hasattr(ticker, 'cashflow') and not ticker.cashflow.empty and "Free Cash Flow" in ticker.cashflow.index:
            cf_series = ticker.cashflow.loc["Free Cash Flow"]
            if len(cf_series) > 0:
                fcf = cf_series.iloc[0] / 1e8
                if len(cf_series) >= 2 and cf_series.iloc[1] != 0:
                    growth_rate = ((cf_series.iloc[0] / cf_series.iloc[1] - 1) * 100)
    except Exception as e:
        print(f"⚠️ {stock_symbol} 無法取得折現現金流數據 (可能為ETF): {e}")

    try:
        tnx = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 10
    except:
        tnx = 3.5
    
    discount_rate = tnx / 100 + 0.05
    
    # 計算內在價值
    try:
        if not np.isnan(fcf):
            intrinsic_value = fcf * (1 + growth_rate/100) / (discount_rate - growth_rate/100)
        else:
            intrinsic_value = np.nan
    except:
        intrinsic_value = np.nan

    # 即時價格
    try:
        hist_1d = ticker.history(period="1d")
        current_price = hist_1d["Close"].iloc[-1] if not hist_1d.empty else np.nan
    except:
        current_price = np.nan
        
    margin = (current_price - intrinsic_value) / intrinsic_value * 100 if intrinsic_value and intrinsic_value > 0 and not np.isnan(current_price) else np.nan

    # 💡 【核心安全修正 2】：安全計算 ROE 與護城河，防止財務報表為空時系統崩潰
    roe = np.nan
    try:
        if hasattr(ticker, 'financials') and "Net Income" in ticker.financials.index and hasattr(ticker, 'balance_sheet') and "Total Stockholder Equity" in ticker.balance_sheet.index:
            netinc = ticker.financials.loc["Net Income"]
            equity = ticker.balance_sheet.loc["Total Stockholder Equity"]
            roe_series = (netinc / equity * 100).dropna()
            if len(roe_series) > 0:
                roe = float(roe_series.head(5).mean())
    except Exception as e:
        print(f"⚠️ {stock_symbol} 無法取得淨利或股東權益數據 (可能為ETF): {e}")
        
    if np.isnan(roe):
        roe = info.get("returnOnEquity", 0.10) * 100
        
    moat = "強" if roe > 15 else "中" if roe > 5 else "弱"

    # 技術指標運算
    try:
        hist60 = ticker.history(period="60d")["Close"]
    except:
        hist60 = pd.Series(dtype=float)
        
    if len(hist60) > 14:
        rsi = float(RSIIndicator(hist60, window=14).rsi().iloc[-1])
        rsi_signal = "超買" if rsi > 70 else "超賣" if rsi < 30 else "中性"
    else:
        rsi, rsi_signal = np.nan, "暫無訊號"
        
    if len(hist60) > 30:
        macd_ind = MACD(hist60)
        macd_val = float(macd_ind.macd().iloc[-1])
        macd_signal = float(macd_ind.macd_signal().iloc[-1])
        macd_signal_text = "買入" if macd_val > macd_signal else "賣出"
    else:
        macd_val, macd_signal, macd_signal_text = np.nan, np.nan, "暫無訊號"

    return {
        "eps": eps if not np.isnan(eps) else "N/A", 
        "pe_ratio": pe_ratio if not np.isnan(pe_ratio) else "N/A", 
        "industry_pe": pe_ratio * 0.95 if pe_ratio and not np.isnan(pe_ratio) else "N/A",
        "intrinsic_value": intrinsic_value if not np.isnan(intrinsic_value) else 0.0, 
        "current_price": current_price if not np.isnan(current_price) else 0.0, 
        "margin": margin if not np.isnan(margin) else 0.0,
        "roe": roe if not np.isnan(roe) else 0.0, 
        "moat": moat,
        "rsi": rsi if not np.isnan(rsi) else 0.0, 
        "rsi_signal": rsi_signal,
        "macd": macd_val if not np.isnan(macd_val) else 0.0, 
        "macd_signal": macd_signal if not np.isnan(macd_signal) else 0.0, 
        "macd_signal_text": macd_signal_text,
        "interest_rate": tnx, 
        "rate_position": "偏高" if tnx > 3 else "持平", 
        "avg_sent": "中性"
    }