# app.py
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import yfinance as yf
from datetime import datetime
import numpy as np  # 確保引入 numpy

# 加載環境變數
load_dotenv(override=True)

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# 延遲導入自訂模組，確保環境變數已正確載入
from config import STOCK_MAP
from services.data_fetch import gather_news, get_index_metrics, get_stock_metrics
from services.rag_engine import run_rag_and_gpt


def clean_nan(obj):
    """💡 核心深度清洗函式：遞迴遍歷所有資料結構，將非法的 NaN 完全抹除，避免前端 JSON 崩潰"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0  # 或者改為 None / "N/A"
        return obj
    return obj


@app.route("/api/market-summary", methods=["GET"])
def market_summary():
    """首頁大盤資料介面：三大指數、加權指數走勢圖、大盤財經新聞"""
    try:
        idx_twii = get_index_metrics("^TWII")
        idx_spx  = get_index_metrics("^GSPC")
        idx_ixic = get_index_metrics("^IXIC")
        
        market_news = gather_news(["台股 大盤", "加權指數", "美股 NASDAQ", "S&P 500"], num_results=15)
        
        twii_ticker = yf.Ticker("^TWII")
        twii_hist = twii_ticker.history(period="12d").dropna(subset=["Close"]) # 先拿多一點日子防捏造
        twii_hist = twii_hist[twii_hist["Close"] > 0].reset_index().tail(10)  # 只取最新 10 筆健康的收盤價

        twii_chart = {
            "dates": twii_hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices": [round(x, 2) for x in twii_hist["Close"].tolist()]
        }
        
        response_data = {
            "index_twii": idx_twii,
            "index_spx": idx_spx,
            "index_ixic": idx_ixic,
            "twii_chart": twii_chart,
            "news": {
                "market": [
                    {
                        "title": n["title"], "link": n["link"], "sentiment": n["sentiment"],
                        "score": n["score"], "query": n["query"],
                        "published": n["published"].isoformat() if n.get("published") else None
                    } for n in market_news
                ]
            }
        }
        # 進行大盤資料清洗
        return jsonify(clean_nan(response_data))
    except Exception as e:
        print(f"❌ 大盤數據加載異常: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<ticker_id>", methods=["GET"])
def stock_detail(ticker_id):
    """動態個股預測介面：配合前端路徑，自動依據傳入的標的代號跑 RAGenius 預測流程"""
    try:
        if ticker_id and not ticker_id.endswith(".TW") and not ticker_id.startswith("^"):
            final_ticker_id = f"{ticker_id}.TW"
        else:
            final_ticker_id = ticker_id

        if final_ticker_id not in STOCK_MAP:
            return jsonify({"error": f"不支援的股票代號: {final_ticker_id}"}), 400
        
        stock_name = STOCK_MAP[final_ticker_id]
        print(f"🔍 收到前端路由參數: {ticker_id} -> 實際對接代號: {final_ticker_id} ({stock_name})")
        
        from services.rag_engine import get_search_queries
        queries = get_search_queries(final_ticker_id, stock_name)
        stock_news = gather_news(queries)
        market_news = gather_news(["台股 大盤", "加權指數"], num_results=5)
        
        metrics = get_stock_metrics(final_ticker_id)
        idx_twii = get_index_metrics("^TWII")
        idx_spx  = get_index_metrics("^GSPC")
        idx_ixic = get_index_metrics("^IXIC")
        
        gpt_analysis, avg_sent = run_rag_and_gpt(
            final_ticker_id, stock_name, stock_news, market_news, 
            metrics, idx_twii, idx_spx, idx_ixic
        )
        
        ticker = yf.Ticker(final_ticker_id)
        stock_hist = ticker.history(period="12d").dropna(subset=["Close"])
        stock_hist = stock_hist[stock_hist["Close"] > 0].reset_index().tail(10) # 確保拿到的 10 天都有真實股價
        stock_chart = {
            "dates": stock_hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices": [round(x, 2) for x in stock_hist["Close"].tolist()]
        }
        
        response_data = {
            "gpt": gpt_analysis,
            "sentiment_summary": avg_sent,
            "stock_chart": stock_chart,
            "news": [
                {
                    "title": n["title"], "link": n["link"], "sentiment": n["sentiment"], 
                    "score": n["score"], "published": n["published"].isoformat() if n.get("published") else None
                } for n in stock_news[:10]
            ]
        }
        # 💡 【核心修正】：個股資料回傳前進行深度的 clean_nan 清洗，徹底消除 Unexpected token 'N' 報錯！
        return jsonify(clean_nan(response_data))
    except Exception as e:
        print(f"❌ 個股數據加載異常 ({ticker_id}): {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)