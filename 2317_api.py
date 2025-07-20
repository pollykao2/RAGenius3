from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import ta

app = Flask(__name__)
CORS(app)

@app.route("/api/stock")
def stock_data():
    stock_symbol = "2317.TW"
    ticker = yf.Ticker(stock_symbol)
    df = ticker.history(period="10d")

    # 收盤價與日期
    dates = df.index.strftime("%m/%d").tolist()
    prices = df["Close"].round(2).tolist()

    # 技術指標計算
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_signal"] = ta.trend.macd_signal(df["Close"])

    try:
        rsi = round(df["RSI"].iloc[-1], 2)
        macd = round(df["MACD"].iloc[-1], 2)
        macd_signal = round(df["MACD_signal"].iloc[-1], 2)
    except:
        rsi = macd = macd_signal = "NaN"

    return jsonify({
        "chart": {
            "dates": dates,
            "prices": prices
        },
        "gpt": f"RSI={rsi}, MACD={macd}，短期偏多，建議分批佈局。",
        "news": [
            { "title": "鴻海擴大AI伺服器投資", "sentiment": "POSITIVE", "score": 0.92 },
            { "title": "美中關係緊張影響出口", "sentiment": "NEGATIVE", "score": 0.81 }
        ]
    })

if __name__ == "__main__":
    app.run(port=5000)
