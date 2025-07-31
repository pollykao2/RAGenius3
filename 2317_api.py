from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

os.environ.pop("OPENAI_API_KEY", None)
load_dotenv(dotenv_path="C:/Users/user/Desktop/RAGenius3/.env")

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

# 抓取新聞內容並過濾近30天

def extract_news_content(url):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = ' '.join(paragraphs).strip()

        # 抓 metadata 發布時間
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
                if dt < datetime.now() - timedelta(days=30):
                    return ""
            except:
                pass
        return content
    except Exception as e:
        print(f"❌ 無法抓取新聞 {url}: {e}")
        return ""

# Google Programmable Search API

def search_google_news(query, num_results=10):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get('items', [])
    except Exception as e:
        print(f"❌ 搜尋失敗：{e}")
        return []

@app.route("/api/stock")
def stock_data():
    start = time.perf_counter()
    tokenizer = TreebankWordTokenizer()

    # Pinecone & 模型初始化
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("news-index")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    # 抓近30天新聞
    date_after = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    query_str = f"鴻海 after:{date_after}"
    results = search_google_news(query_str, num_results=15)

    filtered = []
    for it in results:
        title = it.get('title', '')
        link = it.get('link', '')
        content = extract_news_content(link)
        if content:
            filtered.append({'title': title, 'link': link, 'content': content})
    filtered = filtered[:10]

    # BM25 前處理
    corpus = []
    for e in filtered:
        txt = e['title'] + ' ' + e['content'][:1000]
        toks = [t for t in tokenizer.tokenize(txt.lower()) if t not in string.punctuation]
        corpus.append(toks)
    bm25 = BM25Okapi(corpus)

    # 上傳向量到 Pinecone
    for i, e in enumerate(filtered):
        full = e['title'] + ' ' + e['content'][:1000]
        vec = model.encode(full).tolist()
        s = sentiment_pipe(full)[0]
        index.upsert([{  
            'id': f'news-{i}',
            'values': vec,
            'metadata': {
                'text': full,
                'url': e['link'],
                'sentiment': s['label'],
                'sentiment_score': round(s['score'], 3)
            }
        }])

    # BM25 + Pinecone 查詢
    q = "鴻海AI發展對市場影響"
    qtokens = [t for t in tokenizer.tokenize(q.lower()) if t not in string.punctuation]
    bm_scores = bm25.get_scores(qtokens)
    top5 = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:5]

    qvec = model.encode(q).tolist()
    res = index.query(vector=qvec, top_k=3, include_metadata=True)
    matches = res.get('matches', [])

    pos = sum(1 for m in matches if m['metadata']['sentiment']=='POSITIVE')
    neg = sum(1 for m in matches if m['metadata']['sentiment']=='NEGATIVE')
    avg_sent = '正向' if pos>neg else '負向' if neg>pos else '中性'

        # === 保留原 prompt 並正確包三重引號、用單大括號插變數 ===
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
- 本日新聞情緒平均：{avg_sent}
- 重點新聞摘要如下：
"""  

    # 把每篇 match 的 text 接在 prompt 後面
    for m in matches:
        prompt_content += m["metadata"]["text"] + "\n"

    # 第二段也用三重引號包起來
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

你需要明確的指出
1. 明確新聞來源：在提示中標明新聞平台／標題／時間。
2. 聚焦重點元素：先「摘要新聞要點」，再基於此推測心理。
3. 框架化分析維度：從「情緒(Fear/Greed)」「預期(上漲/下跌)」「動機(逢低布局/獲利了結)」等拆解。
4. 產出格式引導：用條列或分段回覆。

此外，**請務必預測明日的股價方向與預估幅度**（例如：「預估明日小幅上漲約1%」或「預期下跌1~2%」），
最後以1–2句話給出簡短的投資建議（如「可逢低分批佈局」或「建議短期觀望」），
禁止只泛泛而談，必須具體輸出推論與預測數據。

請用繁體中文回答。
"""


    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": prompt_content}]
    )

    return jsonify({
        'news': [{'title': e['title'], 'link': e['link']} for e in filtered],
        'sentiment_summary': avg_sent,
        'gpt': response.choices[0].message.content
    })

if __name__ == '__main__':
    app.run(port=5000)