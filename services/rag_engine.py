# services/rag_engine.py
import os
import string
from nltk.tokenize import TreebankWordTokenizer
from rank_bm25 import BM25Okapi
from openai import OpenAI
from services.models import embed_model, pinecone_index, sentiment_pipe

tokenizer = TreebankWordTokenizer()

# 把這段加在 services/rag_engine.py 的最上方（函式庫匯入的下方）

def get_search_queries(ticker_id: str, stock_name: str) -> list:
    """根據前端傳入的股票代號，客製化優化 RAG 檢索的關鍵字組合"""
    if "2330" in ticker_id:
        return ["台積電", "TSMC 2330", "台積電 先進製程", "台積電 AI晶片"]
    elif "2317" in ticker_id:
        return ["鴻海", "富士康 2317", "鴻海 AI", "鴻海 蘋果訂單"]
    elif "2454" in ticker_id:
        return ["聯發科", "MediaTek 2454", "聯發科 天璣", "聯發科 手機晶片"]
    elif "2382" in ticker_id:
        return ["廣達", "廣達 2382", "廣達 AI伺服器", "林百里"]
    elif "3231" in ticker_id:
        return ["緯創", "緯創 3231", "緯創 AI伺服器", "緯創 財報"]
    elif "0050" in ticker_id:
        return ["元大台灣50", "0050 ETF", "0050 成分股", "台灣50"]
    elif "0056" in ticker_id:
        return ["元大高股息", "0056 ETF", "0056 配息", "高股息 ETF"]
    
    # 其餘股票（如聯電 2303）直接跑通用格式
    return [stock_name, f"{stock_name} 股市", f"{ticker_id.split('.')[0]} 趨勢"]

# services/rag_engine.py

def run_rag_and_gpt(stock_id: str, stock_name: str, stock_news: list, market_news: list, metrics: dict, idx_twii: dict, idx_spx: dict, idx_ixic: dict):
    # 建立 BM25 語料庫
    filtered = stock_news[:10]
    corpus = []
    for e in filtered:
        txt = e['title'] + ' ' + e['content'][:1000]
        toks = [t for t in tokenizer.tokenize(txt.lower()) if t not in string.punctuation]
        corpus.append(toks)

    matches = []
    avg_sent = "中性"
    
    # 執行 RAG 與 向量庫儲存
    if corpus and not all(len(d) == 0 for d in corpus):
        try:
            bm25 = BM25Okapi(corpus)
            for i, e in enumerate(filtered):
                text = e['title'] + ' ' + e['content'][:1000]
                vec = embed_model.encode(text).tolist()
                s = sentiment_pipe(text[:500])[0]
                metadata = {'text': text, 'url': e['link'], 'sentiment': s['label'], 'sentiment_score': round(s['score'], 3)}
                if e.get('published'):
                    metadata['published'] = e['published'].isoformat()

                # 儲存至獨立的 namespace
                pinecone_index.upsert(
                    vectors=[{'id': f'{stock_id}-news-{i}', 'values': vec, 'metadata': metadata}],
                    namespace=stock_id
                )

            query = f"{stock_name} 產能利用率與市場接單動態對台股影響"
            qvec = embed_model.encode(query).tolist()
            
            # 查詢該股票專屬的 namespace
            res = pinecone_index.query(vector=qvec, top_k=3, include_metadata=True, namespace=stock_id)
            matches = res.get('matches', [])
            pos = sum(1 for m in matches if m['metadata']['sentiment'] == 'POSITIVE')
            neg = sum(1 for m in matches if m['metadata']['sentiment'] == 'NEGATIVE')
            avg_sent = '正向' if pos > neg else '負向' if neg > pos else '中性'
        except Exception as e:
            print(f"⚠️ RAG 檢索層發生異常 (將使用純基本面推論): {e}")

    market_headlines = [n["title"] for n in market_news[:5]]
    rag_texts = [m['metadata']['text'][:400] for m in matches if 'metadata' in m and 'text' in m['metadata']]
    
    # 核心 GPT-4o 預測走廊 Prompt
    prompt_content = f"""
    你是專業股市分析師，從以下五面向：財報穩健度、產業週期、估值水位、技術指標、新聞情緒，
    **你的新聞必須是近10日的**，
    結合葛拉漢安全邊際理論、巴菲特護城河、彼得林區選股方法，以及總體經濟利率折現理論，
    針對 {stock_name} 的詳細投資數據與技術指標進行分析。
    **邏輯推論** 與 **具體預測**

    同時請從「大盤」與「個股」兩層做判讀，時間以近10日為主、近3日權重最高。

    [大盤]
    - 台股(^TWII) 收盤：約 {idx_twii.get('last')}, RSI {idx_twii.get('rsi', 0):.1f}（{idx_twii.get('rsi_note')}）, MACD:{idx_twii.get('macd', 0):.2f}/{idx_twii.get('macd_signal', 0):.2f}（{idx_twii.get('macd_note')}）, 趨勢：{idx_twii.get('trend')}
    - S&P500(^GSPC) RSI {idx_spx.get('rsi', 0):.1f}（{idx_spx.get('rsi_note')}）；NASDAQ(^IXIC) RSI {idx_ixic.get('rsi', 0):.1f}（{idx_ixic.get('rsi_note')}）
    - 大盤重點新聞（前5）：
    - """ + "\n  - ".join(market_headlines if market_headlines else ["無"]) + f"""

    [個股：{stock_name} {stock_id}]
    - EPS:{metrics.get('eps')}、PE:{metrics.get('pe_ratio')}、同業PE均值:{metrics.get('industry_pe')}
    - DCF估值:{metrics.get('intrinsic_value', 0):.2f}，現價:{metrics.get('current_price', 0):.2f}，折溢價：{metrics.get('margin', 0):.1f}%
    - ROE:{metrics.get('roe', 0):.1f}%（護城河：{metrics.get('moat')}）
    - RSI:{metrics.get('rsi', 0):.1f}（{metrics.get('rsi_signal')}），MACD:{metrics.get('macd', 0):.2f}/{metrics.get('macd_signal', 0):.2f}（{metrics.get('macd_signal_text')}）
    - 近30日個股新聞情緒：{avg_sent}

    [近似新聞片段（RAG）重點]
    """ + "\n".join(rag_texts if rag_texts else ["無符合的近三日即時新聞片段"]) + """

    請完整融合上述數據，預測明日的股價方向與預估幅度（上漲/下跌/平盤 + 百分比），並以 1–2 句話給出投資建議。用繁體中文回答。
    """

    # 💡 【核心安全修正】：確保 return 絕對會被執行，且縮排完全在函式最外層
    try:
        key = os.getenv("OPENAI_API_KEY")

        print("===== DEBUG =====")
        print(repr(key))
        print("=================")

        client = OpenAI(api_key=key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_content}],
        )

        return response.choices[0].message.content, avg_sent

    except Exception as e:
        print(f"❌ OpenAI API 呼叫失敗: {e}")
        return f"系統暫時無法連線至 OpenAI 進行分析。錯誤訊息: {str(e)}", avg_sent