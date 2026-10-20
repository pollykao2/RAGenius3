# RAGenius 投資導航

## Project Overview

RAGenius 結合財經新聞情緒分析、檢索增強生成（RAG）、基本面與技術指標，協助使用者整理台股資訊並分析短期股價趨勢。

## Project Structure

本系統採前後端分離，程式碼分別存放於兩個 Repository：

- **模型與後端：[RAGenius3](https://github.com/pollykao2/RAGenius3)**
  負責資料蒐集、新聞情緒分析、向量儲存與檢索，以及 GPT-4o 整合分析。

- **前端：[ragenius_react](https://github.com/pollykao2/ragenius_react)**
  使用 Next.js、React 與 TypeScript 建置前端專案。

## System Architecture

1. **資料蒐集**：取得個股新聞、市場新聞、財務數據與市場指標。
2. **新聞處理**：使用 Hugging Face 情緒分析模型判斷新聞情緒，並透過 SentenceTransformer 將新聞轉換為向量。
3. **向量儲存與檢索**：將新聞向量及來源資訊存入 Pinecone，依查詢內容取得相關新聞。
4. **整合分析**：將檢索新聞、情緒、基本面及技術指標交給 GPT-4o，產生繁體中文分析與次日漲跌方向、幅度估計。

## Evaluation

以鴻海（2317.TW）進行每日預測追蹤，記錄 RAGenius 的預測漲跌區間、ARIMA 預測值與實際漲跌幅，作為模型比較的依據。

評估涵蓋方向判斷、區間涵蓋與極端波動案例。

## Current Limitations

- 目前已建立 BM25 語料與物件，尚未將 BM25 分數整合至檢索排序。
- 評估數據仍在核對原始紀錄與統計口徑。
- 極端市場波動與外部 API 服務異常可能影響分析結果。

## Research Achievements

- 獲國科會大專學生研究計畫補助。
- 於第 23 屆 ITAOI 研討會發表研究成果。
