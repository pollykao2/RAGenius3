# services/models.py
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import MODEL_NAME, EMBED_MODEL_NAME

print("⏳ 正在初始化全域 AI 模型（僅加載一次）...")

# 1. 情緒分析模型
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

# 2. 向量嵌入模型
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device='cuda')

# 3. Pinecone 向量資料庫
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("news-index")

print("✅ 所有 AI 模型與資料庫連線初始化成功！")