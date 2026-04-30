"""
百炼 DashScope 向量化封装
解决 OpenAIEmbeddings 与百炼 embedding API 不兼容的问题
"""
from typing import List
from langchain_core.embeddings import Embeddings
import dashscope
from http import HTTPStatus
from app.config import settings


class DashScopeEmbeddings(Embeddings):
    """
    兼容百炼平台的向量化封装，替代 OpenAIEmbeddings
    """
    
    def __init__(self, model: str = None):
        self.model = model or settings.EMBEDDING_MODEL_NAME
        dashscope.api_key = settings.OPENAI_API_KEY
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档（分批处理，每批最多10条）"""
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch
            )
            if resp.status_code == HTTPStatus.OK:
                all_embeddings.extend([item['embedding'] for item in resp.output['embeddings']])
            else:
                raise RuntimeError(f"Embedding failed: {resp}")
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """向量化查询"""
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=text
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise RuntimeError(f"Embedding failed: {resp}")
