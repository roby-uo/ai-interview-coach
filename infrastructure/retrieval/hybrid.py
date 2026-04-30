import logging
import json
import hashlib
import pickle
import shutil
import tempfile
import os
from pathlib import Path
from typing import List, Sequence, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from pydantic import Field

from app.config import settings
from infrastructure.retrieval.embeddings import DashScopeEmbeddings

logging.getLogger("jieba").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_temp_faiss_dir: Optional[Path] = None


def _has_non_ascii(path: str) -> bool:
    try:
        path.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def _get_safe_faiss_path(faiss_dir: Path) -> Path:
    global _temp_faiss_dir
    
    if not _has_non_ascii(str(faiss_dir)):
        return faiss_dir
    
    if _temp_faiss_dir is not None and _temp_faiss_dir.exists():
        return _temp_faiss_dir
    
    _temp_faiss_dir = Path(tempfile.mkdtemp(prefix="faiss_index_"))
    logger.info(f"📁 路径包含非ASCII字符，复制索引到临时目录: {_temp_faiss_dir}")
    
    for file_name in ["index.faiss", "index.pkl"]:
        src = faiss_dir / file_name
        dst = _temp_faiss_dir / file_name
        if src.exists():
            shutil.copy(str(src), str(dst))
    
    return _temp_faiss_dir


class WeightedEnsembleRetriever(BaseRetriever):
    retrievers: Sequence[BaseRetriever] = Field(...)
    weights: List[float] = Field(default=[0.7, 0.3], description="权重列表，如 [FAISS权重, BM25权重]")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_retriever_docs = [retriever.invoke(query) for retriever in self.retrievers]
        
        k = 60
        rrf_scores = {}
        doc_map = {}
        
        for docs, weight in zip(all_retriever_docs, self.weights):
            for rank, doc in enumerate(docs):
                if doc.page_content not in doc_map:
                    doc_map[doc.page_content] = doc
                rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + (weight / (rank + 1 + k))
        
        sorted_docs = sorted(doc_map.values(), key=lambda x: rrf_scores.get(x.page_content, 0), reverse=True)
        return sorted_docs


_RETRIEVER_INSTANCE: Optional[WeightedEnsembleRetriever] = None
_INDEX_HASH_CACHE: dict = {}


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_hash(dir_path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file():
            h.update(file_path.name.encode())
            with open(file_path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def jieba_preprocess(text: str) -> List[str]:
    import jieba
    return list(jieba.lcut(text))


def load_jsonl_to_docs(jsonl_path: str) -> List[Document]:
    docs = []
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning(f"未找到数据文件: {jsonl_path}")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = data.get('question', '') or ''
                inspect_point = data.get('inspect_point', '') or ''
                content = f"面试问题：{question}\n考察考点：{inspect_point}"
                
                if not content.strip() or content.strip() == "面试问题：\n考察考点：":
                    logger.warning(f"跳过空内容行: {line[:50]}...")
                    continue
                
                doc = Document(
                    page_content=content,
                    metadata={"raw_data": data, "job_type": data.get("job_type", "新媒体运营")}
                )
                docs.append(doc)
            except json.JSONDecodeError:
                logger.error(f"解析 JSONL 行失败: {line}")
                continue
    return docs


def _load_bm25_from_json(json_path: Path) -> BM25Retriever:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = [Document(**doc_data) for doc_data in data["documents"]]
    
    bm25_retriever = BM25Retriever.from_documents(
        documents=docs,
        k=settings.BM25_TOP_K,
        preprocess_func=jieba_preprocess
    )
    return bm25_retriever


def _verify_index_integrity(index_dir: Path) -> bool:
    hash_file = index_dir / "index.hash"
    faiss_dir = index_dir / "faiss_index"
    bm25_json_path = index_dir / "bm25.json"
    bm25_pkl_path = index_dir / "bm25.pkl"
    
    if not hash_file.exists():
        logger.warning(f"索引校验文件不存在，将自动生成: {hash_file}")
        try:
            hashes = {}
            if faiss_dir.exists():
                hashes["faiss"] = compute_dir_hash(faiss_dir)
            if bm25_json_path.exists():
                hashes["bm25"] = compute_file_hash(bm25_json_path)
            elif bm25_pkl_path.exists():
                hashes["bm25"] = compute_file_hash(bm25_pkl_path)
            
            with open(hash_file, "w", encoding="utf-8") as f:
                json.dump(hashes, f, indent=2)
            logger.info("✅ 已生成索引校验文件")
            return True
        except Exception as e:
            logger.error(f"生成索引校验文件失败: {e}")
            return False
    
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            stored_hashes = json.load(f)
        
        if faiss_dir.exists():
            current_faiss_hash = compute_dir_hash(faiss_dir)
            if current_faiss_hash != stored_hashes.get("faiss", ""):
                logger.error("FAISS索引校验失败！文件可能被篡改。")
                return False
        
        if bm25_json_path.exists():
            current_bm25_hash = compute_file_hash(bm25_json_path)
            if current_bm25_hash != stored_hashes.get("bm25", ""):
                logger.error("BM25索引校验失败！文件可能被篡改。")
                return False
        elif bm25_pkl_path.exists():
            current_bm25_hash = compute_file_hash(bm25_pkl_path)
            if current_bm25_hash != stored_hashes.get("bm25", ""):
                logger.error("BM25索引校验失败！文件可能被篡改。")
                return False
        
        logger.info("✅ 索引完整性校验通过")
        return True
        
    except Exception as e:
        logger.error(f"索引校验异常: {e}")
        return False


def get_hybrid_retriever(jsonl_path: str = None, force_reload: bool = False) -> WeightedEnsembleRetriever:
    global _RETRIEVER_INSTANCE
    
    if _RETRIEVER_INSTANCE is not None and not force_reload:
        logger.info("⚡ [缓存命中] 直接使用内存中的检索器")
        return _RETRIEVER_INSTANCE

    app_root = settings.get_app_root()
    index_dir = app_root / "data" / "index"
    faiss_dir = index_dir / "faiss_index"
    bm25_json_path = index_dir / "bm25.json"
    bm25_pkl_path = index_dir / "bm25.pkl"

    if faiss_dir.exists() and (bm25_json_path.exists() or bm25_pkl_path.exists()):
        logger.info("📦 检测到本地存在预构建索引，开始加载...")
        
        if not _verify_index_integrity(index_dir):
            raise RuntimeError(
                "索引完整性校验失败！请重新构建索引（运行 python scripts/build_index.py），"
                "或检查索引文件是否被恶意篡改。"
            )
        
        embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        safe_faiss_dir = _get_safe_faiss_path(faiss_dir)
        faiss_vectorstore = FAISS.load_local(
            str(safe_faiss_dir), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": settings.FAISS_TOP_K})
        
        if bm25_json_path.exists():
            bm25_retriever = _load_bm25_from_json(bm25_json_path)
        else:
            logger.warning("⚠️ 使用旧版pickle格式BM25索引，建议重新构建索引以使用更安全的JSON格式")
            with open(bm25_pkl_path, "rb") as f:
                bm25_retriever = pickle.load(f)

        logger.info("✅ 本地全量索引加载完成！")
        
    else:
        logger.warning("⚠️ 未找到本地预构建索引，回退到实时构建模式...")
        
        if jsonl_path is None:
            jsonl_path = str(app_root / "data" / "processed" / "mined_questions.jsonl")
            
        docs = load_jsonl_to_docs(jsonl_path)
        if not docs:
            raise ValueError("没有加载到任何数据，无法构建检索器！")
        
        embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        faiss_vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": settings.FAISS_TOP_K})
        
        bm25_retriever = BM25Retriever.from_documents(
            documents=docs,
            k=settings.BM25_TOP_K,
            preprocess_func=jieba_preprocess
        )

    ensemble_retriever = WeightedEnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    
    _RETRIEVER_INSTANCE = ensemble_retriever
    return ensemble_retriever


def invalidate_retriever_cache():
    global _RETRIEVER_INSTANCE
    _RETRIEVER_INSTANCE = None
    logger.info("检索器缓存已清除")
