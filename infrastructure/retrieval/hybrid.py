import logging
import json
import hashlib
import pickle
import shutil
import tempfile
import os
from pathlib import Path
from typing import List, Sequence, Optional, Dict

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


_temp_faiss_dirs: Dict[str, Path] = {}


def _has_non_ascii(path: str) -> bool:
    try:
        path.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def _get_safe_faiss_path(faiss_dir: Path, job_type: str) -> Path:
    if not _has_non_ascii(str(faiss_dir)):
        return faiss_dir

    if job_type in _temp_faiss_dirs and _temp_faiss_dirs[job_type].exists():
        return _temp_faiss_dirs[job_type]

    safe_prefix = hashlib.md5(job_type.encode()).hexdigest()[:8]
    temp_dir = Path(tempfile.mkdtemp(prefix=f"faiss_{safe_prefix}_"))
    _temp_faiss_dirs[job_type] = temp_dir
    logger.info(f"📁 路径包含非ASCII字符，复制索引到临时目录: {temp_dir}")

    for file_name in ["index.faiss", "index.pkl"]:
        src = faiss_dir / file_name
        dst = temp_dir / file_name
        if src.exists():
            shutil.copy(str(src), str(dst))

    return temp_dir


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


_RETRIEVER_CACHE: Dict[str, WeightedEnsembleRetriever] = {}
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


def load_jsonl_to_docs(jsonl_path: str, job_type: str = None) -> List[Document]:
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

                doc_job_type = data.get("job_type") or job_type or "未知岗位"
                doc = Document(
                    page_content=content,
                    metadata={"raw_data": data, "job_type": doc_job_type}
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


def get_hybrid_retriever(job_type: str, jsonl_path: str = None, force_reload: bool = False) -> WeightedEnsembleRetriever:
    if job_type in _RETRIEVER_CACHE and not force_reload:
        logger.info(f"⚡ [缓存命中] 直接使用内存中的 {job_type} 检索器")
        return _RETRIEVER_CACHE[job_type]

    index_dir = settings.get_job_index_dir(job_type)
    faiss_dir = index_dir / "faiss_index"
    bm25_json_path = index_dir / "bm25.json"
    bm25_pkl_path = index_dir / "bm25.pkl"

    if faiss_dir.exists() and (bm25_json_path.exists() or bm25_pkl_path.exists()):
        logger.info(f"📦 检测到 {job_type} 本地存在预构建索引，开始加载...")

        if not _verify_index_integrity(index_dir):
            raise RuntimeError(
                f"{job_type} 索引完整性校验失败！请重新构建索引"
                f"（运行 python scripts/build_index.py --job-type \"{job_type}\"），"
                "或检查索引文件是否被恶意篡改。"
            )

        embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        safe_faiss_dir = _get_safe_faiss_path(faiss_dir, job_type)
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

        logger.info(f"✅ {job_type} 本地全量索引加载完成！")

    else:
        logger.warning(f"⚠️ 未找到 {job_type} 预构建索引，回退到实时构建模式...")

        if jsonl_path is None:
            jsonl_path = str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")

        docs = load_jsonl_to_docs(jsonl_path, job_type=job_type)
        if not docs:
            raise ValueError(f"{job_type} 没有加载到任何数据，无法构建检索器！请先运行离线流程。")

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

    _RETRIEVER_CACHE[job_type] = ensemble_retriever
    return ensemble_retriever


def invalidate_retriever_cache(job_type: str = None):
    global _RETRIEVER_CACHE
    if job_type:
        _RETRIEVER_CACHE.pop(job_type, None)
        logger.info(f"{job_type} 检索器缓存已清除")
    else:
        _RETRIEVER_CACHE.clear()
        logger.info("所有检索器缓存已清除")
