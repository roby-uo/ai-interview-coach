"""
索引构建脚本

启动方式:
    python scripts/build_index.py --job-type "新媒体运营"

说明:
    该脚本需要直接运行，不依赖 pip install -e .
    因此保留了 sys.path.append 来确保模块导入正常
"""

import sys
import json
import hashlib
import shutil
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from jieba import lcut

from app.config import settings
from domain.job_configs import load_job_config, get_job_config_or_default
from infrastructure.retrieval.hybrid import (
    load_jsonl_to_docs,
    jieba_preprocess,
    compute_file_hash,
    compute_dir_hash
)
from infrastructure.retrieval.embeddings import DashScopeEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _has_non_ascii(path: str) -> bool:
    try:
        path.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def build_and_save_index(job_type: str):
    jsonl_path = str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")
    index_dir = settings.get_job_index_dir(job_type)
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"1. 开始加载 {job_type} JSONL 数据: {jsonl_path}")
    docs = load_jsonl_to_docs(jsonl_path, job_type=job_type)
    logger.info(f"全量数据加载完毕，共 {len(docs)} 条。准备开始耗时操作...")

    logger.info("2. 开始调用 API 构建 FAISS 向量索引（全量，可能需要几分钟，请耐心等待）...")
    embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    faiss_vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    faiss_save_path = index_dir / "faiss_index"

    if _has_non_ascii(str(faiss_save_path)):
        logger.info("检测到路径包含非ASCII字符，使用临时目录保存FAISS索引...")
        temp_dir = Path(tempfile.mkdtemp(prefix="faiss_build_"))
        faiss_vectorstore.save_local(str(temp_dir))

        if faiss_save_path.exists():
            shutil.rmtree(str(faiss_save_path))
        shutil.copytree(str(temp_dir), str(faiss_save_path))
        shutil.rmtree(str(temp_dir))
        logger.info(f"已从临时目录复制到目标目录: {faiss_save_path}")
    else:
        faiss_vectorstore.save_local(str(faiss_save_path))

    logger.info(f"FAISS 索引构建完毕并保存至: {faiss_save_path}")

    logger.info("3. 开始构建 BM25 索引...")
    bm25_retriever = BM25Retriever.from_documents(
        documents=docs,
        k=settings.BM25_TOP_K,
        preprocess_func=jieba_preprocess
    )

    bm25_json_path = index_dir / "bm25.json"
    bm25_docs_data = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(bm25_json_path, "w", encoding="utf-8") as f:
        json.dump({"documents": bm25_docs_data}, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ BM25 索引(JSON格式)构建完毕并保存至: {bm25_json_path}")

    bm25_pkl_path = index_dir / "bm25.pkl"
    with open(bm25_pkl_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    logger.info(f"✅ BM25 索引(pickle格式)备份保存至: {bm25_pkl_path}")

    logger.info("4. 生成索引完整性校验文件...")
    faiss_hash = compute_dir_hash(faiss_save_path)
    bm25_hash = compute_file_hash(bm25_json_path)

    hash_file = index_dir / "index.hash"
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump({
            "faiss": faiss_hash,
            "bm25": bm25_hash,
            "version": "2.0",
            "job_type": job_type
        }, f, indent=2)
    logger.info(f"✅ 索引校验文件保存至: {hash_file}")

    print("\n" + "="*50)
    print(f">>> {job_type} 索引构建完成！你可以启动 chainlit 应用了，加载将是秒级的。")
    print("="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建岗位索引")
    parser.add_argument("--job-type", "-j", default=None, help="岗位类型（如：新媒体运营、产品经理）")

    args = parser.parse_args()

    if args.job_type:
        job_config = load_job_config(args.job_type)
        job_type = job_config.job_type
    else:
        job_config = get_job_config_or_default()
        job_type = job_config.job_type
        logger.info(f"未指定岗位，使用默认: {job_type}")

    build_and_save_index(job_type)
