"""
启动 FastAPI 服务

用法:
    python -m api.run
"""
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,           # 单worker（MemorySaver是内存级，多worker会丢失状态）
        timeout_keep_alive=120,
        log_level="info",
    )