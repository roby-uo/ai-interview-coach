"""
AI 面试教练 - FastAPI 接口层（供 Coze Plugin 调用）

启动方式:
    python -m api.run
    或
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import time
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from .schemas import InterviewRequest, InterviewResponse, ActionType
from .services import interview_service
from .task_store import task_store
from app.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# API Key 认证
# ──────────────────────────────────────────────
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

_raw_keys = os.environ.get("COZE_API_KEYS", "")
VALID_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]


async def verify_api_key(request: Request):
    key = request.headers.get(API_KEY_NAME, "")
    if not VALID_API_KEYS:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return key


# ──────────────────────────────────────────────
# 限流中间件（按 user_id 计数，轻量级）
# ──────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._counts: dict = defaultdict(lambda: {"count": 0, "window_start": time.time()})
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            entry = self._counts[key]
            if now - entry["window_start"] > self._window:
                entry["count"] = 1
                entry["window_start"] = now
                return True
            entry["count"] += 1
            return entry["count"] <= self._max


_rate_limiter = RateLimiter(max_requests=15, window_seconds=60)


# ──────────────────────────────────────────────
# Lifespan（启动预热 + 关闭清理）
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    interview_service.ensure_initialized()
    logger.info("🚀 [Lifespan] 服务启动，引擎预热完成")
    yield
    logger.info("👋 [Lifespan] 服务关闭")


# ──────────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI 面试教练 API",
    description="供 Coze Plugin 调用的面试教练后端接口",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"🌐 {request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)")
    return response


# ──────────────────────────────────────────────
# 路由
# ──────────────────────────────────────────────
@app.post("/api/interview", response_model=InterviewResponse)
async def interview_endpoint(req: InterviewRequest, api_key: str = Security(verify_api_key)):
    if not _rate_limiter.is_allowed(req.user_id):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    try:
        result = await interview_service.handle_request(
            user_id=req.user_id,
            message=req.message,
            action=req.action or ActionType.CHAT,
            job_type=req.job_type,
            file_url=req.file_url,
        )
        return result
    except Exception as e:
        logger.error(f"💀 [API] 未捕获异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@app.get("/api/interview/{task_id}", response_model=InterviewResponse)
async def poll_task(task_id: str, api_key: str = Security(verify_api_key)):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status == "processing":
        return InterviewResponse(
            reply="⏳ 仍在处理中，请稍后再次查询...",
            status="processing",
            task_id=task_id,
        )
    return task.result


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "interview-coach"}


@app.get("/api/jobs")
async def list_jobs(api_key: str = Security(verify_api_key)):
    from domain.job_configs import list_available_jobs, load_job_config
    jobs = []
    for jt in list_available_jobs():
        try:
            config = load_job_config(jt)
            jobs.append({"job_type": jt, "display_name": config.display_name})
        except Exception:
            jobs.append({"job_type": jt, "display_name": jt})
    return {"jobs": jobs}
