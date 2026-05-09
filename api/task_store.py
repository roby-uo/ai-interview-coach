"""
异步任务存储 —— 解决 Coze Plugin 超时问题

Coze HTTP Plugin 超时约30秒，但画像提取/报告生成等操作可能超过30秒。
方案：POST 立即返回 {status: "processing", task_id}，Coze Bot 用 GET 轮询结果。
"""
import uuid
import time
import threading
from typing import Dict, Optional, Any

from .schemas import InterviewResponse


class AsyncTask:
    __slots__ = ("task_id", "status", "result", "created_at")

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "processing"
        self.result: Optional[InterviewResponse] = None
        self.created_at = time.time()


class TaskStore:
    _MAX_TASKS = 2000
    _TTL_SECONDS = 600

    def __init__(self):
        self._tasks: Dict[str, AsyncTask] = {}
        self._lock = threading.Lock()

    def create(self) -> AsyncTask:
        with self._lock:
            self._evict_if_needed()
            task = AsyncTask(task_id=uuid.uuid4().hex[:16])
            self._tasks[task.task_id] = task
            return task

    def get(self, task_id: str) -> Optional[AsyncTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def complete(self, task_id: str, result: InterviewResponse):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "ok" if result.status != "error" else "error"
                task.result = result

    def fail(self, task_id: str, error_msg: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "error"
                task.result = InterviewResponse(
                    reply=f"⚠️ 处理失败：{error_msg[:100]}",
                    status="error",
                )

    def _evict_if_needed(self):
        now = time.time()
        expired = [
            tid for tid, t in self._tasks.items()
            if now - t.created_at > self._TTL_SECONDS
        ]
        for tid in expired:
            del self._tasks[tid]
        if len(self._tasks) > self._MAX_TASKS:
            sorted_keys = sorted(
                self._tasks.keys(),
                key=lambda k: self._tasks[k].created_at,
            )
            for k in sorted_keys[: len(sorted_keys) // 2]:
                del self._tasks[k]


task_store = TaskStore()
