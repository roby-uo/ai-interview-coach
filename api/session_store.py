"""
会话映射：Coze user_id → LangGraph thread_id + 会话元数据

为什么需要这一层？
  Coze每次调用Plugin是独立HTTP请求，无状态。
  我们需要用user_id把多次请求串到同一个LangGraph thread上，
  同时维护前端状态（选了什么岗位、是否在等JD输入等）。
"""
import uuid
import threading
from typing import Dict, Any, Optional


class SessionStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_or_create(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            if user_id not in self._store:
                self._store[user_id] = {
                    "thread_id": f"session_{uuid.uuid4().hex}",
                    "selected_job_type": None,
                    "is_ready": False,
                    "session_ended": False,
                    "is_generating_report": False,
                    "waiting_for_jd": False,
                    "pending_resume": "",
                }
            return self._store[user_id]

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.get(user_id)

    def update(self, user_id: str, updates: Dict[str, Any]):
        with self._lock:
            if user_id not in self._store:
                self._store[user_id] = self._get_defaults()
            self._store[user_id].update(updates)

    def reset(self, user_id: str) -> Optional[str]:
        """重置会话，返回旧thread_id用于清理锁"""
        with self._lock:
            old_thread = None
            if user_id in self._store:
                old_thread = self._store[user_id].get("thread_id")
            self._store[user_id] = self._get_defaults()
            return old_thread

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        return {
            "thread_id": f"session_{uuid.uuid4().hex}",
            "selected_job_type": None,
            "is_ready": False,
            "session_ended": False,
            "is_generating_report": False,
            "waiting_for_jd": False,
            "pending_resume": "",
        }


# 全局单例
session_store = SessionStore()
