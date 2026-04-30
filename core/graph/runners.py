import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from langgraph.checkpoint.memory import MemorySaver

from domain.schemas import InterviewState
from core.nodes import create_initial_state
from .builder import build_interview_graph

logger = logging.getLogger(__name__)


def get_checkpointer():
    return MemorySaver()


def create_compiled_graph(checkpointer=None):
    graph = build_interview_graph()
    
    if checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile()
    
    return compiled


class InterviewGraphRunner:
    _MAX_SESSION_LOCKS = 1000
    
    def __init__(self):
        self._graph = None
        self._graph_lock = asyncio.Lock()
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._sessions_lock = asyncio.Lock()
    
    async def _get_session_lock(self, thread_id: str) -> asyncio.Lock:
        async with self._sessions_lock:
            if thread_id not in self._session_locks:
                if len(self._session_locks) >= self._MAX_SESSION_LOCKS:
                    oldest_keys = list(self._session_locks.keys())[:self._MAX_SESSION_LOCKS // 2]
                    for key in oldest_keys:
                        del self._session_locks[key]
                    logger.info(f"🧹 [锁清理] 清理了 {len(oldest_keys)} 个旧会话锁")
                
                self._session_locks[thread_id] = asyncio.Lock()
            return self._session_locks[thread_id]
    
    async def _ensure_graph(self):
        if self._graph is None:
            async with self._graph_lock:
                if self._graph is None:
                    checkpointer = get_checkpointer()
                    self._graph = create_compiled_graph(checkpointer)
                    logger.info("✅ [LangGraph] 图编译完成，MemorySaver 已就绪")
    
    def initialize(self):
        if self._graph is None:
            checkpointer = get_checkpointer()
            self._graph = create_compiled_graph(checkpointer)
            logger.info("✅ [LangGraph] 图编译完成，MemorySaver 已就绪")

    async def run(self, state: InterviewState, thread_id: str) -> InterviewState:
        await self._ensure_graph()
        
        session_lock = await self._get_session_lock(thread_id)
        
        async with session_lock:
            config = {"configurable": {"thread_id": thread_id}}
            result = await self._graph.ainvoke(state, config)
            return result

    async def get_state(self, thread_id: str) -> InterviewState:
        await self._ensure_graph()
        
        session_lock = await self._get_session_lock(thread_id)
        
        async with session_lock:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self._graph.get_state(config)
            
            if snapshot.values:
                logger.info(f"📖 [获取状态] thread={thread_id[:8]}... is_ready={snapshot.values.get('is_ready')}")
                return snapshot.values
            else:
                logger.info(f"📖 [获取状态] thread={thread_id[:8]}... 无状态，返回初始状态")
                return create_initial_state()

    async def update_state(self, thread_id: str, updates: dict):
        await self._ensure_graph()
        
        session_lock = await self._get_session_lock(thread_id)
        
        async with session_lock:
            config = {"configurable": {"thread_id": thread_id}}
            
            current_snapshot = self._graph.get_state(config)
            if current_snapshot.values:
                current_state = dict(current_snapshot.values)
            else:
                current_state = create_initial_state()
            
            for key, value in updates.items():
                if key == "history" and "history" in current_state:
                    existing_history = list(current_state["history"])
                    if isinstance(value, list):
                        existing_history.extend(value)
                    else:
                        existing_history.append(value)
                    current_state["history"] = existing_history
                else:
                    current_state[key] = value
            
            self._graph.update_state(config, current_state, as_node="route")
            
            verify_snapshot = self._graph.get_state(config)
            logger.info(f"📝 [状态更新] thread={thread_id[:8]}... is_ready={verify_snapshot.values.get('is_ready')} updates={list(updates.keys())}")
    
    async def clear_state(self, thread_id: str):
        await self._ensure_graph()
        
        session_lock = await self._get_session_lock(thread_id)
        
        async with session_lock:
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = create_initial_state()
            self._graph.update_state(config, initial_state)
            logger.info(f"🧹 [状态清理] 会话 {thread_id} 已重置")
    
    def clear_session_lock(self, thread_id: str):
        if thread_id in self._session_locks:
            del self._session_locks[thread_id]
            logger.info(f"🔓 [锁清理] 会话 {thread_id[:8]}... 的锁已清除")
    
    def get_state_sync(self, thread_id: str) -> InterviewState:
        if self._graph is None:
            self.initialize()
        
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self._graph.get_state(config)
        
        if snapshot.values:
            return snapshot.values
        else:
            return create_initial_state()
    
    def update_state_sync(self, thread_id: str, updates: dict):
        if self._graph is None:
            self.initialize()
        
        config = {"configurable": {"thread_id": thread_id}}
        current_snapshot = self._graph.get_state(config)
        
        if current_snapshot.values:
            current_state = dict(current_snapshot.values)
        else:
            current_state = create_initial_state()
        
        for key, value in updates.items():
            if key == "history" and "history" in current_state:
                existing_history = list(current_state["history"])
                if isinstance(value, list):
                    existing_history.extend(value)
                else:
                    existing_history.append(value)
                current_state["history"] = existing_history
            else:
                current_state[key] = value
        
        self._graph.update_state(config, current_state)
    
    def clear_state_sync(self, thread_id: str):
        if self._graph is None:
            self.initialize()
        
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = create_initial_state()
        self._graph.update_state(config, initial_state)
        logger.info(f"🧹 [状态清理] 会话 {thread_id} 已重置")


graph_runner = InterviewGraphRunner()
