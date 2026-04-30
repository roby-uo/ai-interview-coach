import logging
import json
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, HumanMessage

from app.config import settings
from domain.schemas import InterviewState
from .utils import get_message_content, get_message_role
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)

_SUMMARY_CACHE: Dict[str, str] = {}


async def memory_node(state: InterviewState) -> Dict[str, Any]:
    if not state["history"]:
        return {"short_summary": "[]"}

    recent_history = state["history"][-10:] if len(state["history"]) >= 10 else state["history"]
    
    last_summary = state.get("short_summary", "[]")
    new_messages = _get_new_messages(recent_history, last_summary)
    
    if not new_messages:
        logger.info("🧠 [记忆节点] 无新消息，使用缓存摘要")
        return {"short_summary": last_summary}

    llm = llm_factory.get_extractor_llm()

    system_prompt = """你是一个对话结构化提取器。读取最近5轮对话，提取为JSON列表。

输出格式：
[
  {"coach": "教练的提问意图", "user": "用户的核心回应"},
  {"coach": "教练的动作", "user": "用户的状态"}
]

要求：
- 每条不超过20字
- 剥离废话和情绪，只保留核心信息
- 不要输出其他字符，只要纯JSON"""

    try:
        context_str = "\n".join([get_message_content(m) for m in recent_history[-6:]])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"对话记录：\n{context_str}")
        ]

        response = await llm.ainvoke(messages)
        raw_str = response.content.strip()

        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()

        parsed_list = json.loads(raw_str)

        if isinstance(parsed_list, list) and len(parsed_list) > 0 and isinstance(parsed_list[0], dict):
            summary_json = json.dumps(parsed_list, ensure_ascii=False)
            logger.info(f"🧠 [记忆节点] 提取结构化摘要成功")
            return {"short_summary": summary_json}
        else:
            raise ValueError("解析结果不是预期的字典列表格式")

    except Exception as e:
        logger.error(f"摘要提取失败，启用极简兜底: {e}")
        last_two_rounds = recent_history[-4:] if len(recent_history) >= 4 else recent_history
        fallback_str = " | ".join([
            f"{'教练' if get_message_role(m) == 'assistant' else '用户'}: {get_message_content(m)[:30]}"
            for m in last_two_rounds
        ])
        return {"short_summary": f"[提取异常，降级为最近两轮摘要]: {fallback_str}"}


def _get_new_messages(history: list, last_summary: str) -> list:
    if last_summary == "[]":
        return history
    
    if len(history) <= 2:
        return []
    
    return history[-2:] if len(history) >= 2 else history
