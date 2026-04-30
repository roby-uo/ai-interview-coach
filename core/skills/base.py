import logging
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.config import settings
from domain.schemas import InterviewState
from infrastructure.tools.interview_db import search_interview_db, INTERVIEW_TOOLS
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)


async def _execute_skill_with_tools(
    role_prompt: str,
    task_prompt: str,
    weakness_prefix: str = "",
    bind_tools: List = None
) -> str:
    llm_instance = llm_factory.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.3)

    llm = llm_instance.bind_tools(bind_tools) if bind_tools else llm_instance

    if weakness_prefix:
        final_role_prompt = f"{role_prompt}\n\n【全局核心目标 - 必须时刻牢记】：\n{weakness_prefix}"
    else:
        final_role_prompt = role_prompt

    messages = [
        SystemMessage(content=final_role_prompt),
        HumanMessage(content=f"【执行指令】\n{task_prompt}")
    ]

    ai_msg = await llm.ainvoke(messages)

    if ai_msg.tool_calls:
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            tool_result = search_interview_db.invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=settings.STREAM_TIMEOUT_SECONDS
            )
            return response.content
        except asyncio.TimeoutError:
            logger.error("🚨 [Skill执行] 超时，强制切断")
            return "\n\n[系统提示：思考时间过长，连接已重置，请重新发送你的回答]"
    else:
        return ai_msg.content
