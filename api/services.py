"""
核心服务层 —— 把Chainlit的UI交互逻辑，翻译成纯API调用

原则：
  - 100%复用core/引擎代码，不修改一行
  - 用session_store替代cl.user_session
  - 用InterviewResponse替代cl.Message
  - 耗时操作走异步任务，避免Coze Plugin超时
"""
import io
import re
import asyncio
import logging
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import settings
from domain.job_configs import load_job_config, list_available_jobs, get_job_config_or_default
from core.nodes import create_initial_state
from core.graph import graph_runner
from core.profiler.extractor import get_profiler
from core.profiler.gap_analyzer import get_gap_analyzer
from infrastructure.tools.interview_db import (
    set_current_job_type,
    search_interview_db,
)
from core.utils.llm_factory import llm_factory
from infrastructure.parsers.file_parser import extract_text_from_file

from .session_store import session_store
from .schemas import InterviewResponse, ActionButton, ActionType
from .task_store import task_store

logger = logging.getLogger(__name__)


def _sanitize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


class _FileWrapper:
    def __init__(self, content: bytes, name: str):
        self._bio = io.BytesIO(content)
        self.name = name

    def read(self):
        return self._bio.read()


async def _download_file(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        logger.error(f"文件下载失败 {url}: {e}")
        return None


_ACTION_HINT_MAP = {
    "quick_start": "快速体验",
    "generate_report": "生成报告",
    "reset": "重新开始",
    "select_job": "选择岗位",
    "use_default_jd": "使用默认标准",
    "chat": "继续对话",
}


def _append_action_hints(reply: str, actions: list) -> str:
    if not actions:
        return reply
    hints = []
    for btn in actions:
        hint = _ACTION_HINT_MAP.get(btn.action, btn.label)
        if btn.value:
            hint = f"{hint}({btn.value})"
        hints.append(f"- 回复「{hint}」{btn.label}")
    return reply + "\n\n**快捷操作：**\n" + "\n".join(hints)


class InterviewService:
    def __init__(self):
        self._initialized = False

    def ensure_initialized(self):
        if not self._initialized:
            graph_runner.initialize()
            self._initialized = True
            logger.info("✅ [API] InterviewService 初始化完成")

    async def handle_request(
        self,
        user_id: str,
        message: str,
        action: ActionType,
        job_type: Optional[str] = None,
        file_url: Optional[str] = None,
    ) -> InterviewResponse:
        self.ensure_initialized()

        session = session_store.get_or_create(user_id)
        thread_id = session["thread_id"]
        try:
            await graph_runner.get_state(thread_id)
        except Exception:
            await graph_runner.update_state(thread_id, create_initial_state())

        if action == ActionType.RESET:
            return await self._handle_reset(user_id)
        if action == ActionType.SELECT_JOB:
            return await self._handle_select_job(user_id, job_type)
        if action == ActionType.QUICK_START:
            return await self._handle_quick_start(user_id)
        if action == ActionType.GENERATE_REPORT:
            return await self._handle_report(user_id)
        if action == ActionType.USE_DEFAULT_JD:
            return await self._handle_use_default_jd(user_id)
        return await self._handle_chat(user_id, message, file_url)

    async def _handle_reset(self, user_id: str) -> InterviewResponse:
        old_thread = session_store.reset(user_id)
        if old_thread:
            graph_runner.clear_session_lock(old_thread)
        logger.info(f"🔄 [API] 会话重置 user={user_id[:8]}")
        return await self._build_job_selection_response()

    async def _handle_select_job(self, user_id: str, job_type: Optional[str]) -> InterviewResponse:
        if not job_type:
            return InterviewResponse(reply="❌ 未指定岗位类型", status="error")

        try:
            config = load_job_config(job_type)
        except FileNotFoundError:
            return InterviewResponse(reply=f"❌ 未找到岗位配置: {job_type}", status="error")

        session_store.update(user_id, {"selected_job_type": job_type})
        set_current_job_type(job_type)

        actions = [
            ActionButton(label="⚡ 快速体验模式", action="quick_start"),
            ActionButton(label="✅ 结束训练并生成报告", action="generate_report"),
        ]
        reply = (
            f"✅ 已选择岗位：**{config.display_name}**\n\n"
            f"请发送你的 **简历文本** 和 **目标岗位JD**，\n"
            f"或回复「快速体验」使用示例数据开始训练。"
        )
        return InterviewResponse(reply=_append_action_hints(reply, actions), actions=actions)

    async def _handle_quick_start(self, user_id: str) -> InterviewResponse:
        session = session_store.get_or_create(user_id)
        job_type = session.get("selected_job_type")

        if not job_type:
            available = list_available_jobs()
            if available:
                job_type = available[0]
                session_store.update(user_id, {"selected_job_type": job_type})
                set_current_job_type(job_type)

        try:
            config = load_job_config(job_type)
            demo_resume, demo_jd = config.demo_resume, config.demo_jd
        except FileNotFoundError:
            config = get_job_config_or_default()
            demo_resume, demo_jd = config.demo_resume, config.demo_jd

        return await self._dispatch_async_task(
            user_id,
            self._run_initialize_training(user_id, demo_resume, demo_jd, job_type),
            waiting_msg="⏳ 正在使用示例数据初始化训练，请稍候...",
        )

    async def _handle_report(self, user_id: str) -> InterviewResponse:
        session = session_store.get_or_create(user_id)
        thread_id = session["thread_id"]
        state = await graph_runner.get_state(thread_id)

        if len(state["history"]) < 4:
            actions = [ActionButton(label="✅ 结束训练并生成报告", action="generate_report")]
            return InterviewResponse(
                reply=_append_action_hints(
                    "💡 建议至少和我过两招再生成报告，否则报告里没东西可写。", actions
                ),
                actions=actions,
            )

        session_store.update(user_id, {"session_ended": True, "is_generating_report": True})

        return await self._dispatch_async_task(
            user_id,
            self._run_report(user_id),
            waiting_msg="⏳ 正在为你生成面试报告，通常需要10-30秒，请稍候...",
        )

    async def _handle_use_default_jd(self, user_id: str) -> InterviewResponse:
        session = session_store.get_or_create(user_id)
        pending_resume = session.get("pending_resume", "")
        job_type = session.get("selected_job_type")

        try:
            job_config = load_job_config(job_type)
            default_jd = job_config.default_jd or "通用岗位要求"
        except FileNotFoundError:
            default_jd = "通用岗位要求"

        session_store.update(user_id, {"waiting_for_jd": False})
        return await self._dispatch_async_task(
            user_id,
            self._run_initialize_training(user_id, pending_resume, default_jd, job_type),
            waiting_msg="⏳ 正在初始化训练，请稍候...",
        )

    async def _handle_chat(
        self, user_id: str, message: str, file_url: Optional[str] = None
    ) -> InterviewResponse:
        session = session_store.get_or_create(user_id)

        if session.get("session_ended"):
            actions = [ActionButton(label="🔄 开始新一轮训练", action="reset")]
            return InterviewResponse(
                reply=_append_action_hints("✅ 本轮训练已结束。如需开始新训练，请回复「重新开始」。", actions),
                actions=actions,
            )

        if session.get("is_generating_report"):
            return InterviewResponse(reply="⏳ 系统正在为你生成报告，请稍等片刻...")

        if not session.get("selected_job_type"):
            return await self._build_job_selection_response()

        thread_id = session["thread_id"]

        extracted_text = ""
        if file_url:
            file_bytes = await _download_file(file_url)
            if file_bytes:
                try:
                    fake_file = _FileWrapper(content=file_bytes, name="uploaded_file")
                    extracted_text = await asyncio.to_thread(extract_text_from_file, fake_file)
                except Exception as e:
                    return InterviewResponse(
                        reply=f"⚠️ 文件解析失败：{str(e)[:100]}",
                        status="error",
                    )

        user_raw = _sanitize(message) if message else ""
        combined = (extracted_text + "\n" + user_raw).strip()
        state = await graph_runner.get_state(thread_id)

        if not state.get("is_ready"):
            if not combined or len(combined) < 20:
                actions = [ActionButton(label="⚡ 快速体验模式", action="quick_start")]
                return InterviewResponse(
                    reply=_append_action_hints(
                        "请先发送你的 **简历文本** 和 **目标岗位JD**，"
                        "我需要先为你建立能力画像。\n\n"
                        "💡 你可以直接粘贴简历内容，或回复「快速体验」。",
                        actions,
                    ),
                    actions=actions,
                )
            return await self._process_initial_input(user_id, combined, user_raw, bool(file_url))

        if session.get("waiting_for_jd"):
            pending_resume = session.get("pending_resume", "")
            session_store.update(user_id, {"waiting_for_jd": False})
            job_type = session.get("selected_job_type")
            return await self._dispatch_async_task(
                user_id,
                self._run_initialize_training(user_id, pending_resume, user_raw, job_type),
                waiting_msg="⏳ 正在初始化训练，请稍候...",
            )

        if not user_raw:
            return InterviewResponse(reply="请输入你的回答。")

        if len(user_raw) > settings.MAX_INPUT_LENGTH:
            user_raw = user_raw[: settings.MAX_INPUT_LENGTH] + "\n[系统警告：输入超长，已截断]"

        try:
            await graph_runner.update_state(thread_id, {"user_input": user_raw})
            state = await graph_runner.get_state(thread_id)
            result_state = await graph_runner.run(state, thread_id)

            response = result_state.get("response", "")

            asyncio.create_task(self._async_update_summary(thread_id))

            actions = [ActionButton(label="✅ 结束训练并生成报告", action="generate_report")]
            return InterviewResponse(
                reply=_append_action_hints(response, actions),
                actions=actions,
            )

        except Exception as e:
            logger.error(f"💀 [API] 对话处理崩溃: {e}", exc_info=True)
            actions = [ActionButton(label="🔄 重置会话", action="reset")]
            return InterviewResponse(
                reply=_append_action_hints("⚠️ 系统处理失败，请重试。如问题持续，请回复「重新开始」。", actions),
                actions=actions,
                status="error",
            )

    async def _process_initial_input(
        self, user_id: str, combined_text: str, user_raw: str, has_file: bool
    ) -> InterviewResponse:
        session = session_store.get_or_create(user_id)
        job_type = session.get("selected_job_type")

        try:
            from core.profiler import gatekeeper
            gk_result = await gatekeeper.aparse(combined_text)
        except Exception as e:
            logger.warning(f"🚪 [API] 门卫降级: {e}")
            gk_result = None

        if gk_result and gk_result.has_resume:
            final_resume = gk_result.resume_text or combined_text
            final_jd = gk_result.jd_text if gk_result.has_jd else None
        elif gk_result and gk_result.has_jd and not gk_result.has_resume:
            return InterviewResponse(reply="⚠️ 检测到岗位JD，但未识别到简历内容。请先发送你的简历。")
        elif has_file:
            final_resume = combined_text
            final_jd = user_raw if user_raw else None
        else:
            final_resume = combined_text
            final_jd = None

        if not final_jd:
            session_store.update(user_id, {
                "waiting_for_jd": True,
                "pending_resume": final_resume,
            })
            try:
                job_config = load_job_config(job_type)
                label = f"📋 使用{job_config.display_name}通用标准"
            except Exception:
                label = "📋 使用通用标准"

            actions = [ActionButton(label=label, action="use_default_jd")]
            return InterviewResponse(
                reply=_append_action_hints(
                    f"⚠️ 未检测到目标岗位JD。请选择：\n\n"
                    f"- 回复「使用默认标准」使用通用岗位要求\n"
                    f"- 或直接在对话框粘贴你的目标岗位JD",
                    actions,
                ),
                actions=actions,
            )

        return await self._dispatch_async_task(
            user_id,
            self._run_initialize_training(user_id, final_resume, final_jd, job_type),
            waiting_msg="⏳ 正在为你建立能力画像并生成首题，通常需要10-30秒...",
        )

    # ==================== 异步任务调度 ====================
    async def _dispatch_async_task(
        self, user_id: str, coro, waiting_msg: str = "⏳ 处理中，请稍候..."
    ) -> InterviewResponse:
        task = task_store.create()
        asyncio.create_task(self._run_async_task(task.task_id, coro))
        return InterviewResponse(
            reply=waiting_msg,
            status="processing",
            task_id=task.task_id,
        )

    async def _run_async_task(self, task_id: str, coro):
        try:
            result: InterviewResponse = await coro
            task_store.complete(task_id, result)
        except Exception as e:
            logger.error(f"⚡ [AsyncTask] 任务失败: {e}", exc_info=True)
            task_store.fail(task_id, str(e))

    # ==================== 异步执行体 ====================
    async def _run_initialize_training(
        self, user_id: str, resume_text: str, jd_text: str, job_type: str = None
    ) -> InterviewResponse:
        session = session_store.get_or_create(user_id)
        thread_id = session["thread_id"]
        effective_job = job_type or session.get("selected_job_type")
        set_current_job_type(effective_job)

        logger.info(
            f"🧬 [API] 初始化 user={user_id[:8]} resume={len(resume_text)} jd={len(jd_text)} job={effective_job}"
        )

        try:
            job_profiler = get_profiler(effective_job)
            profile = await asyncio.to_thread(job_profiler.extract_full, resume_text, jd_text)
            weakness_prefix = profile.to_prompt_prefix

            updates = {
                "weakness_prefix": weakness_prefix,
                "is_ready": True,
                "resume_text": resume_text,
                "jd_text": jd_text,
                "job_type": effective_job,
                "history": [HumanMessage(content="[用户上传了简历和JD]")],
            }
            await graph_runner.update_state(thread_id, updates)

            job_gap_analyzer = get_gap_analyzer(effective_job)
            gap_result = await asyncio.to_thread(
                job_gap_analyzer.analyze, resume_text, jd_text, profile
            )
            await graph_runner.update_state(thread_id, {"gap_analysis": gap_result.model_dump_json()})

            weakness_lines = "\n".join([f"> - {w}" for w in gap_result.weaknesses])
            suggestion_lines = "\n".join([f"> - {s}" for s in gap_result.resume_suggestions])

            first_question_context = await asyncio.to_thread(
                search_interview_db.invoke,
                {"user_query": weakness_prefix, "scene_mode": "train", "job_type": effective_job},
            )

            try:
                job_config = load_job_config(effective_job)
                interviewer_persona = job_config.interviewer_persona
            except FileNotFoundError:
                interviewer_persona = "你是一个冷酷专业的面试官"

            kick_off_llm = llm_factory.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.7)
            kick_off_msgs = [
                SystemMessage(content=f"{interviewer_persona}。{weakness_prefix}"),
                HumanMessage(
                    content=(
                        f"这是从题库匹配的基础素材：\n{first_question_context}\n\n"
                        f"请基于素材结合用户弱点，抛出第一个刁钻面试问题。只提问，不要解释素材。"
                    )
                ),
            ]
            response = await kick_off_llm.ainvoke(kick_off_msgs)
            first_question = response.content

            await graph_runner.update_state(thread_id, {"history": [AIMessage(content=first_question)]})
            session_store.update(user_id, {"is_ready": True})

            gap_display = (
                f"# 🎯 JD和您的简历Gap(差距)\n\n"
                f"> **⚠️ 您的弱点：**\n{weakness_lines}\n\n"
                f"> **💡 建议简历修改方向：**\n{suggestion_lines}\n\n"
                f"---\n\n**现在正式开始进入面试环节**\n\n"
            )
            full_reply = gap_display + first_question

            actions = [ActionButton(label="✅ 结束训练并生成报告", action="generate_report")]
            return InterviewResponse(reply=_append_action_hints(full_reply, actions), actions=actions)

        except Exception as e:
            logger.error(f"🧬 [API] 初始化致命错误: {e}", exc_info=True)
            actions = [ActionButton(label="⚡ 快速体验模式", action="quick_start")]
            return InterviewResponse(
                reply=_append_action_hints(
                    f"⚠️ 画像提取失败：{str(e)[:100]}\n\n请重试或回复「快速体验」。",
                    actions,
                ),
                actions=actions,
                status="error",
            )

    async def _run_report(self, user_id: str) -> InterviewResponse:
        session = session_store.get_or_create(user_id)
        thread_id = session["thread_id"]

        try:
            await graph_runner.update_state(thread_id, {
                "is_generating_report": True, "should_end": True
            })
            state = await graph_runner.get_state(thread_id)

            from core.nodes import report_node
            result = await report_node(state)

            session_store.update(user_id, {"is_generating_report": False})
            actions = [ActionButton(label="🔄 开始新一轮训练", action="reset")]
            return InterviewResponse(
                reply=_append_action_hints(result["response"], actions),
                actions=actions,
            )

        except Exception as e:
            logger.error(f"📊 [API] 报告生成崩溃: {e}", exc_info=True)
            session_store.update(user_id, {"session_ended": False, "is_generating_report": False})
            await graph_runner.update_state(thread_id, {
                "is_generating_report": False, "should_end": False
            })
            actions = [ActionButton(label="✅ 重新生成报告", action="generate_report")]
            return InterviewResponse(
                reply=_append_action_hints("⚠️ 报告生成失败，请重试。", actions),
                actions=actions,
                status="error",
            )

    # ==================== 辅助方法 ====================
    async def _build_job_selection_response(self) -> InterviewResponse:
        available_jobs = list_available_jobs()
        if not available_jobs:
            return InterviewResponse(reply="❌ 没有找到任何岗位配置", status="error")

        actions = []
        for jt in available_jobs:
            try:
                config = load_job_config(jt)
                actions.append(
                    ActionButton(label=f"📋 {config.display_name}", action="select_job", value=jt)
                )
            except Exception:
                continue

        if not actions:
            return InterviewResponse(reply="❌ 岗位配置加载失败", status="error")

        return InterviewResponse(
            reply=_append_action_hints(
                "🎯 **请选择你要训练的岗位：**\n\n回复对应岗位名称开始训练。",
                actions,
            ),
            actions=actions,
        )

    async def _async_update_summary(self, thread_id: str):
        try:
            from core.nodes.memory import memory_node
            state = await graph_runner.get_state(thread_id)
            summary_update = await memory_node(state)
            if summary_update:
                await graph_runner.update_state(thread_id, summary_update)
        except Exception as e:
            logger.warning(f"📝 [Memory] 摘要更新失败: {e}")


interview_service = InterviewService()
