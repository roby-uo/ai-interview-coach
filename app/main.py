"""
AI 面试教练 - Chainlit 应用入口

启动方式:
    1. 安装依赖: pip install -e .
    2. 启动服务: chainlit run app/main.py

或者直接使用启动脚本:
    启动面试教练.bat
"""

import io
import os
import logging
import asyncio
import uuid
import re
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import settings
from infrastructure.parsers.file_parser import extract_text_from_file
from domain.schemas import InterviewState
from domain.job_configs import load_job_config, list_available_jobs, get_job_config_or_default
from core.nodes import create_initial_state
from core.graph import graph_runner
from core.profiler.extractor import get_profiler
from core.profiler.gap_analyzer import get_gap_analyzer
from infrastructure.tools.interview_db import set_current_job_type, get_current_job_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class FileWrapper:
    def __init__(self, content: bytes, name: str):
        self._bytes_io = io.BytesIO(content)
        self.name = name

    def read(self):
        return self._bytes_io.read()


def get_thread_id() -> str:
    return cl.user_session.get("thread_id", "default_thread")


def sanitize_input(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def format_error_message(error: Exception, context: str = "操作") -> str:
    error_type = type(error).__name__
    error_msg = str(error)
    
    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
        return f"⚠️ {context}超时，请稍后重试。如问题持续，请检查网络连接。"
    elif "rate limit" in error_msg.lower():
        return f"⚠️ 服务繁忙，请稍等片刻后重试。"
    elif "api" in error_msg.lower() or "key" in error_msg.lower():
        return f"⚠️ 服务配置异常，请联系管理员。"
    elif "connection" in error_msg.lower():
        return f"⚠️ 网络连接异常，请检查网络后重试。"
    else:
        safe_msg = error_msg[:80].replace('\n', ' ')
        return f"⚠️ {context}失败：{safe_msg}"


@cl.action_callback("generate_report")
async def generate_report_action(action: cl.Action):
    thread_id = get_thread_id()
    state = await graph_runner.get_state(thread_id)

    if len(state["history"]) < 4:
        await cl.Message(content="💡 建议至少和我过两招再生成报告，否则报告里没东西可写。").send()
        return

    cl.user_session.set("is_generating_report", True)
    cl.user_session.set("session_ended", True)
    await action.remove()

    await cl.Message(content="⏳ 报告生成中，请稍候...\n\n🧠 专属顾问正在抽干记忆，为你整理《面试能力体检报告》...\n\n---").send()

    try:
        from core.nodes import report_node
        updates = {"is_generating_report": True, "should_end": True}
        await graph_runner.update_state(thread_id, updates)

        state = await graph_runner.get_state(thread_id)
        result = await report_node(state)

        await cl.Message(content=result["response"]).send()
        
        reset_action = cl.Action(
            name="reset_session",
            label="🔄 开始新一轮训练",
            payload={"action": "reset"}
        )
        await cl.Message(
            content="\n---\n✅ **训练已完成！** 如需开始新一轮训练，请点击下方按钮：",
            actions=[reset_action]
        ).send()
        
        logger.info("📊 [阶段四] 报告生成完毕，会话已结束。")

    except Exception as e:
        logger.error(f"📊 [阶段四] 报告生成崩溃: {e}")
        await cl.Message(content=format_error_message(e, "报告生成")).send()
        
        cl.user_session.set("is_generating_report", False)
        cl.user_session.set("session_ended", False)
        
        await graph_runner.update_state(thread_id, {
            "is_generating_report": False,
            "should_end": False
        })
        
        retry_action = cl.Action(
            name="generate_report",
            label="✅ 结束训练并生成报告",
            payload={"action": "report"}
        )
        await cl.Message(
            content="❌ 报告生成失败，请重试：",
            actions=[retry_action]
        ).send()


@cl.action_callback("reset_session")
async def reset_session_action(action: cl.Action):
    thread_id = get_thread_id()
    
    new_thread_id = f"session_{uuid.uuid4().hex}"
    cl.user_session.set("thread_id", new_thread_id)
    cl.user_session.set("is_generating_report", False)
    cl.user_session.set("session_ended", False)
    cl.user_session.set("selected_job_type", None)
    
    initial_state = create_initial_state()
    await graph_runner.update_state(new_thread_id, initial_state)
    
    graph_runner.clear_session_lock(thread_id)
    
    await action.remove()
    
    await _show_job_selection()


@cl.action_callback("select_job")
async def select_job_action(action: cl.Action):
    job_type = action.payload.get("job_type") if action.payload else None
    if not job_type:
        await cl.Message(content="❌ 岗位信息获取失败").send()
        return
    cl.user_session.set("selected_job_type", job_type)
    set_current_job_type(job_type)
    await action.remove()

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        await cl.Message(content=f"❌ 未找到岗位配置: {job_type}").send()
        return

    actions = [
        cl.Action(
            name="quick_start",
            label="⚡ 快速体验模式",
            payload={"action": "quick_start"}
        ),
        cl.Action(
            name="generate_report",
            label="✅ 结束训练并生成报告",
            payload={"action": "report"}
        )
    ]

    await cl.Message(
        content=f"✅ 已选择岗位：**{config.display_name}**\n\n"
                f"请发送你的 **简历（PDF/TXT）** 和 **目标岗位 JD**，或点击「⚡ 快速体验模式」使用示例数据开始训练。",
        actions=actions
    ).send()


async def _show_job_selection():
    available_jobs = list_available_jobs()

    if not available_jobs:
        await cl.Message(content="❌ 没有找到任何岗位配置，请先在 domain/job_configs/ 下创建 YAML 配置文件。").send()
        return

    job_actions = []
    for job_type in available_jobs:
        try:
            config = load_job_config(job_type)
            job_actions.append(
                cl.Action(
                    name="select_job",
                    label=f"📋 {config.display_name}",
                    payload={"job_type": job_type}
                )
            )
        except Exception:
            continue

    if not job_actions:
        await cl.Message(content="❌ 岗位配置加载失败，请检查 YAML 文件格式。").send()
        return

    await cl.Message(
        content="🎯 **请选择你要训练的岗位：**",
        actions=job_actions
    ).send()


@cl.on_chat_start
async def start():
    graph_runner.initialize()

    thread_id = f"session_{uuid.uuid4().hex}"
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("is_generating_report", False)
    cl.user_session.set("session_ended", False)
    cl.user_session.set("selected_job_type", None)

    initial_state = create_initial_state()
    await graph_runner.update_state(thread_id, initial_state)

    welcome_content = """# AI 面试教练 👔🎯

你好！我是你的专属 AI 面试教练。

## 我能帮你做什么

- 📄 **简历分析** - 深度解析你的简历，找出潜在弱点
- 🎯 **弱点画像** - 基于 JD 和简历对比，生成个性化能力画像
- 💬 **模拟面试** - 针对你的弱点进行刁钻提问
- 📊 **能力报告** - 生成面试能力体检报告

## 开始使用

1. **选择岗位** 👇 从下方按钮选择你要训练的岗位
2. **发送资料** 📄 把你的简历（PDF/TXT）和目标岗位 JD 粘贴到对话框
3. **开始训练** 🎯 系统会为你建立能力画像，然后开始模拟面试

---

> 💡 **Tips**: 所有问题均针对你的弱点定制，模拟大厂面试官的刁钻提问。卡住时尽管求助，我会给你满分公式化提示，带你一步步拆解思路。

*来头脑风暴吧🧠，你准备好接受挑战了吗？* 💪

"""

    await cl.Message(content=welcome_content).send()

    await _show_job_selection()


@cl.on_message
async def main(message: cl.Message):
    if cl.user_session.get("is_generating_report"):
        await cl.Message(content="⏳ 系统正在为你生成报告，请稍等片刻...").send()
        return
    
    if cl.user_session.get("session_ended"):
        await cl.Message(
            content="✅ 本轮训练已结束。如需开始新训练，请点击上方的「开始新一轮训练」按钮，或刷新页面。"
        ).send()
        return

    selected_job_type = cl.user_session.get("selected_job_type")
    if not selected_job_type:
        available_jobs = list_available_jobs()
        if len(available_jobs) == 1:
            selected_job_type = available_jobs[0]
            cl.user_session.set("selected_job_type", selected_job_type)
            set_current_job_type(selected_job_type)
        else:
            await cl.Message(content="⚠️ 请先选择你要训练的岗位 👆").send()
            await _show_job_selection()
            return

    thread_id = get_thread_id()
    state = await graph_runner.get_state(thread_id)

    user_raw_input = sanitize_input(message.content) if message.content else ""
    logger.info(f"👑 [用户输入] 长度: {len(user_raw_input)} | 含文件: {bool(message.elements)}")

    extracted_text = ""
    has_file = False

    if message.elements:
        has_file = True
        await cl.Message(content="📄 正在解析文件，请稍候...").send()
        for elem in message.elements:
            file_path = getattr(elem, "path", None)
            file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
            max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            
            if file_size > max_size_bytes:
                await cl.Message(
                    content=f"🚫 文件过大（{file_size / 1024 / 1024:.1f}MB），最大支持{settings.MAX_FILE_SIZE_MB}MB。请压缩后重试。"
                ).send()
                return
            
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    fake_file = FileWrapper(content=file_bytes, name=elem.name)
                    extracted_text += await asyncio.to_thread(extract_text_from_file, fake_file) + "\n"
                    logger.info(f"🛡️ [物理网关] 成功解析文件: {elem.name}")
                except ValueError as e:
                    await cl.Message(content=f"🚫 {str(e)}").send()
                    return
                except Exception as e:
                    await cl.Message(content=format_error_message(e, "文件解析")).send()
                    return
            else:
                if elem.content and isinstance(elem.content, str):
                    extracted_text += elem.content + "\n"

    if not state["is_ready"]:
        combined_text = (extracted_text + "\n" + user_raw_input).strip()

        if not combined_text or len(combined_text) < 20:
            await cl.Message(content="请先发送你的简历（PDF或文本）和目标岗位JD，我需要先为你建立能力画像。").send()
            return

        await cl.Message(content="🚪 门卫正在智能识别你的输入内容...").send()
        
        from core.profiler import gatekeeper
        try:
            gatekeeper_result = await gatekeeper.aparse(combined_text)
        except Exception as e:
            logger.warning(f"🚪 [门卫] 识别失败，降级为默认处理: {e}")
            gatekeeper_result = None

        if gatekeeper_result and gatekeeper_result.has_resume:
            final_resume = gatekeeper_result.resume_text or combined_text
            final_jd = gatekeeper_result.jd_text if gatekeeper_result.has_jd else None
        elif gatekeeper_result and gatekeeper_result.has_jd and not gatekeeper_result.has_resume:
            final_resume = ""
            final_jd = gatekeeper_result.jd_text
            await cl.Message(content="⚠️ 检测到岗位JD，但未识别到简历内容。请先发送你的简历。").send()
            return
        elif has_file:
            final_resume = combined_text
            final_jd = user_raw_input if user_raw_input else None
        else:
            final_resume = combined_text
            final_jd = None

        if not final_jd:
            cl.user_session.set("waiting_for_jd_decision", True)
            cl.user_session.set("pending_resume_text", final_resume)
            
            try:
                job_config = load_job_config(selected_job_type)
                default_jd_label = f"📋 使用{job_config.display_name}通用标准"
            except FileNotFoundError:
                default_jd_label = "📋 使用通用标准"

            jd_actions = [
                cl.Action(name="use_default_jd", label=default_jd_label, payload={"action": "default"}),
                cl.Action(name="input_jd", label="✏️ 我来输入JD", payload={"action": "input"})
            ]
            await cl.Message(
                content="⚠️ 未检测到目标岗位JD。请选择：\n\n"
                        f"- {default_jd_label}：使用通用岗位要求进行训练\n"
                        "- ✏️ **我来输入JD**：在对话框中粘贴目标岗位JD",
                actions=jd_actions
            ).send()
            return

        await _initialize_training(thread_id, final_resume, final_jd, selected_job_type)
        return
    
    if cl.user_session.get("waiting_for_jd_decision"):
        decision = user_raw_input.lower().strip()
        
        if decision in ["继续", "继续训练", "跳过", "使用通用标准", "1", "default"]:
            pending_resume = cl.user_session.get("pending_resume_text", "")
            cl.user_session.set("waiting_for_jd_decision", False)
            try:
                job_config = load_job_config(selected_job_type)
                default_jd = job_config.default_jd or "通用岗位要求"
            except FileNotFoundError:
                default_jd = "通用岗位要求"
            await _initialize_training(thread_id, pending_resume, default_jd, selected_job_type)
            return
        else:
            cl.user_session.set("waiting_for_jd_decision", False)
            await _initialize_training(thread_id, cl.user_session.get("pending_resume_text", ""), user_raw_input, selected_job_type)
            return

    if not user_raw_input:
        return

    logger.info("==================================================")

    try:
        user_input = user_raw_input
        if len(user_input) > settings.MAX_INPUT_LENGTH:
            user_input = user_input[:settings.MAX_INPUT_LENGTH] + "\n[系统警告：输入超长，已截断]"

        await graph_runner.update_state(thread_id, {"user_input": user_input})

        thinking_msg = cl.Message(content="🧠 正在分析你的回答...")
        await thinking_msg.send()

        state = await graph_runner.get_state(thread_id)
        result_state = await graph_runner.run(state, thread_id)

        response = result_state.get("response", "")

        await thinking_msg.remove()
        await cl.Message(content=response).send()

        asyncio.create_task(_async_update_summary(thread_id))

        if not cl.user_session.get("session_ended"):
            actions = [
                cl.Action(
                    name="generate_report",
                    label="✅ 结束训练并生成报告",
                    payload={"action": "report"}
                )
            ]
            await cl.Message(
                content="",
                actions=actions
            ).send()

        logger.info("==================================================\n")

    except Exception as e:
        import traceback
        logger.error(f"💀 [战斗主链路] 崩溃!\n{traceback.format_exc()}")
        await cl.Message(content=format_error_message(e, "系统处理")).send()


async def _async_update_summary(thread_id: str):
    try:
        from core.nodes.memory import memory_node
        state = await graph_runner.get_state(thread_id)
        summary_update = await memory_node(state)
        if summary_update:
            await graph_runner.update_state(thread_id, summary_update)
            logger.info("🧠 [异步摘要] 后台摘要更新完成")
    except Exception as e:
        logger.warning(f"🧠 [异步摘要] 后台摘要更新失败: {e}")


@cl.action_callback("use_default_jd")
async def use_default_jd_action(action: cl.Action):
    thread_id = get_thread_id()
    pending_resume = cl.user_session.get("pending_resume_text", "")
    cl.user_session.set("waiting_for_jd_decision", False)
    await action.remove()

    selected_job_type = cl.user_session.get("selected_job_type")
    try:
        job_config = load_job_config(selected_job_type)
        default_jd = job_config.default_jd or "通用岗位要求"
    except FileNotFoundError:
        default_jd = "通用岗位要求"

    await _initialize_training(thread_id, pending_resume, default_jd, selected_job_type)


@cl.action_callback("input_jd")
async def input_jd_action(action: cl.Action):
    cl.user_session.set("waiting_for_jd_decision", True)
    await action.remove()
    await cl.Message(content="请在对话框中粘贴你的目标岗位JD文本。").send()


@cl.action_callback("quick_start")
async def quick_start_action(action: cl.Action):
    thread_id = get_thread_id()
    await action.remove()

    selected_job_type = cl.user_session.get("selected_job_type")
    if not selected_job_type:
        available_jobs = list_available_jobs()
        if available_jobs:
            selected_job_type = available_jobs[0]
            cl.user_session.set("selected_job_type", selected_job_type)
            set_current_job_type(selected_job_type)

    try:
        config = load_job_config(selected_job_type)
        demo_resume = config.demo_resume
        demo_jd = config.demo_jd
    except FileNotFoundError:
        config = get_job_config_or_default()
        demo_resume = config.demo_resume
        demo_jd = config.demo_jd

    await _initialize_training(thread_id, demo_resume, demo_jd, selected_job_type)


async def _initialize_training(thread_id: str, resume_text: str, jd_text: str, job_type: str = None):
    effective_job_type = job_type or get_current_job_type()
    set_current_job_type(effective_job_type)

    logger.info(f"🧬 [初始化流水线] 通过防线。简历长度: {len(resume_text)}, JD长度: {len(jd_text)}, 岗位: {effective_job_type}")
    await cl.Message(content="🧠 检测到背景资料，正在深度分析弱点画像...").send()

    try:
        job_profiler = get_profiler(effective_job_type)
        profile = await asyncio.to_thread(job_profiler.extract_full, resume_text, jd_text)
        weakness_prefix = profile.to_prompt_prefix

        updates = {
            "weakness_prefix": weakness_prefix,
            "is_ready": True,
            "resume_text": resume_text,
            "jd_text": jd_text,
            "job_type": effective_job_type,
            "history": [HumanMessage(content="[用户上传了简历和JD]")]
        }
        await graph_runner.update_state(thread_id, updates)

        await cl.Message(content="📊 正在生成差距分析报告...").send()
        job_gap_analyzer = get_gap_analyzer(effective_job_type)
        gap_result = await asyncio.to_thread(job_gap_analyzer.analyze, resume_text, jd_text, profile)

        gap_analysis_json = gap_result.model_dump_json()
        await graph_runner.update_state(thread_id, {"gap_analysis": gap_analysis_json})

        weakness_lines = "\n".join([f"> - {w}" for w in gap_result.weaknesses])
        suggestion_lines = "\n".join([f"> - {s}" for s in gap_result.resume_suggestions])

        gap_display = f"""# 🎯 JD和您的简历Gap(差距)

> **⚠️ 您的弱点：**
{weakness_lines}

> **💡 建议简历修改方向：**
{suggestion_lines}

---

<div style="text-align: center; font-size: 1.3em; color: #e87878; font-weight: 600; margin: 16px 0;">现在正式开始进入面试环节</div>

<div style="font-size: 1.1em; color: #c9a96e; font-weight: 500;">你好，我是今天的面试官</div>"""

        await cl.Message(content=gap_display).send()

        logger.info("🎯 [初始化流水线] 正在根据弱点画像去题库检索第一道题...")
        from infrastructure.tools.interview_db import search_interview_db
        first_question_context = await asyncio.to_thread(
            search_interview_db.invoke,
            {"user_query": weakness_prefix, "scene_mode": "train", "job_type": effective_job_type}
        )

        from core.utils.llm_factory import llm_factory

        try:
            job_config = load_job_config(effective_job_type)
            interviewer_persona = job_config.interviewer_persona
        except FileNotFoundError:
            interviewer_persona = "你是一个冷酷专业的面试官"

        kick_off_llm = llm_factory.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.7)
        kick_off_msgs = [
            SystemMessage(content=f"{interviewer_persona}。{weakness_prefix}"),
            HumanMessage(content=f"这是我从题库里为你匹配的基础素材：\n{first_question_context}\n\n请基于上述素材，结合用户的弱点，用你自己的话术，抛出第一个极其刁钻的面试问题。只提问，不要解释素材。")
        ]

        response_msg = cl.Message(content="")
        await response_msg.send()

        full_response = ""
        async for chunk in kick_off_llm.astream(kick_off_msgs):
            if chunk.content:
                await response_msg.stream_token(chunk.content)
                full_response += chunk.content
        await response_msg.update()

        history_updates = {
            "history": [AIMessage(content=full_response)]
        }
        await graph_runner.update_state(thread_id, history_updates)

        return

    except Exception as e:
        import traceback
        logger.error(f"🧬 [初始化] 致命错误: {traceback.format_exc()}")
        await cl.Message(content=format_error_message(e, "画像提取")).send()
        return
