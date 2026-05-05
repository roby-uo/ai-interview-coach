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
from core.nodes import create_initial_state
from core.graph import graph_runner
from core.profiler import profiler, gatekeeper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEMO_RESUME = """姓名：李四
联系方式：13900139000
邮箱：lisi@email.com

教育背景：
2019-2023 暨南大学 新闻学 本科

工作经历：
2023-2026 某MCN机构 新媒体运营专员
- 负责公司抖音账号矩阵运营，粉丝总量从5万增长至30万
- 策划并执行3场直播带货活动，GMV累计50万
- 撰写小红书种草笔记100+篇，平均互动率8%
- 协助品牌客户制定社媒投放方案

技能：
抖音运营、小红书运营、直播带货、文案撰写、数据分析、Canva、剪映"""

DEMO_JD = """岗位名称：资深新媒体运营
岗位职责：
1. 负责品牌全媒体矩阵（抖音/小红书/视频号）的内容策略制定与执行
2. 搭建并管理KOL/KOC资源库，维护达人合作体系
3. 制定数据驱动的运营增长策略，建立ROI评估体系
4. 带领2-3人运营小组，制定内容SOP和培训体系
5. 跨部门协调，推动品牌传播与电商转化联动
任职要求：
1. 3年以上新媒体运营经验，有爆款操盘案例
2. 深度理解各平台算法机制与流量逻辑
3. 具备数据分析能力，能独立完成复盘报告
4. 有团队管理经验，能搭建内容SOP
5. 有品牌方或4A公司经验优先"""


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
            value="reset",
            label="🔄 开始新一轮训练",
            payload={}
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
            value="report",
            label="✅ 结束训练并生成报告",
            payload={}
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
    
    initial_state = create_initial_state()
    await graph_runner.update_state(new_thread_id, initial_state)
    
    graph_runner.clear_session_lock(thread_id)
    
    await action.remove()
    
    actions = [
        cl.Action(
            name="quick_start",
            value="quick_start",
            label="⚡ 快速体验模式",
            payload={}
        ),
        cl.Action(
            name="generate_report",
            value="report",
            label="✅ 结束训练并生成报告",
            payload={}
        )
    ]
    
    await cl.Message(
        content="🔄 **已重置！** 请发送你的简历和目标岗位JD，开始新一轮训练。",
        actions=actions
    ).send()
    
    logger.info(f"🔄 [会话重置] 新会话ID: {new_thread_id}")


@cl.on_chat_start
async def start():
    graph_runner.initialize()

    thread_id = f"session_{uuid.uuid4().hex}"
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("is_generating_report", False)
    cl.user_session.set("session_ended", False)

    initial_state = create_initial_state()
    await graph_runner.update_state(thread_id, initial_state)

    actions = [
        cl.Action(
            name="quick_start",
            value="quick_start",
            label="⚡ 快速体验模式",
            payload={}
        ),
        cl.Action(
            name="generate_report",
            value="report",
            label="✅ 结束训练并生成报告",
            payload={}
        )
    ]

    welcome_content = """# AI 面试教练 👔🎯

你好！我是你的专属 AI 面试教练。

## 我能帮你做什么

- 📄 **简历分析** - 深度解析你的简历，找出潜在弱点
- 🎯 **弱点画像** - 基于 JD 和简历对比，生成个性化能力画像
- 💬 **模拟面试** - 针对你的弱点进行刁钻提问
- 📊 **能力报告** - 生成面试能力体检报告

## 开始使用

直接把你的 **简历（PDF/TXT）** 和 **目标岗位 JD** 粘贴到对话框。

我会先为你建立能力画像，然后开始模拟面试训练！

---

> 💡 **Tips**: 所有问题均针对你的弱点定制，模拟大厂面试官的刁钻提问。卡住时尽管求助，我会给你满分公式化提示，带你一步步拆解思路。

*可以选择快速体验模式👇，系统已准备简历和JD*
*来头脑风暴吧🧠，你准备好接受挑战了吗？* 💪

"""

    await cl.Message(
        content=welcome_content,
        actions=actions
    ).send()


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
            
            jd_actions = [
                cl.Action(name="use_default_jd", value="default", label="📋 使用通用标准", payload={}),
                cl.Action(name="input_jd", value="input", label="✏️ 我来输入JD", payload={})
            ]
            await cl.Message(
                content="⚠️ 未检测到目标岗位JD。请选择：\n\n- 📋 **使用通用标准**：使用通用新媒体运营岗位要求进行训练\n- ✏️ **我来输入JD**：在对话框中粘贴目标岗位JD",
                actions=jd_actions
            ).send()
            return

        await _initialize_training(thread_id, final_resume, final_jd)
        return
    
    if cl.user_session.get("waiting_for_jd_decision"):
        decision = user_raw_input.lower().strip()
        
        if decision in ["继续", "继续训练", "跳过", "使用通用标准", "1", "default"]:
            pending_resume = cl.user_session.get("pending_resume_text", "")
            cl.user_session.set("waiting_for_jd_decision", False)
            await _initialize_training(thread_id, pending_resume, "通用新媒体运营岗位要求")
            return
        else:
            cl.user_session.set("waiting_for_jd_decision", False)
            await _initialize_training(thread_id, cl.user_session.get("pending_resume_text", ""), user_raw_input)
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
                    value="report",
                    label="✅ 结束训练并生成报告",
                    payload={}
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
    await _initialize_training(thread_id, pending_resume, "通用新媒体运营岗位要求")


@cl.action_callback("input_jd")
async def input_jd_action(action: cl.Action):
    cl.user_session.set("waiting_for_jd_decision", True)
    await action.remove()
    await cl.Message(content="请在对话框中粘贴你的目标岗位JD文本。").send()


@cl.action_callback("quick_start")
async def quick_start_action(action: cl.Action):
    thread_id = get_thread_id()
    await action.remove()
    await _initialize_training(thread_id, DEMO_RESUME, DEMO_JD)


async def _initialize_training(thread_id: str, resume_text: str, jd_text: str):
    logger.info(f"🧬 [初始化流水线] 通过防线。简历长度: {len(resume_text)}, JD长度: {len(jd_text)}")
    await cl.Message(content="🧠 检测到背景资料，正在深度分析弱点画像...").send()

    try:
        weakness_prefix = await asyncio.to_thread(profiler.extract, resume_text, jd_text)

        updates = {
            "weakness_prefix": weakness_prefix,
            "is_ready": True,
            "resume_text": resume_text,
            "jd_text": jd_text,
            "history": [HumanMessage(content="[用户上传了简历和JD]")]
        }
        await graph_runner.update_state(thread_id, updates)

        logger.info("🎯 [初始化流水线] 正在根据弱点画像去题库检索第一道题...")
        from infrastructure.tools.interview_db import search_interview_db
        first_question_context = await asyncio.to_thread(
            search_interview_db.invoke,
            {"user_query": weakness_prefix, "scene_mode": "train"}
        )

        from core.utils.llm_factory import llm_factory

        kick_off_llm = llm_factory.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.7)
        kick_off_msgs = [
            SystemMessage(content=f"你是一个冷酷专业的面试官。{weakness_prefix}"),
            HumanMessage(content=f"这是我从题库里为你匹配的基础素材：\n{first_question_context}\n\n请基于上述素材，结合用户的弱点，用你自己的话术，抛出第一个极其刁钻的面试问题。只提问，不要解释素材。")
        ]

        response_msg = cl.Message(content="✅ 画像已锁定，正在根据你的死穴定制开场问题...\n\n")
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
