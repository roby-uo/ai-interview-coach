# 面试教练项目文档

> 生成时间: 2026-05-06 22:48:31
> 项目路径: `C:\Users\20647\Desktop\Agent项目管理\私人专属面试顾问\interview_coach`

---

## 目录

- [项目架构](#项目架构)
- [核心代码](#核心代码)
  - [应用层 (app/)](#应用层-app)
  - [核心层 (core/)](#核心层-core)
  - [领域层 (domain/)](#领域层-domain)
    - [岗位配置 (domain/job_configs/)](#岗位配置-domainjob_configs)
  - [基础设施层 (infrastructure/)](#基础设施层-infrastructure)
  - [脚本 (scripts/)](#脚本-scripts)
- [统计信息](#统计信息)

---

## 项目架构

```
interview_coach/
├── app/
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── core/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── runners.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── history.py
│   │   ├── memory.py
│   │   ├── report.py
│   │   ├── router.py
│   │   ├── state.py
│   │   └── utils.py
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── gap_analyzer.py
│   │   └── gatekeeper.py
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── judge.py
│   │   ├── redirection.py
│   │   ├── scaffold.py
│   │   ├── socratic.py
│   │   └── teaching.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── llm_factory.py
│   └── __init__.py
├── data/
│   ├── index/
│   │   ├── faiss_index/
│   │   │   ├── index.faiss
│   │   │   └── index.pkl
│   │   ├── bm25.json
│   │   ├── bm25.pkl
│   │   └── index.hash
│   ├── processed/
│   │   └── mined_questions.jsonl
│   └── raw/
│       └── raw_textbook.txt
├── domain/
│   ├── job_configs/
│   │   ├── __init__.py
│   │   └── 新媒体运营.yaml
│   ├── __init__.py
│   ├── schemas.py
│   └── validators.py
├── infrastructure/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── file_parser.py
│   │   └── text_parser.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   └── hybrid.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── interview_db.py
│   └── __init__.py
├── public/
│   └── custom.css
├── scripts/
│   ├── build_index.py
│   ├── manage.py
│   └── mine_textbook.py
├── chainlit.md
├── generate_project_doc.py
├── pyproject.toml
├── start.py
├── start.sh
└── 启动服务.bat
```

### 架构说明

| 目录 | 说明 |
|------|------|
| `app/` | 应用入口和配置（Chainlit UI层） |
| `core/` | 核心业务逻辑（LangGraph图、节点、技能、用户画像） |
| `core/utils/` | 工具模块（LLM工厂、通用工具函数） |
| `data/` | 数据层（向量索引、处理后数据、原始数据） |
| `data/jobs/{岗位}/` | 各岗位独立数据目录（raw/processed/index） |
| `domain/` | 领域模型、数据结构和业务验证 |
| `domain/job_configs/` | 岗位配置文件（YAML格式） |
| `infrastructure/` | 基础设施（解析器、检索器、工具） |
| `scripts/` | 构建索引、数据处理、CLI管理工具 |
| `public/` | Chainlit静态资源（CSS等） |
| `.chainlit/` | Chainlit配置 |

---

## 核心代码

### 应用层 (app/)

*Chainlit应用入口和配置*

#### `app\__init__.py`

```python
from .config import settings, Settings

__all__ = ["settings", "Settings"]

```

#### `app\config.py`

```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    ROUTER_MODEL_NAME: str = "deepseek-v3"
    FAST_MODEL_NAME: str = "qwen-turbo"
    OFFLINE_MODEL_NAME: str = "qwen3.6-flash"
    EMBEDDING_MODEL_NAME: str = "text-embedding-v4"

    FAISS_TOP_K: int = 3
    BM25_TOP_K: int = 3

    SHORT_TERM_MEMORY_K: int = 10
    MAX_HISTORY_TURNS: int = 10
    MAX_INPUT_LENGTH: int = 3000
    STREAM_TIMEOUT_SECONDS: int = 90
    MAX_FILE_SIZE_MB: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.OPENAI_API_KEY:
            dashscope_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if dashscope_key:
                self.OPENAI_API_KEY = dashscope_key
            else:
                raise ValueError(
                    "API密钥未配置！请设置环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，或在.env文件中配置OPENAI_API_KEY"
                )

    def get_app_root(self) -> Path:
        return Path(__file__).parent.parent.resolve()

    def get_jobs_data_dir(self) -> Path:
        return self.get_app_root() / "data" / "jobs"

    def get_job_data_dir(self, job_type: str) -> Path:
        return self.get_jobs_data_dir() / job_type

    def get_job_raw_dir(self, job_type: str) -> Path:
        return self.get_job_data_dir(job_type) / "raw"

    def get_job_processed_dir(self, job_type: str) -> Path:
        return self.get_job_data_dir(job_type) / "processed"

    def get_job_index_dir(self, job_type: str) -> Path:
        return self.get_job_data_dir(job_type) / "index"


settings = Settings()

```

#### `app\main.py`

```python
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

```

### 核心层 (core/)

*LangGraph图定义、节点实现、技能模块、用户画像*

#### `core\__init__.py`

```python
from core.nodes import (
    create_initial_state,
    memory_node,
    route_node,
    update_history_node,
    report_node,
    get_message_content,
    get_message_role
)
from core.graph import (
    build_interview_graph,
    graph_runner,
    InterviewGraphRunner
)
from core.profiler import Profiler, get_profiler
from core.skills import SKILL_NODE_MAP

__all__ = [
    "create_initial_state",
    "memory_node",
    "route_node",
    "update_history_node",
    "report_node",
    "get_message_content",
    "get_message_role",
    "build_interview_graph",
    "graph_runner",
    "InterviewGraphRunner",
    "Profiler",
    "get_profiler",
    "SKILL_NODE_MAP"
]

```

#### `core\graph\__init__.py`

```python
from .builder import build_interview_graph
from .runners import (
    create_compiled_graph,
    get_checkpointer,
    InterviewGraphRunner,
    graph_runner
)

__all__ = [
    "build_interview_graph",
    "create_compiled_graph",
    "get_checkpointer",
    "InterviewGraphRunner",
    "graph_runner"
]

```

#### `core\graph\builder.py`

```python
import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from domain.schemas import InterviewState
from core.nodes import route_node, update_history_node, report_node
from core.skills import SKILL_NODE_MAP

logger = logging.getLogger(__name__)


def route_decision(state: InterviewState) -> str:
    if state.get("is_generating_report"):
        return "report"
    
    decision = state.get("decision", {})
    target_skill = decision.get("target_skill", "smart_redirection")
    
    skill_to_node = {
        "socratic_probe": "socratic",
        "scaffold_rescue": "scaffold",
        "judge_strike": "judge",
        "smart_redirection": "redirection",
        "teaching_supplement": "teaching"
    }
    return skill_to_node.get(target_skill, "redirection")


def build_interview_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("route", route_node)
    graph.add_node("socratic", SKILL_NODE_MAP["socratic_probe"])
    graph.add_node("scaffold", SKILL_NODE_MAP["scaffold_rescue"])
    graph.add_node("judge", SKILL_NODE_MAP["judge_strike"])
    graph.add_node("redirection", SKILL_NODE_MAP["smart_redirection"])
    graph.add_node("teaching", SKILL_NODE_MAP["teaching_supplement"])
    graph.add_node("update_history", update_history_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "report": "report",
            "socratic": "socratic",
            "scaffold": "scaffold",
            "judge": "judge",
            "redirection": "redirection",
            "teaching": "teaching"
        }
    )

    for skill_name in ["socratic", "scaffold", "judge", "redirection", "teaching"]:
        graph.add_edge(skill_name, "update_history")

    graph.add_edge("update_history", END)
    graph.add_edge("report", END)

    return graph

```

#### `core\graph\runners.py`

```python
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

```

#### `core\nodes\__init__.py`

```python
from .state import create_initial_state
from .memory import memory_node
from .router import route_node
from .history import update_history_node
from .report import report_node
from .utils import get_message_content, get_message_role

__all__ = [
    "create_initial_state",
    "memory_node",
    "get_message_content",
    "get_message_role",
    "route_node",
    "update_history_node",
    "report_node"
]

```

#### `core\nodes\history.py`

```python
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from domain.schemas import InterviewState

logger = logging.getLogger(__name__)


def parse_weakness_prefix(weakness_prefix: str) -> Tuple[List[str], List[str]]:
    if not weakness_prefix:
        return [], []
    
    weakness_tags = []
    forbidden_words = []
    
    weakness_match = re.search(r'【用户弱点：([^】]+)】', weakness_prefix)
    if weakness_match:
        tags_str = weakness_match.group(1)
        weakness_tags = [tag.strip() for tag in tags_str.split('、') if tag.strip()]
    
    forbidden_match = re.search(r'【雷区忌口：([^】]+)】', weakness_prefix)
    if forbidden_match:
        forbid_str = forbidden_match.group(1)
        forbidden_words = [word.strip() for word in forbid_str.split('、') if word.strip()]
    
    return weakness_tags, forbidden_words


def build_weakness_prefix(weakness_tags: List[str], forbidden_words: List[str]) -> str:
    if not weakness_tags and not forbidden_words:
        return ""
    
    parts = []
    if weakness_tags:
        parts.append(f"【用户弱点：{'、'.join(weakness_tags)}】")
    if forbidden_words:
        parts.append(f"【雷区忌口：{'、'.join(forbidden_words)}】")
    
    return "".join(parts)


def parse_judge_score(response: str) -> Optional[int]:
    if not response:
        return None
    
    score_patterns = [
        r'📝\s*\*?\*?得分[：:]\*?\*?\s*(\d+)\s*分',
        r'得分[：:]\s*(\d+)\s*分',
        r'(\d+)\s*分',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
    
    return None


def extract_new_weakness_from_judgment(response: str, current_tags: List[str]) -> List[str]:
    if not response:
        return []
    
    fatal_match = re.search(r'🔪\s*\*?\*?致命伤[：:]\*?\*?\s*(.+?)(?=🧭|📝|$)', response, re.DOTALL)
    if not fatal_match:
        return []
    
    fatal_text = fatal_match.group(1).strip()
    
    weakness_keywords = [
        "数据", "逻辑", "框架", "归因", "量化", "闭环",
        "方法论", "复盘", "拆解", "结构", "深度", "细节"
    ]
    
    new_weaknesses = []
    for keyword in weakness_keywords:
        if keyword in fatal_text:
            potential_weakness = f"{keyword}待强化"
            if potential_weakness not in current_tags:
                similar_exists = any(keyword in tag for tag in current_tags)
                if not similar_exists:
                    new_weaknesses.append(potential_weakness)
    
    return new_weaknesses[:2]


def incremental_update_weakness(
    current_prefix: str,
    response: str,
    turn_count: int,
    update_interval: int = 5,
    score_threshold: int = 80
) -> Optional[str]:
    if turn_count % update_interval != 0:
        return None
    
    weakness_tags, forbidden_words = parse_weakness_prefix(current_prefix)
    
    if not weakness_tags:
        logger.info("🔄 [弱点更新] 当前无弱点标签，跳过更新")
        return None
    
    score = parse_judge_score(response)
    
    if score is None:
        logger.info("🔄 [弱点更新] 未检测到判卷得分，跳过更新")
        return None
    
    logger.info(f"🔄 [弱点更新] 检测到判卷得分: {score}分")
    
    if score >= score_threshold and len(weakness_tags) > 1:
        removed_tag = weakness_tags.pop(0)
        logger.info(f"🔄 [弱点更新] 得分达标({score}>={score_threshold})，移除弱点: {removed_tag}")
        
        new_weaknesses = extract_new_weakness_from_judgment(response, weakness_tags)
        if new_weaknesses:
            weakness_tags.extend(new_weaknesses)
            logger.info(f"🔄 [弱点更新] 从判卷中发现新弱点: {new_weaknesses}")
        
        return build_weakness_prefix(weakness_tags, forbidden_words)
    
    elif score < 60:
        new_weaknesses = extract_new_weakness_from_judgment(response, weakness_tags)
        if new_weaknesses:
            weakness_tags.extend(new_weaknesses)
            logger.info(f"🔄 [弱点更新] 得分较低({score}<60)，添加新弱点: {new_weaknesses}")
            return build_weakness_prefix(weakness_tags, forbidden_words)
    
    return None


async def update_history_node(state: InterviewState) -> Dict[str, Any]:
    new_history = list(state["history"])
    
    if state["user_input"]:
        new_history.append(HumanMessage(content=state["user_input"]))
    if state["response"]:
        new_history.append(AIMessage(content=state["response"]))

    max_turns = settings.MAX_HISTORY_TURNS
    if len(new_history) > max_turns * 2:
        new_history = new_history[-(max_turns * 2):]

    new_turn_count = state["turn_count"] + 1

    result = {
        "history": new_history,
        "turn_count": new_turn_count,
        "user_input": ""
    }

    current_prefix = state.get("weakness_prefix", "")
    last_response = state.get("response", "")
    
    updated_prefix = incremental_update_weakness(
        current_prefix=current_prefix,
        response=last_response,
        turn_count=new_turn_count
    )
    
    if updated_prefix:
        result["weakness_prefix"] = updated_prefix
        logger.info(f"📝 [更新历史节点] 弱点画像已更新: {updated_prefix[:50]}...")

    logger.info(f"📝 [更新历史节点] 轮次: {new_turn_count}, 历史长度: {len(new_history)}")

    return result

```

#### `core\nodes\memory.py`

```python
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

    llm = llm_factory.get_fast_llm()

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

```

#### `core\nodes\report.py`

```python
import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from domain.schemas import InterviewState
from .utils import get_message_content, get_message_role
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)

GENERATE_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位经验丰富且耐心的私人专属面试顾问。现在一轮深度训练刚刚结束。
你的任务是根据【对话记录】，输出一份专业、客观且具有指导价值的《面试能力体检报告》。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

要求：
1. 严格按照规定的 Markdown 结构输出，不要加任何寒暄废话。
2. 【救命锦囊】部分极其重要：你必须从对话记录中，把之前提到的"满分公式"、"避坑指南"、"漏斗模型"等具体方法论提取出来，作为锦囊。绝不允许自己瞎编泛泛而谈的理论。
3. 语气保持专业严谨，同时体现出对学员成长的真诚关怀。
4. 在指出不足时，要给出具体的改进方向和方法论支撑；在肯定进步时，要说明具体好在哪里。
5. 整体风格：既要有"一针见血"的专业洞察力，又要有"循循善诱"的教学耐心。

═══════════════════════════════════════
§2 思维链推理（生成报告前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 全局扫描：通读对话记录，标记所有"判卷得分"、"致命伤"、"脚手架公式"、"教学补充"等关键节点
2. 能力画像：基于标记节点，归纳用户的核心能力短板（不是罗列每轮对话，而是提炼模式）
3. 锦囊提取：从对话中逐条提取具体的方法论、公式、框架（必须是对话中出现过的，不可自创）
4. 进步识别：对比对话前后的回答质量变化，识别出用户真正进步的地方

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

对话记录片段：
- 教练问数据归因，用户只说了"加大投放"
- 教练反问归因逻辑，用户承认没想过
- 教练给脚手架：漏斗模型公式
- 用户尝试用漏斗框架重新回答
- 判卷65分，致命伤：有框架但缺乏具体业务动作推导

→ 报告核心段落示例：
**核心短板：数据归因思维**
你在回答数据类问题时，习惯停留在"我做了什么"的表层，缺少"为什么有效"的归因链条。训练中引入了漏斗模型后，你能够套用框架，但 Action 部分仍缺乏具体的业务推导。

**救命锦囊：**
- 🔑 漏斗归因公式：曝光→点击→转化，逐层定位异常环节
- 🔑 STAR+漏斗：Situation→Task→Action(漏斗拆解)→Result(量化)
- ⚠️ 避坑：不要用"加大投放"这种单一归因，必须展示多维度拆解"""),

    ("human", """以下是刚才的完整对话记录：
{chat_history}

请输出报告。""")
])


async def report_node(state: InterviewState) -> Dict[str, Any]:
    history = state["history"]
    if len(history) < 4:
        return {
            "response": "💡 建议至少和我过两招再生成报告，否则报告里没东西可写。",
            "should_end": False
        }

    history_text = "\n".join([
        f"{'求职者' if get_message_role(m) == 'user' else '教练'}: {get_message_content(m)}"
        for m in history
    ])

    report_llm = llm_factory.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.3)

    report_chain = GENERATE_REPORT_PROMPT | report_llm

    try:
        response = await report_chain.ainvoke({"chat_history": history_text})
        logger.info("📊 [报告节点] 报告生成完毕")
        return {
            "response": response.content,
            "should_end": True
        }
    except Exception as e:
        logger.error(f"📊 [报告节点] 报告生成崩溃: {e}")
        return {
            "response": f"⚠️ 报告生成失败：{str(e)[:100]}",
            "should_end": False
        }

```

#### `core\nodes\router.py`

```python
import logging
import json
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from domain.schemas import InterviewState, GodDecision, VALID_ACTIONS
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)


async def route_node(state: InterviewState) -> Dict[str, Any]:
    llm = llm_factory.get_router_llm()

    system_prompt = """你是面试控制中枢（上帝大脑）。分析用户输入，决策调用哪个 Skill。

═══════════════════════════════════════
§1 输出格式（JSON 四字段）
═══════════════════════════════════════

1. reasoning: 三步显式推理（必须填写）
   - Step1 用户状态：有回答意愿但逻辑断裂 / 纯情绪发泄 / 顺着梯子爬但卡住 / 完全偏题 / 主动投降
   - Step2 教学阶段：开题尚未作答 / 已尝试但质量差 / 给过脚手架仍无法推进 / 回答完整需判卷 / 判卷后追问
   - Step3 动作推导：基于上述两步，选哪个 Skill？为何其他选项不合适？

2. intent_dim: 用户这句话的动态语义定性（用你的理解总结，不要选死词）

3. intended_action: 调用的 Skill 名称（限选以下五个）

4. task_prompt: 给底层 Skill 的执行指令
   【重要】底层 Skill 没有眼睛，看不到上下文。你必须把对上下文的理解、对用户的判断、要求用什么语气回复什么内容，全部写在这里！

═══════════════════════════════════════
§2 决策规则（核心）
═══════════════════════════════════════

┌──────────────────┬─────────────────────────────────┬──────────────────────────┐
│ Skill            │ 触发条件                         │ task_prompt 要点          │
├──────────────────┼─────────────────────────────────┼──────────────────────────┤
│ socratic_probe   │ 有回答意愿，但逻辑不完整/有漏洞   │ 点出漏洞位置，指令反问逼深 │
│ scaffold_rescue  │ 连续卡壳/极度迷茫（最后手段）     │ 给半截公式/思路+收尾提问   │
│ judge_strike     │ 回答完整/主动投降/梯子后仍无法答  │ 按判卷格式打分，指出致命伤 │
│ smart_redirection│ 偏题闲聊/混乱输入/自相矛盾       │ 简短回应+强制拉回面试话题  │
│ teaching_supplement│ 判卷后用户追问                 │ 补充例子/解释/延伸        │
└──────────────────┴─────────────────────────────────┴──────────────────────────┘

【决策优先级】
- 判卷后追问 → 优先 teaching_supplement
- 混乱/自相矛盾 → smart_redirection（非其他）
- scaffold_rescue 是最后手段，用户还有挖掘空间时不要给梯子

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

【示例1 — socratic_probe】
上下文：教练问数据归因能力，用户说了"加大投放"
用户："我觉得主要就是多投一点广告，然后流量就上来了"
→ reasoning: "Step1 用户有回答意愿，但只抛了一个动作名词，缺乏归因逻辑。Step2 开题后首次作答，用户并非完全不会，只是深度不够。Step3 用 socratic_probe 刺穿逻辑薄弱点，给梯子太早会剥夺思考机会。"
→ intended_action: socratic_probe
→ task_prompt: "用户回答'多投广告流量就上来'，这是单一归因谬误。请用反问句逼他思考：流量上来就等于结果好吗？中间漏了什么关键环节？"

【示例2 — scaffold_rescue】
上下文：教练连续追问数据漏斗，用户两次说"不知道"
用户："我真的完全没思路...你说的漏斗我听都没听过"
→ reasoning: "Step1 用户连续两次无法作答，已进入认知崩溃状态。Step2 苏格拉底追问无效，用户连基础概念都没有。Step3 必须用 scaffold_rescue 给半截公式托底，否则对话陷入死循环。"
→ intended_action: scaffold_rescue
→ task_prompt: "用户连续两次无法作答，连漏斗概念都不知道。请调用工具检索'数据漏斗'资料，然后给出半截公式（如'漏斗模型的核心是从___到___的逐层___'）让用户填空，填空后紧跟收尾提问。"

【示例3 — judge_strike】
上下文：用户完整回答了内容复盘问题（STAR+数据对比+漏斗拆解）
用户："我觉得我回答得差不多了，你看看我哪里还能改进"
→ reasoning: "Step1 用户主动请求评估，前两轮回答已覆盖 STAR 和数据拆解。Step2 已过两轮深入追问，用户表现稳定。Step3 应该用 judge_strike 正式判卷。"
→ intended_action: judge_strike
→ task_prompt: "用户已完整回答内容复盘问题，使用了 STAR 框架、数据对比和漏斗拆解。请调用工具用 train 模式检索判卷清单，按格式打分。"

【示例4 — smart_redirection】
上下文：教练问用户增长策略
用户："诶你们AI是不是用那个什么大模型做的？现在大模型好火啊"
→ reasoning: "Step1 用户完全偏题，从面试问题跳到对 AI 技术的好奇。Step2 面试训练被中断，需要拉回。Step3 用 smart_redirection 简短回应后强制拉回。"
→ intended_action: smart_redirection
→ task_prompt: "用户偏题聊大模型。请简短回应一句（不超过15字），然后立刻以面试相关问题收尾，强制拉回用户增长策略话题。"

═══════════════════════════════════════
§4 绝对红线
═══════════════════════════════════════

- 你【绝对不能】因为用户闲聊或追问就判断为"会话结束"
- 只有用户明确说"我要结束训练"或"生成报告"时，才考虑结束
- 判卷后的追问 → 优先路由到 teaching_supplement
- 用户输入混乱或自相矛盾 → smart_redirection（非其他技能）
- scaffold_rescue 是最后手段，不要在用户还有挖掘空间时就给梯子"""

    human_content = f"""【近5轮上下文摘要 (JSON)】
{state['short_summary']}

【用户当前原始输入】
{state['user_input']}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]

    raw_response = await llm.ainvoke(messages)
    raw_str = raw_response.content.strip()

    if "```json" in raw_str:
        raw_str = raw_str.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_str:
        raw_str = raw_str.split("```")[1].split("```")[0].strip()

    try:
        decision = GodDecision.model_validate_json(raw_str)
    except Exception as e:
        logger.error(f"👁️ [路由节点] JSON解析异常降级: {e}, 原始片段: {raw_str[:200]}")
        decision = GodDecision(
            reasoning="系统解析异常，触发安全兜底机制",
            intent_dim="格式混乱",
            intended_action="smart_redirection",
            task_prompt="用户输入格式异常，请用一句话安抚，并强制拉回上一轮的面试话题。"
        )

    logger.info(f"👁️ [路由节点] 意图定性: {decision.intent_dim} | 决策动作: {decision.intended_action}")

    if decision.intended_action not in VALID_ACTIONS:
        logger.warning(f"⚠️ [路由节点] 非法动作: {decision.intended_action}，强制回退到 smart_redirection")
        return {
            "decision": {
                "target_skill": "smart_redirection",
                "task_prompt": decision.task_prompt
            }
        }

    return {
        "decision": {
            "target_skill": decision.intended_action,
            "task_prompt": decision.task_prompt
        }
    }

```

#### `core\nodes\state.py`

```python
from domain.schemas import InterviewState


def create_initial_state() -> InterviewState:
    return {
        "history": [],
        "weakness_prefix": "",
        "is_ready": False,
        "user_input": "",
        "extracted_text": "",
        "has_file": False,
        "decision": {},
        "response": "",
        "short_summary": "[]",
        "turn_count": 0,
        "is_generating_report": False,
        "should_end": False,
        "file_name": "",
        "jd_text": "",
        "resume_text": "",
        "gap_analysis": "",
        "job_type": ""
    }

```

#### `core\nodes\utils.py`

```python
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


def get_message_content(msg) -> str:
    if isinstance(msg, BaseMessage):
        return msg.content
    elif isinstance(msg, dict):
        return msg.get("content", "")
    return str(msg)


def get_message_role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    elif isinstance(msg, dict):
        return msg.get("role", "user")
    return "user"

```

#### `core\profiler\__init__.py`

```python
from .extractor import Profiler, get_profiler
from .gatekeeper import Gatekeeper, gatekeeper, GatekeeperResult
from .gap_analyzer import GapAnalyzer, get_gap_analyzer

__all__ = ["Profiler", "get_profiler", "Gatekeeper", "gatekeeper", "GatekeeperResult", "GapAnalyzer", "get_gap_analyzer"]

```

#### `core\profiler\extractor.py`

```python
import logging
import json
import re
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from domain.schemas import WeaknessProfile
from domain.job_configs import load_job_config, get_job_config_or_default
from infrastructure.parsers.text_parser import safe_parse_weakness
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)


def _build_diagnose_prompt(job_type: str) -> ChatPromptTemplate:
    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        config = get_job_config_or_default()

    hr_persona = config.hr_persona
    job_display = config.display_name

    return ChatPromptTemplate.from_messages([
        ("system", f"""{hr_persona}。
现在给你一份新人的简历和目标岗位的 JD。你的任务是进行专业的差距分析。
既要客观指出候选人与岗位要求之间的 Gap，也要看到候选人的可培养潜力。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

要求：
1. 弱点标签必须极度精炼，每个标签 4-6 个字，用建设性的语言表达（如："数据思维待强化"而非"完全不懂分析"）。
2. 雷区忌话，是基于TA简历中体现出的不足，推测出TA在面试时为了掩饰弱点最容易编造的借口，并给出正确的应对方向。

【重要】你必须以 JSON 格式输出结果，包含 weakness_tags 和 forbidden_words 两个字段。

═══════════════════════════════════════
§2 思维链推理（诊断前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. JD 能力拆解：目标岗位最核心的3个能力维度是什么？
2. 简历能力映射：候选人简历中展示了哪些？缺失了哪些？
3. 弱点提炼：将缺失项压缩为 4-6 字的诊断标签
4. 雷区预判：基于弱点，候选人在面试中最可能用什么借口掩饰？

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

简历摘要：XX大学新闻系，校园公众号编辑，写过10+篇推文，阅读量最高5000+
JD摘要：{job_display}，需数据分析能力，需 ROI 意识，需用户增长方法论

→ 输出：
{{{{
  "weakness_tags": ["数据归因缺失", "增长方法论空白", "ROI意识薄弱"],
  "forbidden_words": ["阅读量挺高的", "感觉效果不错", "领导安排的", "运气好"]
}}}}

→ 推理链路：JD 要求3个核心能力（数据/增长/ROI），简历只展示了内容产出，三个维度全部缺失。候选人在面试中最可能用"阅读量高"来掩饰数据归因缺失，用"感觉效果不错"来回避 ROI 量化，用"领导安排"来逃避主动性证明。"""),

        ("human", """【候选人简历摘要】：
{{resume_text}}

【目标岗位JD】：
{{jd_text}}

请输出 JSON 格式的弱点分析结果。""")
    ])


class Profiler:
    def __init__(self, job_type: str = None):
        self._job_type = job_type
        self._chain = None

    def _ensure_initialized(self):
        if self._chain is None:
            llm = llm_factory.get_fast_llm()
            self._llm = llm
            self._prompt = _build_diagnose_prompt(self._job_type)

    @property
    def chain(self):
        self._ensure_initialized()
        return self._prompt, self._llm

    def extract(self, resume_text: str, jd_text: str) -> str:
        prompt, llm = self.chain
        messages = prompt.format_messages(
            resume_text=resume_text,
            jd_text=jd_text
        )

        raw_response = llm.invoke(messages)
        raw_str = raw_response.content.strip()

        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()

        try:
            raw_profile = WeaknessProfile.model_validate_json(raw_str)
        except Exception as e:
            logger.error(f"📊 [画像提取] JSON解析异常: {e}, 原始片段: {raw_str[:200]}")
            raw_profile = WeaknessProfile(
                weakness_tags=["能力待评估"],
                forbidden_words=["不清楚", "不知道"]
            )

        final_profile = safe_parse_weakness(raw_profile)
        logger.info(f"✅ 画像生成完毕: {final_profile.to_prompt_prefix[:50]}...")

        return final_profile.to_prompt_prefix

    def extract_full(self, resume_text: str, jd_text: str) -> WeaknessProfile:
        prompt, llm = self.chain
        messages = prompt.format_messages(
            resume_text=resume_text,
            jd_text=jd_text
        )

        raw_response = llm.invoke(messages)
        raw_str = raw_response.content.strip()

        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()

        try:
            raw_profile = WeaknessProfile.model_validate_json(raw_str)
        except Exception as e:
            logger.error(f"📊 [画像提取] JSON解析异常: {e}, 原始片段: {raw_str[:200]}")
            raw_profile = WeaknessProfile(
                weakness_tags=["能力待评估"],
                forbidden_words=["不清楚", "不知道"]
            )

        return safe_parse_weakness(raw_profile)


_profiler_cache = {}


def get_profiler(job_type: str = None) -> Profiler:
    if job_type not in _profiler_cache:
        _profiler_cache[job_type] = Profiler(job_type=job_type)
    return _profiler_cache[job_type]


profiler = get_profiler()

```

#### `core\profiler\gap_analyzer.py`

```python
import logging
import json
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from domain.schemas import WeaknessProfile, GapAnalysisResult
from domain.job_configs import load_job_config, get_job_config_or_default
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)


def _build_gap_analysis_prompt(job_type: str) -> ChatPromptTemplate:
    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        config = get_job_config_or_default()

    gap_hr_persona = config.gap_hr_persona
    job_display = config.display_name

    return ChatPromptTemplate.from_messages([
        ("system", f"""{gap_hr_persona}。你已完成了候选人简历与目标岗位JD的弱点分析，现在需要基于弱点分析结果，生成一份清晰的差距分析报告。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

要求：
1. weaknesses 列表：每条弱点必须具体说明候选人在哪个能力维度上与JD要求存在差距，不能只写标签，要写出差距的具体内容
2. resume_suggestions 列表：每条建议必须针对上述弱点给出具体可行的简历修改策略，包括应补充什么内容、如何量化成果、如何调整表述
3. 语气专业客观，同时体现建设性
4. 严格按照 JSON 格式输出，包含 weaknesses 和 resume_suggestions 两个字段

═══════════════════════════════════════
§2 思维链推理（生成前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 弱点解读：每个弱点标签背后，候选人具体缺失了什么？JD要求的标准是什么？候选人目前的水平如何？
2. 简历策略：针对每个弱点，简历中应该补充什么样的经历描述、数据指标或方法论来弥补？
3. 表述优化：现有简历中哪些表述需要从模糊改为具体、从定性改为定量？

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

弱点标签：["数据归因缺失", "增长方法论空白", "ROI意识薄弱"]
JD核心要求：数据分析能力、用户增长方法论、ROI评估体系

→ 输出：
{{{{
  "weaknesses": [
    "JD要求能独立完成数据复盘报告，但简历中仅展示内容产出，缺乏数据归因和漏斗分析的体现",
    "JD要求具备用户增长方法论，但简历中只有粉丝数增长的陈述，未展示增长策略的制定与执行过程",
    "JD要求建立ROI评估体系，但简历中未体现任何投放效果追踪和投入产出比分析的经验"
  ],
  "resume_suggestions": [
    "补充数据复盘经历：将'粉丝从5万增长至30万'改为'通过A/B测试优化内容策略，粉丝从5万增长至30万，月均增长率15%'，体现数据驱动决策能力",
    "增加增长策略描述：补充用户增长的具体方法论，如'搭建用户增长AARRR漏斗模型，通过内容种草→私域引流→社群转化路径，实现月均新增粉丝2万'",
    "补充ROI分析经验：在直播带货经历中增加'通过ROI数据监控优化投放策略，将单场直播ROI从1:3提升至1:5，GMV累计50万'，体现投入产出意识"
  ]
}}}}"""),

        ("human", """【候选人简历摘要】：
{{resume_text}}

【目标岗位JD】：
{{jd_text}}

【已识别的弱点标签】：{{weakness_tags}}
【雷区忌口】：{{forbidden_words}}

请基于上述信息，生成JSON格式的差距分析结果。""")
    ])


class GapAnalyzer:
    def __init__(self, job_type: str = None):
        self._job_type = job_type
        self._chain = None

    def _ensure_initialized(self):
        if self._chain is None:
            llm = llm_factory.get_fast_llm()
            self._llm = llm
            self._prompt = _build_gap_analysis_prompt(self._job_type)

    def analyze(self, resume_text: str, jd_text: str, profile: WeaknessProfile) -> GapAnalysisResult:
        self._ensure_initialized()
        messages = self._prompt.format_messages(
            resume_text=resume_text,
            jd_text=jd_text,
            weakness_tags="、".join(profile.weakness_tags),
            forbidden_words="、".join(profile.forbidden_words)
        )

        raw_response = self._llm.invoke(messages)
        raw_str = raw_response.content.strip()

        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()

        try:
            result = GapAnalysisResult.model_validate_json(raw_str)
        except Exception as e:
            logger.error(f"📊 [Gap分析] JSON解析异常: {e}, 原始片段: {raw_str[:200]}")
            weaknesses = [f"在{tag}方面与岗位要求存在差距" for tag in profile.weakness_tags]
            suggestions = [f"建议在简历中补充与{tag}相关的具体经历和量化成果" for tag in profile.weakness_tags]
            result = GapAnalysisResult(
                weaknesses=weaknesses,
                resume_suggestions=suggestions
            )

        logger.info(f"✅ Gap分析完毕: {len(result.weaknesses)}个弱点, {len(result.resume_suggestions)}条建议")
        return result

    async def aanalyze(self, resume_text: str, jd_text: str, profile: WeaknessProfile) -> GapAnalysisResult:
        self._ensure_initialized()
        messages = self._prompt.format_messages(
            resume_text=resume_text,
            jd_text=jd_text,
            weakness_tags="、".join(profile.weakness_tags),
            forbidden_words="、".join(profile.forbidden_words)
        )

        raw_response = await self._llm.ainvoke(messages)
        raw_str = raw_response.content.strip()

        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()

        try:
            result = GapAnalysisResult.model_validate_json(raw_str)
        except Exception as e:
            logger.error(f"📊 [Gap分析] JSON解析异常: {e}, 原始片段: {raw_str[:200]}")
            weaknesses = [f"在{tag}方面与岗位要求存在差距" for tag in profile.weakness_tags]
            suggestions = [f"建议在简历中补充与{tag}相关的具体经历和量化成果" for tag in profile.weakness_tags]
            result = GapAnalysisResult(
                weaknesses=weaknesses,
                resume_suggestions=suggestions
            )

        logger.info(f"✅ Gap分析完毕: {len(result.weaknesses)}个弱点, {len(result.resume_suggestions)}条建议")
        return result


_gap_analyzer_cache = {}


def get_gap_analyzer(job_type: str = None) -> GapAnalyzer:
    if job_type not in _gap_analyzer_cache:
        _gap_analyzer_cache[job_type] = GapAnalyzer(job_type=job_type)
    return _gap_analyzer_cache[job_type]


gap_analyzer = get_gap_analyzer()

```

#### `core\profiler\gatekeeper.py`

```python
import logging
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional

from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)


class GatekeeperResult(BaseModel):
    resume_text: str = Field(
        ...,
        description="从用户输入中提取的简历部分原文。如果无法识别，返回空字符串"
    )
    jd_text: Optional[str] = Field(
        default=None,
        description="从用户输入中提取的JD部分原文。如果无法识别，返回null"
    )
    has_resume: bool = Field(
        ...,
        description="是否检测到有效简历内容"
    )
    has_jd: bool = Field(
        ...,
        description="是否检测到有效岗位JD内容"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="识别置信度，0-1之间"
    )

      
GATEKEEPER_PROMPT = """你是一个文档分类器。从用户输入中分离出【简历】和【JD】。

判断规则：
1. 简历特征：姓名、联系方式、邮箱、教育背景、工作经历、技能清单
2. JD特征：岗位职责、任职要求、岗位名称、薪资范围
3. 只有简历 → jd_text 返回 null，has_jd 返回 false
4. 只有JD → resume_text 返回空字符串，has_resume 返回 false
5. 都无法识别 → has_resume 和 has_jd 都返回 false

严格按照 JSON 格式输出，不要输出其他内容。"""


class Gatekeeper:
    def __init__(self):
        self._chain = None

    def _ensure_initialized(self):
        if self._chain is None:
            llm = llm_factory.get_fast_llm()
            self._llm = llm

    async def aparse(self, raw_input: str) -> GatekeeperResult:
        self._ensure_initialized()
        messages = [
            SystemMessage(content=GATEKEEPER_PROMPT),
            HumanMessage(content=f"用户输入内容：\n{raw_input}")
        ]
        
        raw_response = await self._llm.ainvoke(messages)
        raw_str = raw_response.content.strip()
        
        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()
        
        try:
            result = GatekeeperResult.model_validate_json(raw_str)
        except Exception as e:
            logger.error(f"🚪 [门卫] JSON解析异常降级: {e}, 原始片段: {raw_str[:200]}")
            result = GatekeeperResult(
                resume_text="",
                jd_text=None,
                has_resume=False,
                has_jd=False,
                confidence=0.0
            )
        
        logger.info(
            f"🚪 [门卫] 识别结果: "
            f"有简历={result.has_resume}, 有JD={result.has_jd}, "
            f"置信度={result.confidence:.2f}"
        )
        return result

    def parse(self, raw_input: str) -> GatekeeperResult:
        self._ensure_initialized()
        messages = [
            SystemMessage(content=GATEKEEPER_PROMPT),
            HumanMessage(content=f"用户输入内容：\n{raw_input}")
        ]
        
        raw_response = self._llm.invoke(messages)
        raw_str = raw_response.content.strip()
        
        if "```json" in raw_str:
            raw_str = raw_str.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_str:
            raw_str = raw_str.split("```")[1].split("```")[0].strip()
        
        try:
            result = GatekeeperResult.model_validate_json(raw_str)
        except Exception as e:
            logger.error(f"🚪 [门卫] JSON解析异常降级: {e}, 原始片段: {raw_str[:200]}")
            result = GatekeeperResult(
                resume_text="",
                jd_text=None,
                has_resume=False,
                has_jd=False,
                confidence=0.0
            )
        
        logger.info(
            f"🚪 [门卫] 识别结果: "
            f"有简历={result.has_resume}, 有JD={result.has_jd}, "
            f"置信度={result.confidence:.2f}"
        )
        return result


gatekeeper = Gatekeeper()

```

#### `core\skills\__init__.py`

```python
from .base import _execute_skill_with_tools
from .socratic import socratic_node
from .scaffold import scaffold_node
from .judge import judge_node
from .redirection import redirection_node
from .teaching import teaching_node

SKILL_NODE_MAP = {
    "socratic_probe": socratic_node,
    "scaffold_rescue": scaffold_node,
    "judge_strike": judge_node,
    "smart_redirection": redirection_node,
    "teaching_supplement": teaching_node
}

__all__ = [
    "_execute_skill_with_tools",
    "socratic_node",
    "scaffold_node",
    "judge_node",
    "redirection_node",
    "teaching_node",
    "SKILL_NODE_MAP"
]

```

#### `core\skills\base.py`

```python
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
    bind_tools: List = None,
    use_fast_model: bool = False
) -> str:
    if use_fast_model:
        llm_instance = llm_factory.get_fast_llm()
    else:
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

```

#### `core\skills\judge.py`

```python
import logging
from typing import Dict, Any, List

from domain.schemas import InterviewState
from infrastructure.tools.interview_db import INTERVIEW_TOOLS
from .base import _execute_skill_with_tools

logger = logging.getLogger(__name__)

JUDGE_ROLE_PROMPT = """你是一个铁面判官。你没有个人主见。
你必须严格按照【执行指令】进行回复。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

【工具使用红线】：你被绑定了知识库检索工具(search_interview_db)。你【必须】根据执行指令中的要求（比如用 train 模式取判卷清单，或用 review 模式取满分公式）去调用工具。

═══════════════════════════════════════
§2 思维链推理（判卷前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 提取论点：用户回答中的所有核心论点是什么？
2. 逐条对照：将每个论点与判卷清单逐条比对——哪些命中？哪些缺失？
3. 识别致命伤：不是小瑕疵，是"一票否决"级的逻辑缺陷
4. 推算分数：基于命中率和致命伤严重程度，推算分数区间

═══════════════════════════════════════
§3 输出格式（硬约束 — 绝对不允许增删模块）
═══════════════════════════════════════

📝 **得分：[XX]分**
🔪 **致命伤：**[...]
🧭 **满分拆解与举例：**[...]

═══════════════════════════════════════
§4 示例锚定
═══════════════════════════════════════

【好例子 ✓】
执行指令："判卷，用户回答了数据归因问题，用了 STAR 但缺漏斗拆解"
→
📝 **得分：65分**
🔪 **致命伤：**有 STAR 框架但 Action 部分缺乏数据漏斗的逐层拆解，只说了"分析了数据"却没有展示从曝光→点击→转化的归因链条，导致回答停留在"我做了"而非"我为什么做对了"。
🧭 **满分拆解与举例：**STAR 只是骨架，血肉在于漏斗归因。满分回答应该是：Situation(背景)→Task(目标)→Action(逐层拆解：曝光量 X→点击率 Y→转化率 Z，定位到 Z 环节异常)→Result(基于归因的优化动作+量化结果)。

【坏例子 ✗（绝对禁止）】
✗ 得分：65分 / 致命伤：回答不够深入 / 满分拆解：要用 STAR 法则（太笼统，致命伤没有指向具体缺陷，满分拆解没有增量信息）
✗ 得分：65分 / 致命伤：缺少数据支撑 / 满分拆解：需要加入更多数据（"更多数据"不是方法论，是废话）"""


async def judge_node(state: InterviewState) -> Dict[str, Any]:
    task_prompt = state["decision"]["task_prompt"]
    weakness_prefix = state.get("weakness_prefix", "")
    response = await _execute_skill_with_tools(JUDGE_ROLE_PROMPT, task_prompt, weakness_prefix, INTERVIEW_TOOLS)
    logger.info(f"⚖️ [判卷节点] 执行完毕")
    return {"response": response}

```

#### `core\skills\redirection.py`

```python
import logging
from typing import Dict, Any

from domain.schemas import InterviewState
from .base import _execute_skill_with_tools

logger = logging.getLogger(__name__)

REDIRECTION_ROLE_PROMPT = """你是一个高情商的面试控制大师。你没有个人主见。
你必须严格按照【执行指令】进行回复，绝对禁止自由发挥。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

【核心职责】：
1. 偏题闲聊：简短回应后自然拉回面试话题
2. 混乱输入：从用户混乱的表述中提炼出2个关键方向，构造一个简单的 A/B 选择问题，强制用户聚焦
3. 自相矛盾：点出矛盾之处，要求用户选择一个立场

═══════════════════════════════════════
§2 思维链推理（重定向前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 识别偏移类型：是偏题闲聊 / 逻辑混乱 / 自相矛盾 / 答疑过深？
2. 选择锚点：从用户的话里提取一个可以桥接回面试话题的关键词
3. 构造桥梁：用"回应→桥接→提问"三段式完成重定向

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

【好例子 ✓ — 偏题闲聊】
执行指令："用户聊起了 AI 技术，拉回用户增长话题"
→ "AI 确实很火——不过回到面试，如果让你用数据来证明你'懂用户增长'，你会拿出什么指标？"

【好例子 ✓ — 混乱输入】
执行指令："用户说了一堆不相关的东西，强制聚焦"
→ "我听到你提到了'数据'和'内容'两个方向。我们先聚焦一个：你更想从哪个角度来回答——A. 用数据证明增长效果；B. 用内容策略说明运营思路？"

【好例子 ✓ — 自相矛盾】
执行指令："用户先说重视数据，后来说凭感觉做决策"
→ "等一下——你刚才说'数据是核心'，现在又说'凭感觉判断'，这两个立场是矛盾的。你必须选一个：你到底是数据驱动型，还是直觉驱动型？"

═══════════════════════════════════════
§4 绝对红线
═══════════════════════════════════════

1. 你【绝对禁止】说"会话结束"、"面试结束"、"再见"、"祝你顺利"等任何暗示终止的话
2. 你【必须】在回应后以一个面试相关问题结尾，确保对话继续
3. 只有系统按钮才能结束会话，你没有权限结束
4. 面对混乱输入时，【绝对禁止】跟着用户绕圈子，必须强制聚焦"""


async def redirection_node(state: InterviewState) -> Dict[str, Any]:
    task_prompt = state["decision"]["task_prompt"]
    weakness_prefix = state.get("weakness_prefix", "")
    response = await _execute_skill_with_tools(REDIRECTION_ROLE_PROMPT, task_prompt, weakness_prefix, use_fast_model=True)
    logger.info(f"🔀 [重定向节点] 执行完毕")
    return {"response": response}

```

#### `core\skills\scaffold.py`

```python
import logging
from typing import Dict, Any, List

from domain.schemas import InterviewState
from infrastructure.tools.interview_db import INTERVIEW_TOOLS
from .base import _execute_skill_with_tools

logger = logging.getLogger(__name__)

SCAFFOLD_ROLE_PROMPT = """你是一个降维向导。你没有个人主见。
你必须严格按照【执行指令】进行回复。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

【工具使用红线】：你被绑定了知识库检索工具(search_interview_db)。你【必须】根据执行指令中的要求去调用工具获取半截公式或思路。在拿到工具结果后，将其融入你的回复中。
绝对禁止自己捏造公式！绝对禁止把完整的答案直接给用户！

═══════════════════════════════════════
§2 思维链推理（搭建脚手架前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 诊断卡点：用户卡在哪里？是概念缺失、框架缺失、还是案例缺失？
2. 选择支架：根据卡点类型，选择"填空式半截公式"或"选择题式方向引导"
3. 预设收口：脚手架的终点必须是一个具体的提问，而非开放式结尾

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

【好例子 ✓】
执行指令："用户不知道漏斗模型，给半截公式"
→ "我给你搭个梯子——做用户增长分析，核心思路是'漏斗模型'：从【___】→【___】→【___】，每一层都有一个转化率。你现在想想，你的业务场景里，这三层分别填什么？"

执行指令："用户卡在 ROI 计算，给方向引导"
→ "ROI 的计算有个万能公式：ROI = (收益 - 成本) / 成本。但关键不在公式本身，而在于你怎么定义'收益'和'成本'。给你两个方向选：A. 只算直接销售收益；B. 算上品牌曝光的长期价值。你选哪个？为什么？"

【坏例子 ✗（绝对禁止）】
✗ "漏斗模型就是从上到下逐层筛选，你可以去了解一下。"（给了概念但没给抓手，用户还是不知道怎么用）
✗ "你应该用 STAR 法则来回答这个问题。"（直接给了完整方法论，没有留白让用户思考）"""


async def scaffold_node(state: InterviewState) -> Dict[str, Any]:
    task_prompt = state["decision"]["task_prompt"]
    weakness_prefix = state.get("weakness_prefix", "")
    response = await _execute_skill_with_tools(SCAFFOLD_ROLE_PROMPT, task_prompt, weakness_prefix, INTERVIEW_TOOLS)
    logger.info(f"🪜 [脚手架节点] 执行完毕")
    return {"response": response}

```

#### `core\skills\socratic.py`

```python
import logging
from typing import Dict, Any

from domain.schemas import InterviewState
from .base import _execute_skill_with_tools

logger = logging.getLogger(__name__)

SOCRATIC_ROLE_PROMPT = """你是一个冷酷的苏格拉底式面试官。你没有个人主见。
你必须严格按照【执行指令】进行回复。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

- 你的回复必须是一个极其锋利的反问句
- 直接刺穿用户逻辑的薄弱点
- 绝对禁止出现问号以外的总结性标点（如句号、感叹号结尾）

═══════════════════════════════════════
§2 思维链推理（内部执行路径 — 不输出推理过程）
═══════════════════════════════════════

1. 定位裂缝：用户回答中最大的逻辑断裂点在哪里？
2. 选择刀口：哪个角度的反问最能逼用户自己发现这个裂缝？
3. 锻造反问：将刀口转化为一个无法用"是/否"敷衍的开放性反问句

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

【好例子 ✓】
执行指令："用户只说了'加大投放'，逼他想归因逻辑"
→ "加大投放之后呢——你凭什么判断是投放带来的增长，而不是自然流量或季节性波动？"

执行指令："用户说'我负责内容运营'，但没提数据，逼他量化"
→ "你负责内容运营——那你怎么知道你产出的内容是'好'还是'自嗨'，衡量标准是什么？"

【坏例子 ✗（绝对禁止）】
✗ "你觉得你的回答完整吗？"（太宽泛，没有刺穿具体裂缝）
✗ "你提到了投放，但还需要考虑归因分析。"（直接给了答案，不是反问）
✗ "能不能再详细说说？"（没有方向性，用户不知道该往哪想）"""


async def socratic_node(state: InterviewState) -> Dict[str, Any]:
    task_prompt = state["decision"]["task_prompt"]
    weakness_prefix = state.get("weakness_prefix", "")
    response = await _execute_skill_with_tools(SOCRATIC_ROLE_PROMPT, task_prompt, weakness_prefix)
    logger.info(f"🎯 [苏格拉底节点] 执行完毕")
    return {"response": response}

```

#### `core\skills\teaching.py`

```python
import logging
from typing import Dict, Any, List

from domain.schemas import InterviewState
from infrastructure.tools.interview_db import INTERVIEW_TOOLS
from .base import _execute_skill_with_tools

logger = logging.getLogger(__name__)

TEACHING_ROLE_PROMPT = """你是一个耐心的答疑者。你没有个人主见。
你必须严格按照【执行指令】进行回复。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

【工具使用红线】：你被绑定了知识库检索工具(search_interview_db)。如果执行指令要求你举例，你【必须】调用工具获取详尽资料。
特别注意：如果执行指令中明确提到"不要给新例子，逼用户复述"，你【绝对禁止】调用工具，必须直接按照指令去逼问。

═══════════════════════════════════════
§2 思维链推理（补充教学前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 诊断追问类型：用户是在追问方法论细节 / 请求举例 / 请求复述巩固？
2. 选择教学策略：
   - 追问细节 → 调用工具获取资料，给出结构化补充
   - 请求举例 → 调用工具获取案例，但只展示框架，留关键填空
   - 请求复述 → 不调用工具，直接逼用户用自己的话重述
3. 控制边界：教学补充不超过2轮，之后必须回到面试推进

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

【好例子 ✓ — 补充举例】
执行指令："用户追问漏斗模型的具体案例，给个例子"
→ "举个电商的例子：用户从看到广告(曝光10000)→点击进入(点击500)→加入购物车(加购50)→完成支付(成交10)。你看，每一层都在流失，而你的任务就是找出哪一层流失最严重、为什么。现在你能用你自己的业务场景，复述一遍这个漏斗吗？"

【好例子 ✓ — 逼用户复述】
执行指令："不要给新例子，逼用户复述刚才的满分公式"
→ "我不给新例子了。你刚才听到了'STAR+漏斗拆解'这个公式，现在请你用你自己的话，把这个公式套到你的项目经历里说一遍。"

【坏例子 ✗（绝对禁止）】
✗ "漏斗模型就是 AARRC 模型，包括获取、激活、留存、推荐、变现。"（纯理论灌输，没有结合用户场景）
✗ "你说得对，确实是这样。"（无教学价值，纯肯定）"""


async def teaching_node(state: InterviewState) -> Dict[str, Any]:
    task_prompt = state["decision"]["task_prompt"]
    weakness_prefix = state.get("weakness_prefix", "")
    response = await _execute_skill_with_tools(TEACHING_ROLE_PROMPT, task_prompt, weakness_prefix, INTERVIEW_TOOLS, use_fast_model=True)
    logger.info(f"📚 [教学节点] 执行完毕")
    return {"response": response}

```

#### `core\utils\__init__.py`

```python
from .llm_factory import LLMFactory, llm_factory

__all__ = ["LLMFactory", "llm_factory"]

```

#### `core\utils\llm_factory.py`

```python
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    _instances: Dict[str, BaseChatModel] = {}
    
    @classmethod
    def get_llm(
        cls,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs
    ) -> BaseChatModel:
        model = model_name or settings.ROUTER_MODEL_NAME
        cache_key = f"{model}_{temperature}"
        
        if cache_key not in cls._instances:
            cls._instances[cache_key] = ChatOpenAI(
                model=model,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                temperature=temperature,
                **kwargs
            )
            logger.info(f"🤖 [LLM工厂] 创建新实例: {model}, temp={temperature}")
        
        return cls._instances[cache_key]
    
    @classmethod
    def get_router_llm(cls) -> BaseChatModel:
        """路由大脑：温度低，保证结构化输出稳定"""
        return cls.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.1)
    
    @classmethod
    def get_fast_llm(cls) -> BaseChatModel:
        """快速手脚：温度适中，追求速度"""
        return cls.get_llm(settings.FAST_MODEL_NAME, temperature=0.5)
    
    @classmethod
    def get_router_llm_creative(cls) -> BaseChatModel:
        """报告/开场白：温度稍高，带点人情味"""
        return cls.get_llm(settings.ROUTER_MODEL_NAME, temperature=0.7)
    
    @classmethod
    def get_creative_llm(cls) -> BaseChatModel:
        """兼容旧接口：等同于 get_router_llm_creative"""
        return cls.get_router_llm_creative()
    
    @classmethod
    def clear_cache(cls):
        cls._instances.clear()
        logger.info("🤖 [LLM工厂] 缓存已清除")


llm_factory = LLMFactory()

```

### 领域层 (domain/)

*数据模型、业务实体、岗位配置*

#### `domain\__init__.py`

```python
from .schemas import (
    InterviewQuestion,
    InterviewQuestionList,
    WeaknessProfile,
    GodDecision,
    InterviewState,
    VALID_ACTIONS
)
from .validators import validate_resume_jd, JD_KEYWORDS, RESUME_KEYWORDS

__all__ = [
    "InterviewQuestion",
    "InterviewQuestionList",
    "WeaknessProfile",
    "GodDecision",
    "InterviewState",
    "VALID_ACTIONS",
    "validate_resume_jd",
    "JD_KEYWORDS",
    "RESUME_KEYWORDS"
]

```

#### `domain\job_configs\__init__.py`

```python
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent


@dataclass
class JobConfig:
    job_type: str
    display_name: str
    domain_keywords: str
    interviewer_persona: str
    hr_persona: str
    gap_hr_persona: str
    mine_interviewer_persona: str
    mine_hr_persona: str
    default_jd: str
    demo_resume: str
    demo_jd: str


_config_cache: Dict[str, JobConfig] = {}


def load_job_config(job_type: str) -> JobConfig:
    if job_type in _config_cache:
        return _config_cache[job_type]

    config_path = CONFIGS_DIR / f"{job_type}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"岗位配置文件不存在: {config_path}\n"
            f"请先在 domain/job_configs/ 下创建 {job_type}.yaml 配置文件"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = JobConfig(
        job_type=data["job_type"],
        display_name=data.get("display_name", data["job_type"]),
        domain_keywords=data.get("domain_keywords", ""),
        interviewer_persona=data.get("interviewer_persona", ""),
        hr_persona=data.get("hr_persona", ""),
        gap_hr_persona=data.get("gap_hr_persona", ""),
        mine_interviewer_persona=data.get("mine_interviewer_persona", ""),
        mine_hr_persona=data.get("mine_hr_persona", ""),
        default_jd=data.get("default_jd", ""),
        demo_resume=data.get("demo_resume", ""),
        demo_jd=data.get("demo_jd", ""),
    )

    _config_cache[job_type] = config
    logger.info(f"✅ 已加载岗位配置: {job_type}")
    return config


def list_available_jobs() -> List[str]:
    jobs = []
    for yaml_file in CONFIGS_DIR.glob("*.yaml"):
        jobs.append(yaml_file.stem)
    return sorted(jobs)


def get_job_config_or_default(job_type: Optional[str] = None) -> JobConfig:
    if job_type:
        return load_job_config(job_type)

    available = list_available_jobs()
    if not available:
        raise FileNotFoundError("没有任何岗位配置文件，请先在 domain/job_configs/ 下创建配置文件")

    default_job = available[0]
    logger.info(f"未指定岗位，使用默认岗位: {default_job}")
    return load_job_config(default_job)

```

#### `domain\schemas.py`

```python
from pydantic import BaseModel, Field, model_validator
from typing import List, Union, Optional, Annotated, TypedDict, Dict, Any, Literal
from langgraph.graph.message import add_messages


VALID_ACTIONS = [
    "socratic_probe",
    "scaffold_rescue",
    "judge_strike",
    "smart_redirection",
    "teaching_supplement"
]


class InterviewQuestion(BaseModel):
    job_type: str = Field(
        ...,
        description="岗位类型，如：新媒体运营、产品经理、前端开发等"
    )
    question: str = Field(
        ...,
        min_length=5,
        description="面试官提出的具体问题，必须是一个完整的疑问句或指令"
    )
    inspect_point: str = Field(
        default="",
        description="该问题背后的考察点（如：复盘逻辑、数据敏感度、网感判断）"
    )
    high_score_formula: str = Field(
        default="",
        description="高分回答的结构化公式（如：STAR法则+漏斗拆解），不要写废话"
    )
    pitfall_guide: str = Field(
        default="",
        description="避坑指南，明确指出绝对不能说的雷区话语"
    )


class InterviewQuestionList(BaseModel):
    questions: List["InterviewQuestion"] = Field(
        ...,
        description="从面经中提取的所有面试问题列表"
    )

    @model_validator(mode='before')
    @classmethod
    def handle_list_input(cls, v):
        if isinstance(v, list):
            return {"questions": v}
        if isinstance(v, dict):
            if "interview_questions" in v and "questions" not in v:
                v["questions"] = v.pop("interview_questions")
            if "items" in v and "questions" not in v:
                v["questions"] = v.pop("items")
            if "data" in v and "questions" not in v:
                v["questions"] = v.pop("data")
        return v


class WeaknessProfile(BaseModel):
    weakness_tags: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="提炼出的核心能力弱点标签，每个标签控制在4-6个字（如：['无网感', '缺数据闭环', '逻辑跳跃']）"
    )
    forbidden_words: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="基于其简历推断出的，面试时绝对不能提的词汇或借口（如：['运气不好', '领导安排的']）"
    )

    @model_validator(mode='before')
    @classmethod
    def normalize_list_fields(cls, v):
        if isinstance(v, dict):
            for field in ['weakness_tags', 'forbidden_words']:
                if field not in v:
                    continue
                val = v[field]
                if isinstance(val, list):
                    normalized = []
                    for item in val:
                        if isinstance(item, str):
                            normalized.append(item)
                        elif isinstance(item, dict):
                            for value in item.values():
                                if isinstance(value, str):
                                    normalized.append(value)
                                    break
                    v[field] = normalized
                elif isinstance(val, dict):
                    v[field] = list(val.keys())
                elif isinstance(val, str):
                    v[field] = [val]
        return v

    @property
    def to_prompt_prefix(self) -> str:
        tags_str = "、".join(self.weakness_tags)
        forbid_str = "、".join(self.forbidden_words)
        return f"【用户弱点：{tags_str}】【雷区忌口：{forbid_str}】"


class GodDecision(BaseModel):
    reasoning: str = Field(
        ...,
        description="【思维链推理 — 必须首先填写】在做出决策前，你必须按以下三步显式推理：Step1 用户状态诊断（用户当前处于什么认知状态？）→ Step2 教学阶段判断（当前对话处于面试训练的哪个阶段？）→ Step3 最优动作推导（基于Step1+Step2，哪个Skill最能推进用户进步？为什么其他选项不合适？）"
    )
    intent_dim: str = Field(
        ...,
        description="你对用户这句话的动态语义定性（如：'有回答意愿但逻辑断裂'、'纯粹的情绪发泄'、'顺着梯子在爬'）。不要用死板词汇。"
    )
    intended_action: Literal[
        "socratic_probe",
        "scaffold_rescue",
        "judge_strike",
        "smart_redirection",
        "teaching_supplement"
    ] = Field(
        ...,
        description="你要调用的底层Skill名称。只能从以下5个选：socratic_probe, scaffold_rescue, judge_strike, smart_redirection, teaching_supplement"
    )
    task_prompt: str = Field(
        ...,
        description="【最关键】扔给底层Skill执行的完整指令。必须包含你对上下文的理解、对用户的判断、要求底层怎么回复。底层是个没脑子的复读机，只会一字不差执行这个指令。"
    )


class GapAnalysisResult(BaseModel):
    weaknesses: List[str] = Field(
        ...,
        min_length=1,
        description="从弱点分析中提取的关键弱点描述列表，每条需具体说明差距内容"
    )
    resume_suggestions: List[str] = Field(
        ...,
        min_length=1,
        description="基于弱点分析的针对性简历优化建议列表，每条需给出具体可行的修改策略"
    )


class InterviewState(TypedDict):
    history: Annotated[list, add_messages]
    weakness_prefix: str
    is_ready: bool
    user_input: str
    extracted_text: str
    has_file: bool
    decision: Dict[str, Any]
    response: str
    short_summary: str
    turn_count: int
    is_generating_report: bool
    should_end: bool
    file_name: str
    jd_text: str
    resume_text: str
    gap_analysis: str
    job_type: str

```

#### `domain\validators.py`

```python
from enum import Enum
from typing import Tuple

JD_KEYWORDS = [
    "岗位职责", "任职要求", "岗位要求", "任职资格", "岗位描述",
    "工作职责", "任职条件", "核心职责", "职位描述", "岗位名称",
    "薪资待遇", "福利待遇", "加分项", "硬性要求", "优先条件",
    "需要你", "具备", "期待", "任职于本岗位", "能够", "负责"
]

RESUME_KEYWORDS = [
    "实习经历", "工作经历", "教育背景", "项目经验", "项目描述",
    "个人评价", "自我评价", "个人优势", "专业技能", "掌握技能",
    "在校期间", "毕业院校", "获得奖项", "工作成果", "业绩亮点",
    "求职意向", "自我介绍"
]


class ValidationResult:
    def __init__(self, is_valid: bool, error_msg: str = "", needs_jd: bool = False):
        self.is_valid = is_valid
        self.error_msg = error_msg
        self.needs_jd = needs_jd


def validate_resume_jd(combined_text: str, has_file: bool) -> Tuple[bool, str]:
    has_jd_flag = any(kw in combined_text for kw in JD_KEYWORDS)
    has_resume_flag = any(kw in combined_text for kw in RESUME_KEYWORDS)

    if not has_resume_flag and not has_file:
        return False, "🚫 没看出来这是简历。请直接粘贴你的简历纯文本，或者上传 PDF 文件。"

    if has_resume_flag and not has_jd_flag:
        return True, ""

    if not has_resume_flag and has_jd_flag:
        return False, "🚫 目前只收到了岗位 JD。请把你的个人简历（PDF 或纯文本）也发给我，我需要对比两者才能建立精准画像。"

    return True, ""


def validate_scene_mode(scene_mode: str) -> bool:
    return scene_mode in ["train", "review"]

```

#### 岗位配置 (domain/job_configs/)

*岗位配置文件，定义各岗位的Prompt、示例数据等*

##### `domain\job_configs\新媒体运营.yaml`

```yaml
job_type: "新媒体运营"
display_name: "新媒体运营"

domain_keywords: "ROI、GMV、DAU、MAU、转化率、留存率、漏斗模型、A/B测试、用户增长、内容运营、直播带货、粉丝增长、投放优化、数据复盘、KOL/KOC、种草笔记、完播率、互动率"

interviewer_persona: "你是一位经验丰富的新媒体运营面试官兼数据分析师，擅长从实战角度提炼面试核心要点。"

hr_persona: "你是一位阅人无数的资深新媒体行业HR，同时也是一位善于发现潜力的职业导师。"

gap_hr_persona: "你是一位资深的新媒体行业HR和职业规划师。"

mine_interviewer_persona: "你是一位经验丰富的新媒体运营面试官兼数据分析师，擅长从实战角度提炼面试核心要点。"

mine_hr_persona: "你是一位资深的新媒体 HR 兼教学设计师，现在你要根据一份教科书式的标准参考答案，逆向出一套用来帮助应届生系统提升面试能力的专业题库。"

default_jd: |
  岗位名称：新媒体运营
  岗位职责：
  1. 负责品牌全媒体矩阵（抖音/小红书/视频号）的内容策略制定与执行
  2. 搭建并管理KOL/KOC资源库，维护达人合作体系
  3. 制定数据驱动的运营增长策略，建立ROI评估体系
  4. 策划并执行线上营销活动，追踪效果并持续优化
  5. 跨部门协调，推动品牌传播与电商转化联动
  任职要求：
  1. 1年以上新媒体运营经验，有内容操盘案例
  2. 理解各平台算法机制与流量逻辑
  3. 具备数据分析能力，能独立完成复盘报告
  4. 有品牌方或MCN公司经验优先

demo_resume: |
  姓名：李四
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
  抖音运营、小红书运营、直播带货、文案撰写、数据分析、Canva、剪映

demo_jd: |
  岗位名称：资深新媒体运营
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
  5. 有品牌方或4A公司经验优先

```

### 基础设施层 (infrastructure/)

*文件解析、向量检索、工具集成*

#### `infrastructure\__init__.py`

```python
from infrastructure.retrieval import (
    DashScopeEmbeddings,
    WeightedEnsembleRetriever,
    get_hybrid_retriever
)
from infrastructure.tools import search_interview_db, INTERVIEW_TOOLS
from infrastructure.parsers import safe_parse_weakness, extract_text_from_file

__all__ = [
    "DashScopeEmbeddings",
    "WeightedEnsembleRetriever",
    "get_hybrid_retriever",
    "search_interview_db",
    "INTERVIEW_TOOLS",
    "safe_parse_weakness",
    "extract_text_from_file"
]

```

#### `infrastructure\parsers\__init__.py`

```python
from .text_parser import safe_parse_weakness
from .file_parser import extract_text_from_file

__all__ = ["safe_parse_weakness", "extract_text_from_file"]

```

#### `infrastructure\parsers\file_parser.py`

```python
import io
import re
import logging

logger = logging.getLogger(__name__)

def extract_text_from_file(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()
    
    extracted_text = ""
    
    try:
        if file_name.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif file_name.endswith('.txt'):
            try:
                extracted_text = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    extracted_text = file_bytes.decode('gbk')
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode('utf-8', errors='ignore')
        else:
            raise ValueError("本教练只认 PDF 和 TXT 文件，别拿奇奇怪怪的文件糊弄我！")
            
    except Exception as e:
        logger.error(f"文件解析底层崩溃: {e}")
        raise ValueError(f"文件损坏或格式太复杂。底层报错：{str(e)[:50]}")
        
    if len(extracted_text.strip()) < 100:
        raise ValueError("提取出的文字太少！你的简历可能全是复杂的表格或图片，本教练看不了这种花里胡哨的排版。请直接复制纯文本粘贴到下方！")
        
    if '<w:' in extracted_text or '<v:' in extracted_text:
        raise ValueError("检测到复杂的底层排版代码，解析失败。请直接复制纯文本粘贴到下方！")
    
    if '<html' in extracted_text.lower() or '<!doctype' in extracted_text.lower():
        raise ValueError("检测到HTML格式内容，解析失败。请直接复制纯文本粘贴到下方！")
        
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', extracted_text))
    total_chars = len(extracted_text.replace(" ", "").replace("\n", ""))
    
    if total_chars > 100 and chinese_chars / total_chars < 0.3:
        raise ValueError("提取出的内容乱码太多，大概率是扫描件。请直接复制纯文本粘贴到下方！")
        
    return extracted_text.strip()

```

#### `infrastructure\parsers\text_parser.py`

```python
import re
import json
import logging
from typing import Optional, Union
from domain.schemas import WeaknessProfile

logger = logging.getLogger(__name__)


def safe_parse_weakness(raw_text: Union[str, WeaknessProfile]) -> WeaknessProfile:
    """
    安全解析弱点画像。
    
    如果输入已经是合法的 Pydantic 对象，直接返回；
    如果是 JSON 字符串，尝试解析；
    如果是乱码文本（针对劣质模型的兜底），尝试正则提取；
    如果全失败，直接抛出异常，绝不伪造数据！
    
    Args:
        raw_text: 可能是 WeaknessProfile 对象、JSON 字符串、或乱码文本
        
    Returns:
        WeaknessProfile: 解析后的弱点画像对象
        
    Raises:
        ValueError: 所有解析策略均失败时抛出
    """
    if isinstance(raw_text, WeaknessProfile):
        return raw_text

    text_to_parse = str(raw_text)

    try:
        data = json.loads(text_to_parse)
        return WeaknessProfile(**data)
    except json.JSONDecodeError:
        pass
    
    json_match = re.search(r'\{.*?\}', text_to_parse, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return WeaknessProfile(**data)
        except Exception:
            pass

    logger.warning("⚠️ 模型返回非标准格式，启动正则强行提取...")
    try:
        weakness_matches = re.findall(r'(?:\d+\.\s*|-\s*)([^：:\n]+?)(?:[:：])', text_to_parse)
        weakness_tags = [w.strip() for w in weakness_matches[:3] if len(w.strip()) <= 10]
        
        forbid_matches = re.findall(r'(?:雷区|忌口|不能说|别说)[:：]\s*([^。\n]+)', text_to_parse)
        forbidden_words = [w.strip() for w in forbid_matches[:3]]
        
        if not weakness_tags:
            raise ValueError("正则也无法提取有效弱点标签")
            
        return WeaknessProfile(
            weakness_tags=weakness_tags if weakness_tags else ["能力待评估"],
            forbidden_words=forbidden_words if forbidden_words else ["说主观原因"]
        )
    except Exception as e:
        logger.error(f"所有解析策略均失败: {e}")
        raise ValueError(f"AI 无法理解你的简历格式，请尝试精简简历内容后重试。原始错误: {str(e)[:50]}")

```

#### `infrastructure\retrieval\__init__.py`

```python
from .embeddings import DashScopeEmbeddings
from .hybrid import (
    WeightedEnsembleRetriever,
    get_hybrid_retriever,
    load_jsonl_to_docs,
    jieba_preprocess
)

__all__ = [
    "DashScopeEmbeddings",
    "WeightedEnsembleRetriever",
    "get_hybrid_retriever",
    "load_jsonl_to_docs",
    "jieba_preprocess"
]

```

#### `infrastructure\retrieval\embeddings.py`

```python
"""
百炼 DashScope 向量化封装
解决 OpenAIEmbeddings 与百炼 embedding API 不兼容的问题
"""
from typing import List
from langchain_core.embeddings import Embeddings
import dashscope
from http import HTTPStatus
from app.config import settings


class DashScopeEmbeddings(Embeddings):
    """
    兼容百炼平台的向量化封装，替代 OpenAIEmbeddings
    """
    
    def __init__(self, model: str = None):
        self.model = model or settings.EMBEDDING_MODEL_NAME
        dashscope.api_key = settings.OPENAI_API_KEY
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档（分批处理，每批最多10条）"""
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch
            )
            if resp.status_code == HTTPStatus.OK:
                all_embeddings.extend([item['embedding'] for item in resp.output['embeddings']])
            else:
                raise RuntimeError(f"Embedding failed: {resp}")
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """向量化查询"""
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=text
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise RuntimeError(f"Embedding failed: {resp}")

```

#### `infrastructure\retrieval\hybrid.py`

```python
import logging
import json
import hashlib
import pickle
import shutil
import tempfile
import os
from pathlib import Path
from typing import List, Sequence, Optional, Dict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from pydantic import Field

from app.config import settings
from infrastructure.retrieval.embeddings import DashScopeEmbeddings

logging.getLogger("jieba").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_temp_faiss_dirs: Dict[str, Path] = {}


def _has_non_ascii(path: str) -> bool:
    try:
        path.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def _get_safe_faiss_path(faiss_dir: Path, job_type: str) -> Path:
    if not _has_non_ascii(str(faiss_dir)):
        return faiss_dir

    if job_type in _temp_faiss_dirs and _temp_faiss_dirs[job_type].exists():
        return _temp_faiss_dirs[job_type]

    safe_prefix = hashlib.md5(job_type.encode()).hexdigest()[:8]
    temp_dir = Path(tempfile.mkdtemp(prefix=f"faiss_{safe_prefix}_"))
    _temp_faiss_dirs[job_type] = temp_dir
    logger.info(f"📁 路径包含非ASCII字符，复制索引到临时目录: {temp_dir}")

    for file_name in ["index.faiss", "index.pkl"]:
        src = faiss_dir / file_name
        dst = temp_dir / file_name
        if src.exists():
            shutil.copy(str(src), str(dst))

    return temp_dir


class WeightedEnsembleRetriever(BaseRetriever):
    retrievers: Sequence[BaseRetriever] = Field(...)
    weights: List[float] = Field(default=[0.7, 0.3], description="权重列表，如 [FAISS权重, BM25权重]")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_retriever_docs = [retriever.invoke(query) for retriever in self.retrievers]

        k = 60
        rrf_scores = {}
        doc_map = {}

        for docs, weight in zip(all_retriever_docs, self.weights):
            for rank, doc in enumerate(docs):
                if doc.page_content not in doc_map:
                    doc_map[doc.page_content] = doc
                rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + (weight / (rank + 1 + k))

        sorted_docs = sorted(doc_map.values(), key=lambda x: rrf_scores.get(x.page_content, 0), reverse=True)
        return sorted_docs


_RETRIEVER_CACHE: Dict[str, WeightedEnsembleRetriever] = {}
_INDEX_HASH_CACHE: dict = {}


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_hash(dir_path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file():
            h.update(file_path.name.encode())
            with open(file_path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def jieba_preprocess(text: str) -> List[str]:
    import jieba
    return list(jieba.lcut(text))


def load_jsonl_to_docs(jsonl_path: str, job_type: str = None) -> List[Document]:
    docs = []
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning(f"未找到数据文件: {jsonl_path}")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = data.get('question', '') or ''
                inspect_point = data.get('inspect_point', '') or ''
                content = f"面试问题：{question}\n考察考点：{inspect_point}"

                if not content.strip() or content.strip() == "面试问题：\n考察考点：":
                    logger.warning(f"跳过空内容行: {line[:50]}...")
                    continue

                doc_job_type = data.get("job_type") or job_type or "未知岗位"
                doc = Document(
                    page_content=content,
                    metadata={"raw_data": data, "job_type": doc_job_type}
                )
                docs.append(doc)
            except json.JSONDecodeError:
                logger.error(f"解析 JSONL 行失败: {line}")
                continue
    return docs


def _load_bm25_from_json(json_path: Path) -> BM25Retriever:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [Document(**doc_data) for doc_data in data["documents"]]

    bm25_retriever = BM25Retriever.from_documents(
        documents=docs,
        k=settings.BM25_TOP_K,
        preprocess_func=jieba_preprocess
    )
    return bm25_retriever


def _verify_index_integrity(index_dir: Path) -> bool:
    hash_file = index_dir / "index.hash"
    faiss_dir = index_dir / "faiss_index"
    bm25_json_path = index_dir / "bm25.json"
    bm25_pkl_path = index_dir / "bm25.pkl"

    if not hash_file.exists():
        logger.warning(f"索引校验文件不存在，将自动生成: {hash_file}")
        try:
            hashes = {}
            if faiss_dir.exists():
                hashes["faiss"] = compute_dir_hash(faiss_dir)
            if bm25_json_path.exists():
                hashes["bm25"] = compute_file_hash(bm25_json_path)
            elif bm25_pkl_path.exists():
                hashes["bm25"] = compute_file_hash(bm25_pkl_path)

            with open(hash_file, "w", encoding="utf-8") as f:
                json.dump(hashes, f, indent=2)
            logger.info("✅ 已生成索引校验文件")
            return True
        except Exception as e:
            logger.error(f"生成索引校验文件失败: {e}")
            return False

    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            stored_hashes = json.load(f)

        if faiss_dir.exists():
            current_faiss_hash = compute_dir_hash(faiss_dir)
            if current_faiss_hash != stored_hashes.get("faiss", ""):
                logger.error("FAISS索引校验失败！文件可能被篡改。")
                return False

        if bm25_json_path.exists():
            current_bm25_hash = compute_file_hash(bm25_json_path)
            if current_bm25_hash != stored_hashes.get("bm25", ""):
                logger.error("BM25索引校验失败！文件可能被篡改。")
                return False
        elif bm25_pkl_path.exists():
            current_bm25_hash = compute_file_hash(bm25_pkl_path)
            if current_bm25_hash != stored_hashes.get("bm25", ""):
                logger.error("BM25索引校验失败！文件可能被篡改。")
                return False

        logger.info("✅ 索引完整性校验通过")
        return True

    except Exception as e:
        logger.error(f"索引校验异常: {e}")
        return False


def get_hybrid_retriever(job_type: str, jsonl_path: str = None, force_reload: bool = False) -> WeightedEnsembleRetriever:
    if job_type in _RETRIEVER_CACHE and not force_reload:
        logger.info(f"⚡ [缓存命中] 直接使用内存中的 {job_type} 检索器")
        return _RETRIEVER_CACHE[job_type]

    index_dir = settings.get_job_index_dir(job_type)
    faiss_dir = index_dir / "faiss_index"
    bm25_json_path = index_dir / "bm25.json"
    bm25_pkl_path = index_dir / "bm25.pkl"

    if faiss_dir.exists() and (bm25_json_path.exists() or bm25_pkl_path.exists()):
        logger.info(f"📦 检测到 {job_type} 本地存在预构建索引，开始加载...")

        if not _verify_index_integrity(index_dir):
            raise RuntimeError(
                f"{job_type} 索引完整性校验失败！请重新构建索引"
                f"（运行 python scripts/build_index.py --job-type \"{job_type}\"），"
                "或检查索引文件是否被恶意篡改。"
            )

        embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        safe_faiss_dir = _get_safe_faiss_path(faiss_dir, job_type)
        faiss_vectorstore = FAISS.load_local(
            str(safe_faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": settings.FAISS_TOP_K})

        if bm25_json_path.exists():
            bm25_retriever = _load_bm25_from_json(bm25_json_path)
        else:
            logger.warning("⚠️ 使用旧版pickle格式BM25索引，建议重新构建索引以使用更安全的JSON格式")
            with open(bm25_pkl_path, "rb") as f:
                bm25_retriever = pickle.load(f)

        logger.info(f"✅ {job_type} 本地全量索引加载完成！")

    else:
        logger.warning(f"⚠️ 未找到 {job_type} 预构建索引，回退到实时构建模式...")

        if jsonl_path is None:
            jsonl_path = str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")

        docs = load_jsonl_to_docs(jsonl_path, job_type=job_type)
        if not docs:
            raise ValueError(f"{job_type} 没有加载到任何数据，无法构建检索器！请先运行离线流程。")

        embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        faiss_vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": settings.FAISS_TOP_K})

        bm25_retriever = BM25Retriever.from_documents(
            documents=docs,
            k=settings.BM25_TOP_K,
            preprocess_func=jieba_preprocess
        )

    ensemble_retriever = WeightedEnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    _RETRIEVER_CACHE[job_type] = ensemble_retriever
    return ensemble_retriever


def invalidate_retriever_cache(job_type: str = None):
    global _RETRIEVER_CACHE
    if job_type:
        _RETRIEVER_CACHE.pop(job_type, None)
        logger.info(f"{job_type} 检索器缓存已清除")
    else:
        _RETRIEVER_CACHE.clear()
        logger.info("所有检索器缓存已清除")

```

#### `infrastructure\tools\__init__.py`

```python
from .interview_db import search_interview_db, INTERVIEW_TOOLS

__all__ = ["search_interview_db", "INTERVIEW_TOOLS"]

```

#### `infrastructure\tools\interview_db.py`

```python
import logging
from typing import List, Literal
from langchain_core.tools import tool

from infrastructure.retrieval.hybrid import get_hybrid_retriever
from app.config import settings
from domain.validators import validate_scene_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SceneMode = Literal["train", "review"]

_CURRENT_JOB_TYPE: str = ""


def set_current_job_type(job_type: str):
    global _CURRENT_JOB_TYPE
    _CURRENT_JOB_TYPE = job_type
    logger.info(f"🔧 [Tool] 当前岗位已切换为: {job_type}")


def get_current_job_type() -> str:
    global _CURRENT_JOB_TYPE
    if not _CURRENT_JOB_TYPE:
        from domain.job_configs import list_available_jobs
        available = list_available_jobs()
        if available:
            _CURRENT_JOB_TYPE = available[0]
            logger.info(f"🔧 [Tool] 未设置岗位，自动使用默认岗位: {_CURRENT_JOB_TYPE}")
    return _CURRENT_JOB_TYPE


@tool
def search_interview_db(
    user_query: str,
    scene_mode: SceneMode = "train",
    job_type: str = ""
) -> str:
    """
    当用户的话里包含任何【业务名词】时，无论用户是在抱怨、闲聊、还是在正经回答，都必须调用此工具！
    因为用户的随口一句抱怨，往往暴露了最真实的认知盲区。

    【红色警报】如果调用此工具后，返回的结果包含"未检索到"字样，你绝对不允许捏造或猜测任何面经内容！

    ═══════════════════════════════════════
    §2 思维链推理（调用前必须执行）
    ═══════════════════════════════════════

    1. 扫描业务名词：用户的话里有没有业务关键词？
    2. 判断调用必要性：即使看似闲聊，只要涉及业务概念就必须检索
    3. 选择场景模式：train（训练模式，隐藏公式）或 review（复盘模式，展示公式）

    ═══════════════════════════════════════
    §3 示例锚定
    ═══════════════════════════════════════

    用户说"我觉得数据也没啥好看的" → 必须调用，因为"数据"是业务名词，暴露了数据思维缺失
    用户说"ROI怎么算啊" → 必须调用，因为"ROI"是业务名词
    用户说"你好" → 不需要调用，无业务名词

    参数:
    user_query: 用户当前说的话，或者你提炼出的需要检索的核心考点。
    scene_mode: 场景模式开关。
                 - 传入 "train" 时：返回隐藏公式的【判卷清单】（用于模拟训练，防泄题）。
                 - 传入 "review" 时：返回包含标准答案的【满分公式】（用于面试复盘，做对比）。
    job_type: 岗位类型，用于检索对应岗位的题库。如果不传，使用当前会话的岗位。
    返回:
    从面经库中检索出的结构化字符串。
    """
    if not validate_scene_mode(scene_mode):
        logger.warning(f"⚠️ 非法scene_mode: {scene_mode}，强制使用train模式")
        scene_mode = "train"

    effective_job_type = job_type or _CURRENT_JOB_TYPE

    logger.info(f"🛠️ [Tool 触发] 准备检索，查询词: {user_query}, 模式: {scene_mode}, 岗位: {effective_job_type}")

    retriever = get_hybrid_retriever(effective_job_type)

    docs = retriever.invoke(user_query)

    if not docs:
        return "未检索到相关面经数据。"

    results = []
    for doc in docs:
        raw_data = doc.metadata.get("raw_data", {})
        formula_concept = raw_data.get('high_score_formula', '结构化框架')

        if scene_mode == "review":
            result_str = (
                f"【考题】{raw_data.get('question')}\n"
                f"【考点】{raw_data.get('inspect_point')}\n"
                f"【满分公式(直接亮底牌)】: {raw_data.get('high_score_formula')}\n"
                f"【一票否决雷区】: {raw_data.get('pitfall_guide')}"
            )
        else:
            result_str = (
                f"【考题】{raw_data.get('question')}\n"
                f"【考点】{raw_data.get('inspect_point')}\n"
                f"【判卷清单(仅作判别用，严禁外泄)】:\n"
                f"- [ ] 是否脱离了单一指标，展示了多维度拆解能力？\n"
                f"- [ ] 是否提到了类似'{formula_concept}'的底层结构？\n"
                f"- [ ] 是否给出了具体的业务动作推导，而不是纯理论？\n"
                f"【雷区红线】: {raw_data.get('pitfall_guide')}"
            )
        results.append(result_str)

    return "\n\n---\n\n".join(results)

INTERVIEW_TOOLS = [search_interview_db]

```

### 脚本 (scripts/)

*构建索引、数据处理、CLI管理工具*

#### `scripts\build_index.py`

```python
"""
索引构建脚本

启动方式:
    python scripts/build_index.py --job-type "新媒体运营"

说明:
    该脚本需要直接运行，不依赖 pip install -e .
    因此保留了 sys.path.append 来确保模块导入正常
"""

import sys
import json
import hashlib
import shutil
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from jieba import lcut

from app.config import settings
from domain.job_configs import load_job_config, get_job_config_or_default
from infrastructure.retrieval.hybrid import (
    load_jsonl_to_docs,
    jieba_preprocess,
    compute_file_hash,
    compute_dir_hash
)
from infrastructure.retrieval.embeddings import DashScopeEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _has_non_ascii(path: str) -> bool:
    try:
        path.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def build_and_save_index(job_type: str):
    jsonl_path = str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")
    index_dir = settings.get_job_index_dir(job_type)
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"1. 开始加载 {job_type} JSONL 数据: {jsonl_path}")
    docs = load_jsonl_to_docs(jsonl_path, job_type=job_type)
    logger.info(f"全量数据加载完毕，共 {len(docs)} 条。准备开始耗时操作...")

    logger.info("2. 开始调用 API 构建 FAISS 向量索引（全量，可能需要几分钟，请耐心等待）...")
    embeddings = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    faiss_vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    faiss_save_path = index_dir / "faiss_index"

    if _has_non_ascii(str(faiss_save_path)):
        logger.info("检测到路径包含非ASCII字符，使用临时目录保存FAISS索引...")
        temp_dir = Path(tempfile.mkdtemp(prefix="faiss_build_"))
        faiss_vectorstore.save_local(str(temp_dir))

        if faiss_save_path.exists():
            shutil.rmtree(str(faiss_save_path))
        shutil.copytree(str(temp_dir), str(faiss_save_path))
        shutil.rmtree(str(temp_dir))
        logger.info(f"已从临时目录复制到目标目录: {faiss_save_path}")
    else:
        faiss_vectorstore.save_local(str(faiss_save_path))

    logger.info(f"FAISS 索引构建完毕并保存至: {faiss_save_path}")

    logger.info("3. 开始构建 BM25 索引...")
    bm25_retriever = BM25Retriever.from_documents(
        documents=docs,
        k=settings.BM25_TOP_K,
        preprocess_func=jieba_preprocess
    )

    bm25_json_path = index_dir / "bm25.json"
    bm25_docs_data = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(bm25_json_path, "w", encoding="utf-8") as f:
        json.dump({"documents": bm25_docs_data}, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ BM25 索引(JSON格式)构建完毕并保存至: {bm25_json_path}")

    bm25_pkl_path = index_dir / "bm25.pkl"
    with open(bm25_pkl_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    logger.info(f"✅ BM25 索引(pickle格式)备份保存至: {bm25_pkl_path}")

    logger.info("4. 生成索引完整性校验文件...")
    faiss_hash = compute_dir_hash(faiss_save_path)
    bm25_hash = compute_file_hash(bm25_json_path)

    hash_file = index_dir / "index.hash"
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump({
            "faiss": faiss_hash,
            "bm25": bm25_hash,
            "version": "2.0",
            "job_type": job_type
        }, f, indent=2)
    logger.info(f"✅ 索引校验文件保存至: {hash_file}")

    print("\n" + "="*50)
    print(f">>> {job_type} 索引构建完成！你可以启动 chainlit 应用了，加载将是秒级的。")
    print("="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建岗位索引")
    parser.add_argument("--job-type", "-j", default=None, help="岗位类型（如：新媒体运营、产品经理）")

    args = parser.parse_args()

    if args.job_type:
        job_config = load_job_config(args.job_type)
        job_type = job_config.job_type
    else:
        job_config = get_job_config_or_default()
        job_type = job_config.job_type
        logger.info(f"未指定岗位，使用默认: {job_type}")

    build_and_save_index(job_type)

```

#### `scripts\manage.py`

```python
"""
岗位管理 CLI 工具

用法:
    python scripts/manage.py list-jobs
    python scripts/manage.py add-job --name "产品经理" --input "题库/产品经理.pdf" "题库/面经2.pdf"
    python scripts/manage.py add-job --name "产品经理" --input "题库/"          (整个目录)
    python scripts/manage.py mine --job-type "产品经理" --input "题库/产品经理.pdf" "题库/面经2.pdf"
    python scripts/manage.py build-index --job-type "产品经理"
    python scripts/manage.py rebuild --job-type "新媒体运营"
    python scripts/manage.py remove-job --job-type "产品经理"
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging

from app.config import settings
from domain.job_configs import load_job_config, list_available_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cmd_list_jobs(args):
    available_jobs = list_available_jobs()
    jobs_data_dir = settings.get_jobs_data_dir()

    if not available_jobs:
        print("没有找到任何岗位配置。")
        print(f"   请在 domain/job_configs/ 下创建 YAML 配置文件。")
        return

    print(f"\n{'='*60}")
    print(f"  已配置岗位列表")
    print(f"{'='*60}\n")

    for job_type in available_jobs:
        try:
            config = load_job_config(job_type)
        except Exception as e:
            print(f"  X {job_type} - 配置加载失败: {e}")
            continue

        job_dir = settings.get_job_data_dir(job_type)
        has_raw = (settings.get_job_raw_dir(job_type).exists() and
                   any(settings.get_job_raw_dir(job_type).iterdir()))
        has_processed = (settings.get_job_processed_dir(job_type).exists() and
                         (settings.get_job_processed_dir(job_type) / "mined_questions.jsonl").exists())
        has_index = (settings.get_job_index_dir(job_type).exists() and
                     (settings.get_job_index_dir(job_type) / "faiss_index").exists() and
                     (settings.get_job_index_dir(job_type) / "bm25.json").exists())

        status_parts = []
        status_parts.append("[raw] OK" if has_raw else "[raw] --")
        status_parts.append("[processed] OK" if has_processed else "[processed] --")
        status_parts.append("[index] OK" if has_index else "[index] --")

        status_str = " | ".join(status_parts)

        print(f"  {config.display_name} ({job_type})")
        print(f"     Status: {status_str}")
        print(f"     Dir:    {job_dir}")
        print()

    print(f"{'='*60}\n")


def cmd_add_job(args):
    from scripts.mine_textbook import mine_textbook_to_questions
    from scripts.build_index import build_and_save_index

    job_type = args.name
    input_paths = args.input

    try:
        config = load_job_config(job_type)
        logger.info(f"Found job config: {job_type}")
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        print(f"  Please create the config file first.")
        return

    raw_dir = settings.get_job_raw_dir(job_type)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for p in input_paths:
        if not Path(p).exists():
            print(f"X Input path does not exist: {p}")
            return

    print(f"\n{'='*60}")
    print(f"  Add Job: {config.display_name} ({job_type})")
    print(f"  Input:   {', '.join(input_paths)}")
    print(f"  Model:   {settings.OFFLINE_MODEL_NAME} (offline)")
    print(f"{'='*60}\n")

    print("Step 1/3: Extract text from files...")
    print("Step 2/3: LLM mining, extracting interview questions...")
    output_path = str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")
    mine_textbook_to_questions(input_paths, output_path, job_type=job_type)

    print("Step 3/3: Build search index...")
    build_and_save_index(job_type)

    print(f"\n{'='*60}")
    print(f"  Done! Job {config.display_name} added successfully!")
    print(f"{'='*60}\n")


def cmd_mine(args):
    from scripts.mine_textbook import mine_textbook_to_questions

    job_type = args.job_type
    input_paths = args.input

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        return

    output_path = args.output or str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")

    print(f"Mining: {config.display_name} ({job_type})")
    print(f"Input:  {', '.join(input_paths)}")
    print(f"Model:  {settings.OFFLINE_MODEL_NAME} (offline)")
    mine_textbook_to_questions(input_paths, output_path, job_type=job_type)


def cmd_build_index(args):
    from scripts.build_index import build_and_save_index

    job_type = args.job_type

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        return

    print(f"Building index: {config.display_name} ({job_type})")
    build_and_save_index(job_type)


def cmd_rebuild(args):
    from scripts.build_index import build_and_save_index

    job_type = args.job_type

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        return

    index_dir = settings.get_job_index_dir(job_type)
    if index_dir.exists():
        import shutil
        shutil.rmtree(str(index_dir))
        logger.info(f"Deleted old index: {index_dir}")

    print(f"Rebuilding index: {config.display_name} ({job_type})")
    build_and_save_index(job_type)


def cmd_remove_job(args):
    import shutil

    job_type = args.job_type

    job_dir = settings.get_job_data_dir(job_type)
    if not job_dir.exists():
        print(f"X Job data directory not found: {job_dir}")
        return

    if not args.force:
        confirm = input(f"Confirm delete all data for '{job_type}'? (y/N): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    shutil.rmtree(str(job_dir))
    print(f"Deleted job data: {job_dir}")
    print(f"  Note: YAML config domain/job_configs/{job_type}.yaml needs to be deleted manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Interview Coach - Job Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage.py list-jobs
  python scripts/manage.py add-job --name "产品经理" --input "题库/产品经理.pdf"
  python scripts/manage.py add-job --name "产品经理" --input "题库/1.pdf" "题库/2.pdf" "题库/3.txt"
  python scripts/manage.py add-job --name "产品经理" --input "题库/"
  python scripts/manage.py mine --job-type "产品经理" --input "题库/1.pdf" "题库/2.pdf"
  python scripts/manage.py build-index --job-type "产品经理"
  python scripts/manage.py rebuild --job-type "新媒体运营"
  python scripts/manage.py remove-job --job-type "产品经理" --force
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    list_parser = subparsers.add_parser("list-jobs", help="List all configured jobs and their status")
    list_parser.set_defaults(func=cmd_list_jobs)

    add_parser = subparsers.add_parser("add-job", help="Add a new job (extract -> mine -> build index)")
    add_parser.add_argument("--name", "-n", required=True, help="Job name (must match YAML config filename)")
    add_parser.add_argument("--input", "-i", nargs="+", required=True,
                            help="Input file paths (PDF/TXT), supports multiple files and directories")
    add_parser.set_defaults(func=cmd_add_job)

    mine_parser = subparsers.add_parser("mine", help="Run the mining step only")
    mine_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    mine_parser.add_argument("--input", "-i", nargs="+", required=True,
                            help="Input file paths (PDF/TXT), supports multiple files and directories")
    mine_parser.add_argument("--output", "-o", default=None, help="Output path")
    mine_parser.set_defaults(func=cmd_mine)

    build_parser = subparsers.add_parser("build-index", help="Run the index building step only")
    build_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    build_parser.set_defaults(func=cmd_build_index)

    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild index for a job")
    rebuild_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    rebuild_parser.set_defaults(func=cmd_rebuild)

    remove_parser = subparsers.add_parser("remove-job", help="Remove a job's data")
    remove_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    remove_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    remove_parser.set_defaults(func=cmd_remove_job)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

```

#### `scripts\mine_textbook.py`

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import InterviewQuestion, InterviewQuestionList
from domain.job_configs import load_job_config
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_mine_questions_prompt(job_config) -> ChatPromptTemplate:
    interviewer_persona = job_config.mine_interviewer_persona
    job_type = job_config.job_type

    return ChatPromptTemplate.from_messages([
        ("system", f"""{interviewer_persona}
你的任务是从一篇杂乱的{job_type}面经长文中，精准提取出面试问题及其背后的底层逻辑。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

你必须严格按照提供的 JSON Schema 格式输出，不要输出任何解释性文字，只要纯 JSON。

提取要求：
1. 问题必须是面试官的真实提问，不要总结。
2. 考察点 要一针见血（如：数据归因能力，而不是泛泛的"分析能力"）。
3. 高分公式 必须具备可执行性（如：STAR+漏斗模型+具体Action）。
4. 避坑指南 必须指出具体的"送命题"话术，并用引导性的语气说明为什么这样回答会扣分。

重要：每个字段都必须填写，不能为空字符串。如果原文没有直接给出，请根据你的专业知识推断并填写。

═══════════════════════════════════════
§2 思维链推理（提取前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 定位提问：从面经中找到面试官的原话提问
2. 逆向推导：这个问题表面在问 X，实际在考察 Y（Y 才是 inspect_point）
3. 构造公式：基于 Y，设计一个可复用的回答框架（high_score_formula）
4. 预判雷区：基于 Y，推测应届生最容易犯的回答误区（pitfall_guide）

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

面经片段："面试官问我怎么做一个活动的复盘，我说就是看看数据哪里好哪里不好，然后面试官皱了皱眉"

→ 输出：
{{{{
  "question": "你是怎么做活动复盘的？",
  "inspect_point": "结构化复盘能力（非直觉式总结）",
  "high_score_formula": "目标回溯→数据漏斗拆解（逐层归因）→关键发现→Action项+量化验证",
  "pitfall_guide": "千万别用'看看数据哪里好哪里不好'这种模糊表述——这暴露了你只会看表面数字，不会做归因。应该说'我通过漏斗拆解定位到转化率最低的环节是X，原因是Y'"
}}}}"""),

        ("human", "下面是待提取的面经长文内容：\n\n{{raw_text}}")
    ])


def _build_mine_textbook_prompt(job_config) -> ChatPromptTemplate:
    hr_persona = job_config.mine_hr_persona
    job_type = job_config.job_type

    return ChatPromptTemplate.from_messages([
        ("system", f"""{hr_persona}
注意：给你的原材料可能是干瘪的、正确的废话。你的任务是将其转化为【生动易懂且具有实操指导价值】的内容！

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

提取与再创作要求：
1. question：保留原题的核心考点，但如果原题太长，精简为口语化的面试提问。
2. inspect_point：一针见血，不要写"综合能力"，要写具体的能力维度。
3. high_score_formula：【核心】不要抄原答案！把原答案里的步骤，浓缩成朗朗上口的"万能公式"或"解题套路"。
4. pitfall_guide：【核心】原答案里绝对没有这个！你必须基于这个考点，自己脑补出一种"应届生最容易犯的回答误区"。

═══════════════════════════════════════
§2 思维链推理（逆向工程前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 解构原答案：标准答案的核心步骤是什么？每步在解决什么问题？
2. 浓缩公式：将步骤序列压缩为一个可记忆的"口诀"或"框架"
3. 口语化提问：把学术化/书面化的题目翻译成面试官真正会问的口语
4. 预判踩坑：站在应届生视角，他们最可能在哪里翻车？

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

原文片段："内容运营的核心在于通过优质内容吸引用户，建立品牌认知，最终实现转化。运营者需要关注内容的选题策划、创作质量、分发渠道和效果评估四个环节。"

→ 输出：
{{{{
  "question": "你做内容运营，怎么判断自己产出的内容是不是'好内容'？",
  "inspect_point": "内容效果评估体系（非主观判断）",
  "high_score_formula": "选题(热点匹配度+用户需求度)→创作(完播率/阅读完成率)→分发(渠道ROI对比)→评估(北极星指标+归因分析)",
  "pitfall_guide": "很多同学会说'我觉得内容好就是好'或'阅读量高就是好内容'——前者是主观自嗨，后者是单一指标陷阱。好内容必须有可量化的评估体系，且不同阶段看不同指标。"
}}}}"""),

        ("human", "下面是一段从题库中截取的【题目】和【标准答案】片段，请进行逆向提取：\n\n{{raw_text}}")
    ])


def _resolve_input_paths(input_paths: list) -> list:
    resolved = []
    for p in input_paths:
        path = Path(p)
        if path.is_file():
            resolved.append(path)
        elif path.is_dir():
            for f in sorted(path.iterdir()):
                if f.suffix.lower() in ('.pdf', '.txt'):
                    resolved.append(f)
        else:
            logger.warning(f"跳过不存在的路径: {p}")
    return resolved


def _extract_text_from_file(file_path: Path) -> str:
    if file_path.suffix.lower() == '.pdf':
        from infrastructure.parsers.file_parser import extract_text_from_file

        class FileWrapper:
            def __init__(self, content: bytes, name: str):
                import io
                self._bytes_io = io.BytesIO(content)
                self.name = name
            def read(self):
                return self._bytes_io.read()

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        wrapper = FileWrapper(content=file_bytes, name=file_path.name)
        return extract_text_from_file(wrapper)
    elif file_path.suffix.lower() == '.txt':
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.warning(f"不支持的文件格式: {file_path.suffix}，跳过: {file_path}")
        return ""


def mine_textbook_to_questions(
    input_paths,
    output_path: str,
    job_type: str = None,
    chunk_size: int = 2000
):
    job_config = load_job_config(job_type) if job_type else None

    if not job_config:
        from domain.job_configs import get_job_config_or_default
        job_config = get_job_config_or_default()

    effective_job_type = job_config.job_type

    if isinstance(input_paths, (str, Path)):
        input_paths = [str(input_paths)]

    files = _resolve_input_paths(input_paths)
    if not files:
        logger.error("没有找到可处理的文件！")
        return

    logger.info(f"📚 共找到 {len(files)} 个文件待处理:")
    for f in files:
        logger.info(f"   - {f.name}")

    all_text = ""
    for f in files:
        logger.info(f"📄 正在提取: {f.name}")
        text = _extract_text_from_file(f)
        if text:
            all_text += f"\n\n===== 来源: {f.name} =====\n\n{text}"

    if not all_text.strip():
        logger.error("所有文件提取结果为空！")
        return

    raw_dir = settings.get_job_raw_dir(effective_job_type)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_text_path = raw_dir / "raw_textbook.txt"
    with open(raw_text_path, "w", encoding="utf-8") as f:
        f.write(all_text)
    logger.info(f"✅ 原始文本已保存至: {raw_text_path}")

    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

    llm = ChatOpenAI(
        model=settings.OFFLINE_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.1
    )

    is_textbook = any(
        any(kw in str(f).lower() for kw in ["textbook", "教科书", "题库", "pdf"])
        for f in files
    )
    if is_textbook:
        prompt_template = _build_mine_textbook_prompt(job_config)
    else:
        prompt_template = _build_mine_questions_prompt(job_config)

    chain = prompt_template | llm.with_structured_output(InterviewQuestionList)

    all_questions = []

    for i, chunk in enumerate(chunks):
        logger.info(f"正在处理第 {i+1}/{len(chunks)} 个文本块...")

        try:
            result = chain.invoke({"raw_text": chunk})

            if isinstance(result, InterviewQuestionList):
                for q in result.questions:
                    if isinstance(q, InterviewQuestion) and not q.job_type:
                        q.job_type = effective_job_type
                all_questions.extend(result.questions)
            elif isinstance(result, dict) and "questions" in result:
                all_questions.extend(result["questions"])

        except Exception as e:
            logger.error(f"处理第 {i+1} 个文本块时出错: {e}")
            continue

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            if isinstance(q, InterviewQuestion):
                if not q.job_type:
                    q.job_type = effective_job_type
                f.write(q.model_dump_json() + "\n")
            elif isinstance(q, dict):
                if "job_type" not in q or not q["job_type"]:
                    q["job_type"] = effective_job_type
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"✅ 挖掘完成，共提取 {len(all_questions)} 个问题，保存至: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从教科书/面经逆向挖掘面试问题")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="输入文件路径，支持多个文件或目录（PDF/TXT）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出路径（默认: data/jobs/{job_type}/processed/mined_questions.jsonl）")
    parser.add_argument("--job-type", "-j", default=None, help="岗位类型（如：新媒体运营、产品经理）")
    parser.add_argument("--chunk-size", "-c", type=int, default=2000, help="文本块大小")

    args = parser.parse_args()

    job_config = load_job_config(args.job_type) if args.job_type else None
    if not job_config:
        from domain.job_configs import get_job_config_or_default
        job_config = get_job_config_or_default()

    output_path = args.output
    if not output_path:
        output_path = str(settings.get_job_processed_dir(job_config.job_type) / "mined_questions.jsonl")

    mine_textbook_to_questions(args.input, output_path, job_type=args.job_type, chunk_size=args.chunk_size)

```

### 其他文件

*根目录和其他文件*

#### `generate_project_doc.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目文档生成器
自动扫描项目结构并生成架构图和代码文档
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()
    OUTPUT_FILE: str = "PROJECT_DOCUMENTATION.md"
    
    EXCLUDE_DIRS: Set[str] = field(default_factory=lambda: {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".chainlit", ".files", ".vscode", ".idea", ".pytest_cache",
        "dist", "build", "*.egg-info", "jobs"
    })
    
    EXCLUDE_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        ".json", ".jsonl", ".toml", ".txt", ".pkl", ".faiss",
        ".pyc", ".pyo", ".bat", ".md", ".css", ".pdf", ".png", 
        ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".lock", ".log"
    })
    
    INCLUDE_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        ".py", ".yaml", ".yml"
    })
    
    EXCLUDE_FILES: Set[str] = field(default_factory=lambda: {
        ".env", ".gitignore", ".python-version", ".env.example",
        "PROJECT_DOCUMENTATION.md"
    })
    
    SENSITIVE_PATTERNS: List[str] = field(default_factory=lambda: [
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'password\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'OPENAI_API_KEY\s*=\s*["\'][^"\']+["\']',
    ])


config = Config()


def sanitize_content(content: str, file_path: Path) -> str:
    for pattern in config.SENSITIVE_PATTERNS:
        content = re.sub(pattern, '***REDACTED***', content, flags=re.IGNORECASE)
    return content


def should_exclude_dir(dir_name: str) -> bool:
    if dir_name in config.EXCLUDE_DIRS:
        return True
    for pattern in config.EXCLUDE_DIRS:
        if '*' in pattern and re.match(pattern.replace('*', '.*'), dir_name):
            return True
    return False


def should_include_file(file_name: str) -> bool:
    if file_name in config.EXCLUDE_FILES:
        return False
    
    ext = Path(file_name).suffix.lower()
    return ext in config.INCLUDE_EXTENSIONS


def get_language_by_extension(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    lang_map = {
        '.py': 'python',
        '.yaml': 'yaml',
        '.yml': 'yaml',
    }
    return lang_map.get(ext, 'text')


class TreeGenerator:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.output_lines = []
    
    def generate(self) -> str:
        self.output_lines = [f"{self.root_path.name}/"]
        self._generate_recursive(self.root_path, "", True)
        return "\n".join(self.output_lines)
    
    def _generate_recursive(self, current_path: Path, prefix: str, is_last: bool):
        try:
            items = sorted([
                item for item in current_path.iterdir()
                if not (item.is_dir() and should_exclude_dir(item.name))
                and item.name not in config.EXCLUDE_FILES
            ], key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "└── " if is_last_item else "├── "
            
            if item.is_dir():
                self.output_lines.append(f"{prefix}{connector}{item.name}/")
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                self._generate_recursive(item, new_prefix, is_last_item)
            else:
                self.output_lines.append(f"{prefix}{connector}{item.name}")


class CodePackager:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.files: List[Path] = []
    
    def scan(self) -> List[Path]:
        self.files = []
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
            for file in files:
                if should_include_file(file):
                    file_path = Path(root) / file
                    if file_path.name not in config.EXCLUDE_FILES:
                        self.files.append(file_path)
        self.files.sort()
        return self.files
    
    def get_file_content(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return sanitize_content(content, file_path)
        except UnicodeDecodeError:
            return "[Binary file - cannot display]"
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    def get_file_stats(self, file_path: Path) -> Dict:
        content = self.get_file_content(file_path)
        lines = content.split('\n') if not content.startswith('[') else []
        return {
            'lines': len(lines),
            'size': file_path.stat().st_size if file_path.exists() else 0
        }


class DocumentationGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.tree_generator = TreeGenerator(config.PROJECT_ROOT)
        self.code_packager = CodePackager(config.PROJECT_ROOT)
    
    def generate(self) -> str:
        sections = []
        
        sections.extend(self._generate_header())
        sections.extend(self._generate_toc())
        sections.extend(self._generate_architecture())
        sections.extend(self._generate_code_section())
        sections.extend(self._generate_footer())
        
        return "\n".join(sections)
    
    def _generate_header(self) -> List[str]:
        return [
            "# 面试教练项目文档",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 项目路径: `{self.config.PROJECT_ROOT}`",
            "",
            "---",
            ""
        ]
    
    def _generate_toc(self) -> List[str]:
        return [
            "## 目录",
            "",
            "- [项目架构](#项目架构)",
            "- [核心代码](#核心代码)",
            "  - [应用层 (app/)](#应用层-app)",
            "  - [核心层 (core/)](#核心层-core)",
            "  - [领域层 (domain/)](#领域层-domain)",
            "    - [岗位配置 (domain/job_configs/)](#岗位配置-domainjob_configs)",
            "  - [基础设施层 (infrastructure/)](#基础设施层-infrastructure)",
            "  - [脚本 (scripts/)](#脚本-scripts)",
            "- [统计信息](#统计信息)",
            "",
            "---",
            ""
        ]
    
    def _generate_architecture(self) -> List[str]:
        tree = self.tree_generator.generate()
        return [
            "## 项目架构",
            "",
            "```",
            tree,
            "```",
            "",
            "### 架构说明",
            "",
            "| 目录 | 说明 |",
            "|------|------|",
            "| `app/` | 应用入口和配置（Chainlit UI层） |",
            "| `core/` | 核心业务逻辑（LangGraph图、节点、技能、用户画像） |",
            "| `core/utils/` | 工具模块（LLM工厂、通用工具函数） |",
            "| `data/` | 数据层（向量索引、处理后数据、原始数据） |",
            "| `data/jobs/{岗位}/` | 各岗位独立数据目录（raw/processed/index） |",
            "| `domain/` | 领域模型、数据结构和业务验证 |",
            "| `domain/job_configs/` | 岗位配置文件（YAML格式） |",
            "| `infrastructure/` | 基础设施（解析器、检索器、工具） |",
            "| `scripts/` | 构建索引、数据处理、CLI管理工具 |",
            "| `public/` | Chainlit静态资源（CSS等） |",
            "| `.chainlit/` | Chainlit配置 |",
            "",
            "---",
            ""
        ]
    
    def _generate_code_section(self) -> List[str]:
        files = self.code_packager.scan()
        sections = ["## 核心代码\n"]
        
        grouped_files = self._group_files_by_directory(files)
        
        dir_order = [
            ('app', '应用层 (app/)', 'Chainlit应用入口和配置'),
            ('core', '核心层 (core/)', 'LangGraph图定义、节点实现、技能模块、用户画像'),
            ('domain', '领域层 (domain/)', '数据模型、业务实体、岗位配置'),
            ('infrastructure', '基础设施层 (infrastructure/)', '文件解析、向量检索、工具集成'),
            ('scripts', '脚本 (scripts/)', '构建索引、数据处理、CLI管理工具'),
        ]
        
        for dir_name, title, description in dir_order:
            if dir_name in grouped_files:
                sections.extend(self._generate_directory_section(
                    dir_name, title, description, grouped_files[dir_name]
                ))
        
        other_files = []
        for file_path in files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            parts = rel_path.parts
            if parts[0] not in [d[0] for d in dir_order]:
                other_files.append(file_path)
        
        if other_files:
            sections.extend(self._generate_directory_section(
                'other', '其他文件', '根目录和其他文件', other_files
            ))
        
        sections.extend(self._generate_statistics(files))
        
        return sections
    
    def _group_files_by_directory(self, files: List[Path]) -> Dict[str, List[Path]]:
        grouped = {}
        for file_path in files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            parts = rel_path.parts
            top_dir = parts[0] if len(parts) > 1 else 'root'
            if top_dir not in grouped:
                grouped[top_dir] = []
            grouped[top_dir].append(file_path)
        return grouped
    
    def _generate_directory_section(self, dir_name: str, title: str, 
                                     description: str, files: List[Path]) -> List[str]:
        sections = [
            f"### {title}",
            "",
            f"*{description}*",
            ""
        ]
        
        job_config_files = []
        other_files = []
        
        for file_path in files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            if 'job_configs' in str(rel_path) and file_path.suffix in ('.yaml', '.yml'):
                job_config_files.append(file_path)
            else:
                other_files.append(file_path)
        
        for file_path in other_files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            content = self.code_packager.get_file_content(file_path)
            lang = get_language_by_extension(file_path)
            
            sections.append(f"#### `{rel_path}`")
            sections.append("")
            sections.append(f"```{lang}")
            sections.append(content)
            sections.append("```")
            sections.append("")
        
        if job_config_files:
            sections.append("#### 岗位配置 (domain/job_configs/)")
            sections.append("")
            sections.append("*岗位配置文件，定义各岗位的Prompt、示例数据等*")
            sections.append("")
            
            for file_path in job_config_files:
                rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
                content = self.code_packager.get_file_content(file_path)
                
                sections.append(f"##### `{rel_path}`")
                sections.append("")
                sections.append("```yaml")
                sections.append(content)
                sections.append("```")
                sections.append("")
        
        return sections
    
    def _generate_statistics(self, files: List[Path]) -> List[str]:
        total_lines = 0
        total_size = 0
        py_count = 0
        yaml_count = 0
        
        for file_path in files:
            stats = self.code_packager.get_file_stats(file_path)
            total_lines += stats['lines']
            total_size += stats['size']
            
            ext = file_path.suffix.lower()
            if ext == '.py':
                py_count += 1
            elif ext in ('.yaml', '.yml'):
                yaml_count += 1
        
        return [
            "---",
            "",
            "## 统计信息",
            "",
            f"- **Python 文件数量**: {py_count}",
            f"- **YAML 配置文件数量**: {yaml_count}",
            f"- **总文件数量**: {len(files)}",
            f"- **总代码行数**: {total_lines:,}",
            f"- **总文件大小**: {total_size / 1024:.1f} KB",
            ""
        ]
    
    def _generate_footer(self) -> List[str]:
        return [
            "---",
            "",
            "*本文档由自动化脚本生成，敏感信息已脱敏处理*",
            ""
        ]
    
    def save(self, output_path: Optional[Path] = None):
        content = self.generate()
        output = output_path or self.config.PROJECT_ROOT / self.config.OUTPUT_FILE
        
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        
        return output


def main():
    print("=" * 60)
    print("项目文档生成器")
    print("=" * 60)
    print()
    
    print(f"[1/3] 扫描项目目录: {config.PROJECT_ROOT}")
    generator = DocumentationGenerator(config)
    
    print("[2/3] 生成项目架构图...")
    tree = generator.tree_generator.generate()
    print(f"      发现 {len(tree.splitlines())} 个目录/文件")
    
    print("[3/3] 打包核心代码...")
    files = generator.code_packager.scan()
    py_count = sum(1 for f in files if f.suffix == '.py')
    yaml_count = sum(1 for f in files if f.suffix in ('.yaml', '.yml'))
    print(f"      共 {py_count} 个 Python 文件, {yaml_count} 个 YAML 配置")
    
    output_path = generator.save()
    print()
    print("=" * 60)
    print(f"[完成] 文档已生成: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

#### `start.py`

```python
#!/usr/bin/env python3
"""
跨平台启动脚本
支持 Windows / macOS / Linux
"""

import subprocess
import sys
import os
import socket
from pathlib import Path

def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def kill_port_process(port: int) -> bool:
    """终止占用指定端口的进程（仅 Windows）"""
    if sys.platform != "win32":
        return False
    
    try:
        result = subprocess.run(
            f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            pids = set()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    pids.add(parts[-1])
            
            for pid in pids:
                if pid.isdigit():
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                    print(f"[清理] 已终止占用端口 {port} 的进程 (PID: {pid})")
            return True
    except Exception:
        pass
    return False

def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 50)
    print("[启动] 私人专属面试顾问 V1.0")
    print("=" * 50)
    print()
    
    env_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not env_key:
        print("[警告] 未检测到 API 密钥环境变量！")
        print("请设置环境变量：")
        print("  Windows: set DASHSCOPE_API_KEY=your_api_key")
        print("  macOS/Linux: export DASHSCOPE_API_KEY=your_api_key")
        print()
        print("或者创建 .env 文件配置 OPENAI_API_KEY")
        print()
    
    port = 8000
    
    if not is_port_available(port):
        print(f"[提示] 端口 {port} 已被占用，正在清理...")
        kill_port_process(port)
        import time
        time.sleep(1)
    
    print(f"正在启动服务... (端口: {port})")
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "chainlit", "run", "app/main.py", "--port", str(port)],
            cwd=script_dir,
            check=True
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except FileNotFoundError:
        print("[错误] 未找到 chainlit，请先安装依赖：")
        print("   pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 启动失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

```

---

## 统计信息

- **Python 文件数量**: 45
- **YAML 配置文件数量**: 1
- **总文件数量**: 46
- **总代码行数**: 4,536
- **总文件大小**: 172.8 KB

---

*本文档由自动化脚本生成，敏感信息已脱敏处理*
