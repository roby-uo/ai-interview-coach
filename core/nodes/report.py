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
