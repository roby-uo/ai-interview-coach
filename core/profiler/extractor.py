import logging
import json
import re
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from domain.schemas import WeaknessProfile
from infrastructure.parsers.text_parser import safe_parse_weakness
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)

DIAGNOSE_WEAKNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位阅人无数的资深新媒体 HR，同时也是一位善于发现潜力的职业导师。
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
JD摘要：新媒体运营，需数据分析能力，需 ROI 意识，需用户增长方法论

→ 输出：
{{
  "weakness_tags": ["数据归因缺失", "增长方法论空白", "ROI意识薄弱"],
  "forbidden_words": ["阅读量挺高的", "感觉效果不错", "领导安排的", "运气好"]
}}

→ 推理链路：JD 要求3个核心能力（数据/增长/ROI），简历只展示了内容产出，三个维度全部缺失。候选人在面试中最可能用"阅读量高"来掩饰数据归因缺失，用"感觉效果不错"来回避 ROI 量化，用"领导安排"来逃避主动性证明。"""),

    ("human", """【候选人简历摘要】：
{resume_text}

【目标岗位JD】：
{jd_text}

请输出 JSON 格式的弱点分析结果。""")
])


class Profiler:
    def __init__(self):
        self._chain = None

    def _ensure_initialized(self):
        if self._chain is None:
            llm = llm_factory.get_fast_llm()
            self._llm = llm
            self._prompt = DIAGNOSE_WEAKNESS_PROMPT

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


_profiler = None


def get_profiler() -> Profiler:
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


profiler = get_profiler()
