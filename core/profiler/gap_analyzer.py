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
