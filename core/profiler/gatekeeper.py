import logging
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
            llm = llm_factory.get_extractor_llm()
            self._chain = llm.with_structured_output(GatekeeperResult)

    async def aparse(self, raw_input: str) -> GatekeeperResult:
        self._ensure_initialized()
        messages = [
            SystemMessage(content=GATEKEEPER_PROMPT),
            HumanMessage(content=f"用户输入内容：\n{raw_input}")
        ]
        result = await self._chain.ainvoke(messages)
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
        result = self._chain.invoke(messages)
        logger.info(
            f"🚪 [门卫] 识别结果: "
            f"有简历={result.has_resume}, 有JD={result.has_jd}, "
            f"置信度={result.confidence:.2f}"
        )
        return result


gatekeeper = Gatekeeper()
