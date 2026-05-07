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
        default="",
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
            if "questions" not in v and "question" in v:
                v = {"questions": [v]}
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
