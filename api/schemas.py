"""
Coze API 请求/响应模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ActionType(str, Enum):
    CHAT = "chat"
    SELECT_JOB = "select_job"
    QUICK_START = "quick_start"
    GENERATE_REPORT = "generate_report"
    RESET = "reset"
    USE_DEFAULT_JD = "use_default_jd"


class InterviewRequest(BaseModel):
    user_id: str = Field(..., description="Coze会话ID，用于映射LangGraph thread")
    message: str = Field(default="", description="用户消息文本")
    action: ActionType = Field(default=ActionType.CHAT, description="操作类型")
    job_type: Optional[str] = Field(default=None, description="岗位类型(选岗时必填)")
    file_url: Optional[str] = Field(default=None, description="文件URL(简历等)")


class ActionButton(BaseModel):
    label: str = Field(..., description="按钮显示文本")
    action: str = Field(..., description="点击时触发的action")
    value: Optional[str] = Field(default=None, description="附加参数(如job_type)")


class InterviewResponse(BaseModel):
    reply: str = Field(..., description="回复内容(Markdown格式)")
    actions: List[ActionButton] = Field(default_factory=list, description="可用操作按钮")
    status: str = Field(default="ok", description="状态: ok / processing / waiting / error")
    task_id: Optional[str] = Field(default=None, description="异步任务ID，processing时返回，用于轮询")
