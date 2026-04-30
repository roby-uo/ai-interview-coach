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
