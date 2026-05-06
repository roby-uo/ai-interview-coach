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
