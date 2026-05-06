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
