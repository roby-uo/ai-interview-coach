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
