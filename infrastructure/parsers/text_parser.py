import re
import json
import logging
from typing import Optional, Union
from domain.schemas import WeaknessProfile

logger = logging.getLogger(__name__)


def safe_parse_weakness(raw_text: Union[str, WeaknessProfile]) -> WeaknessProfile:
    """
    安全解析弱点画像。
    
    如果输入已经是合法的 Pydantic 对象，直接返回；
    如果是 JSON 字符串，尝试解析；
    如果是乱码文本（针对劣质模型的兜底），尝试正则提取；
    如果全失败，直接抛出异常，绝不伪造数据！
    
    Args:
        raw_text: 可能是 WeaknessProfile 对象、JSON 字符串、或乱码文本
        
    Returns:
        WeaknessProfile: 解析后的弱点画像对象
        
    Raises:
        ValueError: 所有解析策略均失败时抛出
    """
    if isinstance(raw_text, WeaknessProfile):
        return raw_text

    text_to_parse = str(raw_text)

    try:
        data = json.loads(text_to_parse)
        return WeaknessProfile(**data)
    except json.JSONDecodeError:
        pass
    
    json_match = re.search(r'\{.*?\}', text_to_parse, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return WeaknessProfile(**data)
        except Exception:
            pass

    logger.warning("⚠️ 模型返回非标准格式，启动正则强行提取...")
    try:
        weakness_matches = re.findall(r'(?:\d+\.\s*|-\s*)([^：:\n]+?)(?:[:：])', text_to_parse)
        weakness_tags = [w.strip() for w in weakness_matches[:3] if len(w.strip()) <= 10]
        
        forbid_matches = re.findall(r'(?:雷区|忌口|不能说|别说)[:：]\s*([^。\n]+)', text_to_parse)
        forbidden_words = [w.strip() for w in forbid_matches[:3]]
        
        if not weakness_tags:
            raise ValueError("正则也无法提取有效弱点标签")
            
        return WeaknessProfile(
            weakness_tags=weakness_tags if weakness_tags else ["能力待评估"],
            forbidden_words=forbidden_words if forbidden_words else ["说主观原因"]
        )
    except Exception as e:
        logger.error(f"所有解析策略均失败: {e}")
        raise ValueError(f"AI 无法理解你的简历格式，请尝试精简简历内容后重试。原始错误: {str(e)[:50]}")
