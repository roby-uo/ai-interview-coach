import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from domain.schemas import InterviewState

logger = logging.getLogger(__name__)


def parse_weakness_prefix(weakness_prefix: str) -> Tuple[List[str], List[str]]:
    if not weakness_prefix:
        return [], []
    
    weakness_tags = []
    forbidden_words = []
    
    weakness_match = re.search(r'【用户弱点：([^】]+)】', weakness_prefix)
    if weakness_match:
        tags_str = weakness_match.group(1)
        weakness_tags = [tag.strip() for tag in tags_str.split('、') if tag.strip()]
    
    forbidden_match = re.search(r'【雷区忌口：([^】]+)】', weakness_prefix)
    if forbidden_match:
        forbid_str = forbidden_match.group(1)
        forbidden_words = [word.strip() for word in forbid_str.split('、') if word.strip()]
    
    return weakness_tags, forbidden_words


def build_weakness_prefix(weakness_tags: List[str], forbidden_words: List[str]) -> str:
    if not weakness_tags and not forbidden_words:
        return ""
    
    parts = []
    if weakness_tags:
        parts.append(f"【用户弱点：{'、'.join(weakness_tags)}】")
    if forbidden_words:
        parts.append(f"【雷区忌口：{'、'.join(forbidden_words)}】")
    
    return "".join(parts)


def parse_judge_score(response: str) -> Optional[int]:
    if not response:
        return None
    
    score_patterns = [
        r'📝\s*\*?\*?得分[：:]\*?\*?\s*(\d+)\s*分',
        r'得分[：:]\s*(\d+)\s*分',
        r'(\d+)\s*分',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
    
    return None


def extract_new_weakness_from_judgment(response: str, current_tags: List[str]) -> List[str]:
    if not response:
        return []
    
    fatal_match = re.search(r'🔪\s*\*?\*?致命伤[：:]\*?\*?\s*(.+?)(?=🧭|📝|$)', response, re.DOTALL)
    if not fatal_match:
        return []
    
    fatal_text = fatal_match.group(1).strip()
    
    weakness_keywords = [
        "数据", "逻辑", "框架", "归因", "量化", "闭环",
        "方法论", "复盘", "拆解", "结构", "深度", "细节"
    ]
    
    new_weaknesses = []
    for keyword in weakness_keywords:
        if keyword in fatal_text:
            potential_weakness = f"{keyword}待强化"
            if potential_weakness not in current_tags:
                similar_exists = any(keyword in tag for tag in current_tags)
                if not similar_exists:
                    new_weaknesses.append(potential_weakness)
    
    return new_weaknesses[:2]


def incremental_update_weakness(
    current_prefix: str,
    response: str,
    turn_count: int,
    update_interval: int = 5,
    score_threshold: int = 80
) -> Optional[str]:
    if turn_count % update_interval != 0:
        return None
    
    weakness_tags, forbidden_words = parse_weakness_prefix(current_prefix)
    
    if not weakness_tags:
        logger.info("🔄 [弱点更新] 当前无弱点标签，跳过更新")
        return None
    
    score = parse_judge_score(response)
    
    if score is None:
        logger.info("🔄 [弱点更新] 未检测到判卷得分，跳过更新")
        return None
    
    logger.info(f"🔄 [弱点更新] 检测到判卷得分: {score}分")
    
    if score >= score_threshold and len(weakness_tags) > 1:
        removed_tag = weakness_tags.pop(0)
        logger.info(f"🔄 [弱点更新] 得分达标({score}>={score_threshold})，移除弱点: {removed_tag}")
        
        new_weaknesses = extract_new_weakness_from_judgment(response, weakness_tags)
        if new_weaknesses:
            weakness_tags.extend(new_weaknesses)
            logger.info(f"🔄 [弱点更新] 从判卷中发现新弱点: {new_weaknesses}")
        
        return build_weakness_prefix(weakness_tags, forbidden_words)
    
    elif score < 60:
        new_weaknesses = extract_new_weakness_from_judgment(response, weakness_tags)
        if new_weaknesses:
            weakness_tags.extend(new_weaknesses)
            logger.info(f"🔄 [弱点更新] 得分较低({score}<60)，添加新弱点: {new_weaknesses}")
            return build_weakness_prefix(weakness_tags, forbidden_words)
    
    return None


async def update_history_node(state: InterviewState) -> Dict[str, Any]:
    new_history = list(state["history"])
    
    if state["user_input"]:
        new_history.append(HumanMessage(content=state["user_input"]))
    if state["response"]:
        new_history.append(AIMessage(content=state["response"]))

    max_turns = settings.MAX_HISTORY_TURNS
    if len(new_history) > max_turns * 2:
        new_history = new_history[-(max_turns * 2):]

    new_turn_count = state["turn_count"] + 1

    result = {
        "history": new_history,
        "turn_count": new_turn_count,
        "user_input": ""
    }

    current_prefix = state.get("weakness_prefix", "")
    last_response = state.get("response", "")
    
    updated_prefix = incremental_update_weakness(
        current_prefix=current_prefix,
        response=last_response,
        turn_count=new_turn_count
    )
    
    if updated_prefix:
        result["weakness_prefix"] = updated_prefix
        logger.info(f"📝 [更新历史节点] 弱点画像已更新: {updated_prefix[:50]}...")

    logger.info(f"📝 [更新历史节点] 轮次: {new_turn_count}, 历史长度: {len(new_history)}")

    return result
