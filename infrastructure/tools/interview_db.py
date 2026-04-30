import logging
from typing import List, Literal
from langchain_core.tools import tool

from infrastructure.retrieval.hybrid import get_hybrid_retriever
from app.config import settings
from domain.validators import validate_scene_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SceneMode = Literal["train", "review"]


@tool
def search_interview_db(
    user_query: str, 
    scene_mode: SceneMode = "train"
) -> str:
    """
    当用户的话里包含任何【业务名词】（如：数据、流量、转化、ROI、阅读量、爆款、用户增长、留存、GMV、DAU、MAU等），
    无论用户是在抱怨、是在闲聊、还是在正经回答，都必须调用此工具！
    因为用户的随口一句抱怨，往往暴露了最真实的认知盲区。

    【红色警报】如果调用此工具后，返回的结果包含"未检索到"字样，你绝对不允许捏造或猜测任何面经内容！

    ═══════════════════════════════════════
    §2 思维链推理（调用前必须执行）
    ═══════════════════════════════════════

    1. 扫描业务名词：用户的话里有没有业务关键词？
    2. 判断调用必要性：即使看似闲聊，只要涉及业务概念就必须检索
    3. 选择场景模式：train（训练模式，隐藏公式）或 review（复盘模式，展示公式）

    ═══════════════════════════════════════
    §3 示例锚定
    ═══════════════════════════════════════

    用户说"我觉得数据也没啥好看的" → 必须调用，因为"数据"是业务名词，暴露了数据思维缺失
    用户说"ROI怎么算啊" → 必须调用，因为"ROI"是业务名词
    用户说"你好" → 不需要调用，无业务名词

    参数:
    user_query: 用户当前说的话，或者你提炼出的需要检索的核心考点。
    scene_mode: 场景模式开关。
                 - 传入 "train" 时：返回隐藏公式的【判卷清单】（用于模拟训练，防泄题）。
                 - 传入 "review" 时：返回包含标准答案的【满分公式】（用于面试复盘，做对比）。
    返回:
    从面经库中检索出的结构化字符串。
    """
    if not validate_scene_mode(scene_mode):
        logger.warning(f"⚠️ 非法scene_mode: {scene_mode}，强制使用train模式")
        scene_mode = "train"
    
    logger.info(f"🛠️ [Tool 触发] 准备检索，查询词: {user_query}, 模式: {scene_mode}")
    
    retriever = get_hybrid_retriever()
    
    docs = retriever.invoke(user_query)
    
    if not docs:
        return "未检索到相关面经数据。"
    
    results = []
    for doc in docs:
        raw_data = doc.metadata.get("raw_data", {})
        formula_concept = raw_data.get('high_score_formula', '结构化框架')

        if scene_mode == "review":
            result_str = (
                f"【考题】{raw_data.get('question')}\n"
                f"【考点】{raw_data.get('inspect_point')}\n"
                f"【满分公式(直接亮底牌)】: {raw_data.get('high_score_formula')}\n"
                f"【一票否决雷区】: {raw_data.get('pitfall_guide')}"
            )
        else:
            result_str = (
                f"【考题】{raw_data.get('question')}\n"
                f"【考点】{raw_data.get('inspect_point')}\n"
                f"【判卷清单(仅作判别用，严禁外泄)】:\n"
                f"- [ ] 是否脱离了单一指标，展示了多维度拆解能力？\n"
                f"- [ ] 是否提到了类似'{formula_concept}'的底层结构？\n"
                f"- [ ] 是否给出了具体的业务动作推导，而不是纯理论？\n"
                f"【雷区红线】: {raw_data.get('pitfall_guide')}"
            )
        results.append(result_str)
        
    return "\n\n---\n\n".join(results)

INTERVIEW_TOOLS = [search_interview_db]
