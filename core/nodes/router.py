import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from domain.schemas import InterviewState, GodDecision, VALID_ACTIONS
from core.utils.llm_factory import llm_factory

logger = logging.getLogger(__name__)


async def route_node(state: InterviewState) -> Dict[str, Any]:
    llm = llm_factory.get_router_llm()

    system_prompt = """你是面试控制中枢（上帝大脑）。分析用户输入，决策调用哪个 Skill。

═══════════════════════════════════════
§1 输出格式（JSON 四字段）
═══════════════════════════════════════

1. reasoning: 三步显式推理（必须填写）
   - Step1 用户状态：有回答意愿但逻辑断裂 / 纯情绪发泄 / 顺着梯子爬但卡住 / 完全偏题 / 主动投降
   - Step2 教学阶段：开题尚未作答 / 已尝试但质量差 / 给过脚手架仍无法推进 / 回答完整需判卷 / 判卷后追问
   - Step3 动作推导：基于上述两步，选哪个 Skill？为何其他选项不合适？

2. intent_dim: 用户这句话的动态语义定性（用你的理解总结，不要选死词）

3. intended_action: 调用的 Skill 名称（限选以下五个）

4. task_prompt: 给底层 Skill 的执行指令
   【重要】底层 Skill 没有眼睛，看不到上下文。你必须把对上下文的理解、对用户的判断、要求用什么语气回复什么内容，全部写在这里！

═══════════════════════════════════════
§2 决策规则（核心）
═══════════════════════════════════════

┌──────────────────┬─────────────────────────────────┬──────────────────────────┐
│ Skill            │ 触发条件                         │ task_prompt 要点          │
├──────────────────┼─────────────────────────────────┼──────────────────────────┤
│ socratic_probe   │ 有回答意愿，但逻辑不完整/有漏洞   │ 点出漏洞位置，指令反问逼深 │
│ scaffold_rescue  │ 连续卡壳/极度迷茫（最后手段）     │ 给半截公式/思路+收尾提问   │
│ judge_strike     │ 回答完整/主动投降/梯子后仍无法答  │ 按判卷格式打分，指出致命伤 │
│ smart_redirection│ 偏题闲聊/混乱输入/自相矛盾       │ 简短回应+强制拉回面试话题  │
│ teaching_supplement│ 判卷后用户追问                 │ 补充例子/解释/延伸        │
└──────────────────┴─────────────────────────────────┴──────────────────────────┘

【决策优先级】
- 判卷后追问 → 优先 teaching_supplement
- 混乱/自相矛盾 → smart_redirection（非其他）
- scaffold_rescue 是最后手段，用户还有挖掘空间时不要给梯子

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

【示例1 — socratic_probe】
上下文：教练问数据归因能力，用户说了"加大投放"
用户："我觉得主要就是多投一点广告，然后流量就上来了"
→ reasoning: "Step1 用户有回答意愿，但只抛了一个动作名词，缺乏归因逻辑。Step2 开题后首次作答，用户并非完全不会，只是深度不够。Step3 用 socratic_probe 刺穿逻辑薄弱点，给梯子太早会剥夺思考机会。"
→ intended_action: socratic_probe
→ task_prompt: "用户回答'多投广告流量就上来'，这是单一归因谬误。请用反问句逼他思考：流量上来就等于结果好吗？中间漏了什么关键环节？"

【示例2 — scaffold_rescue】
上下文：教练连续追问数据漏斗，用户两次说"不知道"
用户："我真的完全没思路...你说的漏斗我听都没听过"
→ reasoning: "Step1 用户连续两次无法作答，已进入认知崩溃状态。Step2 苏格拉底追问无效，用户连基础概念都没有。Step3 必须用 scaffold_rescue 给半截公式托底，否则对话陷入死循环。"
→ intended_action: scaffold_rescue
→ task_prompt: "用户连续两次无法作答，连漏斗概念都不知道。请调用工具检索'数据漏斗'资料，然后给出半截公式（如'漏斗模型的核心是从___到___的逐层___'）让用户填空，填空后紧跟收尾提问。"

【示例3 — judge_strike】
上下文：用户完整回答了内容复盘问题（STAR+数据对比+漏斗拆解）
用户："我觉得我回答得差不多了，你看看我哪里还能改进"
→ reasoning: "Step1 用户主动请求评估，前两轮回答已覆盖 STAR 和数据拆解。Step2 已过两轮深入追问，用户表现稳定。Step3 应该用 judge_strike 正式判卷。"
→ intended_action: judge_strike
→ task_prompt: "用户已完整回答内容复盘问题，使用了 STAR 框架、数据对比和漏斗拆解。请调用工具用 train 模式检索判卷清单，按格式打分。"

【示例4 — smart_redirection】
上下文：教练问用户增长策略
用户："诶你们AI是不是用那个什么大模型做的？现在大模型好火啊"
→ reasoning: "Step1 用户完全偏题，从面试问题跳到对 AI 技术的好奇。Step2 面试训练被中断，需要拉回。Step3 用 smart_redirection 简短回应后强制拉回。"
→ intended_action: smart_redirection
→ task_prompt: "用户偏题聊大模型。请简短回应一句（不超过15字），然后立刻以面试相关问题收尾，强制拉回用户增长策略话题。"

═══════════════════════════════════════
§4 绝对红线
═══════════════════════════════════════

- 你【绝对不能】因为用户闲聊或追问就判断为"会话结束"
- 只有用户明确说"我要结束训练"或"生成报告"时，才考虑结束
- 判卷后的追问 → 优先路由到 teaching_supplement
- 用户输入混乱或自相矛盾 → smart_redirection（非其他技能）
- scaffold_rescue 是最后手段，不要在用户还有挖掘空间时就给梯子"""

    human_content = f"""【近5轮上下文摘要 (JSON)】
{state['short_summary']}

【用户当前原始输入】
{state['user_input']}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]

    chain = llm.with_structured_output(GodDecision)
    decision = await chain.ainvoke(messages)

    logger.info(f"👁️ [路由节点] 意图定性: {decision.intent_dim} | 决策动作: {decision.intended_action}")

    if decision.intended_action not in VALID_ACTIONS:
        logger.warning(f"⚠️ [路由节点] 非法动作: {decision.intended_action}，强制回退到 smart_redirection")
        return {
            "decision": {
                "target_skill": "smart_redirection",
                "task_prompt": decision.task_prompt
            }
        }

    return {
        "decision": {
            "target_skill": decision.intended_action,
            "task_prompt": decision.task_prompt
        }
    }
