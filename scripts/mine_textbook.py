import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import InterviewQuestion, InterviewQuestionList
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. 面经洗矿 Prompt (长文 -> 结构化JSON)
# ==========================================
MINE_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位经验丰富的新媒体运营面试官兼数据分析师，擅长从实战角度提炼面试核心要点。
你的任务是从一篇杂乱的面经长文中，精准提取出面试问题及其背后的底层逻辑。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

你必须严格按照提供的 JSON Schema 格式输出，不要输出任何解释性文字，只要纯 JSON。

提取要求：
1. 问题必须是面试官的真实提问，不要总结。
2. 考察点 要一针见血（如：数据归因能力，而不是泛泛的"分析能力"）。
3. 高分公式 必须具备可执行性（如：STAR+漏斗模型+具体Action）。
4. 避坑指南 必须指出具体的"送命题"话术，并用引导性的语气说明为什么这样回答会扣分。

重要：每个字段都必须填写，不能为空字符串。如果原文没有直接给出，请根据你的专业知识推断并填写。

═══════════════════════════════════════
§2 思维链推理（提取前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 定位提问：从面经中找到面试官的原话提问
2. 逆向推导：这个问题表面在问 X，实际在考察 Y（Y 才是 inspect_point）
3. 构造公式：基于 Y，设计一个可复用的回答框架（high_score_formula）
4. 预判雷区：基于 Y，推测应届生最容易犯的回答误区（pitfall_guide）

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

面经片段："面试官问我怎么做一个活动的复盘，我说就是看看数据哪里好哪里不好，然后面试官皱了皱眉"

→ 输出：
{{
  "question": "你是怎么做活动复盘的？",
  "inspect_point": "结构化复盘能力（非直觉式总结）",
  "high_score_formula": "目标回溯→数据漏斗拆解（逐层归因）→关键发现→Action项+量化验证",
  "pitfall_guide": "千万别用'看看数据哪里好哪里不好'这种模糊表述——这暴露了你只会看表面数字，不会做归因。应该说'我通过漏斗拆解定位到转化率最低的环节是X，原因是Y'"
}}"""),

    ("human", "下面是待提取的面经长文内容：\n\n{raw_text}")
])

# ==========================================
# 2. 教科书逆向工程 Prompt (专供 800 题 PDF)
# ==========================================
MINE_TEXTBOOK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深的新媒体 HR 兼教学设计师，现在你要根据一份"教科书式的标准参考答案"，逆向出一套用来帮助应届生系统提升面试能力的专业题库。
注意：给你的原材料可能是干瘪的、正确的废话。你的任务是将其转化为【生动易懂且具有实操指导价值】的内容！

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

提取与再创作要求：
1. question：保留原题的核心考点，但如果原题太长，精简为口语化的面试提问。
2. inspect_point：一针见血，不要写"综合能力"，要写"数据漏斗拆解能力"、"危机公关预案意识"。
3. high_score_formula：【核心】不要抄原答案！把原答案里的步骤，浓缩成朗朗上口的"万能公式"或"解题套路"。
4. pitfall_guide：【核心】原答案里绝对没有这个！你必须基于这个考点，自己脑补出一种"应届生最容易犯的回答误区"。

═══════════════════════════════════════
§2 思维链推理（逆向工程前必须执行 — 不输出推理过程）
═══════════════════════════════════════

1. 解构原答案：标准答案的核心步骤是什么？每步在解决什么问题？
2. 浓缩公式：将步骤序列压缩为一个可记忆的"口诀"或"框架"
3. 口语化提问：把学术化/书面化的题目翻译成面试官真正会问的口语
4. 预判踩坑：站在应届生视角，他们最可能在哪里翻车？

═══════════════════════════════════════
§3 示例锚定
═══════════════════════════════════════

原文片段："内容运营的核心在于通过优质内容吸引用户，建立品牌认知，最终实现转化。运营者需要关注内容的选题策划、创作质量、分发渠道和效果评估四个环节。"

→ 输出：
{{
  "question": "你做内容运营，怎么判断自己产出的内容是不是'好内容'？",
  "inspect_point": "内容效果评估体系（非主观判断）",
  "high_score_formula": "选题(热点匹配度+用户需求度)→创作(完播率/阅读完成率)→分发(渠道ROI对比)→评估(北极星指标+归因分析)",
  "pitfall_guide": "很多同学会说'我觉得内容好就是好'或'阅读量高就是好内容'——前者是主观自嗨，后者是单一指标陷阱。好内容必须有可量化的评估体系，且不同阶段看不同指标。"
}}"""),

    ("human", "下面是一段从题库中截取的【题目】和【标准答案】片段，请进行逆向提取：\n\n{raw_text}")
])


def mine_textbook_to_questions(textbook_path: str, output_path: str, chunk_size: int = 2000):
    """
    从教科书 PDF 中逆向挖掘面试问题
    """
    with open(textbook_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    llm = ChatOpenAI(
        model=settings.ROUTER_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.1
    )

    chain = MINE_TEXTBOOK_PROMPT | llm.with_structured_output(InterviewQuestionList)

    all_questions = []

    for i, chunk in enumerate(chunks):
        logger.info(f"正在处理第 {i+1}/{len(chunks)} 个文本块...")

        try:
            result = chain.invoke({"raw_text": chunk})

            if isinstance(result, InterviewQuestionList):
                all_questions.extend(result.questions)
            elif isinstance(result, dict) and "questions" in result:
                all_questions.extend(result["questions"])

        except Exception as e:
            logger.error(f"处理第 {i+1} 个文本块时出错: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            if isinstance(q, InterviewQuestion):
                f.write(q.model_dump_json() + "\n")
            elif isinstance(q, dict):
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"✅ 挖掘完成，共提取 {len(all_questions)} 个问题，保存至: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从教科书逆向挖掘面试问题")
    parser.add_argument("--input", "-i", default="./data/raw/raw_textbook.txt", help="教科书文本路径")
    parser.add_argument("--output", "-o", default="./data/processed/mined_questions.jsonl", help="输出路径")
    parser.add_argument("--chunk-size", "-c", type=int, default=2000, help="文本块大小")

    args = parser.parse_args()

    mine_textbook_to_questions(args.input, args.output, args.chunk_size)
