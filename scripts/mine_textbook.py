import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import InterviewQuestion, InterviewQuestionList
from domain.job_configs import load_job_config
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_mine_questions_prompt(job_config) -> ChatPromptTemplate:
    interviewer_persona = job_config.mine_interviewer_persona
    job_type = job_config.job_type

    return ChatPromptTemplate.from_messages([
        ("system", f"""{interviewer_persona}
你的任务是从一篇杂乱的{job_type}面经长文中，精准提取出【所有】面试问题及其背后的底层逻辑。

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

你必须严格按照提供的 JSON Schema 格式输出，不要输出任何解释性文字，只要纯 JSON。
输出格式为：{{{{"questions": [问题1, 问题2, ...]}}}}

⚠️ 关键要求：必须提取文本中的【每一个】面试问题，不要遗漏！不要只总结一个泛泛的问题！

提取要求：
1. 问题必须是面试官的真实提问，不要总结合并。原文有N个问题就输出N条。
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
{{{{
  "question": "你是怎么做活动复盘的？",
  "inspect_point": "结构化复盘能力（非直觉式总结）",
  "high_score_formula": "目标回溯→数据漏斗拆解（逐层归因）→关键发现→Action项+量化验证",
  "pitfall_guide": "千万别用'看看数据哪里好哪里不好'这种模糊表述——这暴露了你只会看表面数字，不会做归因。应该说'我通过漏斗拆解定位到转化率最低的环节是X，原因是Y'"
}}}}"""),

        ("human", "下面是待提取的面经长文内容：\n\n{{raw_text}}")
    ])


def _build_mine_textbook_prompt(job_config) -> ChatPromptTemplate:
    hr_persona = job_config.mine_hr_persona
    job_type = job_config.job_type

    return ChatPromptTemplate.from_messages([
        ("system", f"""{hr_persona}
注意：给你的原材料可能是干瘪的、正确的废话。你的任务是将其转化为【生动易懂且具有实操指导价值】的内容！

═══════════════════════════════════════
§1 基座规则
═══════════════════════════════════════

你必须严格按照提供的 JSON Schema 格式输出，不要输出任何解释性文字，只要纯 JSON。
输出格式为：{{{{"questions": [问题1, 问题2, ...]}}}}

⚠️ 关键要求：必须提取文本中的【每一个】面试问题，不要遗漏！原文有N道题就输出N条，不要合并或只总结一个！

提取与再创作要求：
1. question：保留原题的核心考点，但如果原题太长，精简为口语化的面试提问。
2. inspect_point：一针见血，不要写"综合能力"，要写具体的能力维度。
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
{{{{
  "question": "你做内容运营，怎么判断自己产出的内容是不是'好内容'？",
  "inspect_point": "内容效果评估体系（非主观判断）",
  "high_score_formula": "选题(热点匹配度+用户需求度)→创作(完播率/阅读完成率)→分发(渠道ROI对比)→评估(北极星指标+归因分析)",
  "pitfall_guide": "很多同学会说'我觉得内容好就是好'或'阅读量高就是好内容'——前者是主观自嗨，后者是单一指标陷阱。好内容必须有可量化的评估体系，且不同阶段看不同指标。"
}}}}"""),

        ("human", "下面是一段从题库中截取的【题目】和【标准答案】片段，请进行逆向提取：\n\n{{raw_text}}")
    ])


def _resolve_input_paths(input_paths: list) -> list:
    resolved = []
    for p in input_paths:
        path = Path(p)
        if path.is_file():
            resolved.append(path)
        elif path.is_dir():
            for f in sorted(path.iterdir()):
                if f.suffix.lower() in ('.pdf', '.txt'):
                    resolved.append(f)
        else:
            logger.warning(f"跳过不存在的路径: {p}")
    return resolved


def _extract_text_from_file(file_path: Path) -> str:
    if file_path.suffix.lower() == '.pdf':
        from infrastructure.parsers.file_parser import extract_text_from_file

        class FileWrapper:
            def __init__(self, content: bytes, name: str):
                import io
                self._bytes_io = io.BytesIO(content)
                self.name = name
            def read(self):
                return self._bytes_io.read()

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        wrapper = FileWrapper(content=file_bytes, name=file_path.name)
        return extract_text_from_file(wrapper)
    elif file_path.suffix.lower() == '.txt':
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.warning(f"不支持的文件格式: {file_path.suffix}，跳过: {file_path}")
        return ""


def mine_textbook_to_questions(
    input_paths,
    output_path: str,
    job_type: str = None,
    chunk_size: int = 2000
):
    job_config = load_job_config(job_type) if job_type else None

    if not job_config:
        from domain.job_configs import get_job_config_or_default
        job_config = get_job_config_or_default()

    effective_job_type = job_config.job_type

    if isinstance(input_paths, (str, Path)):
        input_paths = [str(input_paths)]

    files = _resolve_input_paths(input_paths)
    if not files:
        logger.error("没有找到可处理的文件！")
        return

    logger.info(f"📚 共找到 {len(files)} 个文件待处理:")
    for f in files:
        logger.info(f"   - {f.name}")

    all_text = ""
    for f in files:
        logger.info(f"📄 正在提取: {f.name}")
        text = _extract_text_from_file(f)
        if text:
            all_text += f"\n\n===== 来源: {f.name} =====\n\n{text}"

    if not all_text.strip():
        logger.error("所有文件提取结果为空！")
        return

    raw_dir = settings.get_job_raw_dir(effective_job_type)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_text_path = raw_dir / "raw_textbook.txt"
    with open(raw_text_path, "w", encoding="utf-8") as f:
        f.write(all_text)
    logger.info(f"✅ 原始文本已保存至: {raw_text_path}")

    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

    llm = ChatOpenAI(
        model=settings.FAST_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.1
    )

    is_textbook = any(
        any(kw in str(f).lower() for kw in ["textbook", "教科书", "题库", "pdf"])
        for f in files
    )
    if is_textbook:
        prompt_template = _build_mine_textbook_prompt(job_config)
    else:
        prompt_template = _build_mine_questions_prompt(job_config)

    chain = prompt_template | llm.with_structured_output(InterviewQuestionList, method="json_mode")

    all_questions = []

    for i, chunk in enumerate(chunks):
        logger.info(f"正在处理第 {i+1}/{len(chunks)} 个文本块...")

        try:
            result = chain.invoke({"raw_text": chunk})

            if isinstance(result, InterviewQuestionList):
                for q in result.questions:
                    if isinstance(q, InterviewQuestion) and not q.job_type:
                        q.job_type = effective_job_type
                all_questions.extend(result.questions)
            elif isinstance(result, dict) and "questions" in result:
                all_questions.extend(result["questions"])

        except Exception as e:
            logger.error(f"处理第 {i+1} 个文本块时出错: {e}")
            continue

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            if isinstance(q, InterviewQuestion):
                if not q.job_type:
                    q.job_type = effective_job_type
                f.write(q.model_dump_json() + "\n")
            elif isinstance(q, dict):
                if "job_type" not in q or not q["job_type"]:
                    q["job_type"] = effective_job_type
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"✅ 挖掘完成，共提取 {len(all_questions)} 个问题，保存至: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从教科书/面经逆向挖掘面试问题")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="输入文件路径，支持多个文件或目录（PDF/TXT）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出路径（默认: data/jobs/{job_type}/processed/mined_questions.jsonl）")
    parser.add_argument("--job-type", "-j", default=None, help="岗位类型（如：新媒体运营、产品经理）")
    parser.add_argument("--chunk-size", "-c", type=int, default=2000, help="文本块大小")

    args = parser.parse_args()

    job_config = load_job_config(args.job_type) if args.job_type else None
    if not job_config:
        from domain.job_configs import get_job_config_or_default
        job_config = get_job_config_or_default()

    output_path = args.output
    if not output_path:
        output_path = str(settings.get_job_processed_dir(job_config.job_type) / "mined_questions.jsonl")

    mine_textbook_to_questions(args.input, output_path, job_type=args.job_type, chunk_size=args.chunk_size)
