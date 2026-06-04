# 私人专属面试顾问 (AI Interview Coach)

基于 **LangGraph + Chainlit** 的 AI 面试训练系统。上传简历和 JD，系统分析能力短板 → 检索针对性题目 → 多策略自适应面试 → 生成能力报告。

## 核心特性

- **个性化弱点画像** — 对比简历与 JD，自动识别能力差距，所有提问围绕你的短板展开
- **5 策略自适应面试** — Router 上帝大脑每轮分析你的回答状态，动态选择：苏格拉底追问 / 脚手架托底 / 判卷打分 / 偏题拉回 / 知识补充
- **混合检索题库** — FAISS 语义检索 + BM25 关键词检索（RRF 融合），精准匹配考察点
- **弱点动态演化** — 每 5 轮根据得分自动更新弱点标签，攻克一个换下一个
- **双通道接入** — Chainlit Web UI（浏览器直接使用）+ FastAPI Coze Plugin（对接外部平台）

## 快速开始

### 环境要求
- Python >= 3.10

### 安装与启动

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 API 密钥
# 方式一：设置环境变量
export DASHSCOPE_API_KEY="your-api-key"
# 方式二：在 .env 文件中配置 OPENAI_API_KEY

# 3. 启动服务
python start.py
# 或：chainlit run app/main.py --port 8000

# 4. （可选）启动 FastAPI Coze Plugin 服务
python api/run.py --port 8001
```

### 准备题库数据

题库 PDF 需自行准备（按岗位分类放入 `data/题库/{岗位名}/`），然后运行离线数据管线：

```bash
# 从面经 PDF 挖掘结构化题目
python scripts/mine_textbook.py --job-type "电商运营"

# 构建 FAISS + BM25 检索索引
python scripts/build_index.py --job-type "电商运营"

# 或一键完成
python scripts/manage.py add-job --job-type "电商运营" --textbook "data/题库/电商运营/面经.pdf"
python scripts/manage.py rebuild --job-type "电商运营"
```

## 架构概览

```
用户 → Chainlit UI → LangGraph 工作流 → LLM
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
     Router 上帝大脑   5 个面试策略   混合检索引擎
     (意图分析+决策)   (苏格拉底/脚手架  (FAISS+BM25)
                       判卷/偏题拉回/
                       知识补充)
```

### 四阶段训练管线

1. **初始化** — 上传简历+JD → 门卫识别 → 弱点画像 → 差距分析 → 检索首题
2. **主循环** — 用户回答 → Router 决策 → Skill 执行 → 历史记录
3. **异步摘要** — 后台压缩对话历史，控制上下文窗口
4. **报告生成** — 聚合完整对话 → 输出能力体检报告

### 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Chainlit（Web UI，流式输出） |
| 工作流 | LangGraph（状态图，条件路由） |
| LLM | DeepSeek-V3（路由）+ Qwen-Turbo（轻量）+ text-embedding-v4（嵌入） |
| 检索 | FAISS 语义检索 + BM25 关键词检索（jieba 分词） |
| API | FastAPI（Coze Plugin 接口） |
| 数据 | 按岗位 YAML 驱动配置 |

## 项目结构

```
interview_coach/
├── app/                    # Chainlit 入口 + 配置
├── api/                    # FastAPI Coze Plugin 接口
├── core/
│   ├── graph/              # LangGraph 工作流（图构建 + 运行器）
│   ├── nodes/              # 节点：Router / History / Memory / Report
│   ├── skills/             # 5 个面试策略 Skill
│   ├── profiler/           # 画像管线：Gatekeeper / Profiler / GapAnalyzer
│   └── utils/              # LLM 工厂
├── domain/
│   ├── schemas.py          # 核心类型定义
│   ├── validators.py       # 验证器
│   └── job_configs/        # 岗位 YAML 配置（面试官人设、JD 等）
├── infrastructure/
│   ├── retrieval/          # 混合检索引擎 + 嵌入模型
│   ├── parsers/            # PDF/TXT 解析 + 弱点安全解析
│   └── tools/              # 题库检索 LangChain Tool
├── scripts/                # 数据管线：挖掘 / 建索引 / 岗位管理
├── deploy/                 # Coze Plugin + Nginx 部署配置
└── data/                   # 题库 PDF + 构建产物（按岗位隔离）
```

## 新增岗位

```bash
# 创建岗位配置文件
# domain/job_configs/新岗位.yaml

# 准备数据
python scripts/manage.py add-job --job-type "新岗位" --textbook "data/题库/新岗位/面经.pdf"
```

## License

MIT
