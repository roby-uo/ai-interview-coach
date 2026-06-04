# 私人专属面试顾问 (AI Interview Coach)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 **LangGraph + Chainlit** 的 AI 面试训练系统。上传简历和目标岗位 JD，系统自动分析能力短板、从题库检索针对性问题，以五种面试策略进行自适应对话训练，最终生成详尽的《面试能力体检报告》。

---

## 目录

- [核心亮点](#核心亮点)
- [快速开始](#快速开始)
- [使用流程](#使用流程)
- [架构设计](#架构设计)
- [配置说明](#配置说明)
- [数据管线](#数据管线)
- [新增岗位](#新增岗位)
- [部署](#部署)
- [项目结构](#项目结构)
- [已知局限](#已知局限)
- [License](#license)

---

## 核心亮点

### 个性化弱点画像

大多数面试工具随机出题，与用户的真实背景无关。本系统的做法是：

1. 用户上传简历 + 目标岗位 JD
2. **Gatekeeper（门卫）** 用 LLM 智能识别并分离简历和 JD 内容
3. **Profiler（画像师）** 对比两者，输出弱点标签（如「数据归因缺失」「增长方法论空白」）和禁语清单（预判用户可能用来掩饰短板的借口，如「感觉效果不错」「领导安排的」）
4. **GapAnalyzer（差距分析师）** 针对每条弱点生成具体的差距描述和简历修改建议

所有后续面试提问都围绕画像展开——同一岗位、不同背景的用户，拿到的第一道题完全不同。

### Router + 5 Skill 自适应决策系统

这不是一个固定对话模板，而是一套模拟真人面试官的决策架构：

```
每轮对话：
  用户回答 → Router（上帝大脑）分析状态 → 从 5 种策略中选出最优 → Skill 执行
```

| Skill | 触发场景 | 行为 |
|-------|---------|------|
| **Socratic Probe**（苏格拉底追问） | 用户有回答意愿，但逻辑有漏洞 | 镜像回放 → 诊断定位 → 穿刺反问，绝不给答案 |
| **Scaffold Rescue**（脚手架托底） | 用户连续卡壳、极度迷茫 | 给出半截公式或方向提示（最后手段） |
| **Judge Strike**（判卷打分） | 用户回答完整或主动投降 | 对照题库判卷清单打分，指出致命伤，展示满分拆解 |
| **Smart Redirection**（偏题拉回） | 用户偏题闲聊或输入混乱 | 简短安抚后强制拉回面试话题 |
| **Teaching Supplement**（知识补充） | 判卷后用户追问细节 | 补充例子、解释、延伸知识 |

**Router 的三步显式推理**：每个决策都经过「用户状态诊断 → 教学阶段判断 → 最优动作推导」三步推理链，大幅降低误判率。

### 弱点动态演化

每 5 轮对话，系统根据判卷得分自动调整弱点标签：

- 得分 ≥ 80 → 该弱点已攻克，移除并提取更深层问题
- 得分 < 60 → 从致命伤中发现新弱点并追加
- 60 ≤ 得分 < 80 → 维持当前弱点列表

### 混合检索引擎

题库检索采用 **FAISS 语义检索 + BM25 关键词检索** 融合（RRF 算法，权重 7:3），兼顾语义相似度和关键词精确匹配。支持 `train`（训练模式，隐藏答案）和 `review`（复盘模式，展示满分公式）双模式检索。

### 双通道接入

| 通道 | 技术 | 端口 | 用途 |
|------|------|------|------|
| Web UI | Chainlit | 8000 | 浏览器直接使用，支持文件上传、流式输出 |
| REST API | FastAPI | 8001 | Coze Bot / 飞书 等外部平台集成，API Key 认证 + 限流 |

---

## 快速开始

### 前置要求

- **Python** >= 3.10
- **API 密钥**：阿里云 DashScope API Key（[免费申请](https://dashscope.console.aliyun.com/)）

### 安装

```bash
git clone https://github.com/roby-uo/ai-interview-coach.git
cd ai-interview-coach
pip install -e .
```

### 配置

创建 `.env` 文件（或在环境变量中设置）：

```env
# API 密钥（二选一）
DASHSCOPE_API_KEY=your-dashscope-api-key
# OPENAI_API_KEY=your-openai-api-key

# 可选：自定义模型
ROUTER_MODEL_NAME=deepseek-v3
FAST_MODEL_NAME=qwen-turbo
OFFLINE_MODEL_NAME=qwen3.6-flash
EMBEDDING_MODEL_NAME=text-embedding-v4
```

### 启动

```bash
# 一键启动 Chainlit Web UI（端口 8000）
python start.py

# 或者直接使用 chainlit
chainlit run app/main.py --port 8000
```

浏览器打开 `http://localhost:8000`，选择岗位 → 上传简历 → 开始面试。

### 配套 FastAPI 服务（可选）

```bash
python api/run.py --port 8001
```

端点：
- `POST /api/interview` — 发起对话（异步模式）
- `GET /api/interview/{task_id}` — 轮询异步结果
- `GET /api/jobs` — 可用岗位列表
- `GET /api/health` — 健康检查

---

## 使用流程

### 完整用户路径

```
打开页面 → 选择目标岗位 → 上传简历 + JD → 查看差距分析报告 
→ 面试对话循环（多轮） → 生成能力体检报告
```

### 各阶段说明

**1. 岗位选择** — 系统从 `domain/job_configs/` 读取 YAML 配置动态生成岗位按钮。支持 5 个岗位：AI产品经理、AI产品运营、电商运营、新媒体运营、用户增长运营。

**2. 上传简历 + JD** — 支持 PDF / TXT 文件（最多 5 个，单个 ≤ 10MB）。也可以直接粘贴文本。如果只有简历没有 JD，系统提供默认 JD 选项。

**3. 差距分析** — Gatekeeper 自动识别简历和 JD 内容 → Profiler 输出弱点标签和禁语清单 → GapAnalyzer 生成详细差距报告。

**4. 面试对话** — AI 面试官根据你的弱点画像出第一道题，之后每轮根据你的回答质量动态调整追问策略。对话历史通过后台异步摘要控制上下文窗口大小。

**5. 生成报告** — 对话达到 4 轮后可随时生成。报告包含三个板块：
- **核心短板诊断**：归纳能力弱点模式
- **救命锦囊**：从对话中提取的实际方法论、公式、避坑指南
- **进步识别**：对比对话前后回答质量的变化

### 快速体验模式

如果没有简历，点击「快速体验」按钮，系统使用内置示例简历和 JD 演示完整流程。

---

## 架构设计

### 六层架构

```
┌──────────────────────────────────────────┐
│  表现层    Chainlit Web UI（端口 8000）    │
├──────────────────────────────────────────┤
│  接入层    FastAPI + Coze Plugin          │
├──────────────────────────────────────────┤
│  应用层    app/main.py + config.py        │
├──────────────────────────────────────────┤
│  核心层    LangGraph（Router + 5 Skills）  │
│            Profiler Pipeline              │
│            Report / Memory                │
├──────────────────────────────────────────┤
│  领域层    InterviewState / GodDecision   │
│            JobConfig (YAML)               │
├──────────────────────────────────────────┤
│  基础设施层  Hybrid Retrieval             │
│             File Parser / LLM Factory     │
└──────────────────────────────────────────┘
```

### LangGraph 状态图

```
                  ┌─────────┐
                  │  START  │
                  └────┬────┘
                  ┌────▼────┐
                  │  route  │ (Router 上帝大脑)
                  └────┬────┘
         ┌─────────────┼─────────────┬──────────┬──────────┐
         ▼             ▼             ▼          ▼          ▼
    ┌────────┐  ┌─────────┐  ┌──────────┐ ┌─────────┐ ┌────────┐
    │ report │  │socratic │  │ scaffold │ │ judge   │ │redirect│
    └───┬────┘  └────┬────┘  └─────┬────┘ │teaching │ │        │
        │            │              │      └────┬────┘ └────┬───┘
        │            └──────────────┼───────────┘          │
        │                           │                      │
        │                   ┌───────▼───────────┐          │
        │                   │  update_history   │          │
        │                   └───────┬───────────┘          │
        └───────────────────────────┼──────────────────────┘
                                    ▼
                                  ┌───┐
                                  │END│
                                  └───┘
```

**8 个节点**，**7 条条件边**，全部由 `route_decision()` 函数根据 Router 输出的 `GodDecision.target_skill` 决定分派。

### 技术选型

| 层级 | 技术 | 选择理由 |
|------|------|---------|
| 前端 | Chainlit | 专为 LLM 对话设计，内置流式输出、文件上传、会话管理 |
| 工作流 | LangGraph StateGraph | 原生支持条件路由和状态持久化，比 LangChain LCEL 适合分支多的场景 |
| 路由模型 | DeepSeek-V3（temp=0.1） | 强推理能力，低温度保证决策稳定性 |
| 轻量模型 | Qwen-Turbo（temp=0.5） | 画像、摘要、分类等轻量任务，速度快成本低 |
| 离线挖掘模型 | Qwen3.6-Flash（temp=0.1） | 批量结构化输出，处理面经挖掘 |
| 嵌入模型 | text-embedding-v4（1024维） | DashScope 国内访问稳定 |
| 向量检索 | FAISS-CPU | 零依赖本地运行，面试题库规模不需要分布式 |
| 关键词检索 | BM25 + jieba | 中文分词，补充精确匹配 |
| API 框架 | FastAPI | 高性能异步，原生 Pydantic 支持 |
| 模型接口 | DashScope 兼容 OpenAI API | 国内网络稳定，模型选择丰富 |

---

## 配置说明

### 三层配置体系

```
1. 环境变量 / .env              ← API Key、Base URL
2. app/config.py（Settings）    ← 模型名、检索参数、超时限制
3. domain/job_configs/*.yaml    ← 岗位特定配置
```

### 主要可配参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ROUTER_MODEL_NAME` | `deepseek-v3` | 路由决策主模型 |
| `FAST_MODEL_NAME` | `qwen-turbo` | 轻量任务模型 |
| `FAISS_TOP_K` | 3 | 语义检索返回数 |
| `BM25_TOP_K` | 3 | 关键词检索返回数 |
| `MAX_HISTORY_TURNS` | 10 | 最大对话轮数 |
| `STREAM_TIMEOUT_SECONDS` | 90 | Skill 工具调用超时（秒） |
| `MAX_FILE_SIZE_MB` | 10 | 上传文件大小限制（MB） |

### 岗位配置文件示例

```yaml
# domain/job_configs/电商运营.yaml
job_type: 电商运营
display_name: 电商运营
domain_keywords: GMV、ROI、转化率、流量、客单价、复购
interviewer_persona: |
  你是一位资深电商运营总监，拥有8年行业经验...
hr_persona: |
  你是一位电商行业的HRBP...
default_jd: |
  岗位职责：...
demo_resume: |
  教育背景：...
```

---

## 数据管线

### 离线准备流程

题库不是实时从互联网搜索的——为了质量和性能，数据经过离线预处理：

```
原始面经 PDF/TXT（放入 data/题库/{岗位}/）
    │
    ▼  mine_textbook.py（LLM 挖掘，qwen3.6-flash）
    │   - 分块（2000字/块）
    │   - 结构化输出（考题 + 考察点 + 满分公式 + 避坑指南）
    │
    ▼  mined_questions.jsonl
    │
    ▼  build_index.py
    │   - FAISS 向量索引（DashScope text-embedding-v4）
    │   - BM25 关键词索引（jieba 分词）
    │   - SHA-256 完整性校验
    │
    ▼  data/jobs/{岗位}/index/
       ├── faiss_index/
       ├── bm25.json
       └── index.hash
```

### 为什么离线而不是实时 RAG？

| 维度 | 离线管道 | 实时 RAG |
|------|---------|---------|
| 题目质量 | 每道题有结构化公式和避坑指南 | 返回原文片段，无加工 |
| 检索延迟 | 毫秒级（预建索引） | 取决于文档量 |
| 数据更新 | 需重新跑管道 | 实时 |
| 适合场景 | 面试题库（相对固定，质量优先） | 频繁更新的知识库 |

### 岗位管理 CLI

```bash
python scripts/manage.py list-jobs                         # 列出所有岗位
python scripts/manage.py add-job --job-type "电商运营" \
    --textbook "data/题库/电商运营/面经.pdf"                 # 添加新岗位
python scripts/manage.py mine --job-type "电商运营"          # 从 PDF 挖掘题目
python scripts/manage.py build-index --job-type "电商运营"    # 构建检索索引
python scripts/manage.py rebuild --job-type "电商运营"       # 一键重建
python scripts/manage.py remove-job --job-type "电商运营"    # 删除岗位
```

---

## 新增岗位

1. 准备面经 PDF，放入 `data/题库/{新岗位名}/`
2. 创建配置文件 `domain/job_configs/{新岗位名}.yaml`（参考已有文件）
3. 运行数据管线：

```bash
python scripts/manage.py add-job --job-type "新岗位名" --textbook "data/题库/新岗位名/面经.pdf"
```

无需修改任何代码，新增岗位只需 YAML 配置 + 题库数据。

---

## 部署

### Coze Bot 插件部署

详见 [`deploy/README.md`](deploy/README.md)。大致步骤：

1. 设置环境变量（`COZE_API_KEYS`、`DASHSCOPE_API_KEY`）
2. 启动 FastAPI 服务：`python api/run.py --port 8001`
3. 配置 Nginx 反向代理（参考 `deploy/nginx.conf`）
4. 在 Coze 平台导入 `deploy/coze_plugin.yaml`
5. 配置 Bot System Prompt 并按需调用工具

### 注意事项

- FastAPI 必须 `workers=1`（MemorySaver 内存状态，单进程约束）
- Coze 平台 30 秒超时 → 系统采用 POST 立即返回 + GET 轮询异步结果
- 内置速率限制：15 次 / 分钟

---

## 项目结构

```
interview_coach/
├── start.py                     # 跨平台启动脚本
├── 启动服务.bat                   # Windows 一键启动
├── pyproject.toml               # 项目依赖
│
├── app/                         # Chainlit 应用层
│   ├── main.py                  # UI 入口（欢迎页、消息处理、生命周期）
│   └── config.py                # 全局配置（Settings）
│
├── api/                         # FastAPI Coze Plugin 接入层
│   ├── app.py                   # FastAPI 应用（认证、限流、中间件）
│   ├── services.py              # 核心编排（请求翻译 → LangGraph）
│   ├── session_store.py         # user_id → thread_id 映射
│   ├── task_store.py            # 异步任务存储（上限 2000，TTL 600s）
│   ├── schemas.py               # API Pydantic 模型
│   └── run.py                   # uvicorn 启动
│
├── core/                        # 核心业务层
│   ├── graph/
│   │   ├── builder.py           # LangGraph 状态图定义
│   │   └── runners.py           # InterviewGraphRunner 单例
│   ├── nodes/
│   │   ├── router.py            # Router 上帝大脑（三步推理 + GodDecision）
│   │   ├── history.py           # 历史维护 + 弱点动态演化
│   │   ├── memory.py            # 异步对话摘要
│   │   ├── report.py            # 能力体检报告生成
│   │   ├── state.py             # 初始状态创建
│   │   └── utils.py             # 消息处理工具
│   ├── skills/
│   │   ├── base.py              # _execute_skill_with_tools 统一引擎
│   │   ├── socratic.py          # 苏格拉底反问
│   │   ├── scaffold.py          # 脚手架托底
│   │   ├── judge.py             # 判卷打分
│   │   ├── redirection.py       # 偏题拉回
│   │   └── teaching.py          # 教学补充
│   ├── profiler/
│   │   ├── gatekeeper.py        # 简历/JD 门卫分类器
│   │   ├── extractor.py         # Profiler 弱点画像
│   │   └── gap_analyzer.py      # GapAnalyzer 差距分析
│   └── utils/
│       └── llm_factory.py       # LLM 工厂（三档温度管理）
│
├── domain/                      # 领域层
│   ├── schemas.py               # InterviewState / GodDecision / WeaknessProfile
│   ├── validators.py            # 简历/JD 关键词验证
│   └── job_configs/             # 岗位 YAML 配置（5 个岗位）
│
├── infrastructure/              # 基础设施层
│   ├── retrieval/
│   │   ├── hybrid.py            # FAISS + BM25 混合检索（RRF 融合）
│   │   └── embeddings.py        # DashScopeEmbeddings
│   ├── parsers/
│   │   ├── file_parser.py       # PDF/TXT 文件解析
│   │   └── text_parser.py       # 弱点安全解析（多层降级）
│   └── tools/
│       └── interview_db.py      # search_interview_db（LangChain Tool）
│
├── scripts/                     # 数据管线
│   ├── manage.py                # 岗位管理 CLI
│   ├── mine_textbook.py         # LLM 挖掘题目
│   └── build_index.py           # FAISS + BM25 索引构建
│
├── public/
│   └── custom.css               # Chainlit 深空主题定制（781 行）
│
├── deploy/                      # 部署配置
│   ├── coze_plugin.yaml         # Coze Bot OpenAPI Schema
│   ├── nginx.conf               # Nginx 代理配置
│   └── README.md                # 部署指南
│
└── data/                        # 运行时数据（需自行准备）
    └── 题库/                     # 原始面经 PDF（放入此目录）
```

---

## 已知局限

| 局限 | 原因 |
|------|------|
| 会话状态存内存（MemorySaver） | 服务重启后所有会话丢失，不影响单次使用 |
| 仅 5 个岗位 | 每个岗位需独立准备题库，扩展是体力活 |
| 题库依赖离线挖掘 | 新增岗位需先跑数据管线 |
| 无自动化测试 | MVP 阶段优先验证核心链路 |
| 单进程架构 | MemorySaver 约束，无法多 worker 水平扩展 |

以上每项都是 V0→V1 阶段的有意识的取舍，均已在架构演进路线中规划。

---

## License

MIT © [roby-uo](https://github.com/roby-uo)
