# AI面试教练 - Coze Plugin 部署指南

## 目录结构

```
deploy/
├── coze_plugin.yaml   # Coze Plugin OpenAPI Schema
├── nginx.conf         # Nginx 反向代理配置
└── README.md          # 本文档
```

## 部署步骤

### 1. 环境变量配置

在服务器上设置环境变量：

```bash
# API Key（Coze Plugin 鉴权用）
export COZE_API_KEYS="your-secret-key-here"

# LLM API Key（已有）
export DASHSCOPE_API_KEY="your-dashscope-key"
```

或写入 `.env` 文件：

```env
COZE_API_KEYS=your-secret-key-here
DASHSCOPE_API_KEY=your-dashscope-key
```

### 2. 启动 FastAPI 服务

```bash
cd /path/to/interview_coach

# 安装依赖
pip install -e .

# 启动 API 服务（端口 8000）
python -m api.run

# 或使用 uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 3. Nginx 配置

```bash
# 复制配置文件
sudo cp deploy/nginx.conf /etc/nginx/sites-available/coze-api.conf

# 修改域名
sudo vim /etc/nginx/sites-available/coze-api.conf
# 将 your-domain.com 替换为你的实际域名

# 启用配置
sudo ln -s /etc/nginx/sites-available/coze-api.conf /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重载 Nginx
sudo systemctl reload nginx
```

### 4. SSL 证书（Let's Encrypt）

```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 申请证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

### 5. Coze Plugin 配置

1. 登录 [Coze 平台](https://www.coze.cn/)
2. 创建 Bot → 添加 Plugin → 创建自定义 Plugin
3. 选择 "从 OpenAPI 导入"
4. 将 `deploy/coze_plugin.yaml` 内容粘贴进去
5. 配置 API Key：
   - 在 Plugin 设置中添加 Header：`X-API-Key`
   - 值为你在服务器设置的 `COZE_API_KEYS`

### 6. Coze Bot Prompt 配置

在 Coze Bot 的 System Prompt 中加入：

```
## 工具调用规则

1. 每次用户发消息，必须调用 interviewChat 工具，不要自己回答面试相关问题
2. 如果 interviewChat 返回 status=processing，立即调用 pollTask 工具轮询结果
3. 轮询间隔3-5秒，直到 status 变为 ok 或 error
4. 收到 ok 结果后，将 reply 内容原样展示给用户
5. 当 reply 中包含「快捷操作」提示时，告知用户可以回复对应关键词触发操作
6. 用户说"快速体验"→ action=quick_start，"生成报告"→ action=generate_report，"重新开始"→ action=reset
7. 用户选择岗位时，先调用 listJobs 获取可用岗位，再根据用户选择设置 job_type
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/interview` | POST | 主对话接口 |
| `/api/interview/{task_id}` | GET | 轮询异步任务结果 |
| `/api/jobs` | GET | 获取可用岗位列表 |
| `/api/health` | GET | 健康检查 |

## 异步任务流程

```
用户发消息 → POST /api/interview
    ↓
返回 {status: "processing", task_id: "xxx"}
    ↓
Coze Bot 调用 GET /api/interview/{task_id}（3-5秒后）
    ↓
返回 {status: "processing"} → 继续轮询
    ↓
返回 {status: "ok", reply: "..."} → 展示给用户
```

## 故障排查

### 401 Unauthorized
- 检查 `COZE_API_KEYS` 环境变量是否设置
- 检查 Coze Plugin 配置的 Header 是否正确

### 429 Too Many Requests
- 触发了限流，等待 60 秒后重试
- 或调整 `nginx.conf` 中的 `limit_req` 参数

### 504 Gateway Timeout
- Nginx 超时，检查 `proxy_read_timeout` 设置
- 确认后端服务正常运行

### 任务一直 processing
- 检查后端日志：`journalctl -u interview-coach -f`
- 确认 LLM API 可用
