#!/bin/bash
# 私人专属面试顾问启动脚本 (macOS/Linux)

echo "=================================================="
echo "🚀 私人专属面试顾问 V1.0"
echo "=================================================="
echo ""

# 检查API密钥
if [ -z "$DASHSCOPE_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告：未检测到 API 密钥环境变量！"
    echo "请设置环境变量："
    echo "  export DASHSCOPE_API_KEY=your_api_key"
    echo ""
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "正在启动服务..."
echo ""

# 启动chainlit
python3 -m chainlit run app/main.py
