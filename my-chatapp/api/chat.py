import os
import sys
import logging
from flask import Flask, request, jsonify
from openai import OpenAI

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

app = Flask(__name__)

# 初始化DeepSeek客户端（带错误处理）
def init_deepseek_client():
    try:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            logging.error("DEEPSEEK_API_KEY 环境变量未设置")
            return None
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=30  # 增加超时时间
        )
        
        # 测试客户端连接
        models = client.models.list()
        logging.info(f"DeepSeek 客户端初始化成功，可用模型: {[m.id for m in models.data]}")
        return client
    except Exception as e:
        logging.error(f"初始化DeepSeek客户端失败: {str(e)}")
        return None

# 全局客户端实例
client = init_deepseek_client()

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        # 检查客户端是否初始化成功
        if not client:
            return jsonify({
                "error": "服务未准备好",
                "message": "DeepSeek客户端初始化失败，请检查日志"
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供JSON数据"}), 400
            
        user_message = data.get('message')
        if not user_message:
            return jsonify({"error": "请输入消息内容"}), 400
        
        logging.info(f"收到用户消息: {user_message}")
        
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个乐于助人且友好的AI助手"},
                {"role": "user", "content": user_message}
            ],
            stream=False,
            temperature=0.7
        )
        
        ai_reply = response.choices[0].message.content
        logging.info(f"生成回复: {ai_reply}")
        
        # 返回响应
        return jsonify({
            "reply": ai_reply,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        })
    
    except Exception as e:
        logging.exception("处理请求时发生异常")
        return jsonify({
            "error": "服务器内部错误",
            "details": str(e)
        }), 500

# 健康检查端点（用于验证部署）
@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "active",
        "service": "AI Chat API",
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "environment": os.getenv("VERCEL_ENV", "development")
    })

# Vercel 必需的处理函数
def handler(request):
    with app.app_context():
        response = app.full_dispatch_request()
        return response

# 启动信息（本地运行时显示）
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("=" * 60)
    print(f" * 启动 AI 聊天服务")
    print(f" * 时间: {logging._time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" * Python 版本: {sys.version}")
    print(f" * 工作目录: {os.getcwd()}")
    print(f" * 监听地址: http://{host}:{port}")
    print(f" * DeepSeek 状态: {'已连接' if client else '未连接'}")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)