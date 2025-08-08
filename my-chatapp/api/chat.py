import os
import sys
import logging
from flask import Flask, request, jsonify
from openai import OpenAI, APIError, AuthenticationError, APIConnectionError, Timeout
from functools import wraps
import time

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

app = Flask(__name__)

# 速率限制配置
RATE_LIMIT = 10  # 每IP限制请求数
RATE_LIMIT_WINDOW = 60  # 时间窗口（秒）
request_timestamps = {}  # 存储IP的请求时间戳

def rate_limit(f):
    """装饰器：实现基本的速率限制"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        now = time.time()
        
        # 初始化或清理过期的时间戳
        if client_ip not in request_timestamps:
            request_timestamps[client_ip] = []
        request_timestamps[client_ip] = [t for t in request_timestamps[client_ip] if now - t < RATE_LIMIT_WINDOW]
        
        # 检查是否超过速率限制
        if len(request_timestamps[client_ip]) >= RATE_LIMIT:
            return jsonify({
                "error": "速率限制 exceeded",
                "message": f"每个IP在{ RATE_LIMIT_WINDOW }秒内最多允许{ RATE_LIMIT }个请求"
            }), 429
            
        # 记录当前请求时间
        request_timestamps[client_ip].append(now)
        return f(*args, **kwargs)
    return decorated_function

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
        
        logging.info("DeepSeek 客户端初始化成功")
        return client
    except Exception as e:
        logging.error(f"初始化DeepSeek客户端失败: {str(e)}")
        return None

# 全局客户端实例
client = init_deepseek_client()

@app.route('/api/chat', methods=['POST'])
@rate_limit
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
        if not user_message or not isinstance(user_message, str) or len(user_message.strip()) == 0:
            return jsonify({"error": "请输入有效的消息内容"}), 400
        
        # 日志中不记录完整消息内容，保护隐私
        logging.info(f"收到用户消息，长度: {len(user_message)}")
        
        # 调用DeepSeek API
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个乐于助人且友好的AI助手"},
                    {"role": "user", "content": user_message}
                ],
                stream=False,
                temperature=0.7
            )
        except AuthenticationError:
            logging.error("DeepSeek API 认证失败，请检查API密钥")
            return jsonify({
                "error": "认证失败",
                "message": "API密钥无效或已过期"
            }), 401
        except APIConnectionError:
            logging.error("无法连接到DeepSeek API服务")
            return jsonify({
                "error": "连接失败",
                "message": "无法连接到AI服务，请稍后再试"
            }), 503
        except Timeout:
            logging.error("DeepSeek API 请求超时")
            return jsonify({
                "error": "请求超时",
                "message": "AI服务响应超时，请稍后再试"
            }), 504
        except APIError as e:
            logging.error(f"DeepSeek API 错误: {str(e)}")
            return jsonify({
                "error": "AI服务错误",
                "message": f"处理请求时发生错误: {str(e)}"
            }), 500
        
        ai_reply = response.choices[0].message.content
        logging.info(f"生成回复，长度: {len(ai_reply)}")
        
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
            "details": str(e) if app.debug else "请查看服务器日志获取详细信息"
        }), 500

# 健康检查端点（用于验证部署）
@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "active",
        "service": "AI Chat API",
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "environment": os.getenv("VERCEL_ENV", "development"),
        "deepseek_connected": bool(client)
    })

# Vercel 正确的处理函数
from werkzeug.wrappers import Request, Response

@Request.application
def vercel_handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()

# 启动信息（本地运行时显示）
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    print("=" * 60)
    print(f" * 启动 AI 聊天服务")
    print(f" * 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" * Python 版本: {sys.version.split()[0]}")
    print(f" * 工作目录: {os.getcwd()}")
    print(f" * 监听地址: http://{host}:{port}")
    print(f" * 调试模式: {'开启' if debug else '关闭'}")
    print(f" * DeepSeek 状态: {'已连接' if client else '未连接'}")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=debug)
