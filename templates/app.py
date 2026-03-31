from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 加载轻量版 BERT 情感分析模型（前沿且高效）
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 首页路由：显示前端页面
@app.route('/')
def home():
    return render_template('index.html')

# API 路由：处理情感分析请求
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"error": "文本不能为空"}), 400
    result = sentiment_analyzer(text)[0]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
