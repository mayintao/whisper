from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import subprocess
import whisper

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模型
model = whisper.load_model("base")  # 可换成 tiny / small / medium / large

@app.route("/")
def index():
    return "Whisper 视频/音频 识别 + 字幕 API"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # 临时保存文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    try:
        result = model.transcribe(audio_path)
        return jsonify({'text': result['text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(audio_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 默认 10000
    app.run(host="0.0.0.0", port=port)
