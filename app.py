from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile
import subprocess

app = Flask(__name__)
CORS(app)  # 允许跨域请求

model = whisper.load_model("base")

def is_video(filename):
    return filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))

def extract_audio(video_path, audio_path):
    # 提取音轨为 mp3（ffmpeg 必须可用）
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', audio_path])

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename

    # 保存上传文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as temp_input:
        file.save(temp_input.name)
        input_path = temp_input.name

    try:
        if is_video(filename):
            # 提取音频
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                extract_audio(input_path, temp_audio.name)
                audio_path = temp_audio.name
        else:
            audio_path = input_path

        # 调用 Whisper 识别
        result = model.transcribe(audio_path)
        return jsonify({'text': result['text']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(input_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/")
def index():
    return "Whisper 语音识别 API 正常运行中"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 默认 10000
    app.run(host="0.0.0.0", port=port)
