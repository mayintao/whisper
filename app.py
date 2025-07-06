from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import tempfile
import subprocess

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模型
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/")
def index():
    return "Whisper 视频/音频 识别 + 字幕 API"

def extract_audio_from_video(video_path, output_audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "请上传文件"}), 400

    # 限制文件大小（10MB）
    if file.content_length > 10 * 1024 * 1024:
        return jsonify({"error": "文件过大，请上传小于10MB的文件"}), 400

    # 保存临时文件
    filepath = "temp_input"
    file.save(filepath)

    try:
        # 如果是视频，提取音频
        if file.filename.lower().endswith((".mp4", ".mov", ".avi")):
            audio_path = "temp_audio.wav"
            extract_audio_from_video(filepath, audio_path)
            os.remove(filepath)
            filepath = audio_path

        # 分块处理 + 低 beam_size
        segments, info = model.transcribe(filepath, beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments)

        return jsonify({"text": text, "language": info.language})

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 默认 10000
    app.run(host="0.0.0.0", port=port)
