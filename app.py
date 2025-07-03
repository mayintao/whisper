from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import tempfile

app = Flask(__name__)

# 加载模型（首次会自动下载）
model_size = "base"  # 你也可以改成 tiny / small / medium
model = WhisperModel(model_size, device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "请上传音频文件"}), 400

    file = request.files["file"]

    # 将上传文件保存到临时文件夹
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        file.save(temp_audio.name)
        temp_path = temp_audio.name

    # 调用模型进行识别
    segments, info = model.transcribe(temp_path, beam_size=5)
    result = ""
    for seg in segments:
        result += seg.text.strip() + " "

    os.remove(temp_path)  # 清理临时文件

    return jsonify({
        "language": info.language,
        "text": result.strip()
    })

@app.route("/")
def index():
    return "Whisper 中文语音识别 API 服务运行中"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
