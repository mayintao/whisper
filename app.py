from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import tempfile
import subprocess

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模型
model = WhisperModel("small", device="cpu", compute_type="int8")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return "Whisper 视频/音频 识别 + 字幕 API"


# 格式化 SRT 时间戳
def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# 生成 SRT 字幕内容
def generate_srt(segments) -> str:
    srt = ""
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        srt += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt.strip()

# 提取视频中的音频
def extract_audio_from_video(video_path: str, output_audio_path: str):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@app.route("/transcribe", methods=["POST"])
def transcribe():

    print("transcribe")
    file = request.files.get("file")
    if not file :
        return jsonify({"error": "请上传文件"}), 400
    filename = file.filename.lower()
    print(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # 识别
    segments, info = model.transcribe(filepath, beam_size=2)

    full_text = ""
    seg_list = []
    for seg in segments:
        full_text += seg.text.strip() + " "
        seg_list.append(seg)

    srt_content = generate_srt(seg_list)
    print(srt_content)

    # 清理临时文件
    os.remove(filepath)

    return jsonify({
        "language": info.language,
        "text": full_text.strip(),
        "srt": srt_content
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 默认 10000
    app.run(host="0.0.0.0", port=port)
