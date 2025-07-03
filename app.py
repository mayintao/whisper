from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import tempfile
import subprocess

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模型
model = WhisperModel("base", device="cpu", compute_type="int8")


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

    # 临时保存文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_input:
        file.save(temp_input.name)
        input_path = temp_input.name

    # 如果是视频，提取音频
    if filename.endswith((".mp4", ".mov", ".mkv", ".avi")):
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        extract_audio_from_video(input_path, temp_audio_path)
    else:
        temp_audio_path = input_path  # 已是音频
    print(temp_audio_path)
    
    # 识别
    segments, info = model.transcribe(temp_audio_path, beam_size=5)

    full_text = ""
    seg_list = []
    for seg in segments:
        full_text += seg.text.strip() + " "
        seg_list.append(seg)

    srt_content = generate_srt(seg_list)
    print(srt_content)

    # 清理临时文件
    os.remove(input_path)
    if os.path.exists(temp_audio_path) and temp_audio_path != input_path:
        os.remove(temp_audio_path)

    return jsonify({
        "language": info.language,
        "text": full_text.strip(),
        "srt": srt_content
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 默认 10000
    app.run(host="0.0.0.0", port=port)
