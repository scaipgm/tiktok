import os
import subprocess
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq

app = FastAPI()

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato SRT HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

@app.get("/")
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/process-video")
async def process_video(file: UploadFile = File(...), api_key: str = Form(None)):
    token = api_key or os.getenv("GROQ_API_KEY")
    if not token:
        raise HTTPException(status_code=400, detail="Falta la API Key de Groq.")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        audio_path = os.path.join(tmpdir, "audio.mp3")
        srt_path = os.path.join(tmpdir, "subtitles.srt")
        output_path = os.path.join(tmpdir, "output.mp4")

        # Guardar el video subido
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # 1. Extraer audio con FFmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Transcribir con Groq Whisper
        client = Groq(api_key=token)
        with open(audio_path, "rb") as af:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), af.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        # 3. Construir subtítulos en formato SRT
        segments = getattr(transcription, "segments", [])
        if not segments:
            raise HTTPException(status_code=400, detail="No se detectó voz en el video.")

        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for idx, seg in enumerate(segments, start=1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                text = seg["text"].strip().upper()  # Estilo shorts en mayúsculas
                srt_file.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

        # 4. Escalar a 9:16 y quemar subtítulos amarillos con borde negro
        sub_filter = (
            f"subtitles='{srt_path}':force_style='"
            "Fontname=Arial,FontSize=18,Bold=1,PrimaryColour=&H0000FFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=60'"
        )

        vf_pipeline = (
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,{sub_filter}"
        )

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf_pipeline,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Devolver el archivo final
        with open(output_path, "rb") as out_file:
            data = out_file.read()

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_out.write(data)
        temp_out.close()
        return FileResponse(temp_out.name, media_type="video/mp4", filename="short_subtitulado.mp4")
