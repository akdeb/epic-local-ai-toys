import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from models import TextRequest, OpenAISpeechRequest
from tts_service import tts_service
from utils import convert_audio_format, get_media_type_and_filename

app = FastAPI(title="NeuTTS Air Streaming API")

@app.on_event("startup")
async def startup_event():
    tts_service.initialize_tts()

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "tts_initialized": tts_service.is_initialized()}

@app.post("/synthesize")
async def synthesize_speech(request: TextRequest):
    try:
        if not tts_service.is_initialized():
            tts_service.initialize_tts()
        
        ref_codes, ref_text = tts_service.get_reference_data(request.ref_codes_path, request.ref_text)
        
        print(f"Generating audio for: {request.text}")
        
        audio_chunks, timing_info = tts_service.generate_audio_with_timing(request.text, ref_codes, ref_text)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        wav_data = tts_service.create_wav_data(audio_chunks)
        
        def generate_wav():
            audio_stream = io.BytesIO(wav_data)
            while True:
                chunk = audio_stream.read(8192)
                if not chunk:
                    break
                yield chunk
        
        return StreamingResponse(
            generate_wav(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Cache-Control": "no-cache",
                "Content-Length": str(len(wav_data)),
                "X-Audio-Latency": f"{timing_info['latency_ms']:.2f}ms",
                "X-Total-Chunks": str(timing_info['total_chunks']),
                "X-Total-Time": f"{timing_info['total_time_ms']:.2f}ms"
            }
        )
        
    except Exception as e:
        print(f"Error in synthesize_speech: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    try:
        if not tts_service.is_initialized():
            tts_service.initialize_tts()
        
        voice_mapping = {"coral": "dave", "dave": "dave"}
        selected_voice = voice_mapping.get(request.voice, "dave")
        
        ref_codes, ref_text = tts_service.get_cached_reference_data(selected_voice)
        if ref_codes is None or ref_text is None:
            raise HTTPException(status_code=500, detail=f"Voice {selected_voice} not found in cache")
        
        print(f"OpenAI API: Generating audio for: {request.input}")
        
        media_type, filename = get_media_type_and_filename(request.response_format)
        
        def generate_streaming_audio():
            if request.response_format == "pcm":
                for chunk in tts_service.tts.infer_stream(request.input, ref_codes, ref_text):
                    audio = (chunk * 32767).astype(np.int16)
                    yield audio.tobytes()
            else:
                audio_chunks, _ = tts_service.generate_audio_with_timing(request.input, ref_codes, ref_text)
                wav_data = tts_service.create_wav_data(audio_chunks)
                
                final_audio_data = convert_audio_format(wav_data, request.response_format)
                
                audio_stream = io.BytesIO(final_audio_data)
                while True:
                    chunk = audio_stream.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        return StreamingResponse(
            generate_streaming_audio(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        print(f"Error in OpenAI API speech synthesis: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.post("/synthesize-with-timing")
async def synthesize_speech_with_timing(request: TextRequest):
    try:
        if not tts_service.is_initialized():
            tts_service.initialize_tts()
        
        ref_codes, ref_text = tts_service.get_reference_data(request.ref_codes_path, request.ref_text)
        
        print(f"Generating audio for: {request.text}")
        
        audio_chunks, timing_info = tts_service.generate_audio_with_timing(request.text, ref_codes, ref_text)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        wav_data = tts_service.create_wav_data(audio_chunks)
        
        audio_base64 = base64.b64encode(wav_data).decode('utf-8')
        
        return {
            "audio_data": audio_base64,
            "timing": {
                "latency_ms": round(timing_info["latency_ms"], 2),
                "total_chunks": timing_info["total_chunks"],
                "total_time_ms": round(timing_info["total_time_ms"], 2)
            }
        }
        
    except Exception as e:
        print(f"Error in synthesize_speech_with_timing: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)