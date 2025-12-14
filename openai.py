import io
import re
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from models import TextRequest, OpenAISpeechRequest
from tts_service import tts_service
from llm_service import llm_service
from utils import convert_audio_format, get_media_type_and_filename

app = FastAPI(title="NeuTTS Air Streaming API")

@app.on_event("startup")
async def startup_event():
    tts_service.initialize_tts()
    llm_service.initialize_llm()

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

@app.get("/chat")
async def chat_page():
    return FileResponse("chat.html")


class SentenceBuffer:
    """Buffers text tokens and yields complete sentences."""
    
    def __init__(self):
        self.buffer = ""
        self.sentence_endings = re.compile(r'([.!?]+[\s]*)')
    
    def add(self, text: str) -> list[str]:
        """Add text and return any complete sentences."""
        self.buffer += text
        sentences = []
        
        while True:
            match = self.sentence_endings.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                if sentence:
                    sentences.append(sentence)
                self.buffer = self.buffer[end_pos:]
            else:
                break
        
        return sentences
    
    def flush(self) -> str:
        """Return any remaining text in buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            voice = data.get("voice", "dave")
            system_prompt = data.get("system_prompt", "You are a helpful assistant. Keep responses concise and conversational.")
            
            if not prompt:
                await websocket.send_json({"type": "error", "message": "No prompt provided"})
                continue
            
            ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
            if ref_codes is None:
                await websocket.send_json({"type": "error", "message": f"Voice {voice} not found"})
                continue
            
            sentence_buffer = SentenceBuffer()
            full_response = ""
            
            try:
                for token in llm_service.generate_stream(prompt, system_prompt):
                    full_response += token
                    await websocket.send_json({"type": "token", "content": token})
                    
                    sentences = sentence_buffer.add(token)
                    for sentence in sentences:
                        audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                        if audio_chunks:
                            wav_data = tts_service.create_wav_data(audio_chunks)
                            audio_b64 = base64.b64encode(wav_data).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio",
                                "content": audio_b64,
                                "sentence": sentence
                            })
                
                remaining = sentence_buffer.flush()
                if remaining:
                    audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
                    if audio_chunks:
                        wav_data = tts_service.create_wav_data(audio_chunks)
                        audio_b64 = base64.b64encode(wav_data).decode('utf-8')
                        await websocket.send_json({
                            "type": "audio",
                            "content": audio_b64,
                            "sentence": remaining
                        })
                
                await websocket.send_json({"type": "done", "full_response": full_response})
                
            except Exception as e:
                print(f"Error in chat generation: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)