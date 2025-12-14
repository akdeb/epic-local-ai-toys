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
from stt_service import stt_service
from utils import convert_audio_format, get_media_type_and_filename

app = FastAPI(title="NeuTTS Air Streaming API")

@app.on_event("startup")
async def startup_event():
    tts_service.initialize_tts()
    llm_service.initialize_llm()
    stt_service.initialize_stt()

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


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """Voice chat: audio in -> STT -> LLM -> TTS -> audio out"""
    await websocket.accept()
    
    voice = "dave"
    system_prompt = "You are a helpful voice assistant. Be concise."
    greeted = False
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break
            
            if "bytes" in message:
                # Send greeting on first audio received
                if not greeted:
                    greeted = True
                    await websocket.send_json({"type": "pause_mic"})
                    await generate_greeting(websocket, voice, system_prompt, is_esp32=False)
                    await websocket.send_json({"type": "resume_mic"})
                    stt_service.reset()
                    continue
                
                # Process audio chunk
                audio_bytes = message["bytes"]
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                result = stt_service.process_audio_chunk(audio, use_vad=True)
                
                if result:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": result.text,
                        "is_final": result.is_final
                    })
                    
                    if result.is_final:
                        # Pause mic while generating response
                        await websocket.send_json({"type": "pause_mic"})
                        
                        # Generate LLM + TTS response
                        ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
                        if ref_codes is None:
                            await websocket.send_json({"type": "resume_mic"})
                            continue
                        
                        sentence_buffer = SentenceBuffer()
                        full_response = ""
                        
                        for token in llm_service.generate_stream(result.text, system_prompt):
                            full_response += token
                            await websocket.send_json({"type": "token", "content": token})
                            
                            for sentence in sentence_buffer.add(token):
                                if len(sentence.strip()) < 3:  # Skip very short text
                                    continue
                                try:
                                    audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                                    if audio_chunks:
                                        wav_data = tts_service.create_wav_data(audio_chunks)
                                        await websocket.send_json({
                                            "type": "audio",
                                            "content": base64.b64encode(wav_data).decode('utf-8'),
                                            "sentence": sentence
                                        })
                                except Exception as e:
                                    print(f"TTS error for '{sentence}': {e}")
                        
                        remaining = sentence_buffer.flush()
                        if remaining and len(remaining.strip()) >= 3:
                            try:
                                audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
                                if audio_chunks:
                                    wav_data = tts_service.create_wav_data(audio_chunks)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "content": base64.b64encode(wav_data).decode('utf-8'),
                                        "sentence": remaining
                                    })
                            except Exception as e:
                                print(f"TTS error for '{remaining}': {e}")
                        
                        await websocket.send_json({"type": "done", "full_response": full_response})
                        await websocket.send_json({"type": "resume_mic"})
                        stt_service.reset()
                        
            elif "text" in message:
                # Config message
                import json
                try:
                    data = json.loads(message["text"])
                    if "voice" in data:
                        voice = data["voice"]
                    if "system_prompt" in data:
                        system_prompt = data["system_prompt"]
                except:
                    pass
                    
    except WebSocketDisconnect:
        print("Voice WebSocket disconnected")
    finally:
        stt_service.reset()


async def generate_greeting(websocket: WebSocket, voice: str, system_prompt: str, is_esp32: bool = False):
    """Generate and send initial greeting from the assistant."""
    ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
    if ref_codes is None:
        return
    
    greeting_prompt = "The user just connected. Greet them warmly and briefly ask how you can help."
    
    sentence_buffer = SentenceBuffer()
    full_response = ""
    first_audio = True
    
    for token in llm_service.generate_stream(greeting_prompt, system_prompt):
        full_response += token
        
        if not is_esp32:
            await websocket.send_json({"type": "token", "content": token})
        
        for sentence in sentence_buffer.add(token):
            if len(sentence.strip()) < 3:
                continue
            
            if is_esp32 and first_audio:
                await websocket.send_json({"type": "server", "msg": "RESPONSE.CREATED"})
                first_audio = False
            
            try:
                audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                if audio_chunks:
                    if is_esp32:
                        pcm_data = np.concatenate(audio_chunks)
                        pcm_16bit = (pcm_data * 32767).astype(np.int16)
                        await websocket.send_bytes(pcm_16bit.tobytes())
                    else:
                        wav_data = tts_service.create_wav_data(audio_chunks)
                        await websocket.send_json({
                            "type": "audio",
                            "content": base64.b64encode(wav_data).decode('utf-8'),
                            "sentence": sentence
                        })
            except Exception as e:
                print(f"Greeting TTS error: {e}")
    
    remaining = sentence_buffer.flush()
    if remaining and len(remaining.strip()) >= 3:
        try:
            audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
            if audio_chunks:
                if is_esp32:
                    pcm_data = np.concatenate(audio_chunks)
                    pcm_16bit = (pcm_data * 32767).astype(np.int16)
                    await websocket.send_bytes(pcm_16bit.tobytes())
                else:
                    wav_data = tts_service.create_wav_data(audio_chunks)
                    await websocket.send_json({
                        "type": "audio",
                        "content": base64.b64encode(wav_data).decode('utf-8'),
                        "sentence": remaining
                    })
        except Exception as e:
            print(f"Greeting TTS error: {e}")
    
    if is_esp32:
        await websocket.send_json({"type": "server", "msg": "RESPONSE.COMPLETE"})
    else:
        await websocket.send_json({"type": "done", "full_response": full_response})
    
    print(f"[Greeting] {full_response}")


@app.websocket("/ws/esp32")
async def websocket_esp32(websocket: WebSocket):
    """
    ESP32 voice pipeline endpoint.
    - Receives: 16-bit PCM audio at 16kHz
    - Sends: Opus-encoded audio + JSON control messages
    """
    await websocket.accept()
    
    voice = "dave"
    system_prompt = "You are a helpful voice assistant. Be concise and conversational."
    input_sample_rate = 16000  # ESP32 mic sample rate
    
    # Send auth response
    await websocket.send_json({
        "type": "auth",
        "volume_control": 70,
        "pitch_factor": 1.0,
        "is_ota": False,
        "is_reset": False
    })
    
    print("[ESP32] Client connected")
    
    # Send initial greeting
    await generate_greeting(websocket, voice, system_prompt, is_esp32=True)
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break
            
            if "bytes" in message:
                # Process PCM audio from ESP32 (16-bit, 16kHz)
                audio_bytes = message["bytes"]
                audio_16k = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample 16kHz -> 24kHz for STT
                ratio = 24000 / input_sample_rate
                output_len = int(len(audio_16k) * ratio)
                audio_24k = np.zeros(output_len, dtype=np.float32)
                for i in range(output_len):
                    src_idx = i / ratio
                    idx0 = int(src_idx)
                    idx1 = min(idx0 + 1, len(audio_16k) - 1)
                    frac = src_idx - idx0
                    audio_24k[i] = audio_16k[idx0] * (1 - frac) + audio_16k[idx1] * frac
                
                # Process through STT
                result = stt_service.process_audio_chunk(audio_24k, use_vad=True)
                
                if result and result.is_final:
                    print(f"[ESP32] Transcript: {result.text}")
                    
                    # Notify audio committed
                    await websocket.send_json({"type": "server", "msg": "AUDIO.COMMITTED"})
                    
                    # Get TTS reference data
                    ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
                    if ref_codes is None:
                        await websocket.send_json({"type": "server", "msg": "RESPONSE.ERROR"})
                        stt_service.reset()
                        continue
                    
                    # Generate LLM response
                    sentence_buffer = SentenceBuffer()
                    full_response = ""
                    
                    for token in llm_service.generate_stream(result.text, system_prompt):
                        full_response += token
                        
                        for sentence in sentence_buffer.add(token):
                            if len(sentence.strip()) < 3:
                                continue
                            
                            # Notify response created (first audio)
                            if not hasattr(websocket, '_sent_response_created'):
                                await websocket.send_json({"type": "server", "msg": "RESPONSE.CREATED"})
                                websocket._sent_response_created = True
                            
                            try:
                                audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                                if audio_chunks:
                                    # Combine chunks and send as raw PCM (ESP32 expects Opus but we'll send PCM for now)
                                    pcm_data = np.concatenate(audio_chunks)
                                    pcm_16bit = (pcm_data * 32767).astype(np.int16)
                                    await websocket.send_bytes(pcm_16bit.tobytes())
                            except Exception as e:
                                print(f"[ESP32] TTS error: {e}")
                    
                    # Flush remaining text
                    remaining = sentence_buffer.flush()
                    if remaining and len(remaining.strip()) >= 3:
                        if not hasattr(websocket, '_sent_response_created'):
                            await websocket.send_json({"type": "server", "msg": "RESPONSE.CREATED"})
                            websocket._sent_response_created = True
                        
                        try:
                            audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
                            if audio_chunks:
                                pcm_data = np.concatenate(audio_chunks)
                                pcm_16bit = (pcm_data * 32767).astype(np.int16)
                                await websocket.send_bytes(pcm_16bit.tobytes())
                        except Exception as e:
                            print(f"[ESP32] TTS error: {e}")
                    
                    # Notify response complete
                    await websocket.send_json({"type": "server", "msg": "RESPONSE.COMPLETE"})
                    delattr(websocket, '_sent_response_created') if hasattr(websocket, '_sent_response_created') else None
                    
                    print(f"[ESP32] Response: {full_response}")
                    stt_service.reset()
                    
            elif "text" in message:
                # Handle JSON config messages
                try:
                    data = json.loads(message["text"])
                    if "voice" in data:
                        voice = data["voice"]
                    if "system_prompt" in data:
                        system_prompt = data["system_prompt"]
                except:
                    pass
                    
    except WebSocketDisconnect:
        print("[ESP32] Client disconnected")
    finally:
        stt_service.reset()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)