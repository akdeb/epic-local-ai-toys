import os
import sys

# Set espeak library path for macOS homebrew if not set
if sys.platform == "darwin":
    if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        possible_paths = [
            "/opt/homebrew/lib/libespeak-ng.dylib",
            "/usr/local/lib/libespeak-ng.dylib"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = path
                print(f"Set PHONEMIZER_ESPEAK_LIBRARY to {path}")
                break
    
    # Hack to help opuslib find libopus on macOS
    import ctypes.util
    _find_library = ctypes.util.find_library
    def find_library(name):
        if name == 'opus':
            paths = [
                '/opt/homebrew/lib/libopus.dylib',
                '/usr/local/lib/libopus.dylib',
            ]
            for path in paths:
                if os.path.exists(path):
                    return path
        return _find_library(name)
    ctypes.util.find_library = find_library

import io
import re
import json
import base64
import asyncio
import numpy as np
from typing import Optional
try:
    import opuslib
    OPUS_AVAILABLE = True
except Exception as e:
    print(f"Opus library not available: {e}")
    OPUS_AVAILABLE = False

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from models import TextRequest, OpenAISpeechRequest, PersonalityCreate, PersonalityUpdate, ConversationLog, UserCreate, UserUpdate
from tts_service import tts_service
from llm_service import llm_service
from stt_service import stt_service
from db_service import db_service
from utils import convert_audio_format, get_media_type_and_filename
from path_utils import get_resource_path

app = FastAPI(title="NeuTTS Air Streaming API")

@app.on_event("startup")
async def startup_event():
    # Lazy init: Models are loaded only when first requested to save resources/battery
    print("NeuTTS Air started. Models will be loaded on first use.")
    pass

@app.get("/")
async def read_root():
    return FileResponse(get_resource_path("index.html"))

@app.get("/chat")
async def read_chat():
    return FileResponse(get_resource_path("chat.html"))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "tts_initialized": tts_service.is_initialized()}

# --- Database API ---

@app.post("/personalities")
async def create_personality(p: PersonalityCreate):
    return db_service.create_personality(
        name=p.name,
        prompt=p.prompt,
        short_description=p.short_description,
        tags=p.tags,
        voice_id=p.voice_id,
        is_visible=p.is_visible
    )

@app.get("/personalities")
async def list_personalities(include_hidden: bool = False):
    return db_service.get_personalities(include_hidden=include_hidden)

@app.get("/personalities/{p_id}")
async def get_personality(p_id: str):
    p = db_service.get_personality(p_id)
    if not p:
        raise HTTPException(status_code=404, detail="Personality not found")
    return p

@app.put("/personalities/{p_id}")
async def update_personality(p_id: str, p: PersonalityUpdate):
    updated = db_service.update_personality(p_id, **p.dict(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Personality not found")
    return updated

@app.delete("/personalities/{p_id}")
async def delete_personality(p_id: str):
    success = db_service.delete_personality(p_id)
    if not success:
        raise HTTPException(status_code=404, detail="Personality not found")
    return {"status": "success"}

@app.get("/conversations")
async def list_conversations(limit: int = 50, offset: int = 0):
    return db_service.get_conversations(limit, offset)

@app.delete("/conversations/{c_id}")
async def delete_conversation(c_id: str):
    success = db_service.delete_conversation(c_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success"}

@app.post("/users")
async def create_user(u: UserCreate):
    return db_service.create_user(
        name=u.name,
        age=u.age,
        dob=u.dob,
        hobbies=u.hobbies,
        personality_type=u.personality_type,
        likes=u.likes,
        current_personality_id=u.current_personality_id,
        user_type=u.user_type,
        device_volume=u.device_volume
    )

@app.get("/users")
async def list_users():
    return db_service.get_users()

@app.get("/users/{u_id}")
async def get_user(u_id: str):
    u = db_service.get_user(u_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u

@app.put("/users/{u_id}")
async def update_user(u_id: str, u: UserUpdate):
    updated = db_service.update_user(u_id, **u.dict(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated

@app.delete("/users/{u_id}")
async def delete_user(u_id: str):
    success = db_service.delete_user(u_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success"}

# --------------------

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
        
        voice_mapping = {"coral": "dave", "dave": "dave", "jo": "jo", "mara": "mara", "santa": "santa"}
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


class OpusStreamer:
    """Buffers PCM audio and encodes it into Opus frames."""
    def __init__(self, sample_rate: int = 24000, channels: int = 1, frame_duration_ms: int = 60):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.buffer = bytearray()
        
        self.enabled = OPUS_AVAILABLE
        if not self.enabled:
            print("OpusStreamer disabled: opuslib not available")
            return
            
        try:
            self.encoder = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_VOIP)
            try:
                self.encoder.bitrate = 24000
            except Exception as e:
                print(f"Warning: Failed to set Opus bitrate: {e}")
            print(f"OpusStreamer initialized: {frame_duration_ms}ms frames ({self.frame_size} samples)")
        except Exception as e:
            print(f"Failed to create Opus encoder: {e}")
            self.enabled = False

    def boost_audio(self, pcm_data: np.ndarray, gain_db: float = 6.0, ceiling: float = 0.89) -> np.ndarray:
        """
        Apply gain, limiting and soft-clipping to audio data.
        Replicates boostLimitPCM16LEInPlace logic.
        """
        if pcm_data.size == 0:
            return pcm_data
            
        # 1. Apply Gain
        g = 10 ** (gain_db / 20.0)
        y = pcm_data * g
        
        # 2. Measure Peak
        peak = np.max(np.abs(y))
        
        # 3. Calculate Scale (prevent hard clipping if above ceiling)
        scale = 1.0
        if peak > ceiling and peak > 0:
            scale = ceiling / peak
            
        # 4. Apply Scale
        y = y * scale
        
        # 5. Cubic Soft-Clip: y = 0.5 * y * (3 - y^2)
        # Prevents harsh square-waving at the limits
        y_sq = y * y
        y = 0.5 * y * (3 - y_sq)
        
        # 6. Hard Clamp (safety)
        y = np.clip(y, -0.999, 0.999)
        
        return y

    def process(self, pcm_data: np.ndarray) -> list[bytes]:
        """Process PCM chunk and return list of Opus packets."""
        if not self.enabled:
            return []

        # Convert to float32 if needed for processing
        if pcm_data.dtype != np.float32:
            if pcm_data.dtype == np.int16:
                pcm_float = pcm_data.astype(np.float32) / 32768.0
            else:
                pcm_float = pcm_data.astype(np.float32)
        else:
            pcm_float = pcm_data

        # Apply Volume Boost / Soft Clip
        pcm_boosted = self.boost_audio(pcm_float)

        # Convert back to int16 for Opus encoding
        audio_int16 = (pcm_boosted * 32767).astype(np.int16)
            
        self.buffer.extend(audio_int16.tobytes())
        
        bytes_per_frame = self.frame_size * 2 * self.channels
        packets = []
        
        while len(self.buffer) >= bytes_per_frame:
            frame_bytes = self.buffer[:bytes_per_frame]
            del self.buffer[:bytes_per_frame]
            
            try:
                packet = self.encoder.encode(bytes(frame_bytes), self.frame_size)
                packets.append(packet)
            except Exception as e:
                print(f"Opus encoding error: {e}")
        
        if packets:
            # print(f"Encoded {len(packets)} Opus packets")
            pass
            
        return packets

    def flush(self) -> list[bytes]:
        """Flush remaining buffer by padding with silence."""
        if not self.enabled or not self.buffer:
            return []
            
        bytes_per_frame = self.frame_size * 2 * self.channels
        padding_needed = bytes_per_frame - len(self.buffer)
        
        if padding_needed > 0:
            self.buffer.extend(b'\x00' * padding_needed)
            
        packets = []
        try:
            packet = self.encoder.encode(bytes(self.buffer), self.frame_size)
            packets.append(packet)
            print("Flushed 1 final Opus packet")
        except Exception as e:
            print(f"Opus flush error: {e}")
            
        self.buffer = bytearray()
        return packets

    def reset(self):
        self.buffer = bytearray()
        if self.enabled:
            try:
                self.encoder.reset_state()
            except:
                pass



@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("[Chat] WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            print(f"[Chat] Received prompt: {prompt}")
            
            # Lazy init services on first request
            if not llm_service.is_initialized():
                await websocket.send_json({"type": "token", "content": " [Initializing models...]\n\n"})
                llm_service.initialize_llm()
            if not tts_service.is_initialized():
                tts_service.initialize_tts()

            voice = data.get("voice", "dave")
            system_prompt = data.get("system_prompt", "You are a helpful assistant. Keep responses concise and conversational.")
            
            if not prompt:
                await websocket.send_json({"type": "error", "message": "No prompt provided"})
                continue
            
            # Log user conversation
            personality = db_service.get_personality_by_voice(voice)
            personality_id = personality.id if personality else None
            db_service.log_conversation(role="user", transcript=prompt, personality_id=personality_id)
            
            ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
            if ref_codes is None:
                await websocket.send_json({"type": "error", "message": f"Voice {voice} not found"})
                continue
            
            sentence_buffer = SentenceBuffer()
            full_response = ""
            
            try:
                print("[Chat] Starting LLM generation...")
                for token in llm_service.generate_stream(prompt, system_prompt):
                    full_response += token
                    await websocket.send_json({"type": "token", "content": token})
                    
                    sentences = sentence_buffer.add(token)
                    for sentence in sentences:
                        print(f"[Chat] Processing sentence: '{sentence}'")
                        audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                        if audio_chunks:
                            print(f"[Chat] Generated {len(audio_chunks)} audio chunks for sentence")
                            wav_data = tts_service.create_wav_data(audio_chunks)
                            audio_b64 = base64.b64encode(wav_data).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio",
                                "content": audio_b64,
                                "sentence": sentence
                            })
                            print(f"[Chat] Sent audio packet ({len(wav_data)} bytes)")
                        else:
                            print("[Chat] No audio chunks generated for sentence")
                
                remaining = sentence_buffer.flush()
                if remaining:
                    print(f"[Chat] Processing remaining text: '{remaining}'")
                    audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
                    if audio_chunks:
                        print(f"[Chat] Generated {len(audio_chunks)} audio chunks for remaining text")
                        wav_data = tts_service.create_wav_data(audio_chunks)
                        audio_b64 = base64.b64encode(wav_data).decode('utf-8')
                        await websocket.send_json({
                            "type": "audio",
                            "content": audio_b64,
                            "sentence": remaining
                        })
                        print(f"[Chat] Sent final audio packet ({len(wav_data)} bytes)")
                
                print("[Chat] Generation complete")
                # Log AI response
                db_service.log_conversation(role="ai", transcript=full_response, personality_id=personality_id)
                await websocket.send_json({"type": "done", "full_response": full_response})
                
            except Exception as e:
                print(f"Error in chat generation: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": str(e)})
    
    except WebSocketDisconnect:
        print("[Chat] WebSocket disconnected")
    finally:
        print("[Chat] Cleaning up chat resources... (Keeping models loaded)")
        # llm_service.unload()
        # tts_service.unload()


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """Voice chat: audio in -> STT -> LLM -> TTS -> audio out"""
    await websocket.accept()
    
    # Lazy init on connection
    if not stt_service.is_initialized():
        print("Initializing STT for voice chat...")
        stt_service.initialize_stt()
    if not llm_service.is_initialized():
        print("Initializing LLM for voice chat...")
        llm_service.initialize_llm()
    if not tts_service.is_initialized():
        print("Initializing TTS for voice chat...")
        tts_service.initialize_tts()
    
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
                    # Fetch personality for logging
                    personality = db_service.get_personality_by_voice(voice)
                    personality_id = personality.id if personality else None
                    
                    await websocket.send_json({"type": "pause_mic"})
                    await generate_greeting(websocket, voice, system_prompt, is_esp32=False, personality_id=personality_id)
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
                        # Log user conversation
                        personality = db_service.get_personality_by_voice(voice)
                        personality_id = personality.id if personality else None
                        db_service.log_conversation(role="user", transcript=result.text, personality_id=personality_id)

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
                        
                        # Log AI response
                        db_service.log_conversation(role="ai", transcript=full_response, personality_id=personality_id)
                        
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
        print("Cleaning up voice resources... (Keeping models loaded)")
        # stt_service.unload()
        # llm_service.unload()
        # tts_service.unload()


async def send_audio_chunked(websocket: WebSocket, pcm_data: np.ndarray, chunk_size: int = 1024) -> bool:
    """
    Send PCM data in chunks to avoid WebSocket payload limits.
    Returns True if sent successfully, False if client disconnected.
    """
    if pcm_data.dtype == np.float32:
        pcm_bytes = (pcm_data * 32767).astype(np.int16).tobytes()
    else:
        pcm_bytes = pcm_data.tobytes()
    
    try:
        for i in range(0, len(pcm_bytes), chunk_size):
            await websocket.send_bytes(pcm_bytes[i:i + chunk_size])
            # Yield to event loop
            await asyncio.sleep(0)
        return True
    except Exception as e:
        print(f"Error sending audio chunk: {e}")
        return False


async def generate_greeting(websocket: WebSocket, voice: str, system_prompt: str, is_esp32: bool = False, opus_streamer: Optional['OpusStreamer'] = None, personality_id: Optional[str] = None):
    """Generate and send initial greeting from the assistant."""
    ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
    if ref_codes is None:
        return
    
    greeting_prompt = "The user just connected. Greet them warmly and briefly ask how you can help."
    
    sentence_buffer = SentenceBuffer()
    full_response = ""
    first_audio = True
    
    try:
        for token in llm_service.generate_stream(greeting_prompt, system_prompt):
            full_response += token
            
            if not is_esp32:
                try:
                    await websocket.send_json({"type": "token", "content": token})
                except Exception:
                    return
            
            for sentence in sentence_buffer.add(token):
                if len(sentence.strip()) < 3:
                    continue
                
                if is_esp32 and first_audio:
                    try:
                        await websocket.send_json({
                            "type": "server", 
                            "msg": "RESPONSE.CREATED",
                            "volume_control": 70
                        })
                    except Exception:
                        return
                    first_audio = False
                
                try:
                    audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                    if audio_chunks:
                        if is_esp32:
                            pcm_data = np.concatenate(audio_chunks)
                            if opus_streamer and opus_streamer.enabled:
                                packets = opus_streamer.process(pcm_data)
                                for packet in packets:
                                    try:
                                        await websocket.send_bytes(packet)
                                        await asyncio.sleep(0)
                                    except Exception:
                                        return
                            else:
                                if not await send_audio_chunked(websocket, pcm_data):
                                    return
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
            if is_esp32 and first_audio:
                try:
                    await websocket.send_json({
                        "type": "server", 
                        "msg": "RESPONSE.CREATED",
                        "volume_control": 70
                    })
                except Exception:
                    return
                first_audio = False

            try:
                audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
                if audio_chunks:
                    if is_esp32:
                        pcm_data = np.concatenate(audio_chunks)
                        if opus_streamer and opus_streamer.enabled:
                            packets = opus_streamer.process(pcm_data)
                            for packet in packets:
                                try:
                                    await websocket.send_bytes(packet)
                                    await asyncio.sleep(0)
                                except Exception:
                                    return
                        else:
                            if not await send_audio_chunked(websocket, pcm_data):
                                return
                    else:
                        wav_data = tts_service.create_wav_data(audio_chunks)
                        await websocket.send_json({
                            "type": "audio",
                            "content": base64.b64encode(wav_data).decode('utf-8'),
                            "sentence": remaining
                        })
            except Exception as e:
                print(f"Greeting TTS error: {e}")
        
        # Log greeting
        db_service.log_conversation(role="ai", transcript=full_response, personality_id=personality_id)

        if is_esp32:
            try:
                await websocket.send_json({"type": "server", "msg": "RESPONSE.COMPLETE"})
            except Exception:
                return
        else:
            try:
                await websocket.send_json({"type": "done", "full_response": full_response})
            except Exception:
                return
        
        print(f"[Greeting] {full_response}")
        
    except Exception as e:
        print(f"Error in generate_greeting: {e}")


@app.websocket("/ws/esp32")
async def websocket_esp32(websocket: WebSocket):
    """
    ESP32 voice pipeline endpoint.
    - Receives: 16-bit PCM audio at 16kHz
    - Sends: Opus-encoded audio + JSON control messages
    """
    await websocket.accept()
    
    # Lazy init on connection
    if not stt_service.is_initialized():
        print("Initializing STT for ESP32...")
        stt_service.initialize_stt()
    if not llm_service.is_initialized():
        print("Initializing LLM for ESP32...")
        llm_service.initialize_llm()
    if not tts_service.is_initialized():
        print("Initializing TTS for ESP32...")
        tts_service.initialize_tts()
    
    voice = "dave"
    system_prompt = "You are a helpful voice assistant. Be concise and conversational."
    input_sample_rate = 16000  # ESP32 mic sample rate
    
    # Initialize Opus Streamer
    opus_streamer = OpusStreamer(sample_rate=24000, frame_duration_ms=60)
    
    # Send auth response
    try:
        await websocket.send_json({
            "type": "auth",
            "volume_control": 70,
            "pitch_factor": 1.0,
            "is_ota": False,
            "is_reset": False
        })
    except Exception:
        print("[ESP32] Client disconnected during auth")
        return
    
    print("[ESP32] Client connected")
    
    # Send initial greeting
    # Fetch personality for logging
    personality = db_service.get_personality_by_voice(voice)
    personality_id = personality.id if personality else None
    
    await generate_greeting(websocket, voice, system_prompt, is_esp32=True, opus_streamer=opus_streamer, personality_id=personality_id)
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception:
                break
            
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
                    
                    # Log user conversation
                    personality = db_service.get_personality_by_voice(voice)
                    personality_id = personality.id if personality else None
                    db_service.log_conversation(role="user", transcript=result.text, personality_id=personality_id)
                    
                    # Notify audio committed
                    try:
                        await websocket.send_json({"type": "server", "msg": "AUDIO.COMMITTED"})
                    except Exception:
                        break
                    
                    # Get TTS reference data
                    ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
                    if ref_codes is None:
                        try:
                            await websocket.send_json({"type": "server", "msg": "RESPONSE.ERROR"})
                        except Exception:
                            break
                        stt_service.reset()
                        continue
                    
                    # Generate LLM response
                    sentence_buffer = SentenceBuffer()
                    full_response = ""
                    client_active = True
                    
                    for token in llm_service.generate_stream(result.text, system_prompt):
                        if not client_active:
                            break
                            
                        full_response += token
                        
                        for sentence in sentence_buffer.add(token):
                            if len(sentence.strip()) < 3:
                                continue
                            
                            # Notify response created (first audio)
                            if not hasattr(websocket, '_sent_response_created'):
                                try:
                                    await websocket.send_json({
                                        "type": "server", 
                                        "msg": "RESPONSE.CREATED",
                                        "volume_control": 70
                                    })
                                    websocket._sent_response_created = True
                                except Exception:
                                    client_active = False
                                    break
                            
                            try:
                                audio_chunks, _ = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                                if audio_chunks:
                                    pcm_data = np.concatenate(audio_chunks)
                                    if opus_streamer.enabled:
                                        packets = opus_streamer.process(pcm_data)
                                        for packet in packets:
                                            try:
                                                await websocket.send_bytes(packet)
                                                await asyncio.sleep(0)
                                            except Exception:
                                                client_active = False
                                                break
                                    else:
                                        if not await send_audio_chunked(websocket, pcm_data):
                                            client_active = False
                                    
                                    if not client_active:
                                        break
                            except Exception as e:
                                print(f"[ESP32] TTS error: {e}")
                        
                        if not client_active:
                            break
                    
                    # Flush remaining text
                    if client_active:
                        remaining = sentence_buffer.flush()
                        if remaining and len(remaining.strip()) >= 3:
                            if not hasattr(websocket, '_sent_response_created'):
                                try:
                                    await websocket.send_json({
                                        "type": "server", 
                                        "msg": "RESPONSE.CREATED",
                                        "volume_control": 70
                                    })
                                    websocket._sent_response_created = True
                                except Exception:
                                    client_active = False
                            
                            if client_active:
                                try:
                                    audio_chunks, _ = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
                                    if audio_chunks:
                                        pcm_data = np.concatenate(audio_chunks)
                                        if opus_streamer.enabled:
                                            packets = opus_streamer.process(pcm_data)
                                            for packet in packets:
                                                try:
                                                    await websocket.send_bytes(packet)
                                                    await asyncio.sleep(0)
                                                except Exception:
                                                    client_active = False
                                                    break
                                        else:
                                            if not await send_audio_chunked(websocket, pcm_data):
                                                client_active = False
                                except Exception as e:
                                    print(f"[ESP32] TTS error: {e}")
                    
                    # Log AI response
                    if client_active:
                        db_service.log_conversation(role="ai", transcript=full_response, personality_id=personality_id)

                    if not client_active:
                        print("[ESP32] Client disconnected during generation")
                        break

                    # Notify response complete
                    try:
                        await websocket.send_json({"type": "server", "msg": "RESPONSE.COMPLETE"})
                    except Exception:
                        break
                    delattr(websocket, '_sent_response_created') if hasattr(websocket, '_sent_response_created') else None
                    
                    print(f"[ESP32] Response: {full_response}")
                    stt_service.reset()
                    opus_streamer.reset()
                    
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
        print("[ESP32] Cleaning up resources... (Keeping models loaded)")
        # stt_service.unload()
        # llm_service.unload()
        # tts_service.unload()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)