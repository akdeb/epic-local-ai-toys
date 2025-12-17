import os
import sys
import signal

# Handle --print-build-info BEFORE heavy imports to avoid DB init
if "--print-build-info" in sys.argv:
    # Import only what we need for build info
    from llm_service import LLMService
    _llm = LLMService()
    print(
        "ELATO_BUILD_INFO "
        f"llm_repo={_llm.model_repo} "
        f"llm_file={_llm.model_file}"
    )
    sys.exit(0)

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
import uuid
import numpy as np
from typing import Optional
try:
    import opuslib
    OPUS_AVAILABLE = True
except Exception as e:
    print(f"Opus library not available: {e}")
    OPUS_AVAILABLE = False

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from models import TextRequest, OpenAISpeechRequest, PersonalityCreate, PersonalityUpdate, UserCreate, UserUpdate, ActiveUserState, AppModeState
from tts_service import tts_service
from llm_service import llm_service
from stt_service import stt_service
from db_service import db_service
from utils import convert_audio_format, get_media_type_and_filename
from path_utils import get_resource_path

app = FastAPI(title="NeuTTS Air Streaming API")


def _sanitize_tts_text(text: str) -> str:
    t = text
    t = t.replace("\r", " ")
    t = t.replace("\n", " ")
    t = t.replace("**", "")
    t = t.replace("__", "")
    t = t.replace("`", "")
    t = re.sub(r"\s+", " ", t).strip()

    # Remove common markdown list prefixes that can sound awkward.
    t = re.sub(r"(^|\s)[-•]\s+", " ", t)

    # Remove emoji / symbols that can cause misreads in some TTS pipelines.
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    return t.strip()


VOICE_SAFE_SYSTEM_PROMPT_SUFFIX = (
    "Output plain spoken text only. "
    "Do not use emojis. "
    "Do not use asterisks (*) or any other markdown formatting. "
    "Do not use bullet lists. "
    "Do not include stage directions like [laughs] or (sighs). "
    "Avoid reading punctuation aloud (e.g., do not say 'asterisk', 'emoji', 'hashtag')."
)


def build_voice_safe_system_prompt(system_prompt: str | None) -> str:
    sp = (system_prompt or "").strip()
    if sp:
        return f"{sp}\n\n{VOICE_SAFE_SYSTEM_PROMPT_SUFFIX}"
    return VOICE_SAFE_SYSTEM_PROMPT_SUFFIX


def _split_tts_chunks(text: str, max_chars: int = 180, min_chars: int = 25) -> list[str]:
    t = _sanitize_tts_text(text)
    if not t:
        return []

    parts = re.split(r"(?<=[,;:\u2014\u2013.!?])\s+", t)
    merged: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf} {p}"
        else:
            merged.append(buf.strip())
            buf = p
    if buf:
        merged.append(buf.strip())

    # Merge tiny trailing chunks to avoid micro-pauses.
    out: list[str] = []
    for p in merged:
        if out and len(p) < min_chars and (len(out[-1]) + 1 + len(p) <= max_chars):
            out[-1] = f"{out[-1]} {p}".strip()
        else:
            out.append(p)

    return [x for x in out if x]


def _split_tts_fallback(text: str, max_chars: int = 80) -> list[str]:
    t = _sanitize_tts_text(text)
    if not t:
        return []
    words = t.split(" ")
    out: list[str] = []
    buf = ""
    for w in words:
        if not w:
            continue
        if not buf:
            buf = w
            continue
        if len(buf) + 1 + len(w) <= max_chars:
            buf = f"{buf} {w}"
        else:
            out.append(buf)
            buf = w
    if buf:
        out.append(buf)
    return out


def _iter_tts_audio_pairs(text: str, ref_codes, ref_text):
    for chunk in _split_tts_chunks(text):
        audio_chunks, _ = tts_service.generate_audio_with_timing(chunk, ref_codes, ref_text)
        if audio_chunks:
            yield chunk, audio_chunks
            continue

        for sub in _split_tts_fallback(chunk):
            audio_chunks, _ = tts_service.generate_audio_with_timing(sub, ref_codes, ref_text)
            if audio_chunks:
                yield sub, audio_chunks

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
        "http://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Lazy init: Models are loaded only when first requested to save resources/battery
    print("NeuTTS Air started. Models will be loaded on first use.")
    pass

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down: unloading models...")
    try:
        stt_service.unload()
    except Exception as e:
        print(f"Warning: failed to unload STT: {e}")
    try:
        tts_service.unload()
    except Exception as e:
        print(f"Warning: failed to unload TTS: {e}")
    try:
        llm_service.unload()
    except Exception as e:
        print(f"Warning: failed to unload LLM: {e}")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import gc

        gc.collect()
    except Exception:
        pass

    print("Shutdown: cleanup complete")


@app.post("/shutdown")
async def shutdown():
    def _terminate_self() -> None:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            os._exit(0)

    try:
        loop = asyncio.get_running_loop()
        loop.call_later(0.1, _terminate_self)
    except Exception:
        _terminate_self()
    return {"status": "shutting_down"}

@app.get("/")
async def read_root():
    return FileResponse(get_resource_path("index.html"))

@app.get("/chat")
async def read_chat():
    return FileResponse(get_resource_path("chat.html"))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "tts_initialized": tts_service.is_initialized()}

@app.get("/active-user")
async def get_active_user():
    return {"user_id": db_service.get_active_user_id()}

@app.put("/active-user")
async def set_active_user(payload: ActiveUserState):
    db_service.set_active_user_id(payload.user_id)
    return {"user_id": db_service.get_active_user_id()}

@app.get("/app-mode")
async def get_app_mode():
    return {"mode": db_service.get_app_mode()}

@app.put("/app-mode")
async def set_app_mode(payload: AppModeState):
    return {"mode": db_service.set_app_mode(payload.mode)}

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
async def list_conversations(limit: int = 50, offset: int = 0, session_id: Optional[str] = None):
    return db_service.get_conversations(limit, offset, session_id=session_id)

@app.get("/sessions")
async def list_sessions(limit: int = 50, offset: int = 0, user_id: Optional[str] = None):
    return db_service.get_sessions(limit=limit, offset=offset, user_id=user_id)

@app.get("/device-status")
async def get_device_status():
    return db_service.get_device_status()

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
    session_id = str(uuid.uuid4())
    websocket._elato_session_id = session_id
    db_service.start_session(session_id=session_id, client_type="computer")
    greeted = False
    assistant_speaks_first = True
    awaiting_tts_done = False
    awaiting_tts_id: int | None = None
    next_tts_id = 0
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break

            msg_bytes = message.get("bytes")
            msg_text = message.get("text")

            if awaiting_tts_done:
                if msg_text is not None:
                    try:
                        data = json.loads(msg_text)
                        if data.get("type") == "tts_done":
                            if awaiting_tts_id is None or data.get("id") == awaiting_tts_id:
                                awaiting_tts_done = False
                                awaiting_tts_id = None
                                await websocket.send_json({"type": "resume_mic"})
                                stt_service.reset()
                                continue
                    except Exception:
                        pass

                # Ignore any audio bytes that arrive while TTS is playing (client may have buffered frames)
                continue

            if not greeted:
                if msg_text is not None:
                    # Config message
                    import json
                    try:
                        data = json.loads(msg_text)
                        if "voice" in data:
                            voice = data["voice"]
                        if "system_prompt" in data:
                            system_prompt = data["system_prompt"]
                        if "assistant_speaks_first" in data:
                            assistant_speaks_first = bool(data["assistant_speaks_first"])
                    except Exception:
                        pass

                greeted = True
                # Fetch personality for logging
                personality = db_service.get_personality_by_voice(voice)
                personality_id = personality.id if personality else None

                db_service.start_session(session_id=session_id, client_type="computer", personality_id=personality_id)

                if assistant_speaks_first:
                    await websocket.send_json({"type": "pause_mic"})
                    await generate_greeting(websocket, voice, system_prompt, is_esp32=False, personality_id=personality_id)
                    next_tts_id += 1
                    awaiting_tts_id = next_tts_id
                    await websocket.send_json({"type": "tts_end", "id": awaiting_tts_id})
                    awaiting_tts_done = True
                else:
                    try:
                        await websocket.send_json({"type": "resume_mic"})
                    except Exception:
                        break
                continue

            if msg_bytes is not None:
                # Process audio chunk
                audio_bytes = msg_bytes
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
                        db_service.start_session(session_id=session_id, client_type="computer", personality_id=personality_id)
                        db_service.log_conversation(role="user", transcript=result.text, session_id=session_id)

                        # Pause mic while generating response
                        await websocket.send_json({"type": "pause_mic"})
                        
                        # Generate LLM + TTS response
                        ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
                        if ref_codes is None:
                            await websocket.send_json({"type": "resume_mic"})
                            continue
                        
                        sentence_buffer = SentenceBuffer()
                        full_response = ""

                        safe_system_prompt = build_voice_safe_system_prompt(system_prompt)
                        for token in llm_service.generate_stream(result.text, safe_system_prompt):
                            full_response += token
                            await websocket.send_json({"type": "token", "content": token})
                            
                            for sentence in sentence_buffer.add(token):
                                if len(sentence.strip()) < 3:  # Skip very short text
                                    continue
                                try:
                                    for spoken, audio_chunks in _iter_tts_audio_pairs(sentence, ref_codes, ref_text):
                                        wav_data = tts_service.create_wav_data(audio_chunks)
                                        await websocket.send_json({
                                            "type": "audio",
                                            "content": base64.b64encode(wav_data).decode('utf-8'),
                                            "sentence": spoken
                                        })
                                except Exception as e:
                                    print(f"TTS error for '{sentence}': {e}")
                        
                        remaining = sentence_buffer.flush()
                        if remaining and len(remaining.strip()) >= 3:
                            try:
                                for spoken, audio_chunks in _iter_tts_audio_pairs(remaining, ref_codes, ref_text):
                                    wav_data = tts_service.create_wav_data(audio_chunks)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "content": base64.b64encode(wav_data).decode('utf-8'),
                                        "sentence": spoken
                                    })
                            except Exception as e:
                                print(f"TTS error for '{remaining}': {e}")
                        
                        # Log AI response
                        db_service.log_conversation(role="ai", transcript=full_response, session_id=session_id)
                        
                        await websocket.send_json({"type": "done", "full_response": full_response})
                        next_tts_id += 1
                        awaiting_tts_id = next_tts_id
                        await websocket.send_json({"type": "tts_end", "id": awaiting_tts_id})
                        awaiting_tts_done = True
                        
            elif msg_text is not None:
                # Config message
                import json
                try:
                    data = json.loads(msg_text)
                    if data.get("type") == "tts_done":
                        if awaiting_tts_id is None or data.get("id") == awaiting_tts_id:
                            awaiting_tts_done = False
                            awaiting_tts_id = None
                            await websocket.send_json({"type": "resume_mic"})
                            stt_service.reset()
                            continue
                    if data.get("type") == "text_prompt" and data.get("prompt"):
                        prompt = str(data.get("prompt"))

                        personality = db_service.get_personality_by_voice(voice)
                        personality_id = personality.id if personality else None
                        db_service.start_session(session_id=session_id, client_type="computer", personality_id=personality_id)
                        db_service.log_conversation(role="user", transcript=prompt, session_id=session_id)

                        await websocket.send_json({"type": "pause_mic"})
                        ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
                        if ref_codes is None:
                            await websocket.send_json({"type": "resume_mic"})
                            continue

                        sentence_buffer = SentenceBuffer()
                        full_response = ""

                        safe_system_prompt = build_voice_safe_system_prompt(system_prompt)
                        for token in llm_service.generate_stream(prompt, safe_system_prompt):
                            full_response += token
                            await websocket.send_json({"type": "token", "content": token})
                            for sentence in sentence_buffer.add(token):
                                if len(sentence.strip()) < 3:
                                    continue
                                try:
                                    for spoken, audio_chunks in _iter_tts_audio_pairs(sentence, ref_codes, ref_text):
                                        wav_data = tts_service.create_wav_data(audio_chunks)
                                        await websocket.send_json({
                                            "type": "audio",
                                            "content": base64.b64encode(wav_data).decode('utf-8'),
                                            "sentence": spoken,
                                        })
                                except Exception as e:
                                    print(f"TTS error for '{sentence}': {e}")

                        remaining = sentence_buffer.flush()
                        if remaining and len(remaining.strip()) >= 3:
                            try:
                                for spoken, audio_chunks in _iter_tts_audio_pairs(remaining, ref_codes, ref_text):
                                    wav_data = tts_service.create_wav_data(audio_chunks)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "content": base64.b64encode(wav_data).decode('utf-8'),
                                        "sentence": spoken,
                                    })
                            except Exception as e:
                                print(f"TTS error for '{remaining}': {e}")

                        db_service.log_conversation(role="ai", transcript=full_response, session_id=session_id)
                        await websocket.send_json({"type": "done", "full_response": full_response})
                        next_tts_id += 1
                        awaiting_tts_id = next_tts_id
                        await websocket.send_json({"type": "tts_end", "id": awaiting_tts_id})
                        awaiting_tts_done = True
                        continue
                    if "voice" in data:
                        voice = data["voice"]
                    if "system_prompt" in data:
                        system_prompt = data["system_prompt"]
                except:
                    pass
                    
    except WebSocketDisconnect:
        print("Voice WebSocket disconnected")
    except Exception as e:
        print(f"Voice WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            reason = str(e)
            if len(reason) > 120:
                reason = reason[:120]
            await websocket.close(code=1011, reason=reason)
        except Exception:
            pass
    finally:
        stt_service.reset()
        try:
            db_service.end_session(session_id)
        except Exception:
            pass
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
        safe_system_prompt = build_voice_safe_system_prompt(system_prompt)
        for token in llm_service.generate_stream(greeting_prompt, safe_system_prompt):
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
                    for spoken, audio_chunks in _iter_tts_audio_pairs(sentence, ref_codes, ref_text):
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
                                "sentence": spoken
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
                for spoken, audio_chunks in _iter_tts_audio_pairs(remaining, ref_codes, ref_text):
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
                            "sentence": spoken
                        })
            except Exception as e:
                print(f"Greeting TTS error: {e}")
        
        # Log greeting
        session_id = getattr(websocket, "_elato_session_id", None)
        db_service.log_conversation(role="ai", transcript=full_response, session_id=session_id)

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

    if db_service.get_app_mode() != "chat":
        try:
            await websocket.send_json({"type": "error", "message": "ESP32 is disabled unless the app is in Chat mode"})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        return
    
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

    session_id = str(uuid.uuid4())
    websocket._elato_session_id = session_id

    try:
        db_service.set_device_status(True, session_id=session_id)
    except Exception:
        pass

    # If UI selected an active user + personality, use it for this session.
    active_user_id = db_service.get_active_user_id()
    if active_user_id:
        active_user = db_service.get_user(active_user_id)
        if active_user and active_user.current_personality_id:
            p = db_service.get_personality(active_user.current_personality_id)
            if p:
                voice = p.voice_id
                system_prompt = p.prompt
    
    # Initialize Opus Streamer
    opus_streamer = OpusStreamer(sample_rate=24000, frame_duration_ms=60)
    
    # Send auth response
    try:
        volume_control = 70
        if active_user_id:
            active_user = db_service.get_user(active_user_id)
            if active_user and active_user.device_volume is not None:
                volume_control = active_user.device_volume

        await websocket.send_json({
            "type": "auth",
            "volume_control": volume_control,
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

    db_service.start_session(session_id=session_id, client_type="device", user_id=active_user_id, personality_id=personality_id)
    
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
                    db_service.start_session(session_id=session_id, client_type="device", user_id=active_user_id, personality_id=personality_id)
                    db_service.log_conversation(role="user", transcript=result.text, session_id=session_id)
                    
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

                    safe_system_prompt = build_voice_safe_system_prompt(system_prompt)
                    for token in llm_service.generate_stream(result.text, safe_system_prompt):
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
                                for spoken, audio_chunks in _iter_tts_audio_pairs(sentence, ref_codes, ref_text):
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
                                    for spoken, audio_chunks in _iter_tts_audio_pairs(remaining, ref_codes, ref_text):
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
                    
                    # Log AI response
                    if client_active:
                        db_service.log_conversation(role="ai", transcript=full_response, session_id=session_id)

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
                except Exception:
                    pass

    except WebSocketDisconnect:
        print("[ESP32] WebSocket disconnected")
    finally:
        try:
            db_service.end_session(session_id)
        except Exception:
            pass
        try:
            db_service.set_device_status(False, session_id=None)
        except Exception:
            pass
        stt_service.reset()
        opus_streamer.reset()
        print("[ESP32] Cleaning up ESP32 resources... (Keeping models loaded)")
        # stt_service.unload()
        # llm_service.unload()
        # tts_service.unload()


if __name__ == "__main__":
    import uvicorn

    port = 8000
    env_port = os.environ.get("ELATO_PORT")
    if env_port:
        try:
            port = int(env_port)
        except Exception:
            port = 8000

    if "--port" in sys.argv:
        try:
            i = sys.argv.index("--port")
            if i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        except Exception:
            pass
    else:
        for arg in sys.argv:
            if arg.startswith("--port="):
                try:
                    port = int(arg.split("=", 1)[1])
                except Exception:
                    pass

    uvicorn.run(app, host="0.0.0.0", port=port)