# main.py
import argparse
import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional

from engine.characters import build_llm_messages, build_runtime_context, build_system_prompt

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from mlx_lm import generate as mx_generate
from mlx_lm.utils import load as load_llm

from mlx_audio.stt.models.whisper import Model as Whisper
from tts import ChatterboxTTS
import db_service  # DB ops exposed via HTTP endpoints
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        silence_threshold=0.03,
        silence_duration=1.5,
        input_sample_rate=16_000,
        output_sample_rate=24_000,
        streaming_interval=1.5,
        frame_duration_ms=30,
        stt_model="mlx-community/whisper-large-v3-turbo",
        llm_model="Qwen/Qwen2.5-0.5B-Instruct-4bit",
        tts_ref_audio: str | None = None,
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.streaming_interval = streaming_interval
        self.frame_duration_ms = frame_duration_ms

        # Hardcoded STT model as requested
        self.stt_model_id = "mlx-community/whisper-large-v3-turbo"
        # LLM model is dynamic
        self.llm_model = llm_model
        self.tts_ref_audio = tts_ref_audio

        self.mlx_lock = asyncio.Lock()

    async def init_models(self):
        logger.info(f"Loading text generation model: {self.llm_model}")
        self.llm, self.tokenizer = await asyncio.to_thread(
            lambda: load_llm(self.llm_model)
        )

        logger.info(f"Loading speech-to-text model: {self.stt_model_id}")
        self.stt = Whisper.from_pretrained(self.stt_model_id)

        # Initialize Chatterbox TTS
        logger.info("Loading Chatterbox TTS...")
        self.tts = ChatterboxTTS(
            model_id="mlx-community/chatterbox-turbo-4bit",
            # ref_audio_path=self.tts_ref_audio,
            output_sample_rate=self.output_sample_rate,
            stream=True,
            streaming_interval=self.streaming_interval,
        )

        await asyncio.to_thread(self.tts.load)

        # # Warm up TTS once at startup
        # logger.info("Warming up TTS...")
        # async with self.mlx_lock:
        #     await asyncio.to_thread(self.tts.warmup)
        # logger.info("TTS warmup done.")

    async def generate_text_simple(self, prompt: str, max_tokens=100) -> str:
        """Helper to generate text from the loaded LLM for internal tasks."""
        if not self.llm or not self.tokenizer:
            # If LLM is not loaded, we can't generate. 
            # In a real app we might want to load it on demand or fail gracefully.
            raise RuntimeError("LLM not initialized")
            
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        async with self.mlx_lock:
             # Run generation in thread to avoid blocking loop
             response = await asyncio.to_thread(
                 lambda: mx_generate(
                     self.llm, 
                     self.tokenizer, 
                     prompt=formatted_prompt, 
                     max_tokens=max_tokens, 
                     verbose=False
                 )
             )
        return response.strip()

    async def transcribe(self, audio_bytes: bytes) -> str:
        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        async with self.mlx_lock:
            result = await asyncio.to_thread(self.stt.generate, mx.array(audio))
        return result.text.strip()

    async def generate_response(
        self,
        text: str,
        system_prompt: str = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate full LLM response text.
        """
        if messages is None:
            sys_content = system_prompt or (
                "You are a helpful voice assistant. You always respond with short "
                "sentences and never use punctuation like parentheses or colons "
                "that wouldn't appear in conversational speech."
            )
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": text},
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        async with self.mlx_lock:
            response = await asyncio.to_thread(
                lambda: mx_generate(
                    self.llm,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
            )
        return response.strip()

    async def synthesize_speech(self, text: str, cancel_event: asyncio.Event = None):
        """
        Generator that yields audio chunks (as int16 PCM bytes) for the given text.
        Can be cancelled by setting cancel_event.
        """
        audio_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _tts_stream():
            # TTS wrapper already returns int16 PCM bytes
            for audio_bytes in self.tts.generate(text):
                if cancel_event and cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(audio_queue.put_nowait, audio_bytes)
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        async with self.mlx_lock:
            tts_task = asyncio.create_task(asyncio.to_thread(_tts_stream))
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    if cancel_event and cancel_event.is_set():
                        break
                    yield chunk
            finally:
                await tts_task


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total connections: {len(self.active_connections)}"
        )


pipeline: VoicePipeline = None
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    
    # Initialize database (already handled by module import, but logging for clarity)
    logger.info("Database service active")

    # Set defaults if not already set (e.g. running via uvicorn directly)
    if not hasattr(app.state, "stt_model"):
        app.state.stt_model = "mlx-community/whisper-large-v3-turbo"
    if not hasattr(app.state, "llm_model"):
        app.state.llm_model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    # if not hasattr(app.state, "tts_ref_audio"):
    #     app.state.tts_ref_audio = os.path.join(os.path.dirname(__file__), "tts", "santa.wav")
    if not hasattr(app.state, "silence_threshold"):
        app.state.silence_threshold = 0.03
    if not hasattr(app.state, "silence_duration"):
        app.state.silence_duration = 1.5
    if not hasattr(app.state, "streaming_interval"):
        app.state.streaming_interval = 3
    if not hasattr(app.state, "output_sample_rate"):
        app.state.output_sample_rate = 24_000
    
    pipeline = VoicePipeline(
        stt_model=app.state.stt_model,
        llm_model=app.state.llm_model,
        # tts_ref_audio=app.state.tts_ref_audio,
        tts_ref_audio=None,
        silence_threshold=app.state.silence_threshold,
        silence_duration=app.state.silence_duration,
        streaming_interval=app.state.streaming_interval,
        output_sample_rate=app.state.output_sample_rate,
    )
    await pipeline.init_models()
    logger.info("Voice pipeline initialized")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Voice Pipeline WebSocket Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HTTP Endpoints for Settings ---

from pydantic import BaseModel
from typing import Optional, Dict, Any

class SettingUpdate(BaseModel):
    value: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/settings")
async def get_all_settings():
    """Get all settings from app_state."""
    return db_service.db_service.get_all_settings()

@app.get("/settings/{key}")
async def get_setting(key: str):
    """Get a specific setting by key."""
    value = db_service.db_service.get_setting(key)
    return {"key": key, "value": value}

@app.put("/settings/{key}")
async def set_setting(key: str, body: SettingUpdate):
    """Set a setting value."""
    db_service.db_service.set_setting(key, body.value)
    return {"key": key, "value": body.value}

@app.delete("/settings/{key}")
async def delete_setting(key: str):
    """Delete a setting."""
    success = db_service.db_service.delete_setting(key)
    return {"deleted": success}

# --- Convenience endpoints for common settings ---

@app.get("/active-user")
async def get_active_user():
    """Get the active user ID."""
    user_id = db_service.db_service.get_active_user_id()
    user = db_service.db_service.get_user(user_id) if user_id else None
    return {
        "user_id": user_id,
        "user": {
            "id": user.id,
            "name": user.name,
            "current_personality_id": user.current_personality_id,
        } if user else None
    }

class ActiveUserUpdate(BaseModel):
    user_id: Optional[str] = None

@app.put("/active-user")
async def set_active_user(body: ActiveUserUpdate):
    """Set the active user ID."""
    db_service.db_service.set_active_user_id(body.user_id)
    return await get_active_user()

@app.get("/app-mode")
async def get_app_mode():
    """Get the current app mode."""
    return {"mode": db_service.db_service.get_app_mode()}

class AppModeUpdate(BaseModel):
    mode: str

@app.put("/app-mode")
async def set_app_mode(body: AppModeUpdate):
    """Set the app mode."""
    mode = db_service.db_service.set_app_mode(body.mode)
    return {"mode": mode}

# --- ESP32 device state ---

class DeviceUpdate(BaseModel):
    mac_address: Optional[str] = None
    volume: Optional[int] = None
    flashed: Optional[bool] = None
    ws_status: Optional[str] = None
    ws_last_seen: Optional[float] = None
    firmware_version: Optional[str] = None

@app.get("/device")
async def get_device():
    """Get current ESP32 device state."""
    return db_service.db_service.get_device_status()

@app.put("/device")
async def update_device(body: DeviceUpdate):
    """Patch ESP32 device state."""
    patch = body.model_dump(exclude_unset=True)
    return db_service.db_service.update_esp32_device(patch)


class FirmwareFlashRequest(BaseModel):
    port: str
    baud: int = 460800
    chip: str = "esp32s3"
    offset: str = "0x10000"


def _list_serial_ports() -> List[str]:
    try:
        from serial.tools import list_ports  # type: ignore

        ports = [p.device for p in list_ports.comports() if getattr(p, "device", None)]
        ports = [p for p in ports if isinstance(p, str) and p]
        return sorted(list(dict.fromkeys(ports)))
    except Exception:
        paths = []
        paths.extend(glob("/dev/tty.*"))
        paths.extend(glob("/dev/cu.*"))
        return sorted(list(dict.fromkeys([p for p in paths if isinstance(p, str) and p])))


def _firmware_bin_path() -> Path:
    base = Path(__file__).resolve().parent
    return base / "firmware" / "firmware.bin"


@app.get("/firmware/ports")
async def firmware_ports():
    return {"ports": _list_serial_ports()}


@app.post("/firmware/flash")
async def firmware_flash(body: FirmwareFlashRequest):
    fw_path = _firmware_bin_path()
    if not fw_path.exists():
        raise HTTPException(status_code=404, detail=f"firmware.bin not found at {fw_path}")

    cmd = [
        sys.executable,
        "-m",
        "esptool",
        "--before",
        "default_reset",
        "--after",
        "hard_reset",
        "--chip",
        body.chip,
        "--port",
        body.port,
        "--baud",
        str(body.baud),
        "write_flash",
        "-z",
        body.offset,
        str(fw_path),
    ]

    def run() -> Dict[str, object]:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "command": " ".join(cmd),
            "output": out,
        }

    return await asyncio.to_thread(run)

# --- ESP32 WebSocket Endpoint ---
# Mirrors /ws but for ESP32 binary audio (no base64, direct PCM bytes)

import webrtcvad

@app.websocket("/ws/esp32")
async def websocket_esp32(websocket: WebSocket):
    """
    ESP32 voice pipeline endpoint.
    Same flow as /ws but:
    - Receives raw PCM bytes (not base64 JSON)
    - Uses VAD for end-of-speech detection
    - Sends raw PCM bytes back (not base64)
    - Sends JSON control messages for device state
    """
    await websocket.accept()
    
    if not pipeline:
        await websocket.close()
        return

    session_id = str(uuid.uuid4())
    
    # Get active user/personality
    active_user_id = db_service.db_service.get_active_user_id()
    personality_id = None
    if active_user_id:
        u = db_service.db_service.get_user(active_user_id)
        personality_id = u.current_personality_id if u else None

    def _build_llm_context(user_text: str) -> List[Dict[str, str]]:
        try:
            convos = db_service.db_service.get_conversations(session_id=session_id)
        except Exception:
            convos = []

        runtime = build_runtime_context()
        personality = None
        if personality_id:
            try:
                personality = db_service.db_service.get_personality(personality_id)
            except Exception:
                personality = None

        user_ctx = None
        try:
            u = db_service.db_service.get_user(active_user_id) if active_user_id else None
            if u:
                user_ctx = {
                    "name": u.name,
                    "age": u.age,
                    "hobbies": u.hobbies or [],
                    "user_type": u.user_type,
                }
        except Exception:
            user_ctx = None

        behavior_constraints = (
            "You always respond with short sentences. "
            "Avoid punctuation like parentheses or colons that would not appear in conversational speech."
        )

        sys_prompt = build_system_prompt(
            personality_name=getattr(personality, "name", None),
            personality_prompt=getattr(personality, "prompt", None),
            user_context=user_ctx,
            runtime=runtime,
            extra_system_prompt=behavior_constraints,
        )

        history_msgs: List[Dict[str, str]] = []
        for c in convos:
            if c.role == "user":
                history_msgs.append({"role": "user", "content": c.transcript})
            elif c.role == "ai":
                history_msgs.append({"role": "assistant", "content": c.transcript})

        return build_llm_messages(
            system_prompt=sys_prompt,
            history=history_msgs,
            user_text=user_text,
            max_history_messages=30,
        )

    # Start session
    db_service.db_service.start_session(
        session_id=session_id,
        client_type="device",
        user_id=active_user_id,
        personality_id=personality_id,
    )
    
    # Send auth to device
    volume = 70
    if active_user_id:
        u = db_service.db_service.get_user(active_user_id)
        if u and u.device_volume is not None:
            volume = u.device_volume
    
    try:
        await websocket.send_json({
            "type": "auth",
            "volume_control": volume,
            "pitch_factor": 1.0,
            "is_ota": False,
            "is_reset": False
        })
    except Exception:
        return

    logger.info(f"[ESP32] Client connected, session={session_id}")

    # Initial greeting
    cancel_event = asyncio.Event()
    try:
        greeting_user_text = "[System] The device just connected. Greet the user with a short friendly sentence (under 8 words)."
        greeting_messages = _build_llm_context(greeting_user_text)
        greeting_text = await pipeline.generate_response(
            greeting_user_text, messages=greeting_messages, max_tokens=50
        )
        greeting_text = greeting_text.strip() or "Hello!"
        try:
            await websocket.send_json({"type": "server", "msg": "RESPONSE.CREATED", "volume_control": volume})
        except Exception:
            pass

        async for audio_chunk in pipeline.synthesize_speech(greeting_text):
            try:
                await websocket.send_bytes(audio_chunk)
            except Exception:
                break

        try:
            await websocket.send_json({"type": "server", "msg": "RESPONSE.COMPLETE"})
        except Exception:
            pass

        try:
            db_service.db_service.log_conversation(role="ai", transcript=greeting_text, session_id=session_id)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"[ESP32] Greeting generation failed: {e}")

    # VAD setup
    vad = webrtcvad.Vad(3)
    audio_buffer = bytearray()
    input_sample_rate = 16000
    vad_frame_ms = 30
    vad_frame_bytes = int(input_sample_rate * vad_frame_ms / 1000) * 2
    
    speech_frames = []
    is_speaking = False
    silence_count = 0
    SILENCE_FRAMES = int(1.5 / (vad_frame_ms / 1000))  # 1.5s
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception:
                break
            
            if message.get("type") == "websocket.disconnect":
                break
            
            # Binary audio from ESP32
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
                
                # VAD processing
                while len(audio_buffer) >= vad_frame_bytes:
                    frame = bytes(audio_buffer[:vad_frame_bytes])
                    audio_buffer = audio_buffer[vad_frame_bytes:]
                    
                    is_speech = vad.is_speech(frame, input_sample_rate)
                    
                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            logger.info("[ESP32] Speech started")
                        speech_frames.append(frame)
                        silence_count = 0
                    elif is_speaking:
                        speech_frames.append(frame)
                        silence_count += 1
                        
                        if silence_count > SILENCE_FRAMES:
                            is_speaking = False
                            logger.info("[ESP32] Speech ended, processing...")
                            
                            # Combine and transcribe
                            full_audio = b"".join(speech_frames)
                            speech_frames = []
                            silence_count = 0
                            
                            transcription = await pipeline.transcribe(full_audio)
                            
                            if transcription and transcription.strip():
                                logger.info(f"[ESP32] Transcript: {transcription}")
                                db_service.db_service.log_conversation(
                                    role="user", transcript=transcription, session_id=session_id
                                )
                                
                                try:
                                    await websocket.send_json({"type": "server", "msg": "AUDIO.COMMITTED"})
                                except Exception:
                                    break
                                
                                # Generate LLM response, then stream TTS
                                cancel_event.clear()
                                llm_messages = _build_llm_context(transcription)
                                
                                full_response = await pipeline.generate_response(
                                    transcription, messages=llm_messages
                                )
                                
                                if cancel_event.is_set() or not full_response.strip():
                                    continue
                                
                                logger.info(f"[ESP32] LLM response: {full_response}")
                                
                                try:
                                    await websocket.send_json({
                                        "type": "server",
                                        "msg": "RESPONSE.CREATED",
                                        "volume_control": volume
                                    })
                                except Exception:
                                    break
                                
                                # Stream TTS audio
                                async for audio_chunk in pipeline.synthesize_speech(full_response, cancel_event):
                                    if cancel_event.is_set():
                                        break
                                    try:
                                        await websocket.send_bytes(audio_chunk)
                                    except Exception:
                                        cancel_event.set()
                                        break
                                
                                # Log AI response
                                db_service.db_service.log_conversation(
                                    role="ai", transcript=full_response, session_id=session_id
                                )
                                
                                try:
                                    await websocket.send_json({"type": "server", "msg": "RESPONSE.COMPLETE"})
                                except Exception:
                                    break
            
            # JSON config messages
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if "system_prompt" in data:
                        session_system_prompt = data["system_prompt"]
                except Exception:
                    pass
                    
    except WebSocketDisconnect:
        logger.info("[ESP32] Disconnected")
    finally:
        db_service.db_service.end_session(session_id)
        logger.info(f"[ESP32] Session ended: {session_id}")

# --- Models endpoint (for frontend Models.tsx) ---

@app.get("/models")
async def get_models():
    """Get current model configuration."""
    return {
        "llm": {
            "backend": "mlx",
            "repo": db_service.db_service.get_setting("llm_model") or "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "file": None,
            "context_window": 4096,
            "loaded": pipeline is not None and pipeline.llm is not None,
        },
        "tts": {
            "backend": "chatterbox",
            "backbone_repo": "mlx-community/chatterbox-turbo-4bit",
            "codec_repo": None,
            "loaded": pipeline is not None and pipeline.tts is not None,
        },
        "stt": {
            "backend": "whisper",
            "repo": "mlx-community/whisper-large-v3-turbo",
            "loaded": pipeline is not None and pipeline.stt is not None,
        }
    }

class ModelsUpdate(BaseModel):
    model_repo: Optional[str] = None

@app.put("/models")
async def set_models(body: ModelsUpdate):
    """Set model configuration (requires restart to take effect)."""
    if body.model_repo:
        db_service.db_service.set_setting("llm_model", body.model_repo)
    return await get_models()

# --- Voices ---

@app.get("/voices")
async def get_voices(include_non_global: bool = True):
    voices = db_service.db_service.get_voices(include_non_global=include_non_global)
    return [
        {
            "voice_id": v.voice_id,
            "gender": v.gender,
            "voice_name": v.voice_name,
            "voice_description": v.voice_description,
            "voice_src": v.voice_src,
            "is_global": v.is_global,
        }
        for v in voices
    ]

# --- Users CRUD ---

@app.get("/users")
async def get_users():
    """Get all users."""
    users = db_service.db_service.get_users()
    return [
        {
            "id": u.id,
            "name": u.name,
            "age": u.age,
            "current_personality_id": u.current_personality_id,
            "user_type": u.user_type,
            "hobbies": u.hobbies or [],
            "device_volume": u.device_volume,
        }
        for u in users
    ]

class UserCreate(BaseModel):
    name: str
    age: Optional[int] = None

@app.post("/users")
async def create_user(body: UserCreate):
    """Create a new user."""
    user = db_service.db_service.create_user(name=body.name, age=body.age)
    return {"id": user.id, "name": user.name}

@app.put("/users/{user_id}")
async def update_user(user_id: str, body: Dict[str, Any]):
    """Update a user."""
    user = db_service.db_service.update_user(user_id, **body)
    if not user:
        return {"error": "User not found"}, 404
    return {"id": user.id, "name": user.name}

# --- Personalities CRUD ---

@app.get("/personalities")
async def get_personalities(include_hidden: bool = False):
    """Get all personalities."""
    personalities = db_service.db_service.get_personalities(include_hidden=include_hidden)
    return [
        {
            "id": p.id,
            "name": p.name,
            "prompt": p.prompt,
            "short_description": p.short_description,
            "tags": p.tags,
            "is_visible": p.is_visible,
            "is_global": p.is_global,
            "voice_id": p.voice_id,
        }
        for p in personalities
    ]

class PersonalityCreate(BaseModel):
    name: str
    prompt: str
    short_description: Optional[str] = ""
    tags: list = []
    voice_id: str = "radio"
    is_global: bool = False

@app.post("/personalities")
async def create_personality(body: PersonalityCreate):
    """Create a new personality."""
    p = db_service.db_service.create_personality(
        name=body.name,
        prompt=body.prompt,
        short_description=body.short_description or "",
        tags=body.tags,
        voice_id=body.voice_id,
        is_global=False,
    )
    return {"id": p.id, "name": p.name}

class GeneratePersonalityRequest(BaseModel):
    description: str

@app.post("/personalities/generate")
async def generate_personality(body: GeneratePersonalityRequest):
    """Generate a personality from a description using the LLM."""
    if not pipeline:
         raise HTTPException(status_code=503, detail="AI engine not ready")
    
    description = body.description
    logger.info(f"Generating personality from description: {description}")
    
    # 1. Generate Name
    name_prompt = f"Based on this description: '{description}', suggest a short, creative name for this character. Output ONLY the name, nothing else."
    name = await pipeline.generate_text_simple(name_prompt, max_tokens=30)
    name = name.strip().strip('"').strip("'").split("\n")[0]
    
    # 2. Generate Short Description
    desc_prompt = f"Based on this description: '{description}', provide a very short (1 sentence) description of this character. Output ONLY the description."
    short_desc = await pipeline.generate_text_simple(desc_prompt, max_tokens=100)
    short_desc = short_desc.strip().strip('"').strip("'")
    
    # 3. Generate System Prompt
    sys_prompt = f"Based on this description: '{description}', write a system prompt for an AI to act as this character. The prompt should start with 'You are [Name]...'. Output ONLY the prompt."
    system_prompt = await pipeline.generate_text_simple(sys_prompt, max_tokens=300)
    system_prompt = system_prompt.strip()
    
    tags: list = []
    
    # Create the personality
    p = db_service.db_service.create_personality(
        name=name,
        prompt=system_prompt,
        short_description=short_desc,
        tags=tags,
        voice_id="radio",
        is_global=False
    )
    
    return {
        "id": p.id,
        "name": p.name,
        "prompt": p.prompt,
        "short_description": p.short_description,
        "tags": p.tags,
        "voice_id": p.voice_id
    }

@app.put("/personalities/{personality_id}")
async def update_personality(personality_id: str, body: Dict[str, Any]):
    """Update a personality."""
    p = db_service.db_service.update_personality(personality_id, **body)
    if not p:
        return {"error": "Personality not found"}, 404
    return {"id": p.id, "name": p.name}


@app.delete("/personalities/{personality_id}")
async def delete_personality(personality_id: str):
    ok = db_service.db_service.delete_personality(personality_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Personality not found or cannot delete global personality")
    return {"ok": True}

# --- Conversations ---

@app.get("/conversations")
async def get_conversations(limit: int = 50, offset: int = 0, session_id: Optional[str] = None):
    """Get conversations."""
    convos = db_service.db_service.get_conversations(limit=limit, offset=offset, session_id=session_id)
    return [
        {
            "id": c.id,
            "role": c.role,
            "transcript": c.transcript,
            "timestamp": c.timestamp,
            "session_id": c.session_id,
        }
        for c in convos
    ]

# --- Sessions ---

@app.get("/sessions")
async def get_sessions(limit: int = 50, offset: int = 0, user_id: Optional[str] = None):
    """Get sessions."""
    sessions = db_service.db_service.get_sessions(limit=limit, offset=offset, user_id=user_id)
    return [
        {
            "id": s.id,
            "started_at": s.started_at,
            "ended_at": s.ended_at,
            "duration_sec": s.duration_sec,
            "client_type": s.client_type,
            "user_id": s.user_id,
            "personality_id": s.personality_id,
        }
        for s in sessions
    ]

# --- Shutdown ---

@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    import signal
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice communication.

    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 encoded int16 PCM audio>"}
    - Client sends: {"type": "end_of_speech"} to signal end of utterance
    - Server sends: {"type": "transcription", "text": "..."}
    - Server sends: {"type": "response", "text": "..."}
    - Server sends: {"type": "audio", "data": "<base64 encoded int16 PCM audio>"}
    - Server sends: {"type": "audio_end"}
    """
    await manager.connect(websocket)

    session_id = str(uuid.uuid4())
    try:
        user_id = db_service.db_service.get_active_user_id()
        personality_id = None
        if user_id:
            u = db_service.db_service.get_user(user_id)
            personality_id = u.current_personality_id if u else None
        db_service.db_service.start_session(
            session_id=session_id,
            client_type="desktop",
            user_id=user_id,
            personality_id=personality_id,
        )
    except Exception as e:
        logger.error(f"Failed to start session: {e}")

    try:
        await websocket.send_text(json.dumps({"type": "session_started", "session_id": session_id}))
    except Exception:
        pass

    audio_buffer = bytearray()
    cancel_event = asyncio.Event()
    current_tts_task = None
    ws_open = True
    
    # Session state
    session_system_prompt = None
    session_voice = "dave"

    # Prebuffer to avoid initial underrun/garble
    PREBUFFER_MS = 300
    PREBUFFER_BYTES = int(pipeline.output_sample_rate * (PREBUFFER_MS / 1000.0) * 2)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "config":
                session_voice = message.get("voice", "dave")
                session_system_prompt = message.get("system_prompt")
                logger.info(f"Config updated: voice={session_voice}, prompt_len={len(session_system_prompt) if session_system_prompt else 0}")

            elif msg_type == "audio":
                audio_data = base64.b64decode(message["data"])
                audio_buffer.extend(audio_data)

                # If user is speaking while we're TTS-ing, cancel current TTS
                if current_tts_task and not current_tts_task.done():
                    cancel_event.set()
                    try:
                        await current_tts_task
                    except asyncio.CancelledError:
                        pass
                    cancel_event.clear()
                    current_tts_task = None

            elif msg_type == "end_of_speech":
                if audio_buffer:
                    logger.info("Processing audio...")
                    try:
                        convos = db_service.db_service.get_conversations(session_id=session_id)
                    except Exception:
                        convos = []

                    transcription = await pipeline.transcribe(bytes(audio_buffer))
                    audio_buffer.clear()

                    if transcription:
                        logger.info(f"Transcribed: {transcription}")
                        try:
                            db_service.db_service.log_conversation(role="user", transcript=transcription, session_id=session_id)
                        except Exception as e:
                            logger.error(f"Failed to log user conversation: {e}")

                        try:
                            await websocket.send_text(
                                json.dumps({"type": "transcription", "text": transcription})
                            )
                        except Exception:
                            break

                        cancel_event.clear()
                        runtime = build_runtime_context()
                        personality = None
                        if personality_id:
                            try:
                                personality = db_service.db_service.get_personality(personality_id)
                            except Exception:
                                personality = None

                        user_ctx = None
                        try:
                            u = db_service.db_service.get_user(user_id) if user_id else None
                            if u:
                                user_ctx = {
                                    "name": u.name,
                                    "age": u.age,
                                    "hobbies": u.hobbies or [],
                                    "user_type": u.user_type,
                                }
                        except Exception:
                            user_ctx = None

                        behavior_constraints = (
                            "You always respond with short sentences. "
                            "Avoid punctuation like parentheses or colons that would not appear in conversational speech."
                        )

                        sys_prompt = build_system_prompt(
                            personality_name=getattr(personality, "name", None),
                            personality_prompt=getattr(personality, "prompt", None),
                            user_context=user_ctx,
                            runtime=runtime,
                            extra_system_prompt=("\n\n".join([p for p in [session_system_prompt, behavior_constraints] if p])),
                        )

                        history_msgs: List[Dict[str, str]] = []
                        for c in convos:
                            if c.role == "user":
                                history_msgs.append({"role": "user", "content": c.transcript})
                            elif c.role == "ai":
                                history_msgs.append({"role": "assistant", "content": c.transcript})

                        llm_messages = build_llm_messages(
                            system_prompt=sys_prompt,
                            history=history_msgs,
                            user_text=transcription,
                            max_history_messages=30,
                        )
                        async def stream_response():
                            if cancel_event.is_set() or not ws_open:
                                return

                            # Generate full LLM response
                            full_response = await pipeline.generate_response(
                                transcription, messages=llm_messages
                            )

                            if cancel_event.is_set() or not ws_open or not full_response.strip():
                                return

                            logger.info(f"LLM response: {full_response}")

                            # Send text response
                            try:
                                await websocket.send_text(
                                    json.dumps({"type": "response", "text": full_response})
                                )
                            except Exception:
                                return

                            # Log conversation
                            try:
                                db_service.db_service.log_conversation(
                                    role="ai", transcript=full_response, session_id=session_id
                                )
                            except Exception as e:
                                logger.error(f"Failed to log ai conversation: {e}")

                            # Stream TTS audio with prebuffering
                            buffered = bytearray()
                            started = False

                            async for audio_chunk in pipeline.synthesize_speech(
                                full_response, cancel_event
                            ):
                                if cancel_event.is_set() or not ws_open:
                                    break

                                if not started:
                                    buffered.extend(audio_chunk)
                                    if len(buffered) < PREBUFFER_BYTES:
                                        continue

                                    try:
                                        await websocket.send_text(
                                            json.dumps({
                                                "type": "audio",
                                                "data": base64.b64encode(bytes(buffered)).decode("utf-8"),
                                            })
                                        )
                                    except Exception:
                                        break
                                    buffered.clear()
                                    started = True
                                else:
                                    try:
                                        await websocket.send_text(
                                            json.dumps({
                                                "type": "audio",
                                                "data": base64.b64encode(audio_chunk).decode("utf-8"),
                                            })
                                        )
                                    except Exception:
                                        break

                            # Flush remaining buffered audio
                            if buffered:
                                try:
                                    await websocket.send_text(
                                        json.dumps({
                                            "type": "audio",
                                            "data": base64.b64encode(bytes(buffered)).decode("utf-8"),
                                        })
                                    )
                                except Exception:
                                    pass

                            try:
                                await websocket.send_text(json.dumps({"type": "audio_end"}))
                            except Exception:
                                pass

                        current_tts_task = asyncio.create_task(stream_response())

            elif msg_type == "cancel":
                if current_tts_task and not current_tts_task.done():
                    cancel_event.set()
                audio_buffer.clear()

    except WebSocketDisconnect:
        ws_open = False
        if current_tts_task and not current_tts_task.done():
            cancel_event.set()
            current_tts_task.cancel()
        manager.disconnect(websocket)
        try:
            db_service.db_service.end_session(session_id)
        except Exception:
            pass
    except Exception as e:
        ws_open = False
        if current_tts_task and not current_tts_task.done():
            cancel_event.set()
            current_tts_task.cancel()
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        try:
            db_service.db_service.end_session(session_id)
        except Exception:
            pass


def main():
    if len(sys.argv) >= 6 and sys.argv[1:5] == ["-B", "-S", "-I", "-c"]:
        code = sys.argv[5]
        if isinstance(code, str) and code.startswith("from multiprocessing."):
            exec(code, {"__name__": "__main__"})
            return

    parser = argparse.ArgumentParser(description="Voice Pipeline WebSocket Server")
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="LLM model",
    )
    # default_ref_audio = os.path.join(os.path.dirname(__file__), "tts", "santa.wav")
    # parser.add_argument(
    #     "--tts_ref_audio",
    #     type=str,
    #     default=default_ref_audio,
    #     help="Reference audio WAV path for voice cloning",
    # )
    parser.add_argument(
        "--silence_duration", type=float, default=1.5, help="Silence duration"
    )
    parser.add_argument(
        "--silence_threshold", type=float, default=0.03, help="Silence threshold"
    )
    parser.add_argument(
        "--streaming_interval", type=int, default=3, help="Streaming interval"
    )
    parser.add_argument(
        "--output_sample_rate",
        type=int,
        default=24_000,
        help="Output sample rate for TTS audio",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    app.state.stt_model = args.stt_model
    app.state.llm_model = args.llm_model
    # app.state.tts_ref_audio = args.tts_ref_audio
    app.state.silence_threshold = args.silence_threshold
    app.state.silence_duration = args.silence_duration
    app.state.streaming_interval = args.streaming_interval
    app.state.output_sample_rate = args.output_sample_rate

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
