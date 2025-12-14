import os
import time
import base64
from typing import Optional, Dict, Any, Tuple
import numpy as np
from fastapi import HTTPException
from neuttsair.neuttsair.neutts import NeuTTSAir
from utils import create_wav_header
from path_utils import get_resource_path


class TTSService:
    def __init__(self):
        self.tts = None
        self.reference_cache = {}
    
    def initialize_tts(self) -> None:
        if self.tts is None:
            print("Initializing NeuTTSAir...")
            self.tts = NeuTTSAir(
                backbone_repo="neuphonic/neutts-air-q4-gguf",
                backbone_device="gpu",
                codec_repo="neuphonic/neucodec-onnx-decoder",
                codec_device="cpu"
            )
            print("NeuTTSAir initialized successfully!")
            
            print("Caching reference audio data...")
            for voice in ["dave", "jo"]:
                ref_codes_path = get_resource_path(f"neuttsair/samples/{voice}.npy")
                ref_text_path = get_resource_path(f"neuttsair/samples/{voice}.txt")
                
                ref_codes = None
                ref_text = None
                
                if os.path.exists(ref_codes_path):
                    ref_codes = np.load(ref_codes_path)
                
                if os.path.exists(ref_text_path):
                    with open(ref_text_path, "r") as f:
                        ref_text = f.read().strip()
                
                self.reference_cache[voice] = {
                    "ref_codes": ref_codes,
                    "ref_text": ref_text
                }
            
            print("Reference data cached successfully!")
    
    def get_cached_reference_data(self, voice: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if voice not in self.reference_cache:
            return None, None
        
        ref_data = self.reference_cache[voice]
        return ref_data["ref_codes"], ref_data["ref_text"]
    
    def get_reference_data(self, ref_codes_path: str, ref_text_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if "dave" in ref_codes_path:
            return self.get_cached_reference_data("dave")
        elif "jo" in ref_codes_path:
            return self.get_cached_reference_data("jo")
        
        ref_codes = None
        ref_text = None
        
        if ref_codes_path and os.path.exists(ref_codes_path):
            if ref_codes_path.endswith('.npy'):
                ref_codes = np.load(ref_codes_path)
            else:
                # Fallback for .pt files if needed
                import torch
                ref_codes = torch.load(ref_codes_path).numpy()
        
        if ref_text_path and os.path.exists(ref_text_path):
            with open(ref_text_path, "r") as f:
                ref_text = f.read().strip()
        
        return ref_codes, ref_text
    
    def generate_audio_with_timing(self, text: str, ref_codes: Optional[np.ndarray], ref_text: Optional[str]) -> Tuple[list, Dict[str, Any]]:
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        audio_chunks = []
        
        try:
            for chunk in self.tts.infer_stream(text, ref_codes, ref_text):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency_ms = (first_chunk_time - start_time) * 1000
                    print(f"⏱️  Time to first audio chunk: {latency_ms:.2f}ms")
                
                audio = (chunk * 32767).astype(np.int16)
                audio_chunks.append(audio)
                total_chunks += 1
                
        except Exception as e:
            print(f"Error in audio generation: {e}")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        print(f"🎵 Generated {total_chunks} audio chunks in {total_time:.2f}ms total")
        
        timing_info = {
            "latency_ms": (first_chunk_time - start_time) * 1000 if first_chunk_time else 0,
            "total_chunks": total_chunks,
            "total_time_ms": total_time
        }
        
        return audio_chunks, timing_info
    
    def create_wav_data(self, audio_chunks: list) -> bytes:
        full_audio = np.concatenate(audio_chunks)
        audio_bytes = full_audio.astype(np.int16).tobytes()
        
        sample_rate = 24000
        num_channels = 1
        bits_per_sample = 16
        data_size = len(audio_bytes)
        
        wav_header = create_wav_header(sample_rate, num_channels, bits_per_sample, data_size)
        return wav_header + audio_bytes
    
    def is_initialized(self) -> bool:
        return self.tts is not None

    def unload(self) -> None:
        if self.tts is not None:
            print("Unloading TTS...")
            del self.tts
            self.tts = None
            import gc
            gc.collect()
            print("TTS unloaded successfully!")


tts_service = TTSService()
