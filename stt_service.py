"""
Kyutai STT Service using MLX (Apple Silicon, no PyTorch).

Dependencies:
    pip install moshi_mlx rustymimi sentencepiece numpy
"""
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
from moshi_mlx import models, utils


SAMPLE_RATE = 24000
BLOCK_SIZE = 1920


@dataclass
class TranscriptionResult:
    text: str
    is_final: bool


class STTService:
    """Kyutai STT using MLX - runs on Apple Silicon without PyTorch."""
    
    def __init__(self):
        self.model = None
        self.gen = None
        self.audio_tokenizer = None
        self.text_tokenizer = None
        self.lm_config = None
        self._initialized = False
        self.current_text = ""
        self.last_was_vad = False
        self.audio_buffer = np.array([], dtype=np.float32)  # Buffer for incoming audio
        self.vad_counter = 0  # Require multiple VAD triggers to confirm end
        
    def initialize_stt(self, hf_repo: str = "kyutai/stt-1b-en_fr-candle", vad: bool = True) -> None:
        if self._initialized:
            return
            
        print(f"Initializing Kyutai STT from {hf_repo}...")
        
        config_path = hf_hub_download(hf_repo, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        mimi_weights = hf_hub_download(hf_repo, config["mimi_name"])
        moshi_name = config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(hf_repo, moshi_name)
        tokenizer_path = hf_hub_download(hf_repo, config["tokenizer_name"])
        
        self.lm_config = models.LmConfig.from_config_dict(config)
        self.model = models.Lm(self.lm_config)
        self.model.set_dtype(mx.bfloat16)
        
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(self.model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(self.model, bits=8, group_size=64)
        
        print(f"Loading model weights from {moshi_weights}")
        if hf_repo.endswith("-candle"):
            self.model.load_pytorch_weights(moshi_weights, self.lm_config, strict=True)
        else:
            self.model.load_weights(moshi_weights, strict=True)
        
        print(f"Loading text tokenizer from {tokenizer_path}")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        
        print(f"Loading audio tokenizer from {mimi_weights}")
        generated_codebooks = self.lm_config.generated_codebooks
        other_codebooks = self.lm_config.other_codebooks
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        self.audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)
        
        print("Warming up model...")
        self.model.warmup()
        
        self.gen = models.LmGen(
            model=self.model,
            max_steps=4096,
            text_sampler=utils.Sampler(top_k=25, temp=0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )
        
        self._initialized = True
        print("STT initialized successfully!")
    
    def process_audio_chunk(self, audio: np.ndarray, use_vad: bool = True) -> Optional[TranscriptionResult]:
        """Process audio - buffers and processes in 1920-sample blocks."""
        if not self._initialized:
            raise RuntimeError("STT not initialized. Call initialize_stt() first.")
        
        # Add to buffer
        if audio.ndim > 1:
            audio = audio.flatten()
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        
        # Process all complete 1920-sample blocks
        result = None
        while len(self.audio_buffer) >= BLOCK_SIZE:
            block = self.audio_buffer[:BLOCK_SIZE]
            self.audio_buffer = self.audio_buffer[BLOCK_SIZE:]
            
            chunk_result = self._process_block(block, use_vad)
            if chunk_result:
                result = chunk_result  # Keep last result
        
        return result
    
    def _process_block(self, block: np.ndarray, use_vad: bool) -> Optional[TranscriptionResult]:
        """Process exactly 1920 samples - matches CLI exactly."""
        # Shape: (1, 1920)
        block = block[None, :]
        
        other_codebooks = self.lm_config.other_codebooks
        other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :other_codebooks]
        
        is_final = False
        if use_vad:
            text_token, vad_heads = self.gen.step_with_extra_heads(other_audio_tokens[0])
            if vad_heads:
                pr_vad = vad_heads[2][0, 0, 0].item()
                if pr_vad > 0.6:  # Higher threshold
                    self.vad_counter += 1
                    # Require 5 consecutive VAD triggers (~400ms silence) and some text
                    if self.vad_counter >= 5 and len(self.current_text.strip()) > 5:
                        is_final = True
                        self.last_was_vad = True
                else:
                    self.vad_counter = 0
                    self.last_was_vad = False
        else:
            text_token = self.gen.step(other_audio_tokens[0])
        
        text_token = text_token[0].item()
        
        new_text = None
        if text_token not in (0, 3):
            new_text = self.text_tokenizer.id_to_piece(text_token)
            new_text = new_text.replace("▁", " ")
            self.current_text += new_text
            self.last_was_vad = False
            print(f"[STT] Text: {new_text}", end="", flush=True)
        
        if is_final and self.current_text.strip():
            print(f"\n[STT] Final: {self.current_text.strip()}")
            result = TranscriptionResult(text=self.current_text.strip(), is_final=True)
            self.current_text = ""
            return result
        elif new_text:
            return TranscriptionResult(text=new_text, is_final=False)
        
        return None
    
    def get_current_text(self) -> str:
        return self.current_text.strip()
    
    def reset(self):
        self.current_text = ""
        self.last_was_vad = False
        self.vad_counter = 0
        self.audio_buffer = np.array([], dtype=np.float32)
        if self.model:
            self.gen = models.LmGen(
                model=self.model,
                max_steps=4096,
                text_sampler=utils.Sampler(top_k=25, temp=0),
                audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                check=False,
            )
    
    def is_initialized(self) -> bool:
        return self._initialized

    def unload(self) -> None:
        if self._initialized:
            print("Unloading STT...")
            self.model = None
            self.gen = None
            self.audio_tokenizer = None
            self.text_tokenizer = None
            self.lm_config = None
            self._initialized = False
            self.current_text = ""
            self.audio_buffer = np.array([], dtype=np.float32)
            import gc
            gc.collect()
            print("STT unloaded successfully!")


stt_service = STTService()