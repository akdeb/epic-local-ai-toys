import sys
import asyncio
import base64
import numpy as np
import os

# Add current directory to path
sys.path.append(os.getcwd())

from llm_service import llm_service
from tts_service import tts_service

async def test_chat_flow():
    print("Testing Full Chat Flow (LLM + TTS)...")
    
    print("1. Initializing LLM...")
    llm_service.initialize_llm()
    print("LLM Initialized.")
    
    print("2. Initializing TTS...")
    tts_service.initialize_tts()
    print("TTS Initialized.")
    
    prompt = "Say hello to the world in a generic way."
    voice = "dave"
    
    print(f"3. Generating response for: '{prompt}'")
    
    ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
    if ref_codes is None:
        print("Error: Voice not found")
        return

    full_response = ""
    audio_generated = False
    
    # Simple sentence buffer simulation
    import re
    sentence_endings = re.compile(r'([.!?]+[\s]*)')
    buffer = ""
    
    for token in llm_service.generate_stream(prompt):
        full_response += token
        print(f"Token: {token}", end="", flush=True)
        
        buffer += token
        while True:
            match = sentence_endings.search(buffer)
            if match:
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                if sentence:
                    print(f"\n[Sentence]: {sentence}")
                    print("  -> Generating Audio...")
                    try:
                        audio_chunks, timing = tts_service.generate_audio_with_timing(sentence, ref_codes, ref_text)
                        if audio_chunks:
                            print(f"  -> Audio generated: {len(audio_chunks)} chunks, {timing['total_time_ms']:.2f}ms")
                            audio_generated = True
                        else:
                            print("  -> No audio generated for sentence.")
                    except Exception as e:
                        print(f"  -> TTS Error: {e}")
                
                buffer = buffer[end_pos:]
            else:
                break
                
    # Flush
    remaining = buffer.strip()
    if remaining:
        print(f"\n[Remaining]: {remaining}")
        print("  -> Generating Audio...")
        try:
            audio_chunks, timing = tts_service.generate_audio_with_timing(remaining, ref_codes, ref_text)
            if audio_chunks:
                print(f"  -> Audio generated: {len(audio_chunks)} chunks")
                audio_generated = True
        except Exception as e:
            print(f"  -> TTS Error: {e}")

    print(f"\n\nFull LLM Response: {full_response}")
    if audio_generated:
        print("\nSUCCESS: Audio was generated during the flow.")
    else:
        print("\nFAILURE: No audio was generated.")

if __name__ == "__main__":
    import os
    asyncio.run(test_chat_flow())
