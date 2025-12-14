import sys
import os
import time
import numpy as np

print("Testing TTS generation...")

try:
    from tts_service import tts_service
    print("Imported tts_service")
except ImportError as e:
    print(f"Failed to import tts_service: {e}")
    sys.exit(1)

print("Initializing TTS...")
try:
    tts_service.initialize_tts()
    print("TTS initialized")
except Exception as e:
    print(f"Failed to initialize TTS: {e}")
    sys.exit(1)

text = "Hello, this is a test of the text to speech system."
voice = "dave"

print(f"Generating audio for: '{text}' with voice '{voice}'")

try:
    ref_codes, ref_text = tts_service.get_cached_reference_data(voice)
    if ref_codes is None:
        print(f"Error: Could not load reference data for {voice}")
        sys.exit(1)
        
    print(f"Reference loaded. Codes shape: {ref_codes.shape}, Text len: {len(ref_text)}")
    
    start_time = time.time()
    audio_chunks, timing = tts_service.generate_audio_with_timing(text, ref_codes, ref_text)
    end_time = time.time()
    
    print(f"Generation took {end_time - start_time:.4f}s")
    print(f"Generated {len(audio_chunks)} chunks")
    print(f"Timing info: {timing}")
    
    if len(audio_chunks) > 0:
        total_samples = sum(len(chunk) for chunk in audio_chunks)
        print(f"Total samples: {total_samples}")
        print(f"Sample rate: 24000 (assumed)")
        print(f"Duration: {total_samples / 24000:.2f}s")
        
        full_audio = np.concatenate(audio_chunks)
        
        # Check for silence
        max_amp = np.max(np.abs(full_audio))
        mean_amp = np.mean(np.abs(full_audio))
        print(f"Max Amplitude: {max_amp}")
        print(f"Mean Amplitude: {mean_amp}")
        
        if max_amp == 0:
            print("ERROR: Generated audio is completely silent!")
        else:
            print("Audio contains signal.")
            
        import soundfile as sf
        sf.write("test_output.wav", full_audio, 24000)
        print("Saved test_output.wav")
    else:
        print("No audio generated!")

except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()
