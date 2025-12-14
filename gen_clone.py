import numpy as np
import sys
import os
import argparse

sys.path.append(os.getcwd())
from neuttsair.neuttsair.neutts import NeuTTSAir

def main():
    parser = argparse.ArgumentParser(description="Generate .npy reference codes for a voice")
    parser.add_argument("voice", help="Name of the voice (must match filename in neuttsair/samples/<voice>.wav)")
    args = parser.parse_args()

    voice = args.voice
    wav_path = f"neuttsair/samples/{voice}.wav"
    npy_path = f"neuttsair/samples/{voice}.npy"
    
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
        return

    print(f"Initializing TTS to encode reference for '{voice}'...")
    try:
        tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air-q4-gguf",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
        
        print(f"Encoding {wav_path}...")
        ref_codes = tts.encode_reference(wav_path)
        
        print(f"Saving to {npy_path}...")
        np.save(npy_path, ref_codes)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
