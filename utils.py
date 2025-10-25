import io
import struct
from pydub import AudioSegment


def create_wav_header(sample_rate: int, num_channels: int, bits_per_sample: int, data_size: int) -> bytes:
    header = b'RIFF'
    header += struct.pack('<I', 36 + data_size)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)
    header += struct.pack('<H', 1)
    header += struct.pack('<H', num_channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8)
    header += struct.pack('<H', num_channels * bits_per_sample // 8)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    return header


def convert_audio_format(audio_data: bytes, format: str) -> bytes:
    try:
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        
        if format == "mp3":
            output_data = io.BytesIO()
            audio_segment.export(output_data, format="mp3", bitrate="128k")
            return output_data.getvalue()
        elif format == "opus":
            output_data = io.BytesIO()
            audio_segment.export(output_data, format="opus", bitrate="128k")
            return output_data.getvalue()
        elif format == "aac":
            output_data = io.BytesIO()
            audio_segment.export(output_data, format="aac", bitrate="128k")
            return output_data.getvalue()
        elif format == "flac":
            output_data = io.BytesIO()
            audio_segment.export(output_data, format="flac")
            return output_data.getvalue()
        else:
            return audio_data
            
    except Exception as e:
        print(f"Error converting audio format: {e}")
        return audio_data


def get_media_type_and_filename(format: str) -> tuple[str, str]:
    format_mapping = {
        "mp3": ("audio/mpeg", "speech.mp3"),
        "opus": ("audio/opus", "speech.opus"),
        "aac": ("audio/aac", "speech.aac"),
        "flac": ("audio/flac", "speech.flac"),
        "pcm": ("audio/pcm", "speech.pcm"),
        "wav": ("audio/wav", "speech.wav")
    }
    return format_mapping.get(format, ("audio/wav", "speech.wav"))
