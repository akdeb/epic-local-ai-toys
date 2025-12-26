import io
import struct
from typing import Callable, Optional
import numpy as np

STT = "mlx-community/whisper-large-v3-turbo"
LLM = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
TTS = "mlx-community/chatterbox-turbo-4bit"

# Audio constants matching ESP32 expectations
OPUS_SAMPLE_RATE = 24000  # 24kHz for TTS output
OPUS_CHANNELS = 1  # Mono
OPUS_FRAME_DURATION_MS = 120  # Frame duration in ms (same as Deno relay)
OPUS_BYTES_PER_SAMPLE = 2  # 16-bit PCM
# Frame size in SAMPLES
OPUS_FRAME_SAMPLES = OPUS_SAMPLE_RATE * OPUS_FRAME_DURATION_MS // 1000  # 2880 samples
# Frame size in BYTES for buffering
OPUS_FRAME_SIZE = OPUS_FRAME_SAMPLES * OPUS_CHANNELS * OPUS_BYTES_PER_SAMPLE  # 5760 bytes


class OpusPacketizer:
    """
    Opus packetizer for streaming PCM audio to ESP32.
    Matches the Deno relay implementation: 24kHz mono, 120ms frames.
    Uses PyAV (bundled FFmpeg) for encoding - no system dependencies needed.
    """

    def __init__(self, send_packet: Callable[[bytes], None], sample_rate: int = OPUS_SAMPLE_RATE):
        self.send_packet = send_packet
        self.sample_rate = sample_rate
        self.pending = bytearray()
        self.closed = False
        self._codec = None
        self._codec_ctx = None

    def _get_encoder(self):
        if self._codec_ctx is None:
            import av
            self._codec = av.Codec('libopus', 'w')
            self._codec_ctx = av.CodecContext.create(self._codec)
            self._codec_ctx.sample_rate = self.sample_rate
            self._codec_ctx.channels = OPUS_CHANNELS
            self._codec_ctx.format = av.AudioFormat('s16')
            self._codec_ctx.layout = 'mono'
            # Set frame size to match our expected duration
            self._codec_ctx.frame_size = OPUS_FRAME_SAMPLES
            self._codec_ctx.open()
        return self._codec_ctx

    def push(self, pcm: bytes) -> None:
        """Add PCM bytes to the buffer and encode complete frames."""
        if self.closed or not pcm:
            return

        self.pending.extend(pcm)

        import av
        encoder = self._get_encoder()

        while len(self.pending) >= OPUS_FRAME_SIZE:
            frame_bytes = bytes(self.pending[:OPUS_FRAME_SIZE])
            self.pending = self.pending[OPUS_FRAME_SIZE:]
            
            try:
                # Convert bytes to numpy array (int16)
                samples = np.frombuffer(frame_bytes, dtype=np.int16)
                
                # Create an AudioFrame
                frame = av.AudioFrame.from_ndarray(
                    samples.reshape(1, -1),  # (channels, samples)
                    format='s16',
                    layout='mono'
                )
                frame.sample_rate = self.sample_rate
                frame.pts = None
                
                # Encode
                for packet in encoder.encode(frame):
                    if packet.size > 0:
                        self.send_packet(bytes(packet))
            except Exception as e:
                print(f"Opus encode failed: {e}")

    def flush(self, pad_final_frame: bool = False) -> None:
        """Flush remaining audio. If pad_final_frame, pad and encode the last partial frame."""
        if self.closed:
            return

        import av
        
        if self.pending and pad_final_frame:
            # Pad the final frame with silence
            padded = bytearray(OPUS_FRAME_SIZE)
            padded[: len(self.pending)] = self.pending
            self.pending.clear()

            try:
                encoder = self._get_encoder()
                samples = np.frombuffer(bytes(padded), dtype=np.int16)
                frame = av.AudioFrame.from_ndarray(
                    samples.reshape(1, -1),
                    format='s16',
                    layout='mono'
                )
                frame.sample_rate = self.sample_rate
                frame.pts = None
                
                for packet in encoder.encode(frame):
                    if packet.size > 0:
                        self.send_packet(bytes(packet))
            except Exception as e:
                print(f"Opus encode failed: {e}")
        else:
            self.pending.clear()

        # Flush the encoder
        if self._codec_ctx:
            try:
                for packet in self._codec_ctx.encode(None):
                    if packet.size > 0:
                        self.send_packet(bytes(packet))
            except Exception:
                pass

    def reset(self) -> None:
        """Clear pending buffer without encoding."""
        self.pending.clear()

    def close(self) -> None:
        """Mark the packetizer as closed."""
        self.closed = True
        self.pending.clear()
        if self._codec_ctx:
            try:
                self._codec_ctx.close()
            except Exception:
                pass
            self._codec_ctx = None

    @property
    def buffered_bytes(self) -> int:
        return len(self.pending)


def create_opus_packetizer(send_packet: Callable[[bytes], None]) -> OpusPacketizer:
    """Factory function matching the Deno API."""
    return OpusPacketizer(send_packet)


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
        from pydub import AudioSegment
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
