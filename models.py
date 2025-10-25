from pydantic import BaseModel
from typing import Optional, Literal


class TextRequest(BaseModel):
    text: str
    ref_codes_path: Optional[str] = "./neuttsair/samples/dave.pt"
    ref_text: Optional[str] = "./neuttsair/samples/dave.txt"
    backbone: Optional[str] = "neuphonic/neutts-air-q4-gguf"


class OpenAISpeechRequest(BaseModel):
    model: str = "gpt-4o-mini-tts"
    input: str
    voice: Literal["coral", "dave"] = "coral"
    instructions: Optional[str] = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
