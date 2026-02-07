from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class TTSRequest(BaseModel):
    model: str = Field(..., description="TTS model to use")
    input: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(
        default="af_sky", 
        description="Voice to use for standard TTS models (e.g., 'af_sky', 'af_bella')"
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Voice description for VoiceDesign models (e.g., 'A cheerful young female voice with high pitch')"
    )
    response_format: Optional[AudioFormat] = Field(default=AudioFormat.WAV)
    speed: Optional[float] = Field(default=1.0)

    class Config:
        extra = "allow"

    def get_extra_params(self) -> Dict[str, Any]:
        standard_fields = {"model", "input", "voice", "instruct", "response_format", "speed"}
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}

    @field_validator("speed")
    def validate_speed(cls, v):
        if v < 0.25 or v > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        return v
