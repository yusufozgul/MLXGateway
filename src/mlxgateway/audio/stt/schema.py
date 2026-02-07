from enum import Enum
from typing import List, Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"


class TimestampGranularity(str, Enum):
    WORD = "word"
    SEGMENT = "segment"


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: Optional[List[TranscriptionWord]] = None


class STTRequestForm:
    def __init__(
        self,
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        temperature: float = Form(0.0),
        timestamp_granularities: Optional[List[TimestampGranularity]] = Form(None),
    ):
        self.file = file
        self.model = model
        self.language = language
        self.prompt = prompt
        self.response_format = response_format
        self.temperature = temperature
        self.timestamp_granularities = timestamp_granularities
