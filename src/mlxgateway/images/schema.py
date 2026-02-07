from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImageSize(str, Enum):
    S256x256 = "256x256"
    S512x512 = "512x512"
    S1024x1024 = "1024x1024"
    S1792x1024 = "1792x1024"
    S1024x1792 = "1024x1792"


class ResponseFormat(str, Enum):
    URL = "url"
    B64_JSON = "b64_json"


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    model: str = "black-forest-labs/FLUX.2-klein-9B"
    n: int = Field(default=1, ge=1, le=10)
    response_format: ResponseFormat = ResponseFormat.B64_JSON
    size: ImageSize = ImageSize.S1024x1024

    model_config = {"extra": "allow"}

    def get_extra_params(self) -> Dict[str, Any]:
        standard = {"prompt", "model", "n", "response_format", "size"}
        return {k: v for k, v in self.model_dump().items() if k not in standard}


class ImageObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]


class ImageEditRequest(BaseModel):
    prompt: str = Field(..., max_length=32000)
    model: str = "flux2-klein-9b-edit"
    n: int = Field(default=1, ge=1, le=10)
    response_format: ResponseFormat = ResponseFormat.B64_JSON
    size: Optional[str] = None  # Auto-detect from input if not provided
    
    # Image handling (populated from form data)
    image_files: List[str] = Field(default_factory=list)  # Temp paths to uploaded images
    mask_file: Optional[str] = None  # Temp path to mask (for DALL-E 2 compat)
    
    # Extended parameters
    model_config = {"extra": "allow"}
    
    def get_extra_params(self) -> Dict[str, Any]:
        standard = {"prompt", "model", "n", "response_format", "size", "image_files", "mask_file"}
        return {k: v for k, v in self.model_dump().items() if k not in standard}
