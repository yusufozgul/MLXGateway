from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall
    index: Optional[int] = None  # Required for streaming


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    
    def get_text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = [
                item.get('text', '')
                for item in self.content
                if isinstance(item, dict) and item.get('type') == 'text'
            ]
            return ' '.join(text_parts) if text_parts else ''
        return ''
    
    def has_multimodal_content(self) -> bool:
        if not isinstance(self.content, list):
            return False
        
        multimodal_types = {'image_url', 'input_image', 'input_audio', 'audio_url', 'input_video', 'video_url'}
        return any(
            isinstance(item, dict) and item.get('type') in multimodal_types 
            for item in self.content
        )


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, int]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class Tool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    
    enable_cache: Optional[bool] = Field(None, description="Enable prompt caching")
    max_kv_size: Optional[int] = Field(None, description="Maximum KV cache size", ge=0)
    
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Penalize tokens based on frequency")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Penalize tokens based on presence")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")

    class Config:
        extra = "allow"

    def get_extra_params(self) -> Dict[str, Any]:
        standard_fields = {
            "model", "messages", "temperature", "top_p", 
            "max_tokens", "stream", "enable_cache", "max_kv_size",
            "frequency_penalty", "presence_penalty", "stop", "tools"
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}
