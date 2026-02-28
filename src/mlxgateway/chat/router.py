import json
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from ..models.cache import get_model_cache
from ..models.error import ErrorDetail, ErrorResponse
from ..utils.logger import logger
from ..vlm.utils import detect_multimodal_content
from .generator import ChatGenerator
from .schema import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ToolCall,
)

router = APIRouter(prefix="/v1", tags=["chat"])


def _has_multimodal_content(messages) -> bool:
    """Check if messages contain multimodal content (images, audio, video - NOT text)."""
    for msg in messages:
        detection = detect_multimodal_content(msg.content)
        if detection.get("has_images") or detection.get("has_audio") or detection.get("has_video"):
            return True
    return False


def _create_error_response(status_code: int, message: str, error_type: str, code: str, param: str = None):
    """Create a standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=message,
                type=error_type,
                code=code,
                param=param
            )
        ).model_dump()
    )


def _parse_tool_calls(tool_calls_data) -> Optional[List[ToolCall]]:
    """Parse tool calls from generator response."""
    if not tool_calls_data:
        return None
    
    logger.debug(f"Processing {len(tool_calls_data)} tool calls")
    return [
        ToolCall(
            id=tc["id"],
            type=tc["type"],
            function={"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
            index=tc.get("index")
        )
        for tc in tool_calls_data
    ]


def _create_chunk(completion_id: str, created: int, model: str, content: str = "", 
                  reasoning: str = None, tool_calls: List[ToolCall] = None, 
                  finish_reason: str = None) -> str:
    """Create a streaming chunk in SSE format."""
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(
            index=0,
            delta=ChatMessage(
                role="assistant",
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
        )],
    )
    return f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        has_multimodal = _has_multimodal_content(request.messages)
        extra_params = request.get_extra_params()
        
        def progress_callback(processed: int, total: int):
            progress_pct = (processed / total * 100) if total > 0 else 0
            logger.debug(f"Prompt processing progress: {processed}/{total} ({progress_pct:.1f}%)")
        
        # Get appropriate generator
        try:
            if has_multimodal:
                generator = get_model_cache().get_vlm_generator(
                    request.model, extra_params.get("adapter_path")
                )
            else:
                use_cache = request.enable_cache if request.enable_cache is not None else True
                generator = get_model_cache().get_generator(
                    request.model, 
                    extra_params.get("adapter_path"),
                    use_cache=use_cache,
                    max_kv_size=request.max_kv_size,
                )
        except Exception as e:
            return _create_error_response(400, str(e), "invalid_request_error", "model_not_found", "model")
        
        # Prepare messages and tools
        messages = [
            {"role": msg.role, "content": msg.content if has_multimodal else msg.get_text_content()}
            for msg in request.messages
        ]
        
        tools = [tool.model_dump() for tool in request.tools] if request.tools else None
        if tools:
            logger.debug(f"Request contains {len(tools)} tools: {[t['function']['name'] for t in tools]}")
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        
        # Build generation kwargs
        gen_kwargs = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
        }
        
        if not has_multimodal:
            if tools:
                gen_kwargs["tools"] = tools
            gen_kwargs["use_cache"] = request.enable_cache if request.enable_cache is not None else True
            gen_kwargs["progress_callback"] = progress_callback
        
        # Non-streaming response
        if not request.stream:
            result = generator.generate(**gen_kwargs)
            tool_calls = _parse_tool_calls(result.get("tool_calls"))
            
            return JSONResponse(content=ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result["text"],
                        reasoning=result.get("reasoning"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                ),
            ).model_dump(exclude_none=True))
        
        # Streaming response
        async def event_generator():
            for response in generator.generate_stream(**gen_kwargs):
                yield _create_chunk(
                    completion_id, created, request.model,
                    response['text'], response.get('reasoning'),
                    _parse_tool_calls(response.get("tool_calls")),
                    response['finish_reason']
                )
                if response['finish_reason']:
                    break
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    
    except ValueError as e:
        return _create_error_response(400, str(e), "invalid_request_error", "invalid_value")
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {str(e)}", exc_info=True)
        return _create_error_response(
            500, 
            "An unexpected error occurred while processing your request.", 
            "server_error", 
            "internal_error"
        )
