from typing import Any, Dict, Generator, List, Optional
import json
import uuid

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from ..models.loader import MLXModel
from ..utils.logger import logger


class ChatGenerator:
    def __init__(self, model: MLXModel, model_id: str):
        self.mlx_model = model
        self.tokenizer = model.tokenizer
        self.model_id = model_id
        self._max_tokens = model.get_max_tokens()
        
        # Detect thinking/reasoning support
        self.has_thinking = getattr(self.tokenizer, 'has_thinking', False)
        self.think_start_id = getattr(self.tokenizer, 'think_start_id', None)
        self.think_end_id = getattr(self.tokenizer, 'think_end_id', None)
        self.think_end = getattr(self.tokenizer, 'think_end', '</think>')
        
        # Validate thinking support
        if self.has_thinking and not (self.think_start_id and self.think_end_id):
            self.has_thinking = False
        
        # Detect tool calling support
        self.has_tool_calling = getattr(self.tokenizer, 'has_tool_calling', False)
        self.tool_call_start = getattr(self.tokenizer, 'tool_call_start', None)
        self.tool_call_end = getattr(self.tokenizer, 'tool_call_end', None)
        self.tool_parser = getattr(self.tokenizer, 'tool_parser', None)
        
        # Validate tool calling support
        if self.has_tool_calling and not (self.tool_call_start and self.tool_call_end and self.tool_parser):
            self.has_tool_calling = False
        
        # Log tool calling capabilities
        if self.has_tool_calling:
            logger.info(
                f"Model {model_id} has tool calling support. "
                f"Start: '{self.tool_call_start}', End: '{self.tool_call_end}'"
            )
        else:
            logger.debug(f"Model {model_id} does not support tool calling")

    def _prepare_prompt(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
        if tools and not self.has_tool_calling:
            logger.warning(
                f"Model {self.model_id} received {len(tools)} tools but does not support tool calling. "
                "Tools will be ignored."
            )
            tools = None
        
        if tools:
            logger.debug(f"Preparing prompt with {len(tools)} tools for model {self.model_id}")
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def _check_initial_reasoning_state(self, prompt_tokens: List[int]) -> bool:
        if not self.has_thinking:
            return False
        
        for i in range(len(prompt_tokens) - 1, -1, -1):
            if prompt_tokens[i] == self.think_end_id:
                return False
            elif prompt_tokens[i] == self.think_start_id:
                return True
        return False
    
    def _process_token(self, response, in_reasoning: bool, in_tool_call: bool, tool_text: str) -> tuple:
        # Handle reasoning mode
        if self.has_thinking and in_reasoning:
            if response.text == self.think_end:
                return "", None, tool_text, False, in_tool_call
            return "", response.text, tool_text, True, in_tool_call
        
        # Check for thinking start
        if self.has_thinking and response.token == self.think_start_id:
            return "", None, tool_text, True, in_tool_call
        
        # Handle tool calling
        if self.has_tool_calling:
            if response.text == self.tool_call_start:
                logger.debug(f"Tool call started. Text: '{response.text}'")
                return "", None, tool_text, in_reasoning, True
            elif in_tool_call:
                if response.text == self.tool_call_end:
                    logger.debug(f"Tool call ended. Full text: {tool_text[:200]}...")
                    return "", None, tool_text, in_reasoning, False
                else:
                    return "", None, tool_text + response.text, in_reasoning, True
        
        return response.text, None, tool_text, False, in_tool_call
    
    def _format_tool_call(self, tool_text: str, tools: Optional[List[Dict]], tool_idx: int = 0, for_streaming: bool = False) -> Dict:
        if not self.has_tool_calling or not tool_text:
            return None
        
        try:
            logger.debug(f"Parsing tool call (streaming={for_streaming}): {tool_text[:200]}...")
            parsed = self.tool_parser(tool_text, tools)
            
            if isinstance(parsed, list):
                logger.debug(f"Parsed {len(parsed)} tool calls")
                result = []
                for idx, tc in enumerate(parsed):
                    tool_call = {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                        }
                    }
                    if for_streaming:
                        tool_call["index"] = tool_idx + idx
                    result.append(tool_call)
                return result
            else:
                logger.debug(f"Single tool call: {parsed.get('name')}")
                tool_call = {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": parsed.get("name"),
                        "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False)
                    }
                }
                if for_streaming:
                    tool_call["index"] = tool_idx
                return [tool_call]
        except Exception as e:
            logger.error(f"Failed to parse tool call: {e}. Tool text: {tool_text[:200]}")
            return None
    
    def _get_stream_params(self, prompt: str, max_tokens: Optional[int], temperature: float, top_p: float, progress_callback=None) -> dict:
        params = {
            "model": self.mlx_model.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens or self._max_tokens,
            "sampler": make_sampler(temp=temperature, top_p=top_p),
        }
        
        # Add prompt_cache if available
        if self.mlx_model.prompt_cache is not None:
            params["prompt_cache"] = self.mlx_model.prompt_cache
        if progress_callback is not None:
            params["prompt_progress_callback"] = progress_callback
            
        return params

    def generate(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_cache: bool = True,
        tools: Optional[List[Dict]] = None,
        progress_callback=None,
    ) -> Dict:
        prompt = self._prepare_prompt(messages, tools)
        prompt_tokens = self.tokenizer.encode(prompt)
        in_reasoning = self._check_initial_reasoning_state(prompt_tokens)
        
        content_text = reasoning_text = ""
        in_tool_call = False
        tool_text = ""
        tool_calls = []
        prompt_tokens_count = completion_tokens_count = 0
        
        for response in stream_generate(**self._get_stream_params(prompt, max_tokens, temperature, top_p, progress_callback)):
            if response.finish_reason:
                prompt_tokens_count = response.prompt_tokens
                completion_tokens_count = response.generation_tokens
                break
            
            content, reasoning, tool_text, in_reasoning, new_tool_state = self._process_token(
                response, in_reasoning, in_tool_call, tool_text
            )
            
            # If tool call just ended, parse it
            if in_tool_call and not new_tool_state and tool_text:
                parsed_tool = self._format_tool_call(tool_text, tools)
                if parsed_tool:
                    tool_calls.extend(parsed_tool if isinstance(parsed_tool, list) else [parsed_tool])
                tool_text = ""
            
            in_tool_call = new_tool_state
            content_text += content
            if reasoning:
                reasoning_text += reasoning
        
        self.mlx_model.save_cache()
        return {
            "text": content_text,
            "reasoning": reasoning_text or None,
            "tool_calls": tool_calls if tool_calls else None,
            "prompt_tokens": prompt_tokens_count,
            "completion_tokens": completion_tokens_count,
        }

    def generate_stream(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_cache: bool = True,
        tools: Optional[List[Dict]] = None,
        progress_callback=None,
    ) -> Generator[Dict, None, None]:
        prompt = self._prepare_prompt(messages, tools)
        prompt_tokens = self.tokenizer.encode(prompt)
        in_reasoning = self._check_initial_reasoning_state(prompt_tokens)
        
        in_tool_call = False
        tool_text = ""
        tool_calls = []
        tool_idx = 0  # Track index for streaming tool calls
        made_tool_call = False  # Track if any tool calls were made
        
        for response in stream_generate(**self._get_stream_params(prompt, max_tokens, temperature, top_p, progress_callback)):
            content, reasoning, tool_text, in_reasoning, new_tool_state = self._process_token(
                response, in_reasoning, in_tool_call, tool_text
            )
            
            # If tool call just ended, parse it
            if in_tool_call and not new_tool_state and tool_text:
                parsed_tool = self._format_tool_call(tool_text, tools, tool_idx, for_streaming=True)
                if parsed_tool:
                    tool_idx += len(parsed_tool)
                    tool_calls.extend(parsed_tool if isinstance(parsed_tool, list) else [parsed_tool])
                    made_tool_call = True
                    yield {
                        "text": "",
                        "reasoning": None,
                        "tool_calls": parsed_tool,
                        "finish_reason": None,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.generation_tokens,
                    }
                tool_text = ""
            
            in_tool_call = new_tool_state
            
            # Determine finish reason
            finish_reason = response.finish_reason
            if response.finish_reason and made_tool_call:
                finish_reason = "tool_calls"
            
            if content or reasoning or finish_reason:
                yield {
                    "text": content,
                    "reasoning": reasoning,
                    "tool_calls": None,
                    "finish_reason": finish_reason,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.generation_tokens,
                }
            
            if response.finish_reason:
                self.mlx_model.save_cache()
                break
