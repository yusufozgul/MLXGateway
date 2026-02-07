from typing import List, Literal, NamedTuple, Optional


StopStringProcessorStatus = Literal["full_stop", "partial_match", "no_match", "multi_byte"]
REPLACEMENT_CHAR = "\ufffd"


class StopStringProcessorResult(NamedTuple):
    status: StopStringProcessorStatus
    stop_string: Optional[str] = None
    stop_tokens: Optional[List[int]] = None


class StopStringProcessor:
    def __init__(self, stop_strings: List[str], tokenizer):
        if not stop_strings:
            raise ValueError("Must provide at least one stop string")
        
        if not all(isinstance(s, str) for s in stop_strings):
            raise TypeError("All stop strings must be strings")
        
        if any(not s for s in stop_strings):
            raise ValueError("Stop strings cannot be empty")
        
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.token_id_buffer: List[int] = []
    
    def process_token(self, token: int) -> StopStringProcessorResult:
        if not self.stop_strings:
            return StopStringProcessorResult(status="no_match", stop_string=None, stop_tokens=None)
        
        self.token_id_buffer.append(token)
        decoded = self.tokenizer.decode(self.token_id_buffer)
        result = self._check_stopping_criteria(decoded)
        
        if result.status == "no_match":
            self.token_id_buffer = []
            return StopStringProcessorResult(status="no_match", stop_string=None, stop_tokens=None)
        
        elif result.status == "partial_match":
            return StopStringProcessorResult(status="partial_match", stop_string=None, stop_tokens=None)
        
        elif result.status == "multi_byte":
            return StopStringProcessorResult(status="multi_byte", stop_string=None, stop_tokens=None)
        
        elif result.status == "full_stop":
            return StopStringProcessorResult(
                status="full_stop",
                stop_string=result.stop_string,
                stop_tokens=self.token_id_buffer.copy()
            )
        
        raise ValueError(f"Unknown status: {result.status}")
    
    class _InternalResult(NamedTuple):
        status: StopStringProcessorStatus
        stop_string: Optional[str] = None
    
    def _check_stopping_criteria(self, string: str) -> _InternalResult:
        if len(string) == 0 or string[-1] == REPLACEMENT_CHAR:
            return self._InternalResult(status="multi_byte")
        
        for stop_string in self.stop_strings:
            if stop_string in string:
                return self._InternalResult(status="full_stop", stop_string=stop_string)
        
        for stop_string in self.stop_strings:
            for i in range(1, min(len(string), len(stop_string)) + 1):
                if string[-i:] == stop_string[:i]:
                    return self._InternalResult(status="partial_match")
        
        return self._InternalResult(status="no_match")
    
    def reset(self):
        self.token_id_buffer = []
