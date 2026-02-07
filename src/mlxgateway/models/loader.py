from typing import Optional

import mlx_lm

from ..utils.logger import logger
from .base_loader import BaseMLXModel


class MLXModel(BaseMLXModel):
    """Wrapper for Language Models loaded via mlx-lm."""
    
    def __init__(self, model, tokenizer, model_id: str, use_cache: bool = False, max_kv_size: Optional[int] = None):
        super().__init__(model, tokenizer, model_id, is_vlm=False, use_cache=use_cache, max_kv_size=max_kv_size)

    @classmethod
    def load(cls, model_id: str, adapter_path: Optional[str] = None, use_cache: bool = False, max_kv_size: Optional[int] = None) -> "MLXModel":
        """Load a Language Model using mlx-lm."""
        try:
            logger.info(f"Loading LLM: {model_id}")
            model, tokenizer = mlx_lm.load(
                model_id,
                adapter_path=adapter_path,
                tokenizer_config={"trust_remote_code": True},
                model_config={"trust_remote_code": True},
            )
            logger.info(f"LLM loaded successfully: {model_id}")
            return cls(model, tokenizer, model_id, use_cache, max_kv_size)
        except Exception as e:
            logger.error(f"LLM load failed: {model_id} - {e}")
            raise
