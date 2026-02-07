"""Base model loader for MLXGateway - unified interface for LLM and VLM models."""

import gc
import os
from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

from ..utils.logger import logger


class BaseMLXModel:
    """Base class for MLX models (both LLM and VLM)."""
    
    def __init__(
        self,
        model,
        tokenizer_or_processor,
        model_id: str,
        is_vlm: bool = False,
        use_cache: bool = False,
        max_kv_size: Optional[int] = None,
    ):
        self.model = model
        self.model_id = model_id
        self.is_vlm = is_vlm
        self._is_loaded = True
        self.config = getattr(model, 'config', None)
        
        # Set tokenizer/processor based on model type
        self.processor = tokenizer_or_processor if is_vlm else None
        self.tokenizer = None if is_vlm else tokenizer_or_processor
        
        # Prompt cache (LLM only)
        self.use_cache = use_cache and not is_vlm
        self.prompt_cache: Optional[List[Any]] = None
        self.cache_file = None
        
        if self.use_cache:
            self.cache_file = os.path.join(
                os.path.expanduser("~/.cache/mlxgateway"),
                f"{model_id.replace('/', '_')}.safetensors"
            )
            
            # Load existing cache or create new one
            if os.path.exists(self.cache_file):
                try:
                    self.prompt_cache = load_prompt_cache(self.cache_file)
                    logger.info(f"Loaded prompt cache from {self.cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
            
            if self.prompt_cache is None:
                self.prompt_cache = make_prompt_cache(model, max_kv_size)
                logger.info(f"Created new prompt cache for {model_id}")
    
    def get_max_tokens(self) -> int:
        """Get maximum token length for the model (defaults to 4096)."""
        config = self.config or getattr(self.model, "config", None)
        if config:
            for attr in ["max_position_embeddings", "n_positions", "max_sequence_length", "model_max_length"]:
                if val := getattr(config, attr, None):
                    return int(val)
        return 4096
    
    def save_cache(self) -> None:
        """Save prompt cache to disk (LLM only)."""
        if not (self.use_cache and self.prompt_cache) or self.is_vlm:
            return
        
        # Check if cache is empty
        if hasattr(self.prompt_cache[0], 'keys') and self.prompt_cache[0].keys.size == 0:
            return
        
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            save_prompt_cache(self.cache_file, self.prompt_cache)
            logger.debug(f"Saved prompt cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._is_loaded:
            return
        
        logger.info(f"Unloading {'VLM' if self.is_vlm else 'LLM'}: {self.model_id}")
        
        if not self.is_vlm:
            self.save_cache()
        
        self.model = self.tokenizer = self.processor = self.prompt_cache = None
        self._is_loaded = False
        mx.clear_cache()
        gc.collect()
