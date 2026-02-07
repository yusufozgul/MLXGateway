"""VLM Model loader for MLXGateway."""

from typing import Optional

from mlx_vlm import load
from mlx_vlm.utils import load_config

from ..utils.logger import logger
from ..models.base_loader import BaseMLXModel


class VLMModel(BaseMLXModel):
    """Wrapper for Vision Language Models loaded via mlx-vlm."""
    
    def __init__(self, model, processor, model_id: str):
        super().__init__(model, processor, model_id, is_vlm=True)
    
    @classmethod
    def load(cls, model_id: str, adapter_path: Optional[str] = None) -> "VLMModel":
        """Load a VLM model using mlx-vlm."""
        try:
            logger.info(f"Loading VLM: {model_id}")
            model, processor = load(model_id, adapter_path=adapter_path)
            
            if model is None or processor is None:
                raise ValueError("Model or processor is None after loading")
            
            logger.info(f"VLM loaded successfully: {model_id}")
            return cls(model, processor, model_id)
            
        except FileNotFoundError as e:
            raise ValueError(f"Model '{model_id}' not found") from e
        
        except (ImportError, ModuleNotFoundError) as e:
            # Try to get model type for better error message
            try:
                config = load_config(model_id)
                model_type = config.get("model_type", "unknown")
                raise ValueError(f"Model type '{model_type}' is not supported by mlx-vlm") from e
            except Exception:
                raise ValueError(f"Model '{model_id}' is not supported by mlx-vlm") from e
        
        except ValueError:
            raise
        
        except Exception as e:
            logger.error(f"VLM load failed: {model_id} - {e}", exc_info=True)
            raise ValueError(f"Failed to load VLM model '{model_id}': {str(e)}") from e
