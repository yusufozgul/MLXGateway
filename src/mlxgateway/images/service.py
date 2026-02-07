import base64
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from ..utils.logger import logger
from .schema import ImageEditRequest, ImageGenerationRequest, ImageObject, ResponseFormat


class MFluxImageGenerator:
    def __init__(self, model_version: str):
        self.model_version = model_version
        self.model_lower = model_version.lower()
        self._generator = None
        self._model_class = None
    
    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name by replacing dots with dashes"""
        return name.replace("flux.1", "flux1").replace("flux.2", "flux2")

    def _get_model_class(self):
        """Determine which mflux model class to use based on model_version"""
        normalized = self._normalize_model_name(self.model_lower)
        
        # Z-Image models (recommended - best all-rounder)
        if "z-image-turbo" in normalized or "z-turbo" in normalized:
            from mflux.models.z_image import ZImageTurbo
            return ZImageTurbo
        elif "z-image" in normalized:
            from mflux.models.z_image import ZImage
            return ZImage
        
        # FLUX.2 models (fastest + smallest) - lazy import per model
        if "flux2-klein" in normalized:
            from mflux.models.flux2.variants import Flux2Klein
            return Flux2Klein
        elif "flux2" in normalized:
            # Default FLUX.2 to Klein variant
            logger.info(f"Model {self.model_version} mapped to Flux2Klein")
            from mflux.models.flux2.variants import Flux2Klein
            return Flux2Klein
        
        # FLUX.1 models (legacy)
        elif "flux1-schnell" in normalized or "flux-schnell" in normalized:
            from mflux.models.flux import Flux1Schnell
            return Flux1Schnell
        elif "flux1-dev" in normalized or "flux-dev" in normalized:
            from mflux.models.flux import Flux1Dev
            return Flux1Dev
        
        # Default to Z-Image Turbo (best all-rounder)
        logger.warning(f"Unknown model: {self.model_version}, defaulting to Z-Image Turbo")
        from mflux.models.z_image import ZImageTurbo
        return ZImageTurbo

    def _get_generator(self, params: dict):
        if self._generator is None:
            logger.info(f"Loading model: {self.model_version}")
            
            self._model_class = self._get_model_class()
            
            # Initialize model with quantization if specified
            init_kwargs = {"quantize": params.get("quantize", 8)}
            
            # Handle FLUX.2-klein 9B variant specifically
            normalized = self._normalize_model_name(self.model_lower)
            if "flux2-klein-9b" in normalized:
                from mflux.models.common.config.model_config import ModelConfig
                init_kwargs["model_config"] = ModelConfig.flux2_klein_9b()
                logger.info("Using FLUX.2-klein-9B model config")
            
            # Add optional parameters if provided
            for key in ["model_path", "lora-paths", "lora-scales"]:
                if key in params:
                    init_kwargs[key.replace("-", "_")] = params[key]
            
            self._generator = self._model_class(**init_kwargs)
            logger.info(f"Model loaded: {type(self._generator).__name__}")
        return self._generator

    def generate(self, request: ImageGenerationRequest, output_path: str, **extra):
        width, height = map(int, request.size.split("x"))
        params = {**request.get_extra_params(), **extra}
        seed = params.pop("seed", random.randint(0, 2**32 - 1))
        
        logger.info(f"Generating: {width}x{height}, seed={seed}, steps={params.get('steps', 4)}")
        
        generator = self._get_generator(params)
        
        # Build generation kwargs
        gen_kwargs = {
            "seed": seed,
            "prompt": request.prompt,
            "num_inference_steps": params.pop("steps", 4),
            "height": height,
            "width": width,
        }
        
        # Handle guidance parameter based on model type
        if hasattr(generator, 'generate_image'):
            import inspect
            sig = inspect.signature(generator.generate_image)
            if "guidance" in sig.parameters:
                # FLUX.2 models require guidance=1.0, others can use custom guidance
                normalized = self._normalize_model_name(self.model_lower)
                gen_kwargs["guidance"] = 1.0 if "flux2" in normalized else params.pop("guidance", 4.0)
                if "flux2" in normalized:
                    logger.debug("Using guidance=1.0 for FLUX.2 model")
        
        image = generator.generate_image(**gen_kwargs)
        
        image.save(path=output_path, export_json_metadata=False)
        logger.info(f"Saved: {output_path}")
        return image


class MFluxImageEditor:
    def __init__(self, model_version: str):
        self.model_version = model_version
        self.model_lower = model_version.lower()
        self._editor = None
        self._model_class = None
    
    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name by replacing dots with dashes"""
        return name.replace("flux.1", "flux1").replace("flux.2", "flux2")
    
    def _get_model_class(self):
        """Determine which editing model to use"""
        normalized = self._normalize_model_name(self.model_lower)
        
        # FLUX.2 Edit (recommended)
        if "flux2-klein-edit" in normalized or "flux2" in normalized and "edit" in normalized:
            from mflux.models.flux2.variants import Flux2KleinEdit
            return Flux2KleinEdit
        
        # OpenAI compatibility - map gpt-image to FLUX.2 Klein Edit
        if "gpt-image" in normalized:
            from mflux.models.flux2.variants import Flux2KleinEdit
            return Flux2KleinEdit
        
        # Qwen Edit
        if "qwen-edit" in normalized or "qwen-image-edit" in normalized or "qwen" in normalized:
            from mflux.models.qwen.variants.edit import QwenImageEdit
            return QwenImageEdit
        
        # Kontext (legacy)
        if "kontext" in normalized:
            from mflux.models.flux import Flux1Kontext
            return Flux1Kontext
        
        # Default to FLUX.2 Klein Edit (best all-rounder)
        logger.warning(f"Unknown edit model: {self.model_version}, defaulting to FLUX.2 Klein Edit")
        from mflux.models.flux2.variants import Flux2KleinEdit
        return Flux2KleinEdit
    
    def _get_editor(self, params: dict):
        if self._editor is None:
            logger.info(f"Loading edit model: {self.model_version}")
            
            self._model_class = self._get_model_class()
            
            # Initialize model with quantization if specified
            init_kwargs = {"quantize": params.get("quantize", 8)}
            
            # Handle FLUX.2-klein 9B variant specifically
            normalized = self._normalize_model_name(self.model_lower)
            if "9b" in normalized:
                try:
                    from mflux.models.common.config.model_config import ModelConfig
                    init_kwargs["model_config"] = ModelConfig.flux2_klein_9b()
                    logger.info("Using FLUX.2-klein-9B model config for editing")
                except Exception as e:
                    logger.warning(f"Could not set 9B config: {e}")
            
            # Add optional parameters if provided
            for key in ["model_path", "lora_paths", "lora_scales"]:
                if key in params:
                    init_kwargs[key] = params[key]
            
            self._editor = self._model_class(**init_kwargs)
            logger.info(f"Edit model loaded: {type(self._editor).__name__}")
        return self._editor
    
    def edit(self, request: ImageEditRequest, output_path: str, **extra):
        params = {**request.get_extra_params(), **extra}
        seed = params.pop("seed", random.randint(0, 2**32 - 1))
        
        # Get size from request or auto-detect from first image
        if request.size:
            width, height = map(int, request.size.split("x"))
        else:
            # Auto-detect from first image
            from PIL import Image
            first_img = Image.open(request.image_files[0])
            width, height = first_img.size
            logger.info(f"Auto-detected size: {width}x{height}")
        
        logger.info(f"Editing: {len(request.image_files)} image(s), {width}x{height}, seed={seed}")
        
        editor = self._get_editor(params)
        
        # Build generation kwargs
        gen_kwargs = {
            "seed": seed,
            "prompt": request.prompt,
            "image_paths": request.image_files,
            "num_inference_steps": params.pop("steps", 4),  # Default 4 for FLUX.2
            "width": width,
            "height": height,
        }
        
        # Handle guidance parameter based on model type
        if hasattr(editor, 'generate_image'):
            import inspect
            sig = inspect.signature(editor.generate_image)
            if "guidance" in sig.parameters:
                # FLUX.2 models require guidance=1.0, others can use custom guidance
                normalized = self._normalize_model_name(self.model_lower)
                gen_kwargs["guidance"] = 1.0 if "flux2" in normalized else params.pop("guidance", 2.5)
                if "flux2" in normalized:
                    logger.debug("Using guidance=1.0 for FLUX.2 edit model")
        
        # Add any other parameters that might be relevant
        for key in ["lora_paths", "lora_scales"]:
            if key in params:
                gen_kwargs[key] = params[key]
        
        image = editor.generate_image(**gen_kwargs)
        
        image.save(path=output_path, export_json_metadata=False)
        logger.info(f"Saved edited image: {output_path}")
        return image


class ImagesService:
    def __init__(self):
        self.output_dir = Path(tempfile.gettempdir()) / "mlxgateway" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, MFluxImageGenerator] = {}
        self._edit_cache: Dict[str, MFluxImageEditor] = {}

    def generate_images(self, request: ImageGenerationRequest) -> List[ImageObject]:
        if request.model not in self._cache:
            logger.info(f"Creating generator: {request.model}")
            self._cache[request.model] = MFluxImageGenerator(request.model)
        
        generator = self._cache[request.model]
        images = []

        for i in range(request.n):
            path = self.output_dir / f"{int(time.time())}_{i}.png"
            logger.info(f"Generating image {i+1}/{request.n}")
            generator.generate(request, str(path), low_memory=True)

            if request.response_format == ResponseFormat.B64_JSON:
                b64 = base64.b64encode(path.read_bytes()).decode()
                path.unlink()
                images.append(ImageObject(b64_json=b64, revised_prompt=request.prompt))
            else:
                images.append(ImageObject(url=f"file://{path}", revised_prompt=request.prompt))

        logger.info(f"Generated {len(images)} image(s)")
        return images

    def edit_images(self, request: ImageEditRequest) -> List[ImageObject]:
        if request.model not in self._edit_cache:
            logger.info(f"Creating editor: {request.model}")
            self._edit_cache[request.model] = MFluxImageEditor(request.model)
        
        editor = self._edit_cache[request.model]
        images = []

        for i in range(request.n):
            path = self.output_dir / f"{int(time.time())}_{i}_edit.png"
            logger.info(f"Editing image {i+1}/{request.n}")
            editor.edit(request, str(path), low_memory=True)

            if request.response_format == ResponseFormat.B64_JSON:
                b64 = base64.b64encode(path.read_bytes()).decode()
                path.unlink()
                images.append(ImageObject(b64_json=b64, revised_prompt=request.prompt))
            else:
                images.append(ImageObject(url=f"file://{path}", revised_prompt=request.prompt))

        logger.info(f"Edited {len(images)} image(s)")
        return images
