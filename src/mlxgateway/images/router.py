import time
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse

from ..models.error import ErrorDetail, ErrorResponse
from ..utils.logger import logger
from .schema import ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ResponseFormat
from .service import ImagesService

router = APIRouter(tags=["images"])


@router.post("/images/generations")
@router.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """
    Creates an image given a prompt.
    """
    try:
        logger.info(f"Image generation request: model={request.model}, size={request.size}, n={request.n}")
        service = ImagesService()
        images = service.generate_images(request)
        
        return ImageGenerationResponse(
            created=int(time.time()), 
            data=images
        )

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(ve),
                    type="invalid_request_error",
                    code="invalid_value"
                )
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Image generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while generating images.",
                    type="server_error",
                    code="internal_error"
                )
            ).model_dump()
        )


@router.post("/images/edits")
@router.post("/v1/images/edits")
async def edit_image(
    request: Request,
    prompt: str = Form(...),
    model: Optional[str] = Form(default="flux2-klein-9b-edit"),
    size: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="b64_json"),
    n: Optional[int] = Form(default=1),
    steps: Optional[int] = Form(default=None),
    guidance: Optional[float] = Form(default=None),
    seed: Optional[int] = Form(default=None),
    quantize: Optional[int] = Form(default=8),
    mask: Optional[UploadFile] = File(default=None),
) -> ImageGenerationResponse:
    """
    Creates an edited image given one or more source images and a prompt.
    
    Compatible with OpenAI Images API /v1/images/edits endpoint.
    Supports FLUX.2, Qwen Edit, and Kontext editing models.
    
    Args:
        prompt: Text description of desired edits (max 32000 characters)
        image: One or more images to edit (up to 16 for FLUX.2)
        mask: Optional mask for DALL-E 2 compatibility (not used by mflux models)
        model: Model to use (flux2-klein-9b-edit, qwen-edit, kontext, gpt-image-1.5)
        n: Number of images to generate (1-10)
        size: Output size (e.g. "1024x1024"), auto-detects from input if not provided
        response_format: "url" or "b64_json"
        steps: Number of inference steps (model-specific defaults)
        guidance: Guidance scale (default 2.5 for editing)
        seed: Random seed for reproducibility
        quantize: Quantization level (4 or 8, default 8)
    """
    temp_dir = None
    temp_files = []
    
    # Get form data to extract image files (handles 'image[]' field name)
    form = await request.form()
    
    # Extract images from form (could be 'image' or 'image[]')
    image_files = []
    for key in form.keys():
        if 'image' in key.lower():
            value = form.getlist(key) if key.endswith('[]') else [form.get(key)]
            for item in value:
                if hasattr(item, 'filename'):
                    image_files.append(item)
    
    if not image_files:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="At least one image file is required",
                    type="invalid_request_error",
                    param="image",
                    code="missing_required_parameter"
                )
            ).model_dump()
        )
    
    logger.info(f"Image Edit: model={model}, images={len(image_files)}, n={n}, size={size}, steps={steps}, guidance={guidance}, seed={seed}, quantize={quantize}")
    
    try:
        # Create temporary directory for uploaded images
        temp_dir = Path(tempfile.mkdtemp(prefix="mlxgateway_edit_"))
        
        # Save uploaded images to temporary files
        image_paths = []
        for idx, img_file in enumerate(image_files):
            temp_path = temp_dir / f"input_{idx}_{img_file.filename}"
            with open(temp_path, "wb") as f:
                content = await img_file.read()
                f.write(content)
            image_paths.append(str(temp_path))
            temp_files.append(temp_path)
            logger.info(f"Saved upload {idx+1}/{len(image_files)}: {img_file.filename}")
        
        # Save mask if provided (for DALL-E 2 compatibility)
        mask_path = None
        if mask:
            mask_path = temp_dir / f"mask_{mask.filename}"
            with open(mask_path, "wb") as f:
                content = await mask.read()
                f.write(content)
            temp_files.append(mask_path)
            logger.info(f"Saved mask: {mask.filename}")
        
        # Validate response format
        try:
            resp_format = ResponseFormat(response_format)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Invalid response_format: {response_format}",
                        type="invalid_request_error",
                        param="response_format",
                        code="invalid_value"
                    )
                ).model_dump()
            )
        
        # Build request object with extra parameters
        extra_params = {}
        if steps is not None:
            extra_params["steps"] = steps
        if guidance is not None:
            extra_params["guidance"] = guidance
        if seed is not None:
            extra_params["seed"] = seed
        if quantize is not None:
            extra_params["quantize"] = quantize
        
        request = ImageEditRequest(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            response_format=resp_format,
            image_files=image_paths,
            mask_file=mask_path,
            **extra_params
        )
        
        # Generate edited images
        service = ImagesService()
        images = service.edit_images(request)
        
        return ImageGenerationResponse(
            created=int(time.time()), 
            data=images
        )
    
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(ve),
                    type="invalid_request_error",
                    code="invalid_value"
                )
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Image edit error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while editing images.",
                    type="server_error",
                    code="internal_error"
                )
            ).model_dump()
        )
    finally:
        # Cleanup temporary files
        if temp_files:
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        # Cleanup temporary directory
        if temp_dir and temp_dir.exists():
            try:
                temp_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
