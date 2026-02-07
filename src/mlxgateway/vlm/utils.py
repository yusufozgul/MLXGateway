"""Utility functions for VLM content detection and error formatting."""

from typing import List

# Supported modalities content type mappings
MODEL_MODALITIES = {
    "image": ["image_url", "input_image"],
    "audio": ["input_audio", "audio_url"],
    "video": ["input_video", "video_url"],
}


def detect_multimodal_content(content) -> dict:
    """
    Detect multimodal content types in a message content.
    
    Args:
        content: Message content (string or list of content items)
        
    Returns:
        Dict with detected modalities: {"has_images": bool, "has_audio": bool, "has_video": bool}
    """
    result = {
        "has_images": False,
        "has_audio": False,
        "has_video": False,
        "has_text": False,
    }
    
    if isinstance(content, str):
        result["has_text"] = True
        return result
    
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            
            content_type = item.get("type", "")
            
            if content_type == "text":
                result["has_text"] = True
            elif content_type in MODEL_MODALITIES["image"]:
                result["has_images"] = True
            elif content_type in MODEL_MODALITIES["audio"]:
                result["has_audio"] = True
            elif content_type in MODEL_MODALITIES["video"]:
                result["has_video"] = True
    
    return result


