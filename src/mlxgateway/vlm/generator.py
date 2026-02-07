"""VLM Generator for handling multimodal chat completions."""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template

from ..utils.logger import logger
from .loader import VLMModel


class VLMGenerator:
    """Generator for Vision Language Models with multimodal support."""
    
    def __init__(self, model: VLMModel):
        self.model = model.model
        self.processor = model.processor
        self.config = model.config
        self.model_id = model.model_id
        self._max_tokens = model.get_max_tokens()
    
    @staticmethod
    def _is_url(path: str) -> bool:
        """Check if a path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def _download_from_url(url: str) -> Optional[str]:
        """Download file from URL and return local path."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            suffix = Path(urlparse(url).path).suffix or '.tmp'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to download from URL {url}: {e}")
            return None
    
    def _extract_media_from_messages(self, messages: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract images, audio, and video from messages.
        
        Args:
            messages: List of chat messages with content
            
        Returns:
            Dict with 'images', 'audio', 'video' lists
        """
        media = {
            "images": [],
            "audio": [],
            "video": [],
        }
        
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            
            for item in content:
                if not isinstance(item, dict):
                    continue
                
                item_type = item.get("type", "")
                
                # Handle images
                if item_type == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url:
                        # Download if URL, otherwise assume local path
                        if self._is_url(url):
                            local_path = self._download_from_url(url)
                            if local_path:
                                media["images"].append(local_path)
                        else:
                            media["images"].append(url)
                
                # Handle audio
                elif item_type == "input_audio":
                    audio_path = item.get("input_audio", "")
                    if audio_path:
                        if self._is_url(audio_path):
                            local_path = self._download_from_url(audio_path)
                            if local_path:
                                media["audio"].append(local_path)
                        else:
                            media["audio"].append(audio_path)
                
                # Handle video
                elif item_type == "input_video":
                    video_path = item.get("input_video", "")
                    if video_path:
                        if self._is_url(video_path):
                            local_path = self._download_from_url(video_path)
                            if local_path:
                                media["video"].append(local_path)
                        else:
                            media["video"].append(video_path)
        
        return media
    
    def _build_text_prompt(self, messages: List[Dict]) -> str:
        """
        Build text prompt from messages, extracting only text content.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Text prompt string
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Extract text from content items
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    prompt_parts.append(f"{role}: {' '.join(text_parts)}")
        
        return "\n".join(prompt_parts)
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        tokenizer = getattr(self.processor, 'tokenizer', None) if self.processor else None
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        return len(text) // 4  # Rough estimation
    
    def generate(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict:
        """
        Generate completion for multimodal messages.
        
        Args:
            messages: List of chat messages with multimodal content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with 'text', 'prompt_tokens', 'completion_tokens'
        """
        # Extract media from messages
        media = self._extract_media_from_messages(messages)
        
        # Build text prompt
        prompt = self._build_text_prompt(messages)
        
        # Apply chat template with media info
        num_images = len(media["images"])
        num_audios = len(media["audio"])
        num_videos = len(media["video"])
        
        # Validate processor and config
        if self.processor is None:
            raise ValueError("Processor is not available for this model")
        if self.config is None:
            raise ValueError("Model config is not available")
        
        try:
            formatted_prompt = apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=num_images,
                num_audios=num_audios,
            )
        except TypeError as e:
            logger.error(f"Chat template error: {e}", exc_info=True)
            raise ValueError(f"This model may not support the requested modalities: {str(e)}")
        
        # Prepare generation kwargs
        gen_kwargs = {
            "model": self.model,
            "processor": self.processor,
            "prompt": formatted_prompt,
            "max_tokens": max_tokens or self._max_tokens,
            "verbose": False,
        }
        
        # Add media if present
        if media["images"]:
            gen_kwargs["image"] = media["images"]
        if media["audio"]:
            gen_kwargs["audio"] = media["audio"]
        if media["video"]:
            gen_kwargs["video"] = media["video"]
        
        # Generate
        try:
            output = generate(**gen_kwargs)
            
            # Count tokens
            prompt_tokens = self._count_tokens(formatted_prompt)
            completion_tokens = self._count_tokens(output)
            
            return {
                "text": output,
                "reasoning": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except Exception as e:
            logger.error(f"VLM generation failed: {e}", exc_info=True)
            raise
