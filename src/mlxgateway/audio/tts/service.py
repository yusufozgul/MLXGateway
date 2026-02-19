from pathlib import Path

from mlx_audio.tts import load_model
from mlx_audio.tts.generate import generate_audio

from ...utils.logger import logger
from .schema import TTSRequest


class TTSService:
    def __init__(self, model: str):
        self.model = model
        self.sample_audio_path = Path("sample.wav")

    async def generate_speech(self, request: TTSRequest) -> bytes:
        try:
            logger.info(f"TTS request - model: {request.model}, voice: {request.voice}")
            logger.info(f"Loading model: {self.model}")
            model = load_model(self.model)
            instruct = (request.instruct or request.voice or "A clear neutral voice.").strip()
            voice = (request.voice or "af_sky").strip()
            params = {
                "text": request.input,
                "model": model,
                "speed": request.speed,
                "file_prefix": str(self.sample_audio_path).rsplit(".", 1)[0],
                "audio_format": request.response_format.value,
                "sample_rate": 12000,
                "join_audio": True,
                "verbose": True,
                "instruct": instruct,
                "voice": voice,
            }
            params.update(request.get_extra_params() or {})
            logger.info(f"Generating audio - instruct: {instruct!r}, voice: {voice!r}")
            generate_audio(**params)

            # Read and return audio
            with open(self.sample_audio_path, "rb") as f:
                audio_content = f.read()
            
            self.sample_audio_path.unlink(missing_ok=True)
            logger.info(f"Audio generated: {len(audio_content)} bytes")
            
            return audio_content
            
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            self.sample_audio_path.unlink(missing_ok=True)
            raise Exception(f"Error generating audio: {str(e)}")
