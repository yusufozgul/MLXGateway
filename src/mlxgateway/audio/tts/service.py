from pathlib import Path

from mlx_audio.tts import load_model
from mlx_audio.tts.generate import generate_audio

from ...utils.logger import logger
from .schema import TTSRequest


class TTSService:
    def __init__(self, model: str):
        self.model = model
        self.sample_audio_path = Path("sample.wav")

    def _get_voice_config(self, voice: str, instruct: str, is_voice_design: bool) -> tuple[str, str, str]:
        """Returns (voice_param, param_type, lang_code)"""
        if instruct:
            lang_code = "tr" if "türkçe" in instruct.lower() or "turkish" in instruct.lower() else "en"
            return instruct, "instruct", lang_code
        
        if is_voice_design:
            voice_map = {
                "TR": ("Türkçe konuşan erkek sesi", "tr"),
                "EN": ("A clear English male voice.", "en"),
            }
            voice_config = voice_map.get(voice.upper(), (voice, "en"))
            return voice_config[0], "instruct", voice_config[1]
        
        # Handle standard TTS models
        voice_map = {
            "TR": ("af_sky", "tr"),
            "EN": ("af_sky", "en"),
        }
        voice_config = voice_map.get(voice.upper(), (voice, "en"))
        return voice_config[0], "voice", voice_config[1]
    
    async def generate_speech(self, request: TTSRequest) -> bytes:
        try:
            logger.info(f"TTS request - model: {request.model}, voice: {request.voice}")
            
            is_voice_design = "voicedesign" in self.model.lower()
            voice_param, param_type, lang_code = self._get_voice_config(
                request.voice, 
                request.instruct, 
                is_voice_design
            )
            
            logger.info(f"Loading model: {self.model}")
            model = load_model(self.model)
            
            # Prepare parameters
            params = {
                "text": request.input,
                "model": model,
                "speed": request.speed,
                "lang_code": lang_code,
                "file_prefix": str(self.sample_audio_path).rsplit(".", 1)[0],
                "audio_format": request.response_format.value,
                "sample_rate": 24000,
                "join_audio": True,
                "verbose": False,
            }
            
            # Add voice or instruct parameter
            params[param_type] = voice_param
            
            # Add extra parameters
            params.update(request.get_extra_params() or {})
            
            logger.info(f"Generating audio - {param_type}: {voice_param}, lang: {lang_code}")
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
