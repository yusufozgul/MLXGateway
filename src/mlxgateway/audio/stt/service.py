import tempfile
from pathlib import Path
from typing import Union

from mlx_audio.stt import load_model
from mlx_audio.stt.generate import generate_transcription

from ...utils.logger import logger
from .schema import (
    ResponseFormat,
    STTRequestForm,
    TranscriptionResponse,
    TranscriptionWord,
)


class STTService:
    async def _save_upload_file(self, file) -> str:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            return tmp.name

    def _format_response(
        self, result: dict, request: STTRequestForm
    ) -> Union[dict, str, TranscriptionResponse]:
        if request.response_format == ResponseFormat.TEXT:
            return result["text"]

        elif request.response_format == ResponseFormat.VERBOSE_JSON:
            return result

        elif request.response_format == ResponseFormat.JSON:
            return {"text": result["text"]}

        else:
            text = result.get("text", "")
            language = result.get("language", "en")

            duration = 0
            if "segments" in result:
                for segment in result["segments"]:
                    if "end" in segment:
                        duration = max(duration, segment["end"])

            words = []
            if request.timestamp_granularities and "word" in [
                g.value for g in request.timestamp_granularities
            ]:
                for segment in result.get("segments", []):
                    for word_data in segment.get("words", []):
                        word = TranscriptionWord(
                            word=word_data["word"],
                            start=word_data["start"],
                            end=word_data["end"],
                        )
                        words.append(word)

            return TranscriptionResponse(
                task="transcribe",
                language=language,
                duration=duration,
                text=text,
                words=words if words else None,
            )

    async def transcribe(
        self, request: STTRequestForm
    ) -> Union[dict, str, TranscriptionResponse]:
        audio_path = None
        try:
            logger.info(f"STT input - model: {request.model}, file: {request.file.filename}, language: {request.language}, temp: {request.temperature}")
            audio_path = await self._save_upload_file(request.file)
            
            # Load model
            logger.info(f"Loading STT model: {request.model}")
            model = load_model(request.model)
            
            # Generate transcription
            logger.info(f"Transcribing audio: {audio_path}")
            result = generate_transcription(
                model=model,
                audio=audio_path,
                language=request.language,
                temperature=request.temperature,
                initial_prompt=request.prompt,
                verbose=False,
            )
            
            # Convert STTOutput to dict format
            result_dict = {
                "text": result.text,
                "language": result.language or "en",
                "segments": result.segments or [],
            }
            
            logger.info(f"STT output - text: {result.text}, language: {result.language or 'en'}, segments: {len(result.segments or [])}")
            
            response = self._format_response(result_dict, request)
            Path(audio_path).unlink(missing_ok=True)
            return response

        except Exception as e:
            if audio_path:
                Path(audio_path).unlink(missing_ok=True)
            raise e
