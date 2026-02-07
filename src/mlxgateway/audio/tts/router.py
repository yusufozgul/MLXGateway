import io

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from ...models.error import ErrorDetail, ErrorResponse
from .schema import AudioFormat, TTSRequest
from .service import TTSService

router = APIRouter(prefix="/v1", tags=["text-to-speech"])


@router.post("/audio/speech")
async def create_speech(request: TTSRequest):
    tts_service = TTSService(request.model)

    try:
        audio_content = await tts_service.generate_speech(request=request)

        content_type_mapping = {
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.OPUS: "audio/opus",
            AudioFormat.AAC: "audio/aac",
            AudioFormat.FLAC: "audio/flac",
            AudioFormat.WAV: "audio/wav",
            AudioFormat.PCM: "audio/pcm",
        }

        return StreamingResponse(
            io.BytesIO(audio_content),
            media_type=content_type_mapping[request.response_format],
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format.value}"'
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while generating speech.",
                    type="server_error",
                    code="internal_error"
                )
            ).model_dump()
        )
