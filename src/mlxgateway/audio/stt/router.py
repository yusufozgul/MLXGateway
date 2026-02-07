from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette.responses import PlainTextResponse

from ...models.error import ErrorDetail, ErrorResponse
from .schema import ResponseFormat, STTRequestForm, TranscriptionResponse
from .service import STTService

router = APIRouter(prefix="/v1", tags=["speech-to-text"])


@router.post("/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(request: STTRequestForm = Depends()):
    stt_service = STTService()
    try:
        result = await stt_service.transcribe(request)

        if request.response_format == ResponseFormat.TEXT:
            return PlainTextResponse(content=result)
        else:
            return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred while transcribing audio.",
                    type="server_error",
                    code="internal_error"
                )
            ).model_dump()
        )
