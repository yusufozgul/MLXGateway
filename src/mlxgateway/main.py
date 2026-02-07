import argparse
import os

import setproctitle
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .audio.stt.router import router as stt_router
from .audio.tts.router import router as tts_router
from .chat.router import router as chat_router
from .config import Config, set_config
from .images.router import router as images_router
from .middleware.logging import RequestResponseLoggingMiddleware
from .models.router import router as models_router
from .utils.logger import logger, set_logger_level

app = FastAPI(title="MLX Gateway")

app.add_middleware(RequestResponseLoggingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.method} {request.url.path}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(images_router)
app.include_router(tts_router)
app.include_router(stt_router)


def build_parser():
    parser = argparse.ArgumentParser(description="MLX Gateway")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8008,
        help="Port to bind the server to (default: 8008)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: info)",
    )
    parser.add_argument(
        "--model-cache-ttl",
        type=int,
        default=600,
        help="Model cache TTL in seconds (default: 600)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=4,
        help="Maximum number of models to cache (default: 4)",
    )
    parser.add_argument(
        "--model-list-cache",
        type=int,
        default=600,
        help="Model list cache TTL in seconds (default: 600)",
    )
    return parser


def start():
    setproctitle.setproctitle("MLXGateway")
    
    parser = build_parser()
    args = parser.parse_args()

    config = Config.from_env(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        model_cache_ttl=args.model_cache_ttl,
        max_models=args.max_models,
        model_list_cache_ttl=args.model_list_cache,
    )
    set_config(config)

    set_logger_level(logger, config.log_level)

    logger.info(f"Starting MLX Gateway on {config.host}:{config.port}")
    logger.info(f"Model cache: max_size={config.max_models}, ttl={config.model_cache_ttl}s")
    logger.info(f"Model list cache: ttl={config.model_list_cache_ttl}s")

    uvicorn.run(
        "mlxgateway.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        use_colors=True,
    )


if __name__ == "__main__":
    start()
