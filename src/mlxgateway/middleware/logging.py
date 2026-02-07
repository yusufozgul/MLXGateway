import json
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logger import logger


def format_body(body: str) -> str:
    try:
        parsed = json.loads(body)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return body


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        body = await self._get_request_body(request)
        
        try:
            body_json = json.loads(body)
            if body_json.get("stream", False):
                start_time = time.time()
                response = await call_next(request)
                process_time = time.time() - start_time

                logger.debug(
                    f"First Stream Response took {process_time:.2f}s:\n"
                    f"Status: {response.status_code}\n"
                )
                return response
        except json.JSONDecodeError:
            pass

        request_id = str(time.time())

        logger.debug(
            f"Request [{request_id}]: {request.method} {request.url}\n"
            f"Body:\n{format_body(body)}",
        )

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        async def body_iterator():
            yield response_body

        response.body_iterator = body_iterator()

        try:
            body_text = response_body.decode()
            body_text = format_body(body_text)
        except UnicodeDecodeError:
            body_text = "<Binary Content>"

        logger.debug(
            f"Response [{request_id}] took {process_time:.2f}s:\n"
            f"Status: {response.status_code}\n"
            f"Body:\n{body_text}",
        )
        return response

    async def _get_request_body(self, request: Request) -> str:
        try:
            body = await request.body()
            return body.decode()
        except Exception:
            return ""
