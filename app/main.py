import os
import uuid
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import httpx
from fastapi import FastAPI, Query, Path, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()
# Logging setup (avoid logging secrets)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("accuweather-proxy")

API_BASE = "http://dataservice.accuweather.com"
TIMEOUT_SECONDS = 10.0
RETRY_ATTEMPTS = 3

# Simple in-memory rate limiting (global, 60 requests per rolling 60 seconds)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
WINDOW_SECONDS = 60
_request_times: list[float] = []
_rate_lock = asyncio.Lock()


def _get_api_key() -> Optional[str]:
    return os.getenv("API_KEY")


def _safe_url(url: str) -> str:
    """Return URL with any apikey parameter redacted."""
    try:
        parts = urlsplit(url)
        qs = parse_qsl(parts.query, keep_blank_values=True)
        redacted = [(k, "REDACTED" if k.lower() == "apikey" else v) for k, v in qs]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted), parts.fragment))
    except Exception:
        return url


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(base_url=API_BASE, timeout=httpx.Timeout(TIMEOUT_SECONDS)) as client:
        app.state.http = client
        yield


app = FastAPI(title="AccuWeather Proxy", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id_and_rate_limit(request: Request, call_next):
    # Add/propagate request id
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    # Basic global rate limit check (60 rpm by default)
    now = time.monotonic()
    async with _rate_lock:
        cutoff = now - WINDOW_SECONDS
        # prune timestamps outside the window
        # use in-place removal for performance on small lists
        idx = 0
        for t in _request_times:
            if t >= cutoff:
                break
            idx += 1
        if idx:
            del _request_times[:idx]

        if len(_request_times) >= RATE_LIMIT:
            logger.warning(f"rate_limit_exceeded req_id={request_id}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(WINDOW_SECONDS), "X-Request-ID": request_id},
            )
        _request_times.append(now)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _extract_error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            for key in ("Message", "message", "error", "detail", "Reason"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # fallback to stringified dict
            return str(data)
        # list or other JSON
        return str(data)
    except Exception:
        # not JSON
        text = (resp.text or "").strip()
        return text or f"HTTP {resp.status_code}"


async def _upstream_get(request: Request, path: str, params: dict) -> httpx.Response:
    client: httpx.AsyncClient = request.app.state.http
    attempts = 0
    backoff = 0.5
    while True:
        attempts += 1
        try:
            resp = await client.get(path, params=params)
            # retry on 5xx
            if 500 <= resp.status_code < 600 and attempts < RETRY_ATTEMPTS:
                logger.error(
                    "upstream_5xx req_id=%s url=%s status=%s attempt=%s",
                    request.state.request_id,
                    _safe_url(str(client.base_url) + path),
                    resp.status_code,
                    attempts,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return resp
        except (httpx.TimeoutException,) as e:
            if attempts < RETRY_ATTEMPTS:
                logger.error(
                    "upstream_timeout req_id=%s url=%s attempt=%s",
                    request.state.request_id,
                    _safe_url(str(client.base_url) + path),
                    attempts,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise
        except httpx.HTTPError as e:
            # network errors (ConnectionError etc.) - retry similarly
            if attempts < RETRY_ATTEMPTS:
                logger.error(
                    "upstream_http_error req_id=%s url=%s attempt=%s err=%s",
                    request.state.request_id,
                    _safe_url(str(client.base_url) + path),
                    attempts,
                    e.__class__.__name__,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise


@app.get("/")
async def root():
    return {"message": "AccuWeather Proxy is running."}


@app.get("/locations")
async def locations(request: Request, q: str = Query(..., min_length=1)):
    api_key = _get_api_key()
    if not api_key:
        return JSONResponse(
            status_code=500, content={"error": "Server misconfiguration: ACCUWEATHER_API_KEY is not set."}
        )

    path = "/locations/v1/cities/autocomplete"
    try:
        upstream = await _upstream_get(request, path, params={"q": q, "apikey": api_key})
    except httpx.TimeoutException:
        logger.error(
            "upstream_timeout_final req_id=%s url=%s",
            request.state.request_id,
            _safe_url(API_BASE + path),
        )
        return JSONResponse(status_code=504, content={"error": "Upstream request timed out."})
    except httpx.HTTPError as e:
        logger.error(
            "upstream_unreachable req_id=%s url=%s err=%s",
            request.state.request_id,
            _safe_url(API_BASE + path),
            e.__class__.__name__,
        )
        return JSONResponse(status_code=502, content={"error": "Upstream service unreachable."})

    if upstream.status_code >= 400:
        msg = _extract_error_message(upstream)
        logger.error(
            "upstream_error req_id=%s url=%s status=%s msg=%s",
            request.state.request_id,
            _safe_url(str(upstream.request.url)),
            upstream.status_code,
            msg,
        )
        return JSONResponse(status_code=upstream.status_code, content={"error": msg})

    return JSONResponse(status_code=200, content=upstream.json())


@app.get("/currentweather/{locationKey}")
async def current_weather(request: Request, locationKey: str = Path(..., min_length=1)):
    api_key = _get_api_key()
    if not api_key:
        return JSONResponse(
            status_code=500, content={"error": "Server misconfiguration: ACCUWEATHER_API_KEY is not set."}
        )

    path = f"/currentconditions/v1/{locationKey}"
    try:
        upstream = await _upstream_get(request, path, params={"apikey": api_key})
    except httpx.TimeoutException:
        logger.error(
            "upstream_timeout_final req_id=%s url=%s",
            request.state.request_id,
            _safe_url(API_BASE + path),
        )
        return JSONResponse(status_code=504, content={"error": "Upstream request timed out."})
    except httpx.HTTPError as e:
        logger.error(
            "upstream_unreachable req_id=%s url=%s err=%s",
            request.state.request_id,
            _safe_url(API_BASE + path),
            e.__class__.__name__,
        )
        return JSONResponse(status_code=502, content={"error": "Upstream service unreachable."})

    if upstream.status_code >= 400:
        msg = _extract_error_message(upstream)
        logger.error(
            "upstream_error req_id=%s url=%s status=%s msg=%s",
            request.state.request_id,
            _safe_url(str(upstream.request.url)),
            upstream.status_code,
            msg,
        )
        return JSONResponse(status_code=upstream.status_code, content={"error": msg})

    return JSONResponse(status_code=200, content=upstream.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
