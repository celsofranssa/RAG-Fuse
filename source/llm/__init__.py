import asyncio
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


async def llm_predict(session: aiohttp.ClientSession, base_url: str, request: dict) -> dict:
    """
    Send a completion request to a local vLLM server (OpenAI-compatible API).

    Args:
        session: aiohttp ClientSession for connection pooling.
        base_url: vLLM server base URL (e.g. "http://localhost:8001/v1").
        request: dict with keys "body", "model", "timeout", and optionally "stop".
    """
    body = request["body"]
    payload = {
        "model": request["model"],
        "prompt": body["prompt"],
        "max_tokens": body.get("max_gen_len", 256),
        "temperature": body.get("temperature", 0.6),
        "top_p": body.get("top_p", 0.9),
    }
    if "stop" in body:
        payload["stop"] = body["stop"]

    async def _do_request():
        async with session.post(
            f"{base_url}/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            request["response"] = data["choices"][0]["text"]
            request["status"] = "success"

    try:
        await asyncio.wait_for(_do_request(), timeout=request.get("timeout", 60))

    except asyncio.TimeoutError:
        request["status"] = "failure"

    except Exception as e:
        request["status"] = "failure"
        logger.error("Exception while predicting via vLLM", exc_info=e)
        time.sleep(2)

    return request


async def process_llm_predict(
    session: aiohttp.ClientSession, base_url: str, requests: list[dict]
) -> list[dict]:
    """Process a batch of requests concurrently."""
    tasks = [llm_predict(session, base_url, req) for req in requests]
    return await asyncio.gather(*tasks)
