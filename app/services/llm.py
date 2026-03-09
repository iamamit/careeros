import json
import logging
import re
from typing import AsyncGenerator, Optional
from groq import AsyncGroq, APIStatusError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
_client: Optional[AsyncGroq] = None

def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set. Get free key at console.groq.com")
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client

def llm_retry(func):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)

@llm_retry
async def chat_completion(messages, system_prompt, temperature=0.1, max_tokens=2000) -> str:
    client = get_client()
    full_messages = [{"role": "system", "content": system_prompt}, *messages]
    logger.info(f"LLM call → model={settings.groq_model}")
    response = await client.chat.completions.create(
        model=settings.groq_model, messages=full_messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return response.choices[0].message.content

@llm_retry
async def chat_completion_json(messages, system_prompt, temperature=0.1, max_tokens=2000) -> dict:
    client = get_client()
    json_system = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation."
    full_messages = [{"role": "system", "content": json_system}, *messages]
    response = await client.chat.completions.create(
        model=settings.groq_model, messages=full_messages,
        temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    logger.info(f"LLM JSON response → tokens={response.usage.total_tokens}")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON: {content[:200]}")

async def chat_completion_stream(messages, system_prompt, temperature=0.4, max_tokens=2000) -> AsyncGenerator[str, None]:
    client = get_client()
    full_messages = [{"role": "system", "content": system_prompt}, *messages]
    try:
        stream = await client.chat.completions.create(
            model=settings.groq_model, messages=full_messages,
            temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except RateLimitError:
        yield "\n\n[Rate limit reached — try again shortly]"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"\n\n[Error: {str(e)}]"
