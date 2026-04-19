import os
import time
from pydantic import BaseModel

from openai import OpenAI

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

class LLMResponse(BaseModel):
    text: str
    output_tokens: int
    latency_s: float


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def generate(client: OpenAI, prompt: str) -> LLMResponse:
    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round(time.perf_counter() - t0, 3)
    return LLMResponse(
        text=completion.choices[0].message.content,
        output_tokens=completion.usage.completion_tokens,
        latency_s=latency,
    )
