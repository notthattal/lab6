import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

import llm

DATA_FILE = Path("preference_data.csv")

FIELDNAMES = [
    "id",
    "timestamp",
    "model",
    "prompt",
    "response_a",
    "response_b",
    "preference",           # "A", "B", or "tie"
    "output_tokens_a",
    "output_tokens_b",
    "latency_s_a",
    "latency_s_b",
]


def save(prompt: str, resp_a: llm.LLMResponse, resp_b: llm.LLMResponse, preference: str) -> None:
    write_header = not DATA_FILE.exists()
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": llm.LLM_MODEL,
        "prompt": prompt,
        "response_a": resp_a.text,
        "response_b": resp_b.text,
        "preference": preference,
        "output_tokens_a": resp_a.output_tokens,
        "output_tokens_b": resp_b.output_tokens,
        "latency_s_a": resp_a.latency_s,
        "latency_s_b": resp_b.latency_s,
    }
    with DATA_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def record_count() -> int:
    if not DATA_FILE.exists():
        return 0
    with DATA_FILE.open() as f:
        return sum(1 for _ in f) - 1  # subtract header


def csv_bytes() -> bytes:
    return DATA_FILE.read_bytes() if DATA_FILE.exists() else b""
