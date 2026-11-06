"""Day 1 — LLM API Foundation (Tasks 1–5 and Bonus A–C).

Run: python starter-code/template.py
Install: python -m pip install -r requirements.txt
Keep real API keys in a local .env file; do not commit it.
"""

import os
import random
import time
from typing import Any, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# USD per 1 million tokens: original LAB reference values, not live prices.
# For new models, add verified pricing before reporting actual costs.
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    # Provisional example only; do not use as verified billing price.
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
LAB_GEMINI_MODEL = "gemini-2.5-flash"  # Fixed reference for LAB cost test
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _gemini_client():
    from google import genai
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    return genai.Client(api_key=key)


def _is_retryable_error(exc: Exception) -> bool:
    """Retry only transient 429/500/502/503/504 API errors."""
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    try:
        return int(code) in (429, 500, 502, 503, 504)
    except (TypeError, ValueError):
        return False


def _retry_transient(fn: Callable[[], Any], max_retries: int = 3,
                     base_delay: float = 2.0) -> Any:
    """Retry transient API failures; never retry invalid keys or invalid model IDs."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), 30.0)
            print(f"[API {getattr(exc, 'code', '?')}] Tạm thời quá tải; "
                  f"thử lại sau {delay:g} giây ({attempt + 1}/{max_retries})...")
            time.sleep(delay)


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY in .env")
    client = OpenAI(api_key=key)
    start = time.perf_counter()
    response = _retry_transient(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ))
    latency = time.perf_counter() - start
    usage = response.usage
    return (
        response.choices[0].message.content or "",
        latency,
        {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
    )


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google.genai import types
    client = _gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.perf_counter()
    response = _retry_transient(lambda: client.models.generate_content(
        model=model, contents=prompt, config=config,
    ))
    latency = time.perf_counter() - start
    metadata = response.usage_metadata
    return (
        response.text or "",
        latency,
        {
            "input_tokens": (metadata.prompt_token_count or 0) if metadata else 0,
            "output_tokens": (metadata.candidates_token_count or 0) if metadata else 0,
        },
    )


def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Missing ANTHROPIC_API_KEY in .env")
    client = anthropic.Anthropic(api_key=key)
    start = time.perf_counter()
    response = _retry_transient(lambda: client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    ))
    latency = time.perf_counter() - start
    answer = "".join(
        block.text for block in response.content
        if isinstance(getattr(block, "text", None), str)
        and (not isinstance(getattr(block, "type", None), str)
             or block.type == "text")
    )
    usage = response.usage
    return (
        answer,
        latency,
        {
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
        },
    )


def compare_models(prompt: str) -> dict:
    configurations = (
        ("gpt4o", OPENAI_MODEL, call_openai),
        ("gpt4o_mini", OPENAI_MINI_MODEL, call_openai),
        ("gemini_flash", LAB_GEMINI_MODEL, call_gemini),
    )
    results = {}
    for name, model, call_fn in configurations:
        response, latency, usage = call_fn(prompt, model=model)
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        prices = PRICING_1M_TOKENS.get(model)
        if prices is None:
            raise ValueError(
                f"No pricing configured for '{model}'. "
                "Add verified input/output prices to PRICING_1M_TOKENS."
            )
        cost = (
            input_tokens * prices["input"] + output_tokens * prices["output"]
        ) / 1_000_000
        results[name] = {
            "response": response,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return results


def streaming_chatbot() -> None:
    from google.genai import types
    client = _gemini_client()
    history: list[dict[str, str]] = []
    print("\n==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print(f"Powered by Google {GEMINI_MODEL}")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not prompt:
            continue

        contents = [
            types.Content(
                role=turn["role"],
                parts=[types.Part.from_text(text=turn["text"])],
            )
            for turn in history[-6:]
        ]
        contents.append(types.Content(
            role="user", parts=[types.Part.from_text(text=prompt)]
        ))

        answer = None
        for attempt in range(4):
            chunks: list[str] = []
            print("Gemini: ", end="", flush=True)
            try:
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL, contents=contents,
                ):
                    piece = chunk.text or ""
                    if piece:
                        print(piece, end="", flush=True)
                        chunks.append(piece)
                print("\n")
                answer = "".join(chunks)
                break
            except Exception as exc:
                print()
                # Never auto-retry after partial streaming output: doing so
                # could duplicate displayed text and incur additional costs.
                if chunks or not _is_retryable_error(exc) or attempt == 3:
                    print(f"[API error] {exc}\n")
                    break
                delay = min(2 ** attempt, 30)
                print(f"[API temporary error] Retrying in {delay}s "
                      f"({attempt + 1}/3)...\n")
                time.sleep(delay)

        # Only store completed responses; failed streams are not history.
        if answer:
            history.extend([
                {"role": "user", "text": prompt},
                {"role": "model", "text": answer},
            ])
            history = history[-6:]


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Bonus A: generic exponential backoff (0.1, 0.2, 0.4 by default)."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


def batch_compare(prompts: list[str]) -> list[dict]:
    """Bonus B: compare three models for each prompt."""
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    """Bonus C: format comparison results as Markdown."""
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        prompt = clean(item["prompt"])
        if len(prompt) > 40:
            prompt = prompt[:37] + "..."
        for model in ("gpt4o", "gpt4o_mini", "gemini_flash"):
            data = item[model]
            response = clean(data["response"])
            if len(response) > 50:
                response = response[:47] + "..."
            lines.append(
                f"| {prompt} | { {'gpt4o': 'GPT-4o', 'gpt4o_mini': 'GPT-4o-Mini', 'gemini_flash': 'Gemini-Flash'}[model] } | {response} | "
                f"{data['latency']:.2f}s | "
                f"{data['input_tokens']}/{data['output_tokens']} | "
                f"${data['cost']:.6f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    prompt = (
        "Hãy giải thích sự khác biệt giữa temperature và top_p "
        "bằng tiếng Việt ngắn gọn trong 2 câu."
    )
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping model comparison: OPENAI_API_KEY is missing.")
        print("You can still test the Gemini streaming chatbot below.")
    else:
        try:
            comparison = compare_models(prompt)
            for name, stats in comparison.items():
                print(f"\n[{name.upper()}]")
                print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
                print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
                print(f"Response: {stats['response']}")
        except Exception as exc:
            print(f"Model comparison unavailable: {exc}")
            print("Check API keys, model access, and pricing configuration.")

    print(f"\n=== Starting {GEMINI_MODEL} Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as exc:
        print(f"Chatbot could not start: {exc}")
