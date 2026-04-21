"""
llm.py — Ollama prompt builder + caller
Builds structured prompts and calls a local Ollama model (llama3:8b / mistral:7b).

⚠️  Never use 13B or 70B models — too heavy for MacBook Air CPU RAM budget.
"""

import json
import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3.2:3b"   # lightest model — change to llama3:8b or mistral:7b if pulled
REQUEST_TIMEOUT = 120              # seconds — local LLM can be slow on CPU
MAX_RETRIES     = 2
RETRY_DELAY     = 3

SYSTEM_PROMPT = """You are The Stylist's Brain — a creative, knowledgeable wardrobe stylist.
You explain your outfit reasoning clearly and always suggest at least one styling technique.
You ONLY respond with valid JSON. No markdown, no prose, no code fences. Raw JSON only."""

# Three persona system prompts for Bug 4 — 3 suggestions with different temperatures
SYSTEM_PROMPTS = {
    "safe": (
        "You are a classic wardrobe stylist. Suggest a clean, reliable, timeless outfit. "
        "Focus on proven combinations that always look polished. "
        "You ONLY respond with valid JSON. No markdown, no prose, no code fences. Raw JSON only."
    ),
    "creative": (
        "You are a fashion-forward stylist. Suggest something current, interesting and on-trend. "
        "Push boundaries while staying wearable. "
        "You ONLY respond with valid JSON. No markdown, no prose, no code fences. Raw JSON only."
    ),
    "experimental": (
        "You are an avant-garde stylist. Suggest an unconventional but wearable outfit that "
        "challenges expectations. Think texture, proportion and unexpected combinations. "
        "You ONLY respond with valid JSON. No markdown, no prose, no code fences. Raw JSON only."
    ),
}


def build_outfit_prompt(
    vibe: str,
    items: list[dict],
    weather: dict | None = None,
    calendar_event: str | None = None,
    locked_combos: list[list[str]] | None = None,
) -> str:
    """
    Assemble the full prompt payload for outfit generation.

    Args:
        vibe:           User's natural-language style request.
        items:          Retrieved wardrobe items (id + metadata).
        weather:        Dict with keys: temp_c, condition, city.
        calendar_event: Plain-text description of today's main event.
        locked_combos:  Item ID combinations to exclude (habit lock).

    Returns:
        Formatted prompt string to send to Ollama.
    """
    context_parts = []

    if weather:
        context_parts.append(
            f"WEATHER: {weather.get('condition', 'unknown')}, "
            f"{weather.get('temp_c', '?')}°C in {weather.get('city', 'unknown city')}"
        )

    if calendar_event:
        context_parts.append(f"TODAY'S EVENT: {calendar_event}")

    if locked_combos:
        context_parts.append(
            f"LOCKED COMBINATIONS (exclude these): {json.dumps(locked_combos)}"
        )

    context_block = "\n".join(context_parts) if context_parts else "No additional context."

    wardrobe_block = "\n".join(
        f"- ID: {item['id']} | Category: {item['metadata'].get('category','?')} "
        f"| Color: {item['metadata'].get('dominant_color','?')} "
        f"| Vibe tags: {item['metadata'].get('vibe_tags','?')}"
        for item in items
    )

    task = (
        f"USER REQUEST: {vibe}\n\n"
        "Using ONLY items listed above, assemble one complete outfit following the "
        "Rule of Three (top + bottom + third piece). Choose one styling technique. "
        "If you cannot form a strong outfit, identify the gap and suggest a DIY hack.\n\n"
        "Return ONLY this JSON structure, nothing else:\n"
        "{\n"
        '  "outfit": ["item_id_1", "item_id_2", "item_id_3"],\n'
        '  "styling_techniques": ["french tuck"],\n'
        '  "vibe_tags": ["minimalist", "sharp"],\n'
        '  "confidence": "safe | bold | experimental",\n'
        '  "why": "one sentence",\n'
        '  "gap": {"missing_item": "...", "diy_hack": "..."}\n'
        "}"
    )

    return f"CONTEXT:\n{context_block}\n\nWARDROBE:\n{wardrobe_block}\n\nTASK:\n{task}"


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> dict:
    """
    Send a prompt to the local Ollama server and parse the JSON response.

    Args:
        prompt:        Full prompt string (use build_outfit_prompt to build it).
        model:         Ollama model tag, e.g. "llama3:8b" or "mistral:7b".
        temperature:   Sampling temperature (0.7 = balanced, 1.2 = experimental).
        system_prompt: Override the default system prompt (for persona variation).

    Returns:
        Parsed JSON dict matching the required outfit schema.

    Raises:
        RuntimeError: If Ollama is unreachable or never returns valid JSON.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "format":  "json",   # Ollama native JSON mode — forces valid JSON output
        "options": {"temperature": temperature},
    }

    last_exc: Exception | None = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            break   # success — exit retry loop
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot reach Ollama at localhost:11434. "
                "Start it with: ollama serve"
            )
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Ollama returned HTTP %d — retrying in %ds (attempt %d/%d)",
                    exc.response.status_code, RETRY_DELAY, attempt + 1, 1 + MAX_RETRIES,
                )
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Ollama HTTP error after {1 + MAX_RETRIES} attempts: {exc}") from exc

    raw     = response.json()

    # Guard against unexpected Ollama response shapes (KeyError) or null content (TypeError)
    content = raw.get("message", {}).get("content")
    if not content or not isinstance(content, str):
        raise RuntimeError(
            f"Ollama returned an unexpected response structure. "
            f"Expected message.content string, got: {str(raw)[:200]}"
        )
    content = content.strip()

    # Primary: clean JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} block (handles preamble text)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.error("Ollama returned non-JSON content: %s", content[:300])
    raise RuntimeError(
        f"Ollama did not return valid JSON after extraction attempt. "
        f"Raw response (first 200 chars): {content[:200]}"
    )
