"""
LLM Backend — Ollama Provider
==============================

Exposes a single function, call_llm(), that sends a conversation turn to
a local Ollama server.

Start Ollama:
    ollama serve
    ollama pull llama3.1:8b

Environment variables:
    OLLAMA_URL          (default: http://localhost:11434)
    OLLAMA_MODEL        (default: llama3.1:8b)
    OLLAMA_TIMEOUT_S    seconds per request (default: 120)
    OLLAMA_MAX_RETRIES  retries on timeout/server error (default: 2)
"""

import os
import sys
import time
import json
import requests

OLLAMA_URL         = os.getenv("OLLAMA_URL",         "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.1:8b")
OLLAMA_TIMEOUT_S   = int(os.getenv("OLLAMA_TIMEOUT_S",   "120"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))

_STOP_SEQUENCES = ["\nYou:", "\nUser:", "\nPatient:"]


def call_llm(system_prompt: str, history: list, stream: bool = True) -> str:
    """
    Send a conversation turn to Ollama and return the reply text.

    When stream=True (default), tokens are printed to stdout as they arrive
    so the user sees output immediately rather than waiting for the full reply.

    Retries up to OLLAMA_MAX_RETRIES times on timeout or 5xx server errors,
    with a short backoff between attempts.

    Args:
        system_prompt: patient context and instructions (the system role message).
        history: list of {"role": "user"|"assistant", "content": "..."} dicts
                 representing the conversation so far.
        stream: if True, print tokens to stdout as they arrive and return the
                full text; if False, return the full text silently.

    Returns:
        The model's reply as a plain string, or an error message prefixed with [Error].
    """
    messages = [{"role": "system", "content": system_prompt}] + history

    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):  # +2 so range gives 1..max+1
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": messages,
                    "stream":   stream,
                    # Stop sequences prevent the model from generating a fake
                    # user turn (e.g. "\nYou: ..."), which would cause it to
                    # simulate a full back-and-forth conversation in one reply.
                    "options": {
                        "stop":           _STOP_SEQUENCES,
                        "temperature":    0.1,
                        "repeat_penalty": 1.15,
                        "top_p":          0.9,
                    },
                },
                timeout=OLLAMA_TIMEOUT_S,
                stream=stream,
            )
            resp.raise_for_status()

            if not stream:
                return resp.json()["message"]["content"]

            # Stream: print each token chunk as it arrives, accumulate full text.
            full_text: list[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    full_text.append(token)
                if chunk.get("done"):
                    break
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(full_text)

        except requests.exceptions.ConnectionError:
            return (
                f"[Error] Cannot reach Ollama at {OLLAMA_URL}. "
                f"Run `ollama serve` and ensure the model is available: "
                f"`ollama pull {OLLAMA_MODEL}`"
            )

        except requests.exceptions.Timeout:
            if attempt <= OLLAMA_MAX_RETRIES:
                time.sleep(2 * attempt)
                continue
            return (
                f"[Error] Ollama timed out after {OLLAMA_TIMEOUT_S}s "
                f"({OLLAMA_MAX_RETRIES} retries). The model may be overloaded."
            )

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status and int(status) >= 500 and attempt <= OLLAMA_MAX_RETRIES:
                time.sleep(2 * attempt)
                continue
            return f"[Error] Ollama returned HTTP {status}: {exc}"

        except Exception as exc:
            return f"[Error] Ollama call failed: {exc}"

    return "[Error] Ollama call failed after all retries."
