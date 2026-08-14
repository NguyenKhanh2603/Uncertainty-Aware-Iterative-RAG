"""Verify that an OpenAI-compatible server exposes token log-probabilities."""

from __future__ import annotations

import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Reply with exactly: ready"}],
        temperature=0.0,
        max_tokens=8,
        logprobs=True,
        top_logprobs=1,
    )
    choice = response.choices[0]
    token_logprobs = choice.logprobs.content if choice.logprobs else None
    if not token_logprobs:
        raise RuntimeError("The endpoint generated text but returned no token log-probabilities")
    print(f"Endpoint OK: {choice.message.content!r}")
    print(f"Token log-probabilities OK: {len(token_logprobs)} tokens")


if __name__ == "__main__":
    main()

