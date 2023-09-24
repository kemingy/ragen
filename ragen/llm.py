from typing import List, Optional

import openai

from ragen.file import Chunk


def generate_prompt(context: List[Chunk], request: Chunk) -> str:
    context_text = "\n".join(chunk.text for chunk in context)
    return f"""Answer the question "{request.text}" with the following context:
{context_text}"""


def ask_llm(
    prompt: str, model: str, api_key: str = "", api_base: Optional[str] = None
) -> str:
    openai.api_key = api_key
    if api_base:
        openai.api_base = api_base
    chat = openai.ChatCompletion.create(
        model=model,
        stream=True,
        max_tokens=1000,
        temperature=1,
        messages=[{
            "role": "user",
            "content": prompt,
        }]
    )
    for result in chat:
        delta = result.choices[0].delta
        print(delta.get("content", ""), end="", flush=True)
    print()
