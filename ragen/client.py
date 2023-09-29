from typing import List, Optional

import openai
import psycopg

from ragen.file import Chunk


def generate_prompt(context: List[Chunk], request: Chunk) -> str:
    context_text = "\n".join(chunk.text for chunk in context)
    return f"""Answer the question "{request.text}" with the following context:
{context_text}"""


class OpenAIClient:
    def __init__(self, api_key: str, api_base: Optional[str] = None) -> None:
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

    def chat(self, model: str, prompt: str):
        chat = openai.ChatCompletion.create(
            model=model,
            stream=True,
            max_tokens=1000,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        for result in chat:
            delta = result.choices[0].delta
            print(delta.get("content", ""), end="", flush=True)
        print()

    def embeddings(self, model: str, text: str) -> List[float]:
        emb = openai.Embedding.create(
            model=model,
            input=text,
        )
        return emb["data"][0]["embedding"]


class PgClient:
    CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    emb vector(%d) NOT NULL,
)
"""

    def __init__(self, host: str, user: str, password: str, port: int) -> None:
        self.conn = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
        )
