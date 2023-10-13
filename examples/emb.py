import base64
import os

import numpy as np
from llmspec import EmbeddingData, EmbeddingRequest, EmbeddingResponse, TokenUsage
from mosec import ClientError, Runtime, Server, Worker

DEFAULT_MODEL = "thenlper/gte-base"


class Embedding(Worker):
    def __init__(self):
        self.model_name = os.environ.get("EMB_MODEL", DEFAULT_MODEL)
        self.dim = 768

    def deserialize(self, req: bytes) -> EmbeddingRequest:
        return EmbeddingRequest.from_bytes(req)

    def serialize(self, resp: EmbeddingResponse) -> bytes:
        return resp.to_json()

    def forward(self, req: EmbeddingRequest) -> EmbeddingResponse:
        if req.model != self.model_name:
            raise ClientError(
                f"the requested model {req.model} is not supported by "
                f"this worker {self.model_name}"
            )
        token_count = 0
        embeddings = np.random.rand(len(req.input), 768)
        if req.encoding_format == "base64":
            embeddings = [
                base64.b64encode(emb.astype(np.float32).tobytes()).decode("utf-8")
                for emb in embeddings
            ]
        else:
            embeddings = [emb.tolist() for emb in embeddings]

        resp = EmbeddingResponse(
            data=[
                EmbeddingData(embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=self.model_name,
            usage=TokenUsage(
                prompt_tokens=token_count,
                # No completions performed, only embeddings generated.
                completion_tokens=0,
                total_tokens=token_count,
            ),
        )
        return resp


if __name__ == "__main__":
    server = Server()
    emb = Runtime(Embedding)
    server.register_runtime(
        {
            "/v1/embeddings": [emb],
            "/embeddings": [emb],
        }
    )
    server.run()
