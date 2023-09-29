from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ragen.file import Chunk


class Embedding:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return self.model.encode(text)

    def retrieve(self, req_emb: Chunk, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            scores[i] = req_emb.cos_sim(chunk)
        index = np.argpartition(scores, -top_k)[-top_k:]
        return [chunks[i] for i in index]
