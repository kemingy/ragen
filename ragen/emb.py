from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from ragen.file import Chunk


class Embedding:
    def __init__(self, model_name) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return self.model.encode(text)

    def retrieve(self, req_emb: Chunk, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            scores[i] = cos_sim(req_emb.emb, chunk.emb)
        index = np.argpartition(scores, -top_k)[-top_k:]
        return [chunks[i] for i in index]
