from dataclasses import dataclass
from typing import Generator

from torch import Tensor
from torch.nn.functional import cosine_similarity


class FileReader:
    def __init__(self, path: str) -> None:
        self.path = path

    def read_all(self) -> str:
        raise NotImplementedError

    def read_line(self) -> Generator[str, None, None]:
        raise NotImplementedError


class TextFileReader(FileReader):
    def read_all(self) -> str:
        with open(self.path, "r") as f:
            return f.read()

    def read_line(self) -> Generator[str, None, None]:
        with open(self.path, "r") as f:
            for line in f:
                yield line.strip()


MarkdownFileReader = TextFileReader

ExtensionToReader = {
    "md": MarkdownFileReader,
    "txt": TextFileReader,
    "": TextFileReader,
}


@dataclass(frozen=True)
class Chunk:
    index: int
    filename: str
    text: str
    emb: Tensor

    def cos_sim(self, other: "Chunk") -> float:
        return cosine_similarity(self.emb, other.emb, dim=0).item()


class ChunkGenerator:
    def __init__(self, size: int) -> None:
        self.size = size

    def generate(self, path) -> Generator[str, None, None]:
        partitions = path.rsplit(".", 1)
        extension = partitions[1] if len(partitions) > 1 else ""
        if extension not in ExtensionToReader:
            raise ValueError(f"unsupported extension: {extension}")
        reader = ExtensionToReader[extension](path)
        yield from self._gen_chunk(reader)

    def _gen_chunk(self, reader: FileReader) -> Generator[str, None, None]:
        chunk = ""
        for line in reader.read_line():
            if not line:
                continue
            chunk += " " + line
            if len(chunk) >= self.size:
                yield chunk
                chunk = ""
        if chunk:
            yield chunk
