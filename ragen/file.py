from dataclasses import dataclass
from typing import Generator, List

from torch import Tensor


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
    text: str
    emb: Tensor


class ChunkGenerator:
    def __init__(self, paths: List[str], size: int) -> None:
        self.paths = paths
        self.size = size

    def generate(self) -> Generator[str, None, None]:
        for path in self.paths:
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
