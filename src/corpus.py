"""Carrega os dispositivos da LGPD (arts. 6º e 7º) como documentos recuperáveis.

Cada dispositivo (caput ou inciso) é um documento. O texto indexado inclui o
rótulo ("Art. 7º, II"), para que consultas por termo exato (número do artigo)
tenham com o que casar.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    id: str
    rotulo: str
    texto: str

    @property
    def indexavel(self) -> str:
        return f"{self.rotulo} {self.texto}"


def load_docs(path: Path) -> list[Doc]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Doc(n["id"], n["rotulo"], n["texto"]) for n in data["nodes"]]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Tokens de conteúdo: minúsculas, sem acento, mantém números e siglas."""
    return re.findall(r"[a-z0-9]+", normalize(text))
