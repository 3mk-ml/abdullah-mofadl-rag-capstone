from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    title: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "title": self.title,
            "metadata": self.metadata or {},
        }
