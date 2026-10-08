"""Trusted adapter registry: world packages name protocols, not concrete code."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .protocol import WorldAdapter


class AdapterRegistry:
    def __init__(self): self._loaders: dict[str, Callable[[Path], WorldAdapter]] = {}

    def register(self, name: str, loader: Callable[[Path], WorldAdapter]):
        if not name or not callable(loader): raise ValueError("adapter registration needs a name and loader")
        self._loaders[str(name)] = loader

    def load(self, name: str, folder: Path):
        if name not in self._loaders:
            raise ValueError(f"world requests unregistered adapter {name!r}; available: {sorted(self._loaders)}")
        return self._loaders[name](Path(folder))

    def names(self): return sorted(self._loaders)
