"""External service clients used by the workflow agents.

The provider implementations rely on optional third-party SDKs. Importing them
eagerly when the :mod:`video_gen.providers` package is loaded made it
impossible to import lightweight utilities—such as the
``OpenAIWorkflowClient``'s parsing helpers—in environments where the optional
dependencies are unavailable (like the execution environment used by the unit
tests in this kata).  To keep the public API intact while avoiding the hard
dependency, the module now exposes the provider classes via ``__getattr__`` and
only imports the corresponding implementation when it is actually requested.
This mirrors the lazy-import pattern used by the Python standard library (for
example :mod:`importlib.resources`).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "OpenAIWorkflowClient",
    "XunfeiTTSClient",
    "DashscopeMusicClient",
    "DashscopeAmbienceClient",
    "DeepSeekWorkflowClient",
    "DoubaoImageClient",
]


if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .openai_client import OpenAIWorkflowClient as OpenAIWorkflowClient
    from .xunfei import XunfeiTTSClient as XunfeiTTSClient
    from .dashscope_music import DashscopeMusicClient as DashscopeMusicClient
    from .dashscope_ambience import (
        DashscopeAmbienceClient as DashscopeAmbienceClient,
    )
    from .deepseek_client import DeepSeekWorkflowClient as DeepSeekWorkflowClient
    from .doubao_client import DoubaoImageClient as DoubaoImageClient


_MODULE_MAP = {
    "OpenAIWorkflowClient": "openai_client",
    "XunfeiTTSClient": "xunfei",
    "DashscopeMusicClient": "dashscope_music",
    "DashscopeAmbienceClient": "dashscope_ambience",
    "DeepSeekWorkflowClient": "deepseek_client",
    "DoubaoImageClient": "doubao_client",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    try:
        module_name = _MODULE_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module 'video_gen.providers' has no attribute {name!r}"
        ) from exc

    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
