"""letheo_orchestration · Capa Python de alto nivel sobre el Cognitive Runtime.

Provee:
- ``Session``: API ergonómica con context-manager, MQL ejecutable y prosa para LLM.
- ``to_prose``: convierte un ``CompressedContext`` en un bloque narrativo listo para inyectar
  en un prompt de LLM (Claude, GPT, llama, etc.).

Local-first: no requiere red ni SDKs externos. La integración con un LLM concreto se hace fuera.
"""
from .session import Session, EvokeResult
from .prose import to_prose, ArcReading, read_arc
from .tokens import count_tokens, count_tokens_method

__all__ = [
    "Session", "EvokeResult", "to_prose", "ArcReading", "read_arc",
    "count_tokens", "count_tokens_method",
    # ``llm`` se importa perezosamente: requiere el extra `[llm]`.
]


def __getattr__(name: str):
    if name in {"LLMClient", "Memorist", "AskResult", "OPENAI", "DEEPSEEK"}:
        from . import llm
        return getattr(llm, name)
    raise AttributeError(name)
