"""Conteo de tokens honesto del bloque de memoria que se inyecta en el LLM.

Contexto (ver ``docs/05-honest-assessment.md``): el runtime Rust expone
``CompressedContext.token_estimate``, una **estimación de asignación** (``nº_vectores ×
tokens_per_vector``, realimentable desde tiktoken) usada *dentro* de ``evoke`` para decidir cuántos
vectores caben. **No** es el número de tokens del
texto que finalmente se envía al modelo. Para gestionar el budget de forma honesta hay que contar el
**texto real** de la prosa con el tokenizador del modelo consumidor.

Estrategia pluggable:
- Si ``tiktoken`` está instalado → conteo **exacto** con el encoding del modelo (OpenAI/DeepSeek).
- Si no → heurística calibrada **sobre el texto real** (conservadora, no infra-estima el budget).
  Es una estimación, declarada como tal vía ``count_tokens_method()``; la ruta a exacto es
  ``pip install tiktoken``.
"""
from __future__ import annotations

from functools import lru_cache

# Encoding por defecto de los modelos GPT-4o/4o-mini modernos. DeepSeek es razonablemente próximo.
_DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _try_tiktoken(model: str | None):
    """Devuelve un encoder de tiktoken o ``None`` si no está disponible offline."""
    try:
        import tiktoken
    except Exception:
        return None
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:
        return None


def count_tokens_method(model: str | None = None) -> str:
    """``"tiktoken"`` (exacto) o ``"heuristic"`` (estimación). Para reportes honestos."""
    return "tiktoken" if _try_tiktoken(model) is not None else "heuristic"


def _heuristic(text: str) -> int:
    """Estimación conservadora sobre el texto real.

    Dos reglas de oro de OpenAI: ~4 caracteres/token y ~0.75 palabras/token (≈ palabras×1.33).
    Tomamos el **máximo** de ambas para no infra-estimar el budget (es peor pasarse del límite real
    que sobrar). Sigue siendo una estimación; lo honesto es decirlo, no fingir exactitud.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, round(max(words * 1.333, chars / 4.0)))


def count_tokens(text: str, model: str | None = None) -> int:
    """Cuenta los tokens del texto. Exacto si hay ``tiktoken``; heurístico si no."""
    enc = _try_tiktoken(model)
    if enc is not None:
        return len(enc.encode(text))
    return _heuristic(text)
