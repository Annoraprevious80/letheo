"""Adapter LLM — la prosa de Mnemosyne inyectada en un LLM real.

Soporta OpenAI y DeepSeek (que es compatible con la API de OpenAI cambiando ``base_url``).
Local-first sigue siendo el default del runtime; el LLM solo se invoca cuando el agente lo pide
explícitamente.

Uso:

    from letheo_orchestration import Session
    from letheo_orchestration.llm import LLMClient, Memorist

    with Session() as mem:
        # ...seed de eventos...
        agent = Memorist(mem, LLMClient.openai())          # o LLMClient.deepseek()
        reply = agent.ask("user:Xolotl", "¿qué tal zapatos de trail?")
        print(reply)

Las credenciales se leen del entorno (``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY``) salvo que se
pasen explícitamente al constructor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .session import Session

if TYPE_CHECKING:
    from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Provider config: una sola clase, dos endpoints.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMConfig:
    """Configuración mínima del provider."""

    name: str
    base_url: str | None
    env_var: str
    default_model: str


OPENAI = LLMConfig(
    name="openai",
    base_url=None,                       # SDK usa su default
    env_var="OPENAI_API_KEY",
    default_model="gpt-4o-mini",
)

DEEPSEEK = LLMConfig(
    name="deepseek",
    base_url="https://api.deepseek.com",
    env_var="DEEPSEEK_API_KEY",
    # `deepseek-chat` (modo no-thinking de v4-flash) se retira el 2026-07-24; usamos el nombre nuevo.
    # Para razonamiento, pasar --model deepseek-v4-pro.
    default_model="deepseek-v4-flash",
)


# ──────────────────────────────────────────────────────────────────────────────
# Cliente delgado sobre el SDK de openai (que sirve para ambos).
# ──────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """Cliente unificado para OpenAI y DeepSeek (vía SDK ``openai``)."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Falta el SDK `openai`. Instala con: pip install 'letheo-orchestration[llm]' "
                "o pip install openai"
            ) from e

        key = api_key or os.environ.get(config.env_var)
        if not key:
            raise RuntimeError(
                f"No se encontró la API key para {config.name}. "
                f"Exporta {config.env_var} o pásala como api_key=..."
            )

        # `timeout` evita que una respuesta estancada cuelgue el proceso para siempre (el default del
        # SDK es 10 min ≈ "infinito" en un bucle de cientos de llamadas). `max_retries` reintenta
        # fallos transitorios (5xx, conexión cortada) con backoff exponencial del propio SDK.
        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout, "max_retries": max_retries}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client: OpenAI = OpenAI(**kwargs)
        self._model = model or config.default_model
        self.config = config
        # Contabilidad de tokens REALES (lo que reporta la API), thread-safe para el pool de la suite.
        import threading
        self._usage_lock = threading.Lock()
        self.usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}

    @classmethod
    def openai(cls, *, api_key: str | None = None, model: str | None = None) -> "LLMClient":
        return cls(OPENAI, api_key=api_key, model=model)

    @classmethod
    def deepseek(cls, *, api_key: str | None = None, model: str | None = None) -> "LLMClient":
        return cls(DEEPSEEK, api_key=api_key, model=model)

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system: str, user: str, *, temperature: float = 0.7) -> str:
        """Una sola ronda. Devuelve el texto de la respuesta."""
        text, _ = self.chat_with_usage(system, user, temperature=temperature)
        return text

    def chat_with_usage(self, system: str, user: str, *, temperature: float = 0.7):
        """Como `chat` pero devuelve también el uso de tokens de ESTA llamada: (texto, usage_dict).

        `usage_dict` = {prompt, completion, total}. Permite contabilidad granular (p. ej. por brazo).
        Acumula además en el total del cliente (`self.usage`).
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = {"prompt": 0, "completion": 0, "total": 0, "cache_hit": 0, "cache_miss": 0}
        # Contabilidad defensiva — nunca debe tumbar una llamada (el `usage` puede faltar, y en tests
        # con cliente mockeado ni existe el lock). Un fallo aquí se ignora.
        try:
            u = resp.usage
            # DeepSeek separa input cacheado (prefijo repetido, ~1/50 del precio) de no cacheado.
            hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
            miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
            prompt = int(u.prompt_tokens)
            if hit == 0 and miss == 0:  # API sin campos de caché → todo a precio de miss (conservador)
                miss = prompt
            usage = {"prompt": prompt, "completion": int(u.completion_tokens),
                     "total": int(u.total_tokens), "cache_hit": hit, "cache_miss": miss}
            with self._usage_lock:
                for k in ("prompt", "completion", "total", "cache_hit", "cache_miss"):
                    self.usage[k] += usage[k]
                self.usage["calls"] += 1
        except Exception:  # noqa: BLE001
            pass
        return (resp.choices[0].message.content or "").strip(), usage

    # Precios DeepSeek oficiales (USD por 1M tokens): (input_cache_miss, input_cache_hit, output).
    # Fuente: api-docs.deepseek.com/quick_start/pricing. El 75% off de v4-pro caducó 2026-05-31.
    PRICES_USD_PER_M = {
        "deepseek-v4-flash": (0.14, 0.0028, 0.28),
        "deepseek-v4-pro":   (1.74, 0.0145, 3.48),
        "deepseek-chat":     (0.14, 0.0028, 0.28),  # = modo no-thinking de v4-flash
    }

    def cost_of(self, cache_miss: int, cache_hit: int, completion: int) -> float:
        """Coste USD exacto separando input cacheado/no cacheado y output, según la tarifa del modelo."""
        miss_r, hit_r, out_r = self.PRICES_USD_PER_M.get(self._model, (0.14, 0.0028, 0.28))
        return cache_miss / 1e6 * miss_r + cache_hit / 1e6 * hit_r + completion / 1e6 * out_r

    def cost_estimate_usd(self) -> float:
        """Coste total según `usage` acumulado (input hit/miss + output)."""
        return self.cost_of(self.usage["cache_miss"], self.usage["cache_hit"], self.usage["completion"])


# ──────────────────────────────────────────────────────────────────────────────
# Memorist: el agente. La memoria de Mnemosyne va al system prompt; el mensaje al user.
# ──────────────────────────────────────────────────────────────────────────────


SYSTEM_TEMPLATE = (
    "Eres un asistente que conoce profundamente a la persona con la que hablas. "
    "Tu conocimiento NO viene de un historial de eventos: viene de una memoria destilada que "
    "resume su huella de comportamiento en pocos vectores densos. Confía en ella como confiarías "
    "en lo que tú mismo recuerdas de alguien cercano; no le pidas contexto que ya tienes. "
    "Sé concreto, personal y conciso.\n\n"
    "{memory_block}"
)


@dataclass
class AskResult:
    """Resultado de una interacción del agente."""

    reply: str
    memory_block: str
    model: str
    provider: str
    represented_events: int
    memory_tokens_estimate: int
    extras: dict = field(default_factory=dict)


class Memorist:
    """Un agente que recuerda usando Mnemosyne y razona con un LLM externo."""

    def __init__(
        self,
        session: Session,
        client: LLMClient,
        *,
        token_budget: int = 800,
    ) -> None:
        self.session = session
        self.client = client
        self.token_budget = token_budget

    def ask(
        self,
        subject: str,
        message: str,
        *,
        token_budget: int | None = None,
        temperature: float = 0.7,
    ) -> AskResult:
        """Recuerda la esencia del sujeto y responde al mensaje desde ese conocimiento."""
        budget = token_budget or self.token_budget
        evoked = self.session.evoke(subject, token_budget=budget)
        system = SYSTEM_TEMPLATE.format(memory_block=evoked.prose)
        reply = self.client.chat(system, message, temperature=temperature)
        return AskResult(
            reply=reply,
            memory_block=evoked.prose,
            model=self.client.model,
            provider=self.client.config.name,
            represented_events=evoked.context.represented,
            memory_tokens_estimate=evoked.context.token_estimate,
        )
