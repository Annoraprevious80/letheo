"""``Session`` — API ergonómica de alto nivel sobre ``letheo.Runtime``.

Pensada para que un agente (o un humano) hable con Mnemosyne como con un órgano de memoria,
no como con una base de datos.

Ejemplo:

    from letheo_orchestration import Session

    with Session() as mem:
        # Estilo declarativo (MQL ejecutable):
        mem.run('''
            PERCEIVE interaction FROM subject "user:Xolotl"
                     AS { act: purchase, object: shoes, hue: nocturnal }
            DISTILL subject "user:Xolotl" INTO intention_vector COMPRESSING BY semantic_variance
            EVOKE essence OF "user:Xolotl" WITHIN budget 800 tokens
        ''')

        # Estilo Python (azúcar sobre PERCEIVE):
        mem.perceive("user:Xolotl", act="purchase", object="running_shoes")

        # Recuerdo + prosa lista para un LLM:
        prompt_block = mem.prompt("user:Xolotl", token_budget=800)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import letheo

from .prose import to_prose
from .tokens import count_tokens, count_tokens_method


@dataclass(frozen=True)
class EvokeResult:
    """Recuerdo evocado: el ``CompressedContext`` crudo + su prosa + el conteo de tokens honesto."""

    context: Any   # letheo.CompressedContext (no anotamos para no acoplar al tipo PyO3)
    prose: str
    #: Tokens REALES del bloque de prosa que se inyecta en el LLM (no la heurística del runtime).
    prose_tokens: int = 0
    #: "tiktoken" (exacto) o "heuristic" (estimación) — para reportes honestos.
    token_method: str = "heuristic"

    def __str__(self) -> str:
        return self.prose

    def fits(self, budget: int) -> bool:
        """¿El texto real cabe en el presupuesto de tokens?"""
        return self.prose_tokens <= budget


class Session:
    """Una sesión cognitiva. Wrapper amistoso sobre ``letheo.Runtime``.

    Lleva un reloj lógico interno (``now``) que avanza con ``tick(seconds)``. Esto evita pasar
    ``now=...`` a cada llamada y mantiene el espíritu del *tiempo como coeficiente de entropía*:
    el reloj corre solo entre interacciones, no es una marca por evento.
    """

    DEFAULT_HALFLIFE = 7 * 24 * 3600.0   # una semana

    def __init__(
        self,
        *,
        halflife_secs: float | None = None,
        persist_path: str | os.PathLike | None = None,
    ) -> None:
        self._rt = letheo.Runtime()
        self._now: float = 0.0
        self._halflife = halflife_secs or self.DEFAULT_HALFLIFE
        self._subjects: set[str] = set()
        self._persist_path = os.fspath(persist_path) if persist_path is not None else None
        # Si hay ruta de persistencia y ya existe memoria, la rehidratamos al abrir.
        if self._persist_path:
            self.load(self._persist_path)

    # ── Persistencia ──────────────────────────────────────────────────────
    def save(self, path: str | os.PathLike | None = None) -> int:
        """Persiste la memoria de largo plazo (snapshot por sujeto). Devuelve cuántos se guardaron.

        Sin argumento usa ``persist_path`` (si se configuró en el constructor).
        """
        target = os.fspath(path) if path is not None else self._persist_path
        if not target:
            raise ValueError("save() necesita un path (o configura persist_path en el constructor)")
        return self._rt.save(target)

    def load(self, path: str | os.PathLike | None = None) -> int:
        """Rehidrata la memoria de largo plazo desde un directorio de snapshots.

        Devuelve cuántos arquetipos se cargaron; los sujetos restaurados quedan disponibles para
        ``evoke``/``breathe`` sin haber sido percibidos en esta sesión.
        """
        target = os.fspath(path) if path is not None else self._persist_path
        if not target:
            raise ValueError("load() necesita un path (o configura persist_path en el constructor)")
        n = self._rt.load(target)
        self._subjects.update(self._rt.subjects)
        return n

    # ── Contexto Pythonico ────────────────────────────────────────────────
    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Si se configuró persistencia y la sesión sale limpia, guardamos un snapshot.
        if self._persist_path and exc_type is None:
            self.save(self._persist_path)
        return None

    # ── Reloj ─────────────────────────────────────────────────────────────
    @property
    def now(self) -> float:
        return self._now

    def tick(self, seconds: float) -> "Session":
        """Avanza el reloj lógico. Encadenable."""
        if seconds < 0:
            raise ValueError("el tiempo no retrocede en Mnemosyne")
        self._now += seconds
        return self

    # ── Verbos biológicos ─────────────────────────────────────────────────
    def perceive(
        self,
        subject: str,
        *,
        salience: float = 0.7,
        halflife_secs: float | None = None,
        **traits: Any,
    ) -> "Session":
        """``PERCEIVE``: asimila un estímulo. Los kwargs son los rasgos (act, object, hue, ...)."""
        if not traits:
            raise ValueError("perceive() necesita al menos un rasgo (kwargs)")
        text = " ".join(f"{k} {v}" for k, v in sorted(traits.items()))
        self._rt.perceive(
            subject,
            text,
            salience=salience,
            halflife_secs=halflife_secs or self._halflife,
            now=self._now,
        )
        self._subjects.add(subject)
        return self

    def perceive_vector(
        self,
        subject: str,
        embedding,
        *,
        text: str = "",
        salience: float = 0.7,
        halflife_secs: float | None = None,
    ) -> "Session":
        """``PERCEIVE`` con un embedding precalculado (oráculo / Candle / sentence-transformers).

        ``text`` es la etiqueta léxica que la destilación retiene para nombrar el contenido.
        """
        self._rt.perceive_with_embedding(
            subject,
            list(embedding),
            text=text,
            salience=salience,
            halflife_secs=halflife_secs or self._halflife,
            now=self._now,
        )
        self._subjects.add(subject)
        return self

    def breathe(self, subjects: Iterable[str] | None = None) -> Any:
        """``DISTILL`` + ``FADE``: un ciclo de sueño. Por defecto, sobre todos los sujetos vistos."""
        targets = list(subjects) if subjects is not None else sorted(self._subjects)
        return self._rt.breathe(targets, now=self._now)

    def evoke(self, subject: str, *, token_budget: int = 800, model: str | None = None) -> EvokeResult:
        """``EVOKE``: recuerdo + prosa + conteo de tokens REAL del bloque inyectado.

        ``model`` selecciona el tokenizador (si ``tiktoken`` está instalado); si no, se usa una
        heurística conservadora sobre el texto real.
        """
        ctx = self._rt.evoke(subject, token_budget=token_budget, now=self._now)
        prose = to_prose(ctx)
        return EvokeResult(
            context=ctx,
            prose=prose,
            prose_tokens=count_tokens(prose, model=model),
            token_method=count_tokens_method(model),
        )

    # ── Ergonomía ─────────────────────────────────────────────────────────
    def prompt(self, subject: str, *, token_budget: int = 800) -> str:
        """Atajo: devuelve solo el bloque de prosa, listo para inyectar en un LLM."""
        return self.evoke(subject, token_budget=token_budget).prose

    # ── Capa-1 (hechos exactos) y memoria generativa ──────────────────────
    def remember(
        self,
        subject: str,
        text: str,
        *,
        provenance: str = "agent",
        salience: float = 0.9,
        halflife_secs: float | None = None,
    ) -> "Session":
        """Capa-1: registra un hecho episódico **verbatim** (sin pérdida), bajo la física del olvido."""
        self._rt.remember(
            subject,
            text,
            provenance=provenance,
            salience=salience,
            halflife_secs=halflife_secs or (30.0 * 86_400.0),
            now=self._now,
        )
        self._subjects.add(subject)
        return self

    def recall(self, subject: str, query: str, *, k: int = 3) -> list[tuple[str, str, float]]:
        """Capa-1: recupera los ``k`` hechos exactos más relevantes (y los refuerza)."""
        return self._rt.recall(subject, query, k=k, now=self._now)

    def evoke_unified(
        self, subject: str, query: str, *, token_budget: int = 800, fact_budget: int = 200
    ) -> dict:
        """``EVOKE`` unificado: carácter (capa-2) **y** nominal (capa-1) en una sola evocación."""
        return self._rt.evoke_unified(
            subject, query, token_budget=token_budget, fact_budget=fact_budget, now=self._now
        )

    def reflect(self, subject: str) -> list[dict]:
        """Insights de orden superior sobre el arco del sujeto (transiciones, revivals)."""
        return self._rt.reflect(subject)

    def dream_reflect(self, subject: str) -> int:
        """Sueño reflexivo: materializa los insights como hechos de alta salience (capa-1)."""
        return self._rt.dream_reflect(subject, now=self._now)

    def resonate(self, query: str, *, k: int = 5) -> list[str]:
        """Búsqueda por **similitud**: los ``k`` sujetos cuya esencia más resuena con la consulta.
        Usa el índice ANN (HNSW) a escala, Flat exacto por debajo del umbral. Para enrutar al sujeto
        más relevante (caso flota)."""
        return self._rt.resonate(query, k=k, now=self._now)

    def validate(self, mql_src: str) -> list[str]:
        """Valida semánticamente un programa MQL sin ejecutarlo. Lista vacía ⇒ válido."""
        return letheo.validate_mql(mql_src)

    def run(self, mql_src: str, *, validate: bool = True) -> list[dict]:
        """Ejecuta un programa MQL. Devuelve una lista de dicts (uno por sentencia).

        Con ``validate=True`` (por defecto) comprueba la semántica antes de ejecutar y lanza
        ``ValueError`` si hay problemas, en vez de ejecutar el programa a medias.
        """
        if validate:
            problems = letheo.validate_mql(mql_src)
            if problems:
                raise ValueError("MQL inválido:\n" + "\n".join(problems))
        return self._rt.execute_mql(mql_src, now=self._now)

    # ── Inspección ────────────────────────────────────────────────────────
    @property
    def short_term_len(self) -> int:
        return self._rt.short_term_len

    @property
    def long_term_len(self) -> int:
        return self._rt.long_term_len

    @property
    def subjects(self) -> list[str]:
        return sorted(self._subjects)

    @property
    def cache_stats(self) -> dict:
        """Estadísticas de la caché de embeddings: ``{hits, misses, entries, hit_rate}``.

        Útil para ver cuánto se ahorra: en flujos con hábitos repetidos, ``hit_rate`` tiende a 1.
        """
        return self._rt.cache_stats()
