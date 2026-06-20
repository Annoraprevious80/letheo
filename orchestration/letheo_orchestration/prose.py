"""Generación de prosa narrativa a partir de un ``CompressedContext``.

El runtime entrega un contexto **estructurado** (núcleo, anomalías, hitos del arco). Para que un
LLM consumidor pueda usarlo como memoria, lo convertimos en un bloque de **prosa narrativa**
ultra-comprimida: lenguaje natural denso, no listas de eventos.

Local-first: este módulo NO llama a un LLM para resumir; *deriva* la prosa de los datos del arco.
La integración con un LLM (Claude/GPT) toma este bloque y lo inyecta en el prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ArcReading:
    """Lectura interpretada del arco evolutivo del sujeto."""

    direction: str    # "rising" | "falling" | "stable" | "volatile"
    drift_total: float
    drift_peak: float
    span_seconds: float
    inflexions: int    # número de cambios de signo en la derivada del drift


def _ascii_sparkline(values: Sequence[float]) -> str:
    """Mini-sparkline de 8 niveles a partir de una secuencia de números."""
    if not values:
        return ""
    levels = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return levels[len(levels) // 2] * len(values)
    return "".join(levels[min(len(levels) - 1, int((v - lo) / (hi - lo) * (len(levels) - 1)))] for v in values)


# Suelo de ruido: la ÚNICA constante dependiente del embedder. Por debajo de un desplazamiento
# coseno de este orden, el movimiento está por debajo de la resolución del modelo y se lee como
# estable. Calibrado para all-MiniLM-L6-v2, que comprime dominios afines y produce picos ~20x
# menores que un embedder ortogonal (la MISMA reversión: 0.375 con oráculo, 0.020 con MiniLM).
# El resto de la clasificación es scale-free (forma relativa, no magnitud absoluta). Ver docs/06 §8.nonies.
ARC_NOISE_FLOOR = 0.015


def read_arc(arc_points: Sequence[tuple[float, float]]) -> ArcReading:
    """Interpreta el arco como una lectura cualitativa de la trayectoria.

    La detección de *returned* (ida y vuelta) es **scale-free**: depende de la FORMA — el pico de
    excursión domina sobre el cambio neto y hubo exactamente una vuelta — no de la magnitud absoluta.
    Así un mismo arco se lee igual con un embedder ortogonal (picos ~0.4) o con uno que comprime
    los dominios (MiniLM, picos ~0.02). La única magnitud absoluta es `ARC_NOISE_FLOOR`.
    """
    if not arc_points:
        return ArcReading("stable", 0.0, 0.0, 0.0, 0)

    times = [t for (t, _) in arc_points]
    drifts = [d for (_, d) in arc_points]
    drift_total = drifts[-1] - drifts[0]
    drift_peak = max((abs(d) for d in drifts), default=0.0)
    span = times[-1] - times[0]

    # Conteo de inflexiones (cambios de signo en la derivada discreta).
    inflexions = 0
    deltas = [drifts[i + 1] - drifts[i] for i in range(len(drifts) - 1)]
    for i in range(len(deltas) - 1):
        if deltas[i] * deltas[i + 1] < 0:
            inflexions += 1

    if drift_peak < ARC_NOISE_FLOOR:
        # Bajo la resolución del embedder: no hay excursión ni dirección reales.
        direction = "stable"
    # Varias oscilaciones de sentido = volátil; UNA sola vuelta con pico que domina el cambio neto =
    # reversión (returned). El criterio es relativo (pico ≫ neto), no un umbral absoluto: un viaje de
    # ida y vuelta NO es "estable" aunque drift_total≈0. (Ver docs/06 §8.ter y §8.nonies.)
    elif inflexions >= 2:
        direction = "volatile"
    elif inflexions == 1 and drift_peak >= 2.0 * abs(drift_total):
        direction = "returned"
    elif drift_total > 0.05:
        direction = "rising"
    elif drift_total < -0.05:
        direction = "falling"
    else:
        direction = "stable"

    return ArcReading(direction, drift_total, drift_peak, span, inflexions)


_DIRECTION_NARRATIVE = {
    "rising": "su identidad ha derivado de forma sostenida hacia una dirección nueva",
    "falling": "su identidad ha regresado hacia el patrón inicial tras explorar otros territorios",
    "returned": "su identidad exploró una dirección distinta y regresó después hacia su patrón inicial",
    "stable": "su identidad se ha mantenido coherente con su línea base",
    "volatile": "su identidad ha oscilado entre direcciones contrapuestas a lo largo del período",
}


def _domain_trend(series: list[float]) -> str | None:
    """Clasifica la trayectoria de prevalencia de UN comportamiento a lo largo de las fases.

    `series` = fracción de actividad del dominio en cada hito (0..1). Devuelve una frase narrativa o
    `None` si nunca fue relevante o no hay patrón claro (para no meter ruido). Cierra el gap de
    reversión por dominio: el arco global no distingue qué comportamiento concreto volvió.
    """
    if not series or len(series) < 2:
        return None
    peak = max(series)
    # Suelo de pico: por debajo es ruido de fondo (dominios casi ausentes ~0.03), no un comportamiento.
    if peak < 0.07:
        return None
    # SCALE-FREE: normalizamos por el propio pico y clasificamos por FORMA, no por magnitud. Esencial
    # porque un dominio se fragmenta en varias plantillas → cada una con ~1/k de la prevalencia real,
    # pero la MISMA forma. (Mismo principio que `read_arc`; ver docs/06 §8.nonies y §11.)
    norm = [x / peak for x in series]
    n = len(norm)
    start, end = norm[0], norm[-1]
    interior_min = min(norm[1:-1]) if n > 2 else min(start, end)
    peak_idx = norm.index(1.0)
    if start >= 0.6 and interior_min <= 0.45 and end >= 0.6:
        return "decayó durante una temporada y después volvió"
    if end < 0.25:
        if peak_idx == 0:
            return "ha decaído respecto a su nivel anterior"
        return "tuvo un periodo de fuerte interés y luego desapareció por completo"
    if end >= 0.85 and start <= 0.55:
        return "ha ido creciendo de forma sostenida"
    # Presente sin patrón marcado: lo narramos igual (cobertura) — una pregunta sobre este dominio
    # necesita verlo aunque no haya subido/caído. El suelo de pico ya filtró el ruido.
    return "se ha mantenido como un interés presente"


# Stopwords mínimas (EN + ruido de dominio reseñas) para que los términos comunes sean informativos.
_STOP = {
    "this", "that", "with", "from", "have", "your", "just", "they", "them", "their", "what", "when",
    "which", "would", "could", "about", "there", "here", "into", "than", "then", "some", "more", "most",
    "very", "really", "much", "many", "like", "good", "great", "best", "well", "also", "even", "still",
    "movie", "movies", "film", "films", "review", "reviewed", "stars", "star", "watch", "watched",
    "story", "really", "dont", "didnt", "isnt", "thing", "things", "make", "made", "show", "shows",
    "text", "phrase",  # prefijos de clave de rasgo, no contenido (ver _clean)
}


def _common_terms(histograms, k: int = 4) -> list[str]:
    """Términos más recurrentes (frecuencia, ponderada por conteo) entre las etiquetas de los hitos.

    Mejora E: en vez de UN título representativo (poco informativo en datos item-céntricos, ver G2),
    deriva las palabras de contenido que se repiten a lo largo del arco. Es presentación (capa de prosa);
    el motor solo aporta los histogramas (campo aditivo). Honesto: es frecuencia + stopwords, no TF-IDF
    completo — suficiente para nombrar el patrón sin pretender más.
    """
    import re
    from collections import Counter
    c: Counter = Counter()
    for hist in histograms or []:
        for label, count in hist:
            for tok in re.findall(r"[a-zA-Záéíóúñ]+", str(label).lower()):
                if len(tok) >= 4 and tok not in _STOP:
                    c[tok] += int(count)
    return [w for w, _ in c.most_common(k)]


def _is_faded_peak(series: list[float]) -> bool:
    """¿Este comportamiento tuvo un pico real y ya se desvaneció? (importó en el pasado, hoy casi nulo).

    Es la señal de `Q_past_peak`: lo que un agente NO debe olvidar que *fue* importante. Scale-free
    (normaliza por el pico), mismo principio que `_domain_trend`.
    """
    if not series or len(series) < 2:
        return False
    peak = max(series)
    if peak < 0.07:  # nunca fue un comportamiento real (ruido de fondo)
        return False
    return (series[-1] / peak) < 0.25  # terminó muy por debajo de su propio pico


def to_prose(ctx, *, span_label: str = "el período observado") -> str:
    """Convierte un ``CompressedContext`` en un bloque de prosa para un LLM.

    El bloque está pensado para inyectarse tal cual en un prompt: encabezado, narrativa del arco,
    métricas de confianza (compresión, eventos representados) y un marcador de cierre.

    Args:
        ctx: instancia de ``letheo.CompressedContext``.
        span_label: texto humano para el período (p.ej. "el último año").
    """
    reading = read_arc(ctx.arc_points)
    spark = _ascii_sparkline([d for (_, d) in ctx.arc_points])

    direction_phrase = _DIRECTION_NARRATIVE.get(reading.direction, "su comportamiento ha evolucionado")

    lines = [
        f"≈ MEMORIA DESTILADA · sujeto «{ctx.subject}» · {span_label}",
        "",
        f"Durante {span_label}, {direction_phrase} (Δ acumulado = {reading.drift_total:+.2f}, "
        f"pico de cambio = {reading.drift_peak:.2f}).",
    ]

    # Contenido léxico: qué le ocupa AHORA (etiqueta del núcleo actual). Sin esto, la prosa narra la
    # deriva pero no nombra el tema — un LLM no podría decir "en qué anda". Ver docs/06 §8.bis.
    core_label = _clean(getattr(ctx, "core_label", ""))
    if core_label:
        lines.append(f"Ahora su comportamiento gravita en torno a: {core_label}.")

    # Mejora E (aditiva): términos recurrentes a lo largo del arco — nombran el PATRÓN, no un solo título.
    # Usa el campo nuevo `arc_label_histograms` si el binding lo expone; si no, no añade nada.
    terms = _common_terms(getattr(ctx, "arc_label_histograms", None))
    if len(terms) >= 2:
        lines.append("Términos recurrentes en su arco: " + ", ".join(terms) + ".")

    # Picos pasados ya desvanecidos (item D, docs/11): comportamientos que IMPORTARON y se apagaron.
    # Sección dedicada y PRONTO en el bloque para sobrevivir al recorte de budget — es el contenido que
    # responde Q_past_peak ("¿qué le importó antes y ya no?"), el único triunfo robusto del motor.
    faded = []
    faded_names = set()
    for label, series in (getattr(ctx, "domain_arcs", []) or []):
        name = _clean(label)
        if name and _is_faded_peak([float(x) for x in series]):
            faded.append(name)
            faded_names.add(name)
    if faded:
        lines.append("Picos pasados ya desvanecidos (importaron y hoy casi no aparecen — relevantes si "
                     "se pregunta por el pasado): " + ", ".join(f"«{n}»" for n in faded[:5]) + ".")

    # Evolución POR comportamiento: la dimensión que el arco global no captura — responde "¿volvió X?"
    # de un comportamiento concreto. Excluimos los ya listados como pico-pasado para no duplicar.
    trend_lines = []
    for label, series in (getattr(ctx, "domain_arcs", []) or []):
        name = _clean(label)
        if not name or name in faded_names:
            continue
        phrase = _domain_trend([float(x) for x in series])
        if phrase:
            trend_lines.append(f"  · «{name}»: {phrase}")
    if trend_lines:
        lines.append("Evolución por comportamiento:")
        lines.extend(trend_lines)

    # Trayectoria nombrada: la secuencia de temas dominantes a lo largo del arco (sin repetir
    # consecutivos), p.ej. "trail running → yoga → trail running" (capta reversiones).
    arc_labels = [_clean(x) for x in getattr(ctx, "arc_labels", []) if _clean(x)]
    path = _dedup_consecutive(arc_labels)
    if len(path) >= 2:
        lines.append("Trayectoria temática: " + " → ".join(path) + ".")

    if ctx.arc_points:
        lines.append(f"Firma temporal del arco: {spark}  ({len(ctx.arc_points)} hitos)")
        if reading.direction == "volatile":
            lines.append(
                f"Se observan {reading.inflexions} inflexiones de sentido — el sujeto cambió de "
                "dirección y volvió a cambiar; un agente cuidadoso debería sondear su estado actual "
                "antes de asumir continuidad."
            )

    if ctx.anomalies_included:
        anom = [_clean(x) for x in getattr(ctx, "anomaly_labels", []) if _clean(x)]
        named = f" (p.ej.: {'; '.join(anom[:3])})" if anom else ""
        lines.append(
            f"Persisten {ctx.anomalies_included} señales atípicas que no encajan con el núcleo "
            f"del comportamiento{named}; trátalas como hipótesis vivas, no como anécdotas "
            "descartables."
        )

    lines.extend([
        "",
        f"Esta memoria representa la huella destilada de {ctx.represented:,} interacciones "
        f"originales, condensadas a {ctx.vectors_returned} vectores densos "
        f"({_fmt_ratio(ctx.compression_ratio)}).",
        "",
        "≈ FIN MEMORIA",
    ])
    return "\n".join(lines)
    # Nota de honestidad: el conteo de tokens REAL del bloque lo expone Session.evoke().prose_tokens
    # (tokenizador real / heurística declarada). No afirmamos aquí un "≤ N" que el texto con
    # etiquetas léxicas puede no cumplir; el budget se gestiona donde existe el texto. Ver docs/05.


def _clean(label: str) -> str:
    """Normaliza una etiqueta léxica para la prosa: quita guiones bajos y el prefijo de rasgo.

    El bloque almacena el rasgo como ``"text <frase>"`` o ``"phrase frase_con_guiones"``; aquí lo
    dejamos legible. Es presentación, no semántica.
    """
    if not label:
        return ""
    s = label.replace("_", " ").strip()
    # Quita un prefijo de clave de rasgo redundante ("text ", "phrase ").
    for pref in ("text ", "phrase ", "act "):
        if s.startswith(pref):
            s = s[len(pref):]
    return s.strip()


def _dedup_consecutive(items: list[str]) -> list[str]:
    """Colapsa repeticiones consecutivas: [a,a,b,a] → [a,b,a] (preserva reversiones)."""
    out: list[str] = []
    for it in items:
        if not out or out[-1] != it:
            out.append(it)
    return out


def _fmt_ratio(r: float) -> str:
    if r >= 1000:
        return f"compresión {r/1000:.1f}k:1"
    return f"compresión {r:.1f}:1"
