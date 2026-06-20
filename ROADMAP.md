# Letheo · Roadmap & decisiones de diseño

> Documento vivo: qué hace el motor, qué sigue, y por qué se tomó cada decisión. La auditoría de
> realidad (qué es real, qué deuda hay) vive en [`docs/05-honest-assessment.md`](docs/05-honest-assessment.md).

## Hecho — el motor

Cada capacidad es aditiva y mantiene `cargo test --workspace` verde (hoy **144 passed, 0 failed,
2 ignored, 0 warnings**), bajo el invariante **VERDAD 100%** (cero mock/fake/hardcode en producto).

| Capacidad | Qué aporta |
|---|---|
| **Arquetipo multi-modal** | El comportamiento se descompone en **modos** (no una media ciega): cada modo tiene su propia física de olvido **y su propio drift** (trayectoria por-modo). Resonancia por modo. |
| **Retrieval físico** | Rankeo por `relevancia · weight(now)` (decay + salience + refuerzo nativos), sin coeficientes α/β/γ a dedo. |
| **Bicapa unificada** | Capa-1 episódica (`factstore`: hechos verbatim, dedup, olvido) + capa-2 semántica bajo **una sola física**. `EVOKE` unificado responde carácter Y nominal a un presupuesto, con coste de token medido/inyectado. |
| **MQL 2.0** | 7 verbos (PERCEIVE · DISTILL · EVOKE · FADE · IMPRINT · RECALL · REINFORCE) + `RESONATING WITH` + predicado vectorial `WHERE resonates > θ`. Nada parseado-e-ignorado. |
| **Memoria generativa** | Reflexión determinista (insights de transición/revival del arco, materializables como hechos) + compresión predictiva como métrica norte interna. |
| **Índice ANN (HNSW)** | A escala (`recall@10 ≥ 0.99` vs Flat exacto); `Retriever` Flat/HNSW por umbral con filtrado por vida. |
| **Persistencia** | JSON inspeccionable (ambas capas) + store embebido **`redb`** transaccional ACID, multi-tenant por sujeto. |
| **Física del refuerzo** | Rendimientos decrecientes + suelo de vida media: la frecuencia de uso no domina el ranking sin techo, nada se vuelve inmortal. |
| **Calibración** | Los umbrales (`θ_fade/θ_red/θ_anom`) se validan contra ground-truth sintético — no son constantes mágicas. |
| **SDK** | Todo el motor expuesto a Python (`import letheo` / `Session`): `perceive`/`breathe`/`evoke`/`remember`/`recall`/`evoke_unified`/`reflect`/`dream_reflect`/`resonate`, y `save`/`load` de ambas capas. |

## Lo que sigue

**Optimizaciones del motor** (no faltan piezas; son afinados a escala):
- Backend `redb` a `bincode` (hoy serde_json) — store más pequeño y rápido.
- Índice ANN incremental/amortizado (hoy se reconstruye al cambiar el tamaño).
- Batching de embeddings en ingesta (un forward por lote, no por evento).
- Lookup por sujeto O(1) (hoy `ArchetypeStore::get` es lineal).
- Compresión predictiva como **controlador**: auto-podar modos de bajo poder + auto-calibrar `θ_mode`/`θ_dedup`.
- `cluster_modes` con 2ª pasada (k-means esférico) para modos más nítidos.

**Substrato de flota** (lo consume un orquestador de agentes, no este motor):
- `run_project` principal→delegación→auditoría (coste del principal ~constante al escalar).
- Mercado de memoria entre agentes (provenance + reputación por reuso).
- "Construye X": el agente construye desde su memoria; LLM solo como fallback.

## 📌 Decisiones de diseño (registro inmutable)

| # | Decisión | Razón |
|---|----------|-------|
| D1  | Rust sobre C++ para el core | PyO3 + Candle + Tokio = stack coherente memory-safe sin GC |
| D2  | Lazy evaluation del peso | `e^x` por-tic es prohibitivo a escala |
| D3  | `semantic_variance` = coseno vs centroide, explícito | No caja negra |
| D4  | Flat search, no HNSW (por defecto) | Decenas de vectores por arquetipo; el ANN (`letheo-index`) entra a escala |
| D5  | Modelo desde disco vía `LETHEO_MODEL_DIR` | Bug hf-hub+ureq en Windows; coherente con local-first |
| D6  | `letheo-py` excluido del workspace | Mantiene `cargo test --workspace` hermético/offline |
| D7  | Arquetipo evolutivo con `ArcMilestone` | Sin esto, no hay trayectoria reconstruible |
| D8  | Ratio "vivido" en métricas, no solo "consolidado" | El voto del evento ya vive en el arquetipo aunque FADE lo barra |
| D9  | Ejecutor en crate separado | Evita acoplar core↔mql |
| D10 | `execute_mql` devuelve dicts por sentencia | Error por-sentencia no aborta el programa |
| D11 | `IMPRINT` consolida la esencia existente | Refuerza la física del arquetipo y sus modos (no es un no-op) |
| D12 | Persistencia JSON + store embebido | JSON inspeccionable/diffable; `redb` transaccional a escala |
| D13 | `read_arc` scale-free (pico ≥ 2·neto) | La magnitud de drift cambia entre embedders (MiniLM comprime) |
| D14 | "Soñar antes de avanzar el reloj" | Si avanzas antes del breathe, los estímulos cruzan FADE sin consolidar |
| D15 | Un solo cliente para OpenAI + DeepSeek | DeepSeek es API-compatible; cambiar base_url basta |
| D16 | Prosa al `system`, mensaje al `user` | El system *conoce* al usuario; el user dice algo ahora |
| D17 | No resumir con LLM la prosa | La prosa ya está destilada; resumir sería ruido |
| D18 | Conteo de tokens vía tiktoken con fallback declarado | Honesto: si no es exacto, lo decimos (`token_method`) |
| D21 | `domain_arcs` (histograma por dominio + hito) | Responder "¿volvió X?" de un comportamiento concreto |
| D22 | Clustering por embedding (no por texto) | Títulos únicos rompen la señal; los modos emergentes la rescatan |
| D23 | Modo representante = texto más cercano al centroide | Etiqueta interpretable por el LLM sin metadatos externos |
| D26 | Una sola física para las dos capas | Capa-1 episódica y capa-2 semántica comparten `EntropyTrace` (CLS literal) |
| D27 | Dedup de hechos por sujeto y dirección (`θ=0.95`) | La repetición consolida un hecho, no infla el store |
| D28 | Evocar es tocar: `recall` refuerza | Spaced repetition nativo (lo evocado sobrevive, lo demás decae) |
| D29 | Una evocación, un presupuesto (EVOKE unificado) | Carácter Y nominal sin coser dos sistemas en Python |
| D30 | El coste de token se mide o se inyecta, nunca se inventa | `tokens_per_vector` realimentable + `fact_cost` inyectado |
| D31 | Cada modo tiene `origin` fijo + `drift` | La resonancia ya usaba modos; la trayectoria también debe |
| D32 | Refuerzo con rendimientos decrecientes + suelo de λ | La frecuencia de uso no domina el ranking ni vuelve nada inmortal |
