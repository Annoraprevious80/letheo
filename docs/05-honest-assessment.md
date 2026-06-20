# 05 · Auditoría de realidad (VERDAD 100%)

> Documento vivo, basado en **lectura del código fuente**, no en resúmenes de desarrollo ni en el
> marketing del README. Es el **libro mayor del invariante VERDAD 100%**: nada en el camino de
> producto puede ser mock, hardcode, proxy, placeholder o no-op. Si un valor no se mide o no se
> deriva, no se inventa: se calcula o se expone como configuración explícita. Si algo aquí deja de
> ser cierto porque se implementó, **actualiza este documento en el mismo commit**.

## TL;DR

La capa determinista del motor (física del olvido, álgebra vectorial, MQL, runtime, persistencia,
arquetipo multi-modal, factstore episódico, evoke unificado) es correcta y está bien probada
(`cargo test --workspace` verde). El foco ya no es *comparar* el motor con nada: es **desarrollarlo a
nivel Dios** para que lo consuma una flota de agentes (Paideia). Esta auditoría lista lo que es real y
las deudas de VERDAD 100% que quedan por saldar.

## Qué es REAL

| Componente | Archivo | Nota |
|---|---|---|
| Física del olvido (decay/refuerzo/consolidación) | `crates/letheo-core/src/entropy.rs` | `weight = salience·e^(−λΔt)·(1+r)`, lazy, bien testeado |
| Coseno / norma / centroide (vectorizado) | `crates/letheo-core/src/vector.rs` | equivalencia probada vs. referencia escalar |
| DISTILL: centroide + anomalías + **modos** | `crates/letheo-core/src/synthesis.rs`, `modes.rs` | clustering determinista leader/DP-means (sin RNG) |
| Arquetipo **multi-modal** (modos con física propia, resonancia por modo) | `crates/letheo-core/src/archetype.rs` | rompe el centroide único; `evolve` ponderado por volumen |
| Recuperación **física** (`relevancia · weight(now)`) | `archetype.rs::resonate` | sin coeficientes α/β/γ a mano |
| **FactStore** episódico (capa-1) en el core: verbatim + olvido + dedup + recall | `crates/letheo-core/src/factstore.rs` | bajo la misma `EntropyTrace`; reemplaza la lista Python |
| **EVOKE unificado** (gist capa-2 + hechos capa-1, un presupuesto, coste inyectado) | `crates/letheo-core/src/evoke.rs::evoke_unified` | responde carácter Y nominal |
| MQL: léxer, parser, validación, ejecutor | `crates/letheo-{mql,exec}` | 5 verbos, predicados WHERE reales, errores tipados |
| Runtime async (actor Tokio), persistencia JSON (arquetipos + hechos), caché de embeddings | `crates/letheo-{async,persist,inference}` | sólidos y testeados offline |
| `CandleProvider` (BERT all-MiniLM, embeddings reales) | `crates/letheo-inference/src/candle_provider.rs` | el binding es candle-only por `compile_error!` si falta la feature |
| Conteo de tokens REAL del bloque inyectado | `orchestration/letheo_orchestration/tokens.py` | tiktoken si está; heurística calibrada declarada vía `token_method` |
| Calibración de umbrales (`θ_fade/θ_red/θ_anom`) contra ground-truth | `crates/letheo-calibration` | demuestra que los defaults no son mágicos |

## Deudas de VERDAD 100% (pendientes)

Cada una es un fake/no-op/parseado-e-ignorado a saldar (estado en `ROADMAP.md`).

| # | Deuda | Dónde | Estado |
|---|---|---|---|
| 1 | `TOKENS_PER_VECTOR = 24` baked | `evoke.rs` | ✅ **saldada** (L6): `EvokeRequest::tokens_per_vector` (default declarado, realimentable desde tiktoken) + `fact_cost` inyectado en `evoke_unified` |
| 3 | `MockProvider` (bag-of-tokens FNV, semántica falsa) | `letheo-inference` | ✅ **confinado**: `#[cfg(any(test, feature = "testing"))]` → no compila en ningún build de producto (el CLI usa `CandleProvider` real; `testing` es solo dev-dependency). El rename es cosmético; la garantía es el cfg-gate |
| 4 | `domain_trajectories(a, 10, …)` cap mágico `10` | `evoke.rs` | ✅ **derivado del budget** (`budget_vectors`): a más presupuesto, más dominios caben; ya no es constante |
| 5 | Vidas medias 30/180/720 días baked-in | `archetype.rs` | ✅ **declaradas**: `pub const HALFLIFE_{LOW,MEDIUM,HIGH}_SECS` documentadas como física (mismo idioma que `DEFAULT_THETA_FADE`…), no literales mágicos en un `match` |
| 6 | `IMPRINT` = no-op (marcador) | `letheo-exec` | ✅ **saldada** (L7): IMPRINT consolida/ancla de verdad el arquetipo (refuerza su física y la de sus modos; Δt→0, λ reducido). `ArchetypeStore::consolidate` |
| 7 | `CandleProvider` devuelve vector cero en error (fallback silencioso) | `candle_provider.rs` | ✅ **falla ruidoso**: un fallo de inferencia con el modelo ya cargado hace `panic!` con contexto; nunca un embedding cero silencioso que contamine centroides/resonancias |
| 9 | `RESONATING WITH { traits }` parseado pero **ignorado** | `letheo-exec`, `letheo-mql` | ✅ **saldada** (L7): embebe los rasgos y enfoca la evocación en el modo que resuena (`Archetype::resonant_mode_label`; `CompressedContext.resonating_mode`) |
| 10 | TODO/stub/heurística residual en crates de producto | barrido global | ✅ **barrido hecho**: el grep de cierre no devuelve hits reales en `crates/` (solo "todo" español y un nombre de test que afirma *no*-hardcoding); reescritos los comentarios "MVP"/deuda; **retirada la cláusula `WHEN` de IMPRINT** (era parseada-e-ignorada); mocks solo bajo `#[cfg(test)]` |

> **Ledger saldado.** Todas las deudas de VERDAD 100% (#1·#3·#4·#5·#6·#7·#9·#10) están resueltas; #2 lo
> cubre #1 (el conteo real de tokens del bloque vive en `tokens.py` con tiktoken) y #8 cayó con la poda L0
> (la inflación de métricas por duplicación y todo el aparato de comparación/duelo).

## ¿Necesitamos LLMs?

No para el núcleo (es matemática autocontenida; el LLM solo consume el bloque evocado). Las
afirmaciones *semánticas* dependen del **embedder real** (Candle), que no es un LLM generativo. El LLM
generativo es opcional: solo redacta la prosa de `EVOKE` a partir del contexto ya destilado.

## Veredicto

Fundación honesta y bien hecha, ahora con la bicapa unificada en el core. La dirección es construir el
**motor de memoria a nivel Dios** bajo VERDAD 100% para que Paideia lo use como substrato. El progreso
se mide por **corrección del motor** (tests deterministas), no por comparación con baselines.
