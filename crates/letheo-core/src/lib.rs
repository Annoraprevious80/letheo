//! # letheo-core · El Cognitive Runtime de Mnemosyne
//!
//! No es una base de datos: es un organismo que **percibe, sueña, evoca y desvanece**.
//!
//! - [`entropy`]  — La física del olvido (`weight(t) = salience·e^(−λ·Δt)·(1+reinforcement)`), lazy.
//! - [`perception`] — Memoria volátil de corto plazo (`PERCEIVE`).
//! - [`synthesis`] — El "sueño": compresión semántica vía centroide + varianza (`DISTILL`).
//! - [`archetype`] — Memoria de largo plazo semántica (capa-2), anclaje de evolución (`IMPRINT`).
//! - [`factstore`] — Memoria episódica (capa-1): hechos verbatim con olvido, dedup e índice.
//! - [`evoke`]     — Resonancia semántica con token budget (`EVOKE`).
//! - [`reflection`] — Memoria generativa: insights del arco + compresión predictiva ("inteligencia = compresión").
//! - [`runtime`]   — El bucle que "respira" y orquesta las capas.
//! - [`vector`]    — Operaciones vectoriales (coseno, centroide), búsqueda Flat.
//!
//! Ver `docs/` para la física, la gramática MQL y el pipeline.

pub mod archetype;
pub mod entropy;
pub mod evoke;
pub mod factstore;
pub mod modes;
pub mod perception;
pub mod reflection;
pub mod runtime;
pub mod synthesis;
pub mod vector;

pub use archetype::{Archetype, ArchetypeStore, Resilience};
pub use entropy::{EntropyTrace, Tick};
pub use evoke::{
    approx_token_count, evoke, evoke_unified, ArcDetail, CompressedContext, EvokeRequest,
    UnifiedContext, DEFAULT_TOKENS_PER_VECTOR,
};
pub use factstore::{Fact, FactStore, RecalledFact, Remember};
pub use modes::{cluster_modes, Mode, ModeConfig, ModeSeed};
pub use perception::{Perception, PerceptionBuffer};
pub use reflection::{materialize, predictive_compression, reflect, Insight, PredictiveScore};
pub use runtime::{BreathReport, CognitiveRuntime, RuntimeConfig};
pub use synthesis::{distill, DistillConfig, IntentionVector};
