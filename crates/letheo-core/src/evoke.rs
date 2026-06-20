//! `EVOKE` — resonancia semántica con token budget.
//!
//! No recorre la historia: resuena con los arquetipos y reconstruye la esencia dentro de un
//! presupuesto de tokens, devolviendo un **bloque de contexto ultra-comprimido** (no una lista de
//! eventos). Ver `docs/03-engine-pipeline.md`.

use crate::archetype::{Archetype, ArchetypeStore};
use crate::entropy::{Tick, DEFAULT_THETA_FADE};
use crate::factstore::{FactStore, RecalledFact};
use crate::vector::cosine;

/// Cuánto detalle del arco evolutivo devolver. Mapea las cláusulas MQL `RESOLUTION`/`PROJECTING`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArcDetail {
    /// Arco completo, recortado al presupuesto (`RESOLUTION arc` / `PROJECTING trajectory`).
    #[default]
    Full,
    /// Solo unos pocos hitos clave (`RESOLUTION summary`).
    Summary,
    /// Sin arco: solo el estado actual (`RESOLUTION point` / `PROJECTING snapshot`).
    None,
}

/// Parámetros de una evocación.
#[derive(Debug, Clone)]
pub struct EvokeRequest {
    pub subject: String,
    /// Presupuesto de tokens: el contexto devuelto no debe excederlo.
    pub token_budget: usize,
    /// Solo hitos del arco con `at ≥ since` (`None` = todo el arco). Mapea `ACROSS span`.
    pub since: Option<Tick>,
    /// Detalle del arco a devolver. Mapea `RESOLUTION`/`PROJECTING`.
    pub arc_detail: ArcDetail,
    /// Coste en tokens de un vector denso renderizado. **Default declarado**
    /// ([`DEFAULT_TOKENS_PER_VECTOR`]), realimentable con la media real medida por el tokenizer
    /// (tiktoken) en la capa de orquestación: el 24 deja de estar baked en el algoritmo (deuda #1 de
    /// VERDAD 100% — el coste se mide o se inyecta, nunca se inventa).
    pub tokens_per_vector: usize,
}

impl EvokeRequest {
    /// Evocación con los valores por defecto (arco completo, sin ventana temporal).
    pub fn new(subject: impl Into<String>, token_budget: usize) -> Self {
        Self {
            subject: subject.into(),
            token_budget,
            since: None,
            arc_detail: ArcDetail::Full,
            tokens_per_vector: DEFAULT_TOKENS_PER_VECTOR,
        }
    }
}

/// El resultado de `EVOKE`: contexto denso y su métrica de compresión. La capa de orquestación lo
/// convierte en prosa para el LLM; el core entrega la estructura.
#[derive(Debug, Clone)]
pub struct CompressedContext {
    pub subject: String,
    /// Nº de percepciones que esta esencia representa (numerador del ratio).
    pub represented: usize,
    /// Nº de vectores devueltos (núcleo + anomalías + hitos del arco). Denominador del ratio.
    pub vectors_returned: usize,
    /// Anomalías (quiebres de patrón) incluidas, recortadas al presupuesto.
    pub anomalies_included: usize,
    /// Hitos del arco evolutivo incluidos (t, deriva): la trayectoria del sujeto en el tiempo.
    pub arc_points: Vec<(f64, f32)>,
    /// Etiqueta léxica del comportamiento dominante **actual** (qué le interesa ahora).
    pub core_label: String,
    /// Etiquetas léxicas alineadas con `arc_points`: qué ocupaba al sujeto en cada hito.
    pub arc_labels: Vec<String>,
    /// Etiquetas léxicas de las anomalías incluidas.
    pub anomaly_labels: Vec<String>,
    /// Trayectorias **por comportamiento**: para los dominios más prevalentes, su fracción de
    /// actividad en cada hito del arco. Permite narrar "X subió, cayó y volvió" de un comportamiento
    /// concreto, no solo del centroide global. Cierra el gap de reversión por dominio (docs/06 §11).
    pub domain_arcs: Vec<(String, Vec<f32>)>,
    /// Histograma `(etiqueta, conteo)` de cada hito devuelto, ALINEADO con `arc_points`. Aditivo
    /// (Mejora E): permite a la capa de prosa derivar una etiqueta por **términos comunes** (TF-IDF)
    /// en vez de un único texto representativo, sin que el core decida la presentación. No altera
    /// ninguna otra señal del motor.
    pub arc_label_histograms: Vec<Vec<(String, usize)>>,
    /// Estimación de tokens del bloque, garantizada ≤ token_budget.
    pub token_estimate: usize,
    /// Etiqueta del modo en el que se enfocó la evocación por `EVOKE … RESONATING WITH { rasgo }`
    /// (el aspecto del sujeto que resuena con el rasgo). `None` si no hubo cláusula RESONATING WITH.
    pub resonating_mode: Option<String>,
    /// **Trayectoria por-modo**: `(etiqueta, drift)` de cada modo vivo — cuánto se ha desplazado ese
    /// comportamiento desde que nació. Complementa a `arc_points` (drift del centroide global) con la
    /// evolución *de cada modo*, no de la media ciega. Ver [`crate::Mode::drift`].
    pub mode_drifts: Vec<(String, f32)>,
}

impl CompressedContext {
    /// Ratio de compresión: percepciones representadas / vectores devueltos.
    pub fn compression_ratio(&self) -> f64 {
        if self.vectors_returned == 0 {
            return 0.0;
        }
        self.represented as f64 / self.vectors_returned as f64
    }
}

/// Coste por defecto, en tokens, de un vector denso al renderizarse para el LLM. Es un **default
/// declarado**, no una constante mágica enterrada en el algoritmo: [`EvokeRequest::tokens_per_vector`]
/// lo sobre-escribe con la media real medida por tiktoken en la capa de orquestación. (Resuelve la
/// deuda #1 de VERDAD 100%: el coste se mide o se inyecta, nunca se inventa.)
pub const DEFAULT_TOKENS_PER_VECTOR: usize = 24;

/// `EVOKE`: resuelve la esencia de un sujeto dentro del presupuesto de tokens.
///
/// Devuelve `None` si el sujeto no tiene arquetipo vivo.
pub fn evoke(store: &ArchetypeStore, req: &EvokeRequest, now: Tick) -> Option<CompressedContext> {
    let a: &Archetype = store.get(&req.subject)?;
    if a.trace.weight(now) < DEFAULT_THETA_FADE {
        return None; // su esencia se ha desvanecido
    }

    // Asignación del presupuesto: núcleo (1) + arco (la trayectoria es la firma del sujeto en el
    // tiempo, prioritaria sobre las anomalías) + anomalías sueltas con lo que sobre.
    let budget_vectors = req.token_budget / req.tokens_per_vector.max(1);
    let core_vectors = 1usize;
    let base_arc_quota = (budget_vectors.saturating_sub(core_vectors)) * 2 / 3;
    // El detalle pedido (RESOLUTION/PROJECTING) modula cuánto arco devolvemos.
    let arc_quota = match req.arc_detail {
        ArcDetail::None => 0,
        ArcDetail::Summary => base_arc_quota.min(4),
        ArcDetail::Full => base_arc_quota,
    };
    let (arc_points, arc_labels, arc_label_histograms) = arc_signature(a, arc_quota, req.since);
    let arc_count = arc_points.len();

    let room_for_anomalies = budget_vectors
        .saturating_sub(core_vectors)
        .saturating_sub(arc_count);
    let anomalies_included = a.anomalies.len().min(room_for_anomalies);
    // Etiquetas de las anomalías incluidas (alineadas; robustas si hay menos labels que vectores).
    let anomaly_labels: Vec<String> = a
        .anomaly_labels
        .iter()
        .take(anomalies_included)
        .cloned()
        .collect();

    let vectors_returned = core_vectors + arc_count + anomalies_included;
    let token_estimate = vectors_returned * req.tokens_per_vector;

    // Trayectorias por dominio: los más relevantes por **pico** (no por popularidad acumulada, ver
    // `domain_trajectories`). El cap se **deriva del presupuesto** (`budget_vectors`), no es una
    // constante mágica: a más budget, más dominios caben en la prosa. (Deuda #4 de VERDAD 100% saldada.)
    let domain_arcs = if req.arc_detail == ArcDetail::None {
        Vec::new()
    } else {
        domain_trajectories(a, budget_vectors.max(1), req.since)
    };

    Some(CompressedContext {
        subject: a.subject.clone(),
        represented: a.represented,
        vectors_returned,
        anomalies_included,
        arc_points,
        core_label: a.core_label.clone(),
        arc_labels,
        anomaly_labels,
        domain_arcs,
        arc_label_histograms,
        token_estimate,
        resonating_mode: None, // lo fija el ejecutor si la sentencia trae RESONATING WITH.
        // Trayectoria por-modo: el drift de cada modo vivo (cuánto cambió ese comportamiento desde su origen).
        mode_drifts: a
            .modes
            .iter()
            .filter(|m| m.trace.weight(now) >= DEFAULT_THETA_FADE)
            .map(|m| (m.label.clone(), m.drift()))
            .collect(),
    })
}

/// Reconstruye, para los `max_domains` comportamientos más relevantes, su **fracción de actividad
/// en cada hito** del arco (serie temporal normalizada por hito). De aquí sale "¿subió/cayó/volvió X?".
///
/// **Ranking (Mejora B, 2026-06-12).** Antes usábamos `total_count` (suma de apariciones). Eso
/// aplanaba los picos: un dominio que apareció uniformemente en todas las fases ganaba siempre, y
/// uno que tuvo un PICO claro en una sola fase quedaba fuera del top-K. Para preguntas tipo
/// "¿fue X importante alguna vez aunque ya no?" eso destruía la señal en verticales item-céntricos
/// (títulos únicos que no se repiten entre fases).
///
/// Nuevo ranking: `score = max_phase × (1 + variance_across_phases)`. Eleva los dominios que
/// tuvieron un pico claro (`max_phase` alto) Y se concentraron en pocas fases (`variance` alto).
/// Castiga a los uniformes y rescata a los desaparecidos.
fn domain_trajectories(
    a: &Archetype,
    max_domains: usize,
    since: Option<Tick>,
) -> Vec<(String, Vec<f32>)> {
    use std::collections::HashSet;
    let milestones: Vec<&crate::archetype::ArcMilestone> = a
        .arc
        .iter()
        .filter(|m| since.is_none_or(|s| m.at >= s))
        .collect();
    if milestones.is_empty() {
        return Vec::new();
    }
    // Universo de etiquetas que aparecieron alguna vez en algún hito.
    let mut universe: HashSet<&str> = HashSet::new();
    for m in &milestones {
        for (label, _) in &m.label_histogram {
            universe.insert(label.as_str());
        }
    }
    // Pre-calcula totales por hito (denominadores) una sola vez.
    let phase_totals: Vec<usize> = milestones
        .iter()
        .map(|m| m.label_histogram.iter().map(|(_, c)| *c).sum())
        .collect();

    // Para cada dominio, computamos la serie normalizada Y el score por pico+varianza.
    let mut scored: Vec<(&str, Vec<f32>, f32)> = universe
        .into_iter()
        .map(|label| {
            let series: Vec<f32> = milestones
                .iter()
                .zip(phase_totals.iter())
                .map(|(m, &total)| {
                    let c = m
                        .label_histogram
                        .iter()
                        .find(|(l, _)| l.as_str() == label)
                        .map(|(_, c)| *c)
                        .unwrap_or(0);
                    if total > 0 {
                        c as f32 / total as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let max_phase = series.iter().cloned().fold(0.0_f32, f32::max);
            // Varianza poblacional simple. Con n=4 fases es estable y barata.
            let n = series.len().max(1) as f32;
            let mean = series.iter().sum::<f32>() / n;
            let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let score = max_phase * (1.0 + variance);
            (label, series, score)
        })
        .collect();

    // Orden estable: por score desc, desempate alfabético para reproducibilidad.
    scored.sort_by(|(la, _, sa), (lb, _, sb)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| la.cmp(lb))
    });
    scored.truncate(max_domains);

    scored
        .into_iter()
        .map(|(label, series, _)| (label.to_string(), series))
        .collect()
}

/// Reduce el arco a `quota` hitos (por *thinning* uniforme) y los proyecta como
/// `(t, deriva_acumulada)` — deriva = `1 - cos(hito_i, hito_0)` ∈ [0, 2]. Hito_0 (origen absoluto
/// de la identidad) sirve de referencia, aunque `since` recorte qué hitos se reportan.
type ArcSignature = (Vec<(f64, f32)>, Vec<String>, Vec<Vec<(String, usize)>>);

fn arc_signature(a: &Archetype, quota: usize, since: Option<Tick>) -> ArcSignature {
    if quota == 0 || a.arc.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let origin = &a.arc[0].direction;
    // `ACROSS span`: solo los hitos dentro de la ventana temporal pedida.
    let window: Vec<&crate::archetype::ArcMilestone> = a
        .arc
        .iter()
        .filter(|m| since.is_none_or(|s| m.at >= s))
        .collect();
    if window.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let n = window.len();
    let mut points = Vec::with_capacity(quota.min(n));
    let mut labels = Vec::with_capacity(quota.min(n));
    let mut histograms = Vec::with_capacity(quota.min(n));
    let mut push = |m: &crate::archetype::ArcMilestone| {
        points.push((m.at, 1.0 - cosine(&m.direction, origin)));
        labels.push(m.label.clone());
        histograms.push(m.label_histogram.clone());
    };
    if n <= quota {
        for m in &window {
            push(m);
        }
    } else {
        // Thinning uniforme: `quota` hitos equiespaciados (incluye primero y último de la ventana).
        for i in 0..quota {
            let idx = (i * (n - 1)) / (quota - 1).max(1);
            push(window[idx]);
        }
    }
    (points, labels, histograms)
}

// ─────────────────────────────────────────────────────────────────────────────
// EVOKE unificado (L6): la bicapa en una sola evocación, bajo un solo presupuesto.
// ─────────────────────────────────────────────────────────────────────────────

/// Estimación de tokens de un texto contando palabras (unidades separadas por espacios). Es una
/// **medición real del texto**, declarada como aproximación: el conteo exacto del tokenizer del LLM
/// se inyecta vía el parámetro `fact_cost` de [`evoke_unified`]. No es una constante inventada — o se
/// mide el texto, o se inyecta tiktoken.
pub fn approx_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Contexto **unificado**: la firma caracterológica (capa-2, gist) y los hechos episódicos exactos
/// (capa-1) que UNA evocación reúne bajo UN presupuesto. Responde carácter Y nominal sin coser dos
/// sistemas a mano en la capa de orquestación — es la bicapa de *Complementary Learning Systems*
/// expuesta por el motor en una sola consulta.
#[derive(Debug, Clone)]
pub struct UnifiedContext {
    /// La esencia caracterológica (capa-2). `None` si no hay arquetipo vivo o el presupuesto restante
    /// no cubre ni el núcleo (los hechos lo coparon).
    pub gist: Option<CompressedContext>,
    /// Hechos exactos (capa-1) incluidos, ordenados por score físico (`relevancia · vida`).
    pub facts: Vec<RecalledFact>,
    /// Tokens consumidos por los hechos (suma del `fact_cost` real de cada texto incluido).
    pub fact_tokens: usize,
    /// Tokens totales del bloque (gist + hechos). Garantizado ≤ `req.token_budget`.
    pub total_tokens: usize,
}

/// `EVOKE` unificado: reparte UN presupuesto entre la capa-1 (hechos exactos) y la capa-2 (gist).
///
/// `fact_budget` (recortado a `req.token_budget`) es la porción reservada a hechos; el resto va al
/// gist. Los hechos se eligen **greedy por score físico** (ver [`FactStore::search`]) rellenando hasta
/// donde quepan bajo `fact_budget` según `fact_cost` — el tokenizer real inyectado por el llamante. El
/// gist se evoca con el presupuesto sobrante y solo si alcanza al menos para el núcleo. Es **read-only**
/// (componible, sin efectos): para el refuerzo por evocación (spaced repetition) usar
/// [`FactStore::recall`]. Garantiza `total_tokens ≤ req.token_budget`.
#[allow(clippy::too_many_arguments)]
pub fn evoke_unified(
    archetypes: &ArchetypeStore,
    facts: &FactStore,
    req: &EvokeRequest,
    query: &[f32],
    fact_budget: usize,
    now: Tick,
    fact_cost: impl Fn(&str) -> usize,
) -> UnifiedContext {
    let fact_budget = fact_budget.min(req.token_budget);

    // Capa-1: hechos por score físico, knapsack greedy bajo `fact_budget` con coste real medido.
    // (Greedy por score rellenando huecos: aproximación honesta, no se afirma óptima.)
    let ranked = facts.search(&req.subject, query, facts.len(), now, DEFAULT_THETA_FADE);
    let mut chosen = Vec::new();
    let mut fact_tokens = 0usize;
    for (score, f) in ranked {
        let cost = fact_cost(&f.text);
        if fact_tokens + cost <= fact_budget {
            fact_tokens += cost;
            chosen.push(RecalledFact {
                text: f.text.clone(),
                provenance: f.provenance.clone(),
                score,
            });
        }
    }

    // Capa-2: el gist con lo que sobra, solo si cubre al menos el núcleo (si no, los hechos coparon).
    let gist_budget = req.token_budget - fact_tokens;
    let gist = if gist_budget >= req.tokens_per_vector {
        let gist_req = EvokeRequest {
            token_budget: gist_budget,
            ..req.clone()
        };
        evoke(archetypes, &gist_req, now)
    } else {
        None
    };

    let total_tokens = fact_tokens + gist.as_ref().map_or(0, |g| g.token_estimate);
    UnifiedContext {
        gist,
        facts: chosen,
        fact_tokens,
        total_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archetype::{ArchetypeStore, Resilience};
    use crate::factstore::FactStore;
    use crate::synthesis::IntentionVector;

    fn store_with(absorbed: usize, anomalies: usize) -> ArchetypeStore {
        let mut s = ArchetypeStore::new();
        s.imprint(
            &IntentionVector {
                subject: "user:Xolotl".into(),
                centroid: vec![1.0, 0.0],
                anomalies: vec![vec![0.0, 1.0]; anomalies],
                core_label: "core".into(),
                anomaly_labels: vec!["novelty".into(); anomalies],
                absorbed,
                redundant: 0,
                label_histogram: vec![("core".into(), absorbed)],
                modes: vec![],
            },
            Resilience::High,
            0.0,
        );
        s
    }

    #[test]
    fn evoke_unknown_subject_is_none() {
        let s = ArchetypeStore::new();
        let req = EvokeRequest::new("ghost", 800);
        assert!(evoke(&s, &req, 0.0).is_none());
    }

    #[test]
    fn unified_evoke_answers_character_and_nominal_under_one_budget() {
        // Capa-2: un sujeto con esencia consolidada (gist de carácter).
        let archetypes = store_with(10_000, 2);
        // Capa-1: un hecho exacto — lo nominal que el gist (una media) nunca podría guardar.
        let mut facts = FactStore::new();
        facts.remember(
            "user:Xolotl",
            "API key is sk-ABC123",
            vec![0.0, 1.0],
            "agent",
            1.0,
            86_400.0,
            0.0,
        );

        let req = EvokeRequest::new("user:Xolotl", 800);
        let u = evoke_unified(
            &archetypes,
            &facts,
            &req,
            &[0.0, 1.0],
            100,
            0.0,
            approx_token_count,
        );

        // Una sola evocación responde AMBAS clases de pregunta:
        assert!(u.gist.is_some(), "carácter: la capa-2 está presente");
        assert_eq!(u.facts.len(), 1, "nominal: el hecho exacto entró");
        assert_eq!(
            u.facts[0].text, "API key is sk-ABC123",
            "capa-1 sin pérdida, verbatim"
        );
        // …bajo UN presupuesto, respetado.
        assert!(u.total_tokens <= 800, "total {} ≤ 800", u.total_tokens);
        assert_eq!(
            u.total_tokens,
            u.fact_tokens + u.gist.as_ref().unwrap().token_estimate
        );
    }

    #[test]
    fn unified_evoke_fact_cost_is_injected_not_hardcoded() {
        // Sin arquetipo: aislamos la lógica de capa-1 (el gist será None, no afecta a los hechos).
        let archetypes = ArchetypeStore::new();
        let mut facts = FactStore::new();
        facts.remember("u", "one two", vec![1.0, 0.0], "a", 1.0, 86_400.0, 0.0);
        facts.remember("u", "three four", vec![0.7, 0.7], "a", 1.0, 86_400.0, 0.0);
        facts.remember("u", "five six", vec![0.0, 1.0], "a", 1.0, 86_400.0, 0.0);
        let req = EvokeRequest::new("u", 800);
        let q = [0.7, 0.7];

        // Coste barato (1 token/hecho) → caben los 3 en fact_budget=3.
        let cheap = evoke_unified(&archetypes, &facts, &req, &q, 3, 0.0, |_| 1);
        assert_eq!(cheap.facts.len(), 3);
        assert_eq!(cheap.fact_tokens, 3);

        // Mismo presupuesto, coste caro (5 > 3) → no cabe ninguno. El coste manda, no un 24 baked.
        let pricey = evoke_unified(&archetypes, &facts, &req, &q, 3, 0.0, |_| 5);
        assert_eq!(pricey.facts.len(), 0);

        // Presupuesto 5, coste 5 → cabe exactamente uno (el de mayor score físico).
        let one = evoke_unified(&archetypes, &facts, &req, &q, 5, 0.0, |_| 5);
        assert_eq!(one.facts.len(), 1);
    }

    #[test]
    fn unified_skips_gist_when_facts_consume_budget() {
        let archetypes = store_with(10_000, 0); // gist disponible para "user:Xolotl"
        let mut facts = FactStore::new();
        facts.remember("user:Xolotl", "x", vec![1.0, 0.0], "a", 1.0, 86_400.0, 0.0);
        let req = EvokeRequest::new("user:Xolotl", 30); // presupuesto pequeño

        // El hecho cuesta 25 → quedan 5 < tokens_per_vector(24) ⇒ el gist se omite, sin reventar el budget.
        let u = evoke_unified(&archetypes, &facts, &req, &[1.0, 0.0], 30, 0.0, |_| 25);
        assert_eq!(u.facts.len(), 1);
        assert!(
            u.gist.is_none(),
            "los hechos coparon el presupuesto: sin gist"
        );
        assert_eq!(u.total_tokens, 25);
        assert!(u.total_tokens <= 30);
    }

    #[test]
    fn evoke_respects_token_budget() {
        let s = store_with(100_000, 1000);
        let req = EvokeRequest::new("user:Xolotl", 800);
        let ctx = evoke(&s, &req, 0.0).unwrap();
        assert!(
            ctx.token_estimate <= 800,
            "el contexto cabe en el presupuesto"
        );
    }

    #[test]
    fn evoke_achieves_massive_compression() {
        let s = store_with(100_000, 5);
        let req = EvokeRequest::new("user:Xolotl", 800);
        let ctx = evoke(&s, &req, 0.0).unwrap();
        assert_eq!(ctx.represented, 100_000);
        assert!(ctx.compression_ratio() > 1000.0, "compresión >> 1000:1");
    }

    #[test]
    fn evoke_reports_per_mode_drift() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};
        let mut s = ArchetypeStore::new();
        // Ciclo 1 en [1,0] → nace el modo "x" (origin [1,0]).
        let ps1: Vec<Perception> = (0..6)
            .map(|_| Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "x"))
            .collect();
        let r1: Vec<&Perception> = ps1.iter().collect();
        s.imprint(
            &distill("u", &r1, DistillConfig::default()).unwrap(),
            Resilience::High,
            0.0,
        );
        // Ciclo 2 desplazado a [0.6,0.8] (funde en el mismo modo) → el modo deriva.
        let ps2: Vec<Perception> = (0..6)
            .map(|_| Perception::new("u", vec![0.6, 0.8], 1.0, 3600.0, 0.0).with_trait("act", "x"))
            .collect();
        let r2: Vec<&Perception> = ps2.iter().collect();
        s.imprint(
            &distill("u", &r2, DistillConfig::default()).unwrap(),
            Resilience::High,
            1.0,
        );

        let ctx = evoke(&s, &EvokeRequest::new("u", 800), 1.0).unwrap();
        assert_eq!(
            ctx.mode_drifts.len(),
            1,
            "un modo vivo → una trayectoria por-modo"
        );
        assert_eq!(ctx.mode_drifts[0].0, "x");
        assert!(
            ctx.mode_drifts[0].1 > 0.0,
            "el modo derivó desde su origen: {:?}",
            ctx.mode_drifts
        );
    }

    /// Store con un arco de varios hitos en tiempos crecientes (para probar span/resolution).
    fn store_with_arc() -> ArchetypeStore {
        let mut s = ArchetypeStore::new();
        // 4 ciclos de sueño en t = 0, 100, 200, 300 con direcciones que derivan.
        let dirs = [
            vec![1.0, 0.0],
            vec![0.8, 0.6],
            vec![0.0, 1.0],
            vec![-0.6, 0.8],
        ];
        for (i, d) in dirs.iter().enumerate() {
            s.imprint(
                &IntentionVector {
                    subject: "u".into(),
                    centroid: d.clone(),
                    anomalies: vec![],
                    core_label: "dom".into(),
                    anomaly_labels: vec![],
                    absorbed: 10,
                    redundant: 0,
                    label_histogram: vec![("dom".into(), 10)],
                    modes: vec![],
                },
                Resilience::High,
                i as f64 * 100.0,
            );
        }
        s
    }

    #[test]
    fn resolution_point_returns_no_arc() {
        let s = store_with_arc();
        let req = EvokeRequest {
            arc_detail: ArcDetail::None,
            ..EvokeRequest::new("u", 800)
        };
        let ctx = evoke(&s, &req, 300.0).unwrap();
        assert!(
            ctx.arc_points.is_empty(),
            "RESOLUTION point / snapshot ⇒ sin arco"
        );
    }

    #[test]
    fn resolution_summary_caps_arc_points() {
        let s = store_with_arc();
        let full = evoke(&s, &EvokeRequest::new("u", 800), 300.0).unwrap();
        let summary = evoke(
            &s,
            &EvokeRequest {
                arc_detail: ArcDetail::Summary,
                ..EvokeRequest::new("u", 800)
            },
            300.0,
        )
        .unwrap();
        assert!(summary.arc_points.len() <= 4);
        assert!(summary.arc_points.len() <= full.arc_points.len());
    }

    #[test]
    fn span_window_filters_old_milestones() {
        let s = store_with_arc();
        // since = 150 ⇒ solo los hitos en t=200 y t=300 entran.
        let req = EvokeRequest {
            since: Some(150.0),
            ..EvokeRequest::new("u", 800)
        };
        let ctx = evoke(&s, &req, 300.0).unwrap();
        assert!(
            ctx.arc_points.iter().all(|(t, _)| *t >= 150.0),
            "{:?}",
            ctx.arc_points
        );
        assert_eq!(ctx.arc_points.len(), 2);
    }

    #[test]
    fn domain_arcs_capture_per_domain_reversal() {
        // El gap que el benchmark adversarial expuso: el arco GLOBAL no distingue qué comportamiento
        // concreto volvió. `domain_arcs` reconstruye la prevalencia por dominio en cada hito.
        // Guion: "yoga" alto → cae → cae → VUELVE; "trail" crece monótono. (Ver docs/06 §11.)
        let cycles = [
            vec![("yoga".to_string(), 7usize), ("trail".to_string(), 3)],
            vec![("yoga".to_string(), 2), ("trail".to_string(), 8)],
            vec![("yoga".to_string(), 1), ("trail".to_string(), 9)],
            vec![("yoga".to_string(), 6), ("trail".to_string(), 4)],
        ];
        let mut s = ArchetypeStore::new();
        for (i, hist) in cycles.iter().enumerate() {
            let absorbed: usize = hist.iter().map(|(_, c)| c).sum();
            s.imprint(
                &IntentionVector {
                    subject: "u".into(),
                    centroid: vec![1.0, 0.0],
                    anomalies: vec![],
                    core_label: hist[0].0.clone(),
                    anomaly_labels: vec![],
                    absorbed,
                    redundant: 0,
                    label_histogram: hist.clone(),
                    modes: vec![],
                },
                Resilience::High,
                i as f64 * 100.0,
            );
        }
        let ctx = evoke(&s, &EvokeRequest::new("u", 800), 400.0).unwrap();

        let yoga = &ctx
            .domain_arcs
            .iter()
            .find(|(l, _)| l == "yoga")
            .expect("serie de yoga")
            .1;
        // Prevalencia de yoga por hito: alta, baja, baja, alta ⇒ ida y vuelta (lo que el arco global perdía).
        assert!(
            yoga[0] > 0.5 && yoga[1] < 0.3 && yoga[3] > 0.4,
            "yoga reversión: {yoga:?}"
        );

        let trail = &ctx
            .domain_arcs
            .iter()
            .find(|(l, _)| l == "trail")
            .expect("serie de trail")
            .1;
        assert!(
            trail[0] < 0.4 && trail[2] > 0.8,
            "trail creciente: {trail:?}"
        );
    }
}
