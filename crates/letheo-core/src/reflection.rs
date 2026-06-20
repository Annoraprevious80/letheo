//! Capa de Reflexión — memoria generativa (L8) y compresión predictiva (L9).
//!
//! "**Inteligencia = compresión**": una memoria mejor **predice mejor** el futuro a partir del pasado
//! comprimido. Dos capacidades, ambas deterministas y **sin LLM** (la reflexión es análisis estructural
//! del arco, no generación de prosa):
//!
//! - **L9 · compresión predictiva** ([`predictive_compression`]): entrena la esencia con el pasado y
//!   mide cuánto resuena el futuro held-out con ella. Si los **modos** predicen mejor que el centroide
//!   único, la descomposición multi-modal *comprende* el comportamiento, no solo lo describe.
//! - **L8 · reflexión** ([`reflect`]): sintetiza **insights** que no están en ningún evento individual
//!   —transiciones dominantes entre ciclos y *revivals* (un comportamiento que tuvo pico, cayó y
//!   volvió)— derivados de la trayectoria. Es la "reflexión" del sueño hecha estructura.

use crate::archetype::{ArcMilestone, Archetype};
use crate::perception::Perception;
use crate::synthesis::{distill, DistillConfig};
use crate::vector::{cosine, Vector};
use std::collections::HashMap;

/// Salience por defecto de un insight materializado como hecho: alta (es sabiduría destilada del arco,
/// no un evento crudo). Declarada, ajustable.
pub const DEFAULT_INSIGHT_SALIENCE: f64 = 0.9;

// ─────────────────────────────────────────────────────────────────────────────
// L9 — Compresión predictiva (la métrica norte interna)
// ─────────────────────────────────────────────────────────────────────────────

/// Cuánto predice la esencia el comportamiento futuro held-out, separado por representación: los
/// **modos** (capa-2 multi-modal) vs el **centroide** único (la media ciega). `modal > centroid` ⇒ la
/// descomposición multi-modal aporta poder predictivo real.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictiveScore {
    /// Resonancia media (coseno, ≥0) de los eventos held-out con el mejor modo.
    pub modal: f64,
    /// Resonancia media de los eventos held-out con el centroide único.
    pub centroid: f64,
    /// Cuántos eventos held-out se evaluaron.
    pub held_out: usize,
}

/// **L9** — entrena la esencia con `events[..k]` (k = `train_frac`) y mide la resonancia media de los
/// eventos restantes (held-out) con los modos y con el centroide. Determinista. `None` si no hay datos
/// suficientes (< 2 eventos) o las dimensiones no permiten destilar.
pub fn predictive_compression(
    events: &[&Perception],
    train_frac: f64,
    cfg: DistillConfig,
) -> Option<PredictiveScore> {
    if events.len() < 2 {
        return None;
    }
    let frac = train_frac.clamp(0.0, 1.0);
    let n_train = ((events.len() as f64 * frac).round() as usize).clamp(1, events.len() - 1);
    let (train, test) = events.split_at(n_train);
    let iv = distill("predict", train, cfg)?;

    let mut modal_sum = 0.0;
    let mut centroid_sum = 0.0;
    for e in test {
        let centroid_res = cosine(&iv.centroid, &e.embedding).max(0.0) as f64;
        let modal_res = if iv.modes.is_empty() {
            centroid_res
        } else {
            iv.modes
                .iter()
                .map(|m| cosine(&m.centroid, &e.embedding))
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0) as f64
        };
        modal_sum += modal_res;
        centroid_sum += centroid_res;
    }
    let n = test.len() as f64;
    Some(PredictiveScore {
        modal: modal_sum / n,
        centroid: centroid_sum / n,
        held_out: test.len(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// L8 — Reflexión (insights de orden superior)
// ─────────────────────────────────────────────────────────────────────────────

/// Prevalencia (fracción del pico) por debajo de la cual un dominio cuenta como "decaído", para
/// detectar revivals. Física declarada, ajustable.
pub const DEFAULT_REVIVAL_FLOOR: f32 = 0.25;

/// Un insight de orden superior: un enunciado que **no está en ningún evento individual**, derivado de
/// la trayectoria (arco) del sujeto.
#[derive(Debug, Clone, PartialEq)]
pub enum Insight {
    /// El sujeto tiende a pasar de `from` a `to` (transición dominante entre ciclos consecutivos).
    Transition {
        from: String,
        to: String,
        support: usize,
    },
    /// Un comportamiento que tuvo un pico, decayó por debajo del suelo, y **volvió** a subir.
    Revival { domain: String },
}

/// **L8** — reflexiona sobre el arco: sintetiza la transición dominante entre etiquetas de ciclos
/// consecutivos y los revivals. Determinista, sin LLM. Vacío si el arco es demasiado corto.
pub fn reflect(arc: &[ArcMilestone]) -> Vec<Insight> {
    let mut out = Vec::new();

    // Transición dominante: el par (label_i → label_{i+1}) con label distinto más frecuente.
    let mut trans: HashMap<(String, String), usize> = HashMap::new();
    for w in arc.windows(2) {
        let (a, b) = (&w[0].label, &w[1].label);
        if a != b && !a.is_empty() && !b.is_empty() {
            *trans.entry((a.clone(), b.clone())).or_insert(0) += 1;
        }
    }
    // Desempate determinista: más soporte, luego orden alfabético de (from, to).
    if let Some(((from, to), support)) = trans
        .into_iter()
        .max_by(|x, y| x.1.cmp(&y.1).then_with(|| (y.0).cmp(&x.0)))
    {
        out.push(Insight::Transition { from, to, support });
    }

    out.extend(detect_revivals(arc, DEFAULT_REVIVAL_FLOOR));
    out
}

/// Predice el siguiente comportamiento dado el actual, con las transiciones aprendidas del arco. Base
/// de la verificación de L8: predecir por transición debe batir a predecir la moda global del arco.
pub fn predict_next(arc: &[ArcMilestone], current: &str) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for w in arc.windows(2) {
        if w[0].label == current && w[1].label != current {
            *counts.entry(w[1].label.as_str()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|x, y| x.1.cmp(&y.1).then_with(|| (y.0).cmp(x.0)))
        .map(|(l, _)| l.to_string())
}

/// Materializa un insight como `(texto, embedding)` para guardarlo como **hecho de alta salience** en
/// la capa-1. El embedding se deriva del comportamiento aludido (centroide del modo, o dirección del
/// hito) **sin necesidad de un provider** —es geometría que el motor ya tiene—. Así una `RECALL` que
/// resuene con ese comportamiento recupera también el insight. `None` si el comportamiento aludido no
/// tiene dirección conocida en el arquetipo.
pub fn materialize(archetype: &Archetype, insight: &Insight) -> Option<(String, Vector)> {
    let (text, label) = match insight {
        Insight::Transition { from, to, .. } => (
            format!("transición de comportamiento: {from} → {to}"),
            to.as_str(),
        ),
        Insight::Revival { domain } => (
            format!("comportamiento recurrente (revival): {domain}"),
            domain.as_str(),
        ),
    };
    let dir = direction_for_label(archetype, label)?;
    Some((text, dir))
}

/// Dirección (embedding) asociada a una etiqueta: el centroide de su modo si existe, o la dirección
/// del hito más reciente con esa etiqueta. Geometría ya presente en el arquetipo, sin provider.
fn direction_for_label(a: &Archetype, label: &str) -> Option<Vector> {
    a.modes
        .iter()
        .find(|m| m.label == label)
        .map(|m| m.centroid.clone())
        .or_else(|| {
            a.arc
                .iter()
                .rev()
                .find(|m| m.label == label)
                .map(|m| m.direction.clone())
        })
}

/// Detecta revivals: para cada dominio, su prevalencia por hito (de `label_histogram`) muestra un pico,
/// luego una caída bajo `floor·pico`, y después un rebote por encima de `floor·pico`.
fn detect_revivals(arc: &[ArcMilestone], floor: f32) -> Vec<Insight> {
    if arc.len() < 3 {
        return Vec::new();
    }
    let totals: Vec<usize> = arc
        .iter()
        .map(|m| m.label_histogram.iter().map(|(_, c)| c).sum())
        .collect();

    // Universo de dominios, en orden determinista.
    let mut domains: Vec<&str> = arc
        .iter()
        .flat_map(|m| m.label_histogram.iter().map(|(l, _)| l.as_str()))
        .collect();
    domains.sort_unstable();
    domains.dedup();

    let mut revivals = Vec::new();
    for d in domains {
        let series: Vec<f32> = arc
            .iter()
            .zip(&totals)
            .map(|(m, &t)| {
                let c = m
                    .label_histogram
                    .iter()
                    .find(|(l, _)| l == d)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);
                if t > 0 {
                    c as f32 / t as f32
                } else {
                    0.0
                }
            })
            .collect();
        let peak = series.iter().cloned().fold(0.0_f32, f32::max);
        if peak <= 0.0 {
            continue;
        }
        let thr = floor * peak;
        let peak_i = series
            .iter()
            .position(|&x| (x - peak).abs() < 1e-6)
            .unwrap();
        // Tras el pico: ¿hay una caída bajo el suelo y luego un rebote por encima?
        let (mut dipped, mut revived) = (false, false);
        for &x in &series[peak_i + 1..] {
            if x < thr {
                dipped = true;
            } else if dipped && x >= thr {
                revived = true;
                break;
            }
        }
        if dipped && revived {
            revivals.push(Insight::Revival {
                domain: d.to_string(),
            });
        }
    }
    revivals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(act: &str, e: Vec<f32>) -> Perception {
        Perception::new("u", e, 1.0, 3600.0, 0.0).with_trait("act", act)
    }

    fn milestone(label: &str, hist: &[(&str, usize)]) -> ArcMilestone {
        ArcMilestone {
            at: 0.0,
            direction: vec![1.0, 0.0],
            absorbed: hist.iter().map(|(_, c)| c).sum(),
            label: label.to_string(),
            label_histogram: hist.iter().map(|(l, c)| (l.to_string(), *c)).collect(),
        }
    }

    fn arc_of(labels: &[&str]) -> Vec<ArcMilestone> {
        labels
            .iter()
            .enumerate()
            .map(|(i, l)| ArcMilestone {
                at: i as f64,
                direction: vec![1.0, 0.0],
                absorbed: 1,
                label: l.to_string(),
                label_histogram: vec![(l.to_string(), 1)],
            })
            .collect()
    }

    #[test]
    fn modes_predict_held_out_better_than_centroid() {
        // Comportamiento bimodal (A y B ortogonales, alternados): la media queda entre ambos y predice
        // peor que recuperar el modo correcto de cada evento futuro.
        let mut ps = Vec::new();
        for _ in 0..20 {
            ps.push(p("A", vec![1.0, 0.0]));
            ps.push(p("B", vec![0.0, 1.0]));
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let s = predictive_compression(&refs, 0.7, DistillConfig::default()).unwrap();
        assert!(s.held_out > 0);
        assert!(
            s.modal > s.centroid + 0.2,
            "los modos predicen el futuro mejor que la media: {s:?}"
        );
    }

    #[test]
    fn unimodal_modes_match_centroid() {
        // Un solo comportamiento: el modo ≈ el centroide → no hay ganancia predictiva (ni pérdida).
        let ps: Vec<Perception> = (0..20)
            .map(|i| p("x", vec![1.0, 0.01 * i as f32]))
            .collect();
        let refs: Vec<&Perception> = ps.iter().collect();
        let s = predictive_compression(&refs, 0.7, DistillConfig::default()).unwrap();
        assert!(
            (s.modal - s.centroid).abs() < 0.1,
            "unimodal: modos ≈ centroide: {s:?}"
        );
    }

    #[test]
    fn predictive_compression_needs_data() {
        let one = p("a", vec![1.0, 0.0]);
        assert!(predictive_compression(&[&one], 0.7, DistillConfig::default()).is_none());
    }

    #[test]
    fn reflect_finds_dominant_transition() {
        // trail→yoga ocurre dos veces; es la transición dominante del arco.
        let arc = arc_of(&["trail", "yoga", "trail", "yoga", "climb"]);
        let insights = reflect(&arc);
        assert!(
            insights
                .iter()
                .any(|i| matches!(i, Insight::Transition { from, to, support }
                if from == "trail" && to == "yoga" && *support == 2)),
            "{insights:?}"
        );
    }

    #[test]
    fn reflect_detects_revival() {
        // yoga: pico (0.8), cae (0.1), vuelve (0.7) → revival; trail no revive (sube monótono).
        let arc = vec![
            milestone("yoga", &[("yoga", 8), ("trail", 2)]),
            milestone("trail", &[("yoga", 1), ("trail", 9)]),
            milestone("yoga", &[("yoga", 7), ("trail", 3)]),
        ];
        let insights = reflect(&arc);
        assert!(
            insights
                .iter()
                .any(|i| matches!(i, Insight::Revival { domain } if domain == "yoga")),
            "{insights:?}"
        );
        assert!(
            !insights
                .iter()
                .any(|i| matches!(i, Insight::Revival { domain } if domain == "trail")),
            "trail no revive: {insights:?}"
        );
    }

    #[test]
    fn transition_prediction_beats_marginal() {
        // Estructura A→B: dado "A", el siguiente real es siempre "B". La moda global no lo capta;
        // la predicción por transición sí.
        let arc = arc_of(&["A", "B", "A", "B", "A", "B"]);
        assert_eq!(
            predict_next(&arc, "A").as_deref(),
            Some("B"),
            "la transición predice B tras A"
        );
    }
}
