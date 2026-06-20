//! Capa de Síntesis — el "sueño" / compresión semántica (`DISTILL`).
//!
//! Toma N percepciones, calcula su **centroide** y mide la **varianza semántica** como similitud
//! del coseno de cada evento respecto al centroide (ver `docs/01-physics.md` §5):
//!
//! - cercano al centroide (`sim ≥ θ_redundancia`)  → ruido redundante → FADE.
//! - centroide en sí                                → la dirección del usuario → retener.
//! - outlier (`sim ≤ θ_anomalía`)                   → novelty / cambio brusco → retener.

use crate::modes::{cluster_modes, ModeConfig, ModeSeed};
use crate::perception::Perception;
use crate::vector::{centroid_refs, cosine, Vector};

/// Umbral por encima del cual un evento se considera redundante (predecible) → FADE.
pub const DEFAULT_THETA_REDUNDANCY: f32 = 0.92;
/// Umbral por debajo del cual un evento es un outlier anómalo (novelty) → retener.
pub const DEFAULT_THETA_ANOMALY: f32 = 0.30;

/// El producto del sueño: un Vector de Intención que comprime muchas percepciones.
#[derive(Debug, Clone)]
pub struct IntentionVector {
    pub subject: String,
    /// La dirección central del comportamiento del usuario (el centroide).
    pub centroid: Vector,
    /// Embeddings anómalos retenidos (novelty / quiebres de patrón).
    pub anomalies: Vec<Vector>,
    /// Texto representativo de la percepción **más central** (la más típica del cluster). Es la
    /// etiqueta léxica del núcleo, para que la prosa nombre el contenido dominante.
    pub core_label: String,
    /// Etiquetas léxicas de cada anomalía, alineadas con `anomalies`.
    pub anomaly_labels: Vec<String>,
    /// Cuántas percepciones se colapsaron en este vector (para el ratio de compresión).
    pub absorbed: usize,
    /// Cuántas se marcaron como ruido redundante (candidatas a FADE).
    pub redundant: usize,
    /// Histograma de etiquetas léxicas del ciclo: `(texto, conteo)` ordenado por frecuencia desc.
    /// Permite reconstruir trayectorias **por comportamiento** a lo largo del arco (no solo el
    /// centroide global) → responder "¿volvió X?" de un comportamiento concreto. (Ver docs/06 §11.)
    pub label_histogram: Vec<(String, usize)>,
    /// **Modos** del ciclo: subgrupos coherentes de comportamiento (clustering determinista). El
    /// `centroid` de arriba es la media GLOBAL (origen del arco, retrocompatible); los modos son la
    /// descomposición multi-modal que evita que la media colapse comportamientos distintos en ruido.
    /// Unimodal ⇒ un solo modo ≈ el centroide global. Ver [`crate::modes`].
    pub modes: Vec<ModeSeed>,
}

/// Parámetros de `DISTILL`.
#[derive(Debug, Clone, Copy)]
pub struct DistillConfig {
    pub theta_redundancy: f32,
    pub theta_anomaly: f32,
    /// Parámetros del clustering multi-modal (ver [`crate::modes`]).
    pub modes: ModeConfig,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            theta_redundancy: DEFAULT_THETA_REDUNDANCY,
            theta_anomaly: DEFAULT_THETA_ANOMALY,
            modes: ModeConfig::default(),
        }
    }
}

/// `DISTILL`: colapsa percepciones en un Vector de Intención.
///
/// Devuelve `None` si no hay percepciones o sus dimensiones no coinciden.
pub fn distill(
    subject: &str,
    perceptions: &[&Perception],
    cfg: DistillConfig,
) -> Option<IntentionVector> {
    if perceptions.is_empty() {
        return None;
    }
    // Centroide **sin clonar** los embeddings: se referencian (antes se clonaba un `Vec<Vector>` de
    // N×dim floats por cada sueño, solo para promediar).
    let refs: Vec<&[f32]> = perceptions.iter().map(|p| p.embedding.as_slice()).collect();
    let c = centroid_refs(&refs)?;

    let mut anomalies = Vec::new();
    let mut anomaly_labels = Vec::new();
    let mut redundant = 0usize;
    // El núcleo se etiqueta por la **moda** (el comportamiento más frecuente), no por la percepción
    // más cercana al centroide: con un centroide mixto, el "más cercano" puede ser un evento poco
    // representativo. La moda refleja qué *domina*. (Ver docs/06 §8.quinquies: bug a escala.)
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in perceptions.iter() {
        *freq.entry(p.representative_text()).or_insert(0) += 1;
        let sim = cosine(&p.embedding, &c);
        if sim >= cfg.theta_redundancy {
            redundant += 1; // ruido predecible → su voto ya vive en el centroide → FADE
        } else if sim <= cfg.theta_anomaly {
            anomalies.push(p.embedding.clone()); // novelty → solo se clona lo que se RETIENE
            anomaly_labels.push(p.representative_text());
        }
    }
    // Histograma ordenado por frecuencia desc (desempate alfabético determinista).
    let mut label_histogram: Vec<(String, usize)> = freq.into_iter().collect();
    label_histogram.sort_by(|(ta, ca), (tb, cb)| cb.cmp(ca).then_with(|| ta.cmp(tb)));
    // Moda = etiqueta más frecuente (cabeza del histograma).
    let core_label = label_histogram
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();

    // Descomposición multi-modal: subgrupos coherentes en vez de una sola media. Si el conjunto es
    // unimodal, esto devuelve un único modo ≈ al centroide global (retrocompatible).
    let modes = cluster_modes(perceptions, cfg.modes);

    Some(IntentionVector {
        subject: subject.to_string(),
        centroid: c,
        anomalies,
        core_label,
        anomaly_labels,
        absorbed: perceptions.len(),
        redundant,
        label_histogram,
        modes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::Perception;

    const HF: f64 = 3600.0;

    fn p(subject: &str, e: Vec<f32>) -> Perception {
        Perception::new(subject, e, 1.0, HF, 0.0)
    }

    #[test]
    fn distill_empty_is_none() {
        assert!(distill("user:X", &[], DistillConfig::default()).is_none());
    }

    #[test]
    fn distill_captures_core_and_anomaly_labels() {
        // Cluster denso etiquetado "trail" + un outlier "crypto". El núcleo debe etiquetarse con el
        // contenido central y la anomalía conservar su texto.
        let trail: Vec<Perception> = (0..8)
            .map(|i| p("u", vec![1.0, 0.0 + i as f32 * 0.001]).with_trait("act", "trail"))
            .collect();
        let outlier = p("u", vec![0.0, 1.0]).with_trait("act", "crypto");
        let mut refs: Vec<&Perception> = trail.iter().collect();
        refs.push(&outlier);
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        assert_eq!(
            iv.core_label, "trail",
            "el núcleo se etiqueta con el contenido central"
        );
        assert!(
            iv.anomaly_labels.contains(&"crypto".to_string()),
            "la anomalía conserva su texto"
        );
        assert_eq!(
            iv.anomaly_labels.len(),
            iv.anomalies.len(),
            "etiquetas alineadas con vectores"
        );
    }

    #[test]
    fn redundant_cluster_collapses_to_centroid() {
        // Cinco eventos casi idénticos: alta redundancia, centroide en su dirección.
        let ps = [
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![0.99, 0.01]),
            p("user:X", vec![1.0, 0.02]),
            p("user:X", vec![0.98, 0.0]),
            p("user:X", vec![1.0, 0.0]),
        ];
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("user:X", &refs, DistillConfig::default()).unwrap();
        assert_eq!(iv.absorbed, 5);
        assert!(
            iv.redundant >= 4,
            "el cluster denso es mayormente redundante"
        );
        assert!(
            iv.anomalies.is_empty(),
            "sin novelty en un cluster homogéneo"
        );
    }

    #[test]
    fn outlier_is_retained_as_anomaly() {
        // Cuatro eventos en una dirección + un outlier ortogonal (cambio brusco de comportamiento).
        let ps = [
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![1.0, 0.0]),
            p("user:X", vec![0.0, 1.0]), // outlier
        ];
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("user:X", &refs, DistillConfig::default()).unwrap();
        assert_eq!(iv.anomalies.len(), 1, "el outlier se retiene como novelty");
    }

    #[test]
    fn compression_ratio_is_meaningful() {
        let ps: Vec<Perception> = (0..1000).map(|_| p("user:X", vec![1.0, 0.0])).collect();
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("user:X", &refs, DistillConfig::default()).unwrap();
        // 1000 percepciones → 1 centroide (+ 0 anomalías): compresión masiva.
        assert_eq!(iv.absorbed, 1000);
        assert!(iv.anomalies.is_empty());
    }
}
