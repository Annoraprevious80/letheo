//! Capa de Modos — el arquetipo **multi-modal**.
//!
//! Un sujeto rara vez es una sola cosa: ve cine negro *y* documentales, programa en Rust *y* escribe
//! prosa. Un único centroide (la media de todo) colapsa esos comportamientos distintos en un punto
//! intermedio que **no representa a ninguno** — la media de "thriller" y "comedia romántica" no es un
//! género, es ruido. Ese era el cuello de botella nº1 del motor: en datos multi-modales la señal se
//! destruía al promediar (ver `docs/05-honest-assessment.md`, fallo del clustering ausente del core).
//!
//! Aquí el comportamiento se descompone en **modos**: subgrupos coherentes de percepciones, cada uno
//! con su propio centroide, etiqueta y **física de olvido independiente** (un modo que no se revisita
//! decae y se desvanece; uno que recurre se refuerza). El clustering es **determinista** (sin RNG):
//! asignación tipo *leader / DP-means* en el orden de llegada — reproducible bit a bit, sin semillas
//! ocultas (coherente con la disciplina VERDAD 100%: nada de aleatoriedad no declarada).

use crate::entropy::{EntropyTrace, Tick};
use crate::perception::Perception;
use crate::vector::{cosine, Vector};
use std::collections::HashMap;

/// Frontera de coseno entre "el mismo modo de comportamiento" y "un modo distinto". Por encima del
/// umbral, dos direcciones se consideran el mismo modo (se fusionan); por debajo, nace un modo nuevo.
/// Física calibrable, declarada (no constante mágica): con embeddings normalizados (all-MiniLM), el
/// mismo tema ronda 0.6–0.9 y temas distintos 0.1–0.4, así que 0.5 separa con holgura.
pub const DEFAULT_MODE_THETA: f32 = 0.5;

/// Número máximo de modos por arquetipo en un ciclo de destilación. Acota el coste y evita que el
/// ruido fragmente la identidad en mil pedazos. Declarado, ajustable vía [`ModeConfig`].
pub const DEFAULT_MAX_MODES: usize = 8;

/// Parámetros del clustering de modos (parte de `DistillConfig`).
#[derive(Debug, Clone, Copy)]
pub struct ModeConfig {
    /// Umbral de coseno para asignar a un modo existente vs crear uno nuevo (ver [`DEFAULT_MODE_THETA`]).
    pub theta: f32,
    /// Tope de modos por ciclo (ver [`DEFAULT_MAX_MODES`]).
    pub max_modes: usize,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            theta: DEFAULT_MODE_THETA,
            max_modes: DEFAULT_MAX_MODES,
        }
    }
}

/// Semilla de un modo recién destilado de un ciclo (aún sin física: el rastro de entropía se ancla en
/// `IMPRINT`, que es quien conoce `now` y la resiliencia). Es el producto del clustering en `DISTILL`.
#[derive(Debug, Clone)]
pub struct ModeSeed {
    /// Centroide del subgrupo (media de sus embeddings).
    pub centroid: Vector,
    /// Etiqueta léxica dominante del modo (moda del subgrupo).
    pub label: String,
    /// Histograma `(texto, conteo)` del modo, ordenado por frecuencia desc.
    pub label_histogram: Vec<(String, usize)>,
    /// Percepciones absorbidas por este modo en el ciclo.
    pub absorbed: usize,
}

impl ModeSeed {
    /// Consolida la semilla en un [`Mode`] vivo, anclándole su física de olvido.
    pub fn into_mode(self, halflife: f64, now: Tick) -> Mode {
        Mode {
            trace: EntropyTrace::new(1.0, halflife, now),
            origin: self.centroid.clone(), // nacimiento = dirección de la semilla; nunca cambia
            centroid: self.centroid,
            label: self.label,
            label_histogram: self.label_histogram,
            absorbed: self.absorbed,
        }
    }
}

/// Un modo consolidado en el arquetipo: un comportamiento estable del sujeto, con su propia vida.
#[derive(Debug, Clone)]
pub struct Mode {
    /// Dirección central del modo (acumulada, ponderada por volumen al evolucionar).
    pub centroid: Vector,
    /// Dirección **de nacimiento** del modo (su `centroid` la primera vez que apareció). Fija — no se
    /// toca al evolucionar. Base del `drift`: cuánto ha cambiado *este comportamiento* desde que surgió.
    pub origin: Vector,
    /// Etiqueta léxica dominante.
    pub label: String,
    /// Histograma `(texto, conteo)` acumulado del modo.
    pub label_histogram: Vec<(String, usize)>,
    /// Total de percepciones que este modo representa.
    pub absorbed: usize,
    /// Física del olvido del modo: si no se revisita, decae; al recurrir, se refuerza.
    pub trace: EntropyTrace,
}

impl Mode {
    /// Funde una semilla nueva en este modo (cuando resuena con él): mueve el centroide hacia la nueva
    /// evidencia **ponderado por volumen** (un ciclo de 3 eventos no desplaza tanto como uno de 3000),
    /// acumula el histograma y **refuerza** la permanencia (recordar alarga la vida media).
    pub fn merge(&mut self, seed: &ModeSeed, now: Tick) {
        if self.centroid.len() == seed.centroid.len() {
            let w_old = self.absorbed.max(1) as f32;
            let w_new = seed.absorbed.max(1) as f32;
            let total = w_old + w_new;
            for (c, x) in self.centroid.iter_mut().zip(&seed.centroid) {
                *c = (*c * w_old + *x * w_new) / total;
            }
        }
        merge_histograms(&mut self.label_histogram, &seed.label_histogram);
        self.label = self
            .label_histogram
            .first()
            .map(|(t, _)| t.clone())
            .unwrap_or_else(|| self.label.clone());
        self.absorbed += seed.absorbed;
        // Consolidación suave: el modo recurrente gana permanencia (como la sinapsis).
        self.trace.reinforce(now, 0.1);
    }

    /// **Drift del modo**: cuánto se ha desplazado su comportamiento desde que nació,
    /// `1 − cos(centroid, origin) ∈ [0, 2]`. La identidad (`origin`) es fija; el `centroid` evoluciona
    /// al recurrir, así que drift alto = el sujeto sigue con este modo pero su forma cambió (p. ej.
    /// "thriller" → "true crime"). Esto da **trayectoria por-modo**, no solo la del centroide global.
    pub fn drift(&self) -> f32 {
        (1.0 - cosine(&self.centroid, &self.origin)).max(0.0)
    }
}

/// Fusiona el histograma `src` dentro de `dst` (suma conteos por etiqueta) y lo reordena por
/// frecuencia desc con desempate alfabético determinista.
fn merge_histograms(dst: &mut Vec<(String, usize)>, src: &[(String, usize)]) {
    let mut map: HashMap<String, usize> = dst.drain(..).collect();
    for (t, c) in src {
        *map.entry(t.clone()).or_insert(0) += *c;
    }
    let mut merged: Vec<(String, usize)> = map.into_iter().collect();
    merged.sort_by(|(ta, ca), (tb, cb)| cb.cmp(ca).then_with(|| ta.cmp(tb)));
    *dst = merged;
}

/// Acumulador interno de un líder durante el clustering (suma sin normalizar + frecuencias de etiqueta).
struct Leader {
    sum: Vec<f32>,
    count: usize,
    freq: HashMap<String, usize>,
}

/// **DISTILL multi-modal**: descompone un conjunto de percepciones en modos coherentes.
///
/// Algoritmo *leader / DP-means* determinista: recorre las percepciones en orden; asigna cada una al
/// líder más cercano por coseno si supera `cfg.theta`, o abre un líder nuevo (hasta `cfg.max_modes`;
/// alcanzado el tope, se asigna al más cercano). El coseno es invariante de escala, así que comparar
/// contra la **suma** del líder equivale a compararla contra su centroide — sin dividir en el bucle.
///
/// Devuelve los modos en orden de aparición (determinista). Vacío si no hay percepciones.
pub fn cluster_modes(perceptions: &[&Perception], cfg: ModeConfig) -> Vec<ModeSeed> {
    if perceptions.is_empty() {
        return Vec::new();
    }
    let dim = perceptions[0].embedding.len();
    let mut leaders: Vec<Leader> = Vec::new();

    for p in perceptions {
        if p.embedding.len() != dim {
            continue; // robustez: ignora dimensiones incompatibles en vez de corromper centroides
        }
        // Líder más cercano por coseno (contra la suma sin normalizar — mismo ranking que el centroide).
        let mut best = f32::NEG_INFINITY;
        let mut best_i: Option<usize> = None;
        for (i, l) in leaders.iter().enumerate() {
            let c = cosine(&p.embedding, &l.sum);
            if c > best {
                best = c;
                best_i = Some(i);
            }
        }
        let target = match best_i {
            Some(i) if best >= cfg.theta => i,
            _ if leaders.len() < cfg.max_modes => {
                leaders.push(Leader {
                    sum: vec![0.0; dim],
                    count: 0,
                    freq: HashMap::new(),
                });
                leaders.len() - 1
            }
            Some(i) => i, // tope alcanzado → al más cercano
            None => unreachable!(
                "sin líderes solo en la primera iteración, cubierta por la rama de creación"
            ),
        };
        let l = &mut leaders[target];
        for (a, x) in l.sum.iter_mut().zip(&p.embedding) {
            *a += *x;
        }
        l.count += 1;
        *l.freq.entry(p.representative_text()).or_insert(0) += 1;
    }

    leaders
        .into_iter()
        .map(|l| {
            let inv = 1.0 / l.count as f32;
            let centroid: Vector = l.sum.iter().map(|x| x * inv).collect();
            let mut hist: Vec<(String, usize)> = l.freq.into_iter().collect();
            hist.sort_by(|(ta, ca), (tb, cb)| cb.cmp(ca).then_with(|| ta.cmp(tb)));
            let label = hist.first().map(|(t, _)| t.clone()).unwrap_or_default();
            ModeSeed {
                centroid,
                label,
                label_histogram: hist,
                absorbed: l.count,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(act: &str, e: Vec<f32>) -> Perception {
        Perception::new("u", e, 1.0, 3600.0, 0.0).with_trait("act", act)
    }

    #[test]
    fn unimodal_set_yields_one_mode() {
        // Un único comportamiento → un solo modo (reproduce el caso del centroide único de antes).
        let ps = [
            p("trail", vec![1.0, 0.0]),
            p("trail", vec![0.99, 0.01]),
            p("trail", vec![1.0, 0.02]),
        ];
        let refs: Vec<&Perception> = ps.iter().collect();
        let modes = cluster_modes(&refs, ModeConfig::default());
        assert_eq!(modes.len(), 1);
        assert_eq!(modes[0].absorbed, 3);
        assert_eq!(modes[0].label, "trail");
    }

    #[test]
    fn three_distinct_behaviors_yield_three_modes() {
        // Tres comportamientos ortogonales → tres modos, no una media en el centro.
        let mut ps = Vec::new();
        for _ in 0..5 {
            ps.push(p("noir", vec![1.0, 0.0, 0.0]));
            ps.push(p("docs", vec![0.0, 1.0, 0.0]));
            ps.push(p("scifi", vec![0.0, 0.0, 1.0]));
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let modes = cluster_modes(&refs, ModeConfig::default());
        assert_eq!(
            modes.len(),
            3,
            "tres modos coherentes, no un promedio ruidoso"
        );
        let labels: Vec<&str> = modes.iter().map(|m| m.label.as_str()).collect();
        assert!(labels.contains(&"noir") && labels.contains(&"docs") && labels.contains(&"scifi"));
        // Cada modo apunta a su dirección, no a la media (que sería ~(0.33,0.33,0.33)).
        for m in &modes {
            let max = m.centroid.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                max > 0.9,
                "el centroide del modo es nítido, no la media: {:?}",
                m.centroid
            );
        }
    }

    #[test]
    fn merge_blends_by_volume_and_reinforces() {
        let halflife = 3600.0;
        let mut mode = ModeSeed {
            centroid: vec![1.0, 0.0],
            label: "a".into(),
            label_histogram: vec![("a".into(), 100)],
            absorbed: 100,
        }
        .into_mode(halflife, 0.0);
        let r0 = mode.trace.reinforcement;
        mode.merge(
            &ModeSeed {
                centroid: vec![0.0, 1.0],
                label: "b".into(),
                label_histogram: vec![("b".into(), 50)],
                absorbed: 50,
            },
            1.0,
        );
        // (100·1, 50·1)/150 = (0.667, 0.333) — ponderado por volumen, no (0.5, 0.5).
        assert!((mode.centroid[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((mode.centroid[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(mode.absorbed, 150);
        assert!(
            mode.trace.reinforcement > r0,
            "fundir un modo recurrente lo refuerza"
        );
    }

    #[test]
    fn mode_drift_grows_as_behavior_shifts_but_origin_is_fixed() {
        let mut mode = ModeSeed {
            centroid: vec![1.0, 0.0],
            label: "a".into(),
            label_histogram: vec![("a".into(), 100)],
            absorbed: 100,
        }
        .into_mode(3600.0, 0.0);
        // Recién nacido: centroid == origin → sin drift.
        assert!(mode.drift() < 1e-6, "modo recién nacido no ha derivado");

        // El comportamiento se desplaza (nueva evidencia mueve el centroide), el origin NO cambia.
        mode.merge(
            &ModeSeed {
                centroid: vec![0.0, 1.0],
                label: "a".into(),
                label_histogram: vec![("a".into(), 100)],
                absorbed: 100,
            },
            1.0,
        );
        // (100·[1,0] + 100·[0,1])/200 = [0.5,0.5]; cos([0.5,0.5],[1,0]) ≈ 0.707 → drift ≈ 0.293.
        assert!(
            mode.drift() > 0.25,
            "el modo derivó desde su origen: {}",
            mode.drift()
        );
        assert_eq!(
            mode.origin,
            vec![1.0, 0.0],
            "el origin sigue siendo la dirección de nacimiento"
        );
    }
}
