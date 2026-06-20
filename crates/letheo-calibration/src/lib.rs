//! # letheo-calibration · Sweep empírico de umbrales
//!
//! Riesgo abierto del plan original: los tres umbrales del runtime —`θ_fade`, `θ_redundancia`,
//! `θ_anomalía`— están puestos a ojo (`0.05`, `0.92`, `0.30`). ¿Son los correctos?
//!
//! Aquí los calibramos contra **datos sintéticos con etiqueta de verdad** (*ground truth*): cada
//! evento sabe qué *debería* ser, así que podemos medir precisión/recall de cada umbral y trazar la
//! **frontera de Pareto** entre objetivos en tensión (no barrer señal real vs. no perder novelty).
//!
//! Todo es determinista (RNG `splitmix64` propio, sin dependencias) → el reporte es reproducible y
//! los tests no dependen de azar del sistema. Es análisis, no runtime: vive en su propio crate.

use letheo_core::entropy::EntropyTrace;
use letheo_core::vector::{centroid, cosine, Vector};

// ─────────────────────────────────────────────────────────────────────────────
// RNG determinista
// ─────────────────────────────────────────────────────────────────────────────

/// Generador `splitmix64`: rápido, determinista, sin dependencias. Suficiente para datos sintéticos.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// f64 uniforme en `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// f64 uniforme en `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Normal estándar vía Box–Muller.
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Métricas
// ─────────────────────────────────────────────────────────────────────────────

/// Precisión / recall / F1 de un clasificador binario contra ground truth.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Metrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
}

impl Metrics {
    pub fn observe(&mut self, predicted_positive: bool, actually_positive: bool) {
        match (predicted_positive, actually_positive) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, true) => self.fn_ += 1,
            (false, false) => self.tn += 1,
        }
    }

    pub fn precision(&self) -> f64 {
        let d = self.tp + self.fp;
        if d == 0 {
            1.0
        } else {
            self.tp as f64 / d as f64
        }
    }

    pub fn recall(&self) -> f64 {
        let d = self.tp + self.fn_;
        if d == 0 {
            1.0
        } else {
            self.tp as f64 / d as f64
        }
    }

    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sweep semántico: θ_redundancia / θ_anomalía
// ─────────────────────────────────────────────────────────────────────────────

/// La clase verdadera de un evento semántico sintético.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrueClass {
    /// Ruido predecible, pegado a la dirección central → *debería* marcarse redundante (FADE).
    Redundant,
    /// Comportamiento legítimo con dispersión moderada → *debería* absorberse (ni FADE ni anomalía).
    Signal,
    /// Quiebre de patrón, dirección nueva → *debería* retenerse como anomalía/novelty.
    Anomaly,
}

// Dimensión de los embeddings sintéticos. Alta a propósito: en baja dimensión dos direcciones
// "nuevas" no son casi-ortogonales (el coseno tiene varianza ~1/√DIM), y los embeddings reales
// (all-MiniLM = 384) sí lo son. 32 basta para reproducir esa separación sin coste.
const DIM: usize = 32;

fn random_unit_vector(rng: &mut Rng) -> Vector {
    let mut v: Vector = (0..DIM).map(|_| rng.gaussian() as f32).collect();
    let n = letheo_core::vector::norm(&v).max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn jitter(base: &[f32], sigma: f64, rng: &mut Rng) -> Vector {
    base.iter()
        .map(|&b| b + (rng.gaussian() * sigma) as f32)
        .collect()
}

/// Un evento semántico etiquetado.
pub struct SemanticEvent {
    pub embedding: Vector,
    pub class: TrueClass,
}

/// Genera una población semántica realista alrededor de una dirección central.
///
/// - `redundant`: pegados al core (σ pequeña) → coseno alto.
/// - `signal`: dispersión moderada (la conducta real del usuario varía) → coseno medio.
/// - `anomaly`: direcciones nuevas independientes → coseno bajo.
pub fn synth_semantic(
    seed: u64,
    n_redundant: usize,
    n_signal: usize,
    n_anomaly: usize,
) -> Vec<SemanticEvent> {
    let mut rng = Rng::new(seed);
    let core = random_unit_vector(&mut rng);
    let mut out = Vec::new();

    for _ in 0..n_redundant {
        out.push(SemanticEvent {
            embedding: jitter(&core, 0.03, &mut rng),
            class: TrueClass::Redundant,
        });
    }
    for _ in 0..n_signal {
        out.push(SemanticEvent {
            embedding: jitter(&core, 0.28, &mut rng),
            class: TrueClass::Signal,
        });
    }
    for _ in 0..n_anomaly {
        // Dirección nueva e independiente del core.
        let novel = random_unit_vector(&mut rng);
        out.push(SemanticEvent {
            embedding: jitter(&novel, 0.05, &mut rng),
            class: TrueClass::Anomaly,
        });
    }
    out
}

/// Resultado de evaluar un par (θ_red, θ_anom) sobre una población semántica.
#[derive(Debug, Clone, Copy)]
pub struct SemanticScore {
    pub theta_redundancy: f32,
    pub theta_anomaly: f32,
    /// Detección de redundancia (positivo = redundante).
    pub redundancy: Metrics,
    /// Detección de anomalía (positivo = anomalía).
    pub anomaly: Metrics,
    /// Fracción de *señal legítima* erróneamente marcada como redundante (riesgo: perder conducta real).
    pub signal_fade_rate: f64,
}

impl SemanticScore {
    /// Objetivo combinado: equilibra ambas detecciones penalizando barrer señal real.
    /// `min(F1_red, F1_anom) · (1 − signal_fade_rate)` — un solo escalar para ordenar.
    pub fn objective(&self) -> f64 {
        self.redundancy.f1().min(self.anomaly.f1()) * (1.0 - self.signal_fade_rate)
    }
}

/// Evalúa un par de umbrales contra la población (usa el centroide real, como hace `distill`).
pub fn score_semantic(events: &[SemanticEvent], theta_red: f32, theta_anom: f32) -> SemanticScore {
    let embeddings: Vec<Vector> = events.iter().map(|e| e.embedding.clone()).collect();
    let c = centroid(&embeddings).expect("población no vacía");

    let mut redundancy = Metrics::default();
    let mut anomaly = Metrics::default();
    let (mut signal_total, mut signal_faded) = (0usize, 0usize);

    for e in events {
        let sim = cosine(&e.embedding, &c);
        let pred_redundant = sim >= theta_red;
        let pred_anomaly = sim <= theta_anom;

        redundancy.observe(pred_redundant, e.class == TrueClass::Redundant);
        anomaly.observe(pred_anomaly, e.class == TrueClass::Anomaly);

        if e.class == TrueClass::Signal {
            signal_total += 1;
            if pred_redundant {
                signal_faded += 1;
            }
        }
    }

    let signal_fade_rate = if signal_total == 0 {
        0.0
    } else {
        signal_faded as f64 / signal_total as f64
    };
    SemanticScore {
        theta_redundancy: theta_red,
        theta_anomaly: theta_anom,
        redundancy,
        anomaly,
        signal_fade_rate,
    }
}

/// Barre la grilla cartesiana de (θ_red, θ_anom).
pub fn sweep_semantic(events: &[SemanticEvent], reds: &[f32], anoms: &[f32]) -> Vec<SemanticScore> {
    let mut out = Vec::with_capacity(reds.len() * anoms.len());
    for &r in reds {
        for &a in anoms {
            out.push(score_semantic(events, r, a));
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Sweep de entropía: θ_fade
// ─────────────────────────────────────────────────────────────────────────────

/// Un evento temporal etiquetado para calibrar θ_fade.
pub struct DecayEvent {
    pub trace: EntropyTrace,
    /// `true` si es ruido que *debería* desvanecerse al horizonte.
    pub is_noise: bool,
}

const HOUR: f64 = 3600.0;
const DAY: f64 = 24.0 * HOUR;

/// Genera eventos con dos clases:
/// - **ruido**: salience baja, vida media corta, sin refuerzo → debe caer.
/// - **memoria**: salience alta, vida media larga, a veces reforzada → debe persistir.
pub fn synth_decay(seed: u64, n_noise: usize, n_memory: usize) -> Vec<DecayEvent> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();

    for _ in 0..n_noise {
        let salience = rng.range(0.05, 0.30);
        let halflife = rng.range(0.5 * HOUR, 4.0 * HOUR);
        out.push(DecayEvent {
            trace: EntropyTrace::new(salience, halflife, 0.0),
            is_noise: true,
        });
    }
    for _ in 0..n_memory {
        let salience = rng.range(0.6, 1.0);
        let halflife = rng.range(1.0 * DAY, 7.0 * DAY);
        let mut trace = EntropyTrace::new(salience, halflife, 0.0);
        // ~40% de las memorias se refuerzan (consolidan) en algún momento del horizonte.
        if rng.unit() < 0.4 {
            trace.reinforce(rng.range(0.0, 1.0 * DAY), 0.3);
        }
        out.push(DecayEvent {
            trace,
            is_noise: false,
        });
    }
    out
}

/// Score de un θ_fade dado, evaluado en `horizon` (positivo = "desvanecido", verdad = "es ruido").
#[derive(Debug, Clone, Copy)]
pub struct FadeScore {
    pub theta_fade: f64,
    pub horizon: f64,
    pub fade: Metrics,
    /// Fracción de *memorias reales* erróneamente desvanecidas (el riesgo grave: amnesia).
    pub memory_loss_rate: f64,
}

impl FadeScore {
    /// Objetivo: F1 de detección de ruido, penalizando fuerte la amnesia (perder memoria real).
    pub fn objective(&self) -> f64 {
        self.fade.f1() * (1.0 - self.memory_loss_rate)
    }
}

pub fn score_fade(events: &[DecayEvent], theta_fade: f64, horizon: f64) -> FadeScore {
    let mut fade = Metrics::default();
    let (mut mem_total, mut mem_lost) = (0usize, 0usize);

    for e in events {
        let faded = e.trace.is_faded(horizon, theta_fade);
        fade.observe(faded, e.is_noise);
        if !e.is_noise {
            mem_total += 1;
            if faded {
                mem_lost += 1;
            }
        }
    }

    let memory_loss_rate = if mem_total == 0 {
        0.0
    } else {
        mem_lost as f64 / mem_total as f64
    };
    FadeScore {
        theta_fade,
        horizon,
        fade,
        memory_loss_rate,
    }
}

pub fn sweep_fade(events: &[DecayEvent], thetas: &[f64], horizon: f64) -> Vec<FadeScore> {
    thetas
        .iter()
        .map(|&t| score_fade(events, t, horizon))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Frontera de Pareto
// ─────────────────────────────────────────────────────────────────────────────

/// Devuelve los índices Pareto-óptimos: ningún otro punto domina (mayor o igual en ambos ejes y
/// estrictamente mayor en al menos uno). Maximiza ambos objetivos.
pub fn pareto_front(points: &[(f64, f64)]) -> Vec<usize> {
    let mut front = Vec::new();
    for (i, &(xi, yi)) in points.iter().enumerate() {
        let dominated = points
            .iter()
            .enumerate()
            .any(|(j, &(xj, yj))| j != i && xj >= xi && yj >= yi && (xj > xi || yj > yi));
        if !dominated {
            front.push(i);
        }
    }
    front
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn gaussian_is_roughly_centered() {
        let mut rng = Rng::new(7);
        let n = 20_000;
        let mean: f64 = (0..n).map(|_| rng.gaussian()).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "media ≈ 0, fue {mean}");
    }

    #[test]
    fn metrics_precision_recall_f1() {
        let m = Metrics {
            tp: 8,
            fp: 2,
            fn_: 2,
            tn: 88,
        };
        assert!((m.precision() - 0.8).abs() < 1e-9);
        assert!((m.recall() - 0.8).abs() < 1e-9);
        assert!((m.f1() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn default_semantic_thresholds_separate_classes_well() {
        let events = synth_semantic(1, 400, 400, 200);
        let s = score_semantic(&events, 0.92, 0.30);
        // Los umbrales por defecto deberían detectar redundancia y anomalía con buen recall...
        assert!(
            s.redundancy.recall() > 0.85,
            "recall redundancia: {:?}",
            s.redundancy
        );
        assert!(
            s.anomaly.recall() > 0.85,
            "recall anomalía: {:?}",
            s.anomaly
        );
        // ...sin barrer apenas señal legítima.
        assert!(
            s.signal_fade_rate < 0.10,
            "fade de señal: {}",
            s.signal_fade_rate
        );
    }

    #[test]
    fn sweep_finds_a_nonworse_optimum_than_default() {
        let events = synth_semantic(2, 400, 400, 200);
        let reds = [0.85f32, 0.88, 0.90, 0.92, 0.95];
        let anoms = [0.20f32, 0.30, 0.40, 0.50];
        let scores = sweep_semantic(&events, &reds, &anoms);
        let best = scores.iter().cloned().fold(scores[0], |acc, s| {
            if s.objective() > acc.objective() {
                s
            } else {
                acc
            }
        });
        let default = score_semantic(&events, 0.92, 0.30);
        assert!(
            best.objective() >= default.objective() - 1e-9,
            "el óptimo del sweep no es peor que el default ({} vs {})",
            best.objective(),
            default.objective()
        );
    }

    #[test]
    fn default_theta_fade_keeps_memories_and_drops_noise() {
        let events = synth_decay(3, 500, 500);
        // A 3 días, el ruido (vida media de horas) ya debería haber caído.
        let s = score_fade(&events, 0.05, 3.0 * DAY);
        assert!(s.fade.recall() > 0.9, "el ruido se desvanece: {:?}", s.fade);
        assert!(
            s.memory_loss_rate < 0.05,
            "casi ninguna memoria real se pierde: {}",
            s.memory_loss_rate
        );
    }

    #[test]
    fn pareto_front_excludes_dominated_points() {
        // (0.9,0.2) y (0.2,0.9) se compensan (ninguno domina al otro); (0.6,0.6) tampoco es
        // dominado; (0.5,0.5) sí lo es por (0.6,0.6).
        let pts = [(0.9, 0.2), (0.2, 0.9), (0.5, 0.5), (0.6, 0.6)];
        let mut front = pareto_front(&pts);
        front.sort();
        assert_eq!(front, vec![0, 1, 3]);
    }
}
