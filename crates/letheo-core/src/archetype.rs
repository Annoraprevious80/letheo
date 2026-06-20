//! Capa de Arquetipo — memoria de largo plazo (`IMPRINT`).
//!
//! Los Vectores de Intención consistentes a través de ciclos se consolidan en un `Archetype`: la
//! esencia del sujeto en pocos vectores densos. **Anclaje de evolución**: no es inmortal — sigue
//! sujeto a la física, pero con resiliencia alta. Almacenamiento embebido con búsqueda lineal Flat
//! (coseno); el índice ANN (HNSW) llega en L3 (ver `docs/04-architecture.md`).

use crate::entropy::{EntropyTrace, Tick};
use crate::modes::{Mode, DEFAULT_MODE_THETA};
use crate::synthesis::IntentionVector;
use crate::vector::{cosine, Vector};

/// Resiliencia de un arquetipo al olvido (modula la vida media base del `IMPRINT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resilience {
    Low,
    Medium,
    High,
}

/// Vidas medias base (segundos) de cada nivel de resiliencia. **Física declarada** (no constantes
/// mágicas enterradas en un `match`): un arquetipo `Low` resiste ~1 mes, `Medium` ~6 meses, `High` ~2
/// años antes de que su peso caiga a la mitad sin refuerzo (la consolidación por evocación las alarga).
/// Públicas y calibrables, en el mismo idioma que el resto de umbrales del motor (`DEFAULT_THETA_FADE`…).
pub const HALFLIFE_LOW_SECS: f64 = 30.0 * 86_400.0;
pub const HALFLIFE_MEDIUM_SECS: f64 = 180.0 * 86_400.0;
pub const HALFLIFE_HIGH_SECS: f64 = 720.0 * 86_400.0;

impl Resilience {
    /// Vida media base (segundos) del nivel. Ver [`HALFLIFE_LOW_SECS`] / [`HALFLIFE_MEDIUM_SECS`] /
    /// [`HALFLIFE_HIGH_SECS`].
    pub fn halflife(self) -> f64 {
        match self {
            Resilience::Low => HALFLIFE_LOW_SECS,
            Resilience::Medium => HALFLIFE_MEDIUM_SECS,
            Resilience::High => HALFLIFE_HIGH_SECS,
        }
    }
}

/// Un hito en el arco evolutivo: la dirección del sujeto en un ciclo de sueño dado.
#[derive(Debug, Clone)]
pub struct ArcMilestone {
    pub at: Tick,
    pub direction: Vector,
    pub absorbed: usize,
    /// Etiqueta léxica dominante de ese ciclo (qué le ocupaba al sujeto entonces).
    pub label: String,
    /// Histograma `(texto, conteo)` del ciclo: la mezcla de comportamientos, no solo el dominante.
    /// Base de las trayectorias por dominio en `evoke` (responder "¿volvió X?").
    pub label_histogram: Vec<(String, usize)>,
}

/// La esencia consolidada de un sujeto.
#[derive(Debug, Clone)]
pub struct Archetype {
    pub subject: String,
    /// Núcleo estable: dirección central acumulada del comportamiento (media GLOBAL). Se conserva
    /// como origen del arco y resonancia retrocompatible; la representación rica vive en `modes`.
    pub core: Vector,
    /// **Modos** del sujeto: comportamientos coherentes distintos, cada uno con su propia física.
    /// La media global (`core`) colapsaba comportamientos dispares en ruido; los modos los separan,
    /// y la resonancia recupera el modo relevante, no el promedio. Ver [`crate::modes`].
    pub modes: Vec<Mode>,
    /// Vectores de novelty retenidos (quiebres de patrón que aún resuenan).
    pub anomalies: Vec<Vector>,
    /// Etiquetas léxicas de las anomalías, alineadas con `anomalies`.
    pub anomaly_labels: Vec<String>,
    /// Etiqueta léxica del comportamiento dominante **actual** (último ciclo consolidado).
    pub core_label: String,
    /// Total de percepciones que esta esencia representa (denominador del ratio de compresión).
    pub represented: usize,
    /// Hitos del arco: una dirección por ciclo de sueño. La trayectoria del sujeto en el tiempo.
    pub arc: Vec<ArcMilestone>,
    /// Física del olvido del propio arquetipo (anclaje de evolución).
    pub trace: EntropyTrace,
}

impl Archetype {
    /// `IMPRINT`: consolida un Vector de Intención en un arquetipo nuevo.
    pub fn imprint(iv: &IntentionVector, resilience: Resilience, now: Tick) -> Self {
        let arc = vec![ArcMilestone {
            at: now,
            direction: iv.centroid.clone(),
            absorbed: iv.absorbed,
            label: iv.core_label.clone(),
            label_histogram: iv.label_histogram.clone(),
        }];
        let halflife = resilience.halflife();
        let modes = iv
            .modes
            .iter()
            .cloned()
            .map(|s| s.into_mode(halflife, now))
            .collect();
        Self {
            subject: iv.subject.clone(),
            core: iv.centroid.clone(),
            modes,
            anomalies: iv.anomalies.clone(),
            anomaly_labels: iv.anomaly_labels.clone(),
            core_label: iv.core_label.clone(),
            represented: iv.absorbed,
            arc,
            trace: EntropyTrace::new(1.0, resilience.halflife(), now),
        }
    }

    /// Evolución del arquetipo: absorbe un nuevo Vector de Intención moviendo el núcleo hacia la
    /// nueva dirección, reforzando su permanencia, y registrando un hito en el arco. La esencia
    /// *evoluciona*, no se reemplaza.
    pub fn evolve(&mut self, iv: &IntentionVector, now: Tick) {
        // Mezcla **ponderada por volumen**: el núcleo se mueve hacia la nueva dirección en
        // proporción a la evidencia que la respalda. Un ciclo de 3 eventos no debe desplazar la
        // identidad tanto como uno de 30.000. (Antes era un promedio simple — ver
        // `docs/05-honest-assessment.md`.)
        if self.core.len() == iv.centroid.len() {
            let w_old = self.represented.max(1) as f32;
            let w_new = iv.absorbed.max(1) as f32;
            let total = w_old + w_new;
            for (c, x) in self.core.iter_mut().zip(&iv.centroid) {
                *c = (*c * w_old + *x * w_new) / total;
            }
        }
        // Modos: cada modo nuevo se funde en el modo existente que más resuena (≥ θ), o nace como
        // modo propio (con la vida media ya consolidada del arquetipo). Así un comportamiento que
        // recurre se refuerza, y uno nuevo se añade sin contaminar a los demás.
        let halflife = crate::entropy::halflife_from_lambda(self.trace.lambda);
        for seed in &iv.modes {
            let mut best = DEFAULT_MODE_THETA;
            let mut best_i: Option<usize> = None;
            for (i, m) in self.modes.iter().enumerate() {
                let c = cosine(&m.centroid, &seed.centroid);
                if c >= best {
                    best = c;
                    best_i = Some(i);
                }
            }
            match best_i {
                Some(i) => self.modes[i].merge(seed, now),
                None => self.modes.push(seed.clone().into_mode(halflife, now)),
            }
        }

        self.anomalies.extend(iv.anomalies.iter().cloned());
        self.anomaly_labels
            .extend(iv.anomaly_labels.iter().cloned());
        // El "interés actual" es el del último ciclo consolidado.
        self.core_label = iv.core_label.clone();
        self.represented += iv.absorbed;
        self.arc.push(ArcMilestone {
            at: now,
            direction: iv.centroid.clone(),
            absorbed: iv.absorbed,
            label: iv.core_label.clone(),
            label_histogram: iv.label_histogram.clone(),
        });
        // Consolidación suave: recordar/reforzar alarga la vida media.
        self.trace.reinforce(now, 0.1);
    }

    /// Resonancia (coseno) entre una consulta y el sujeto. Con modos, es el **máximo** sobre los
    /// modos (la consulta recupera el comportamiento que de verdad le concierne, no la media global
    /// que diluía la señal). Sin modos (arquetipo legado), cae al núcleo global. Base de `EVOKE`.
    pub fn resonance(&self, query: &[f32]) -> f32 {
        if self.modes.is_empty() {
            return cosine(&self.core, query);
        }
        self.modes
            .iter()
            .map(|m| cosine(&m.centroid, query))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Etiqueta del **modo que más resuena** con una consulta (base de `EVOKE … RESONATING WITH`):
    /// la evocación se enfoca en el aspecto del sujeto que concierne al rasgo consultado, no en su
    /// comportamiento dominante global. `None` si el arquetipo no tiene modos (legado).
    pub fn resonant_mode_label(&self, query: &[f32]) -> Option<String> {
        self.modes
            .iter()
            .map(|m| (cosine(&m.centroid, query), &m.label))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, label)| label.clone())
    }

    /// `IMPRINT` real: **consolida/ancla** una esencia ya existente. No crea ni evoluciona (eso es
    /// `DISTILL`/`evolve`): refuerza la física del arquetipo **y de cada modo** (Δt→0, λ reducido por
    /// `consolidation` → vida media mayor), de modo que la esencia gana permanencia frente al olvido.
    /// `consolidation ∈ [0, 1)`: cuánto se reduce λ (0 = solo resetea Δt y suma refuerzo).
    pub fn consolidate(&mut self, now: Tick, consolidation: f64) {
        self.trace.reinforce(now, consolidation);
        for m in &mut self.modes {
            m.trace.reinforce(now, consolidation);
        }
    }
}

/// Memoria de largo plazo: el conjunto de arquetipos vivos. Búsqueda lineal Flat (exacta; el ANN es L3).
#[derive(Debug, Default)]
pub struct ArchetypeStore {
    archetypes: Vec<Archetype>,
}

impl ArchetypeStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// `IMPRINT` evolutivo: si ya existe un arquetipo para el sujeto, lo evoluciona; si no, lo crea.
    pub fn imprint(&mut self, iv: &IntentionVector, resilience: Resilience, now: Tick) {
        if let Some(a) = self.archetypes.iter_mut().find(|a| a.subject == iv.subject) {
            a.evolve(iv, now);
        } else {
            self.archetypes
                .push(Archetype::imprint(iv, resilience, now));
        }
    }

    /// Arquetipo de un sujeto, si existe.
    pub fn get(&self, subject: &str) -> Option<&Archetype> {
        self.archetypes.iter().find(|a| a.subject == subject)
    }

    /// Recorre los arquetipos (para snapshots / persistencia).
    pub fn iter(&self) -> impl Iterator<Item = &Archetype> {
        self.archetypes.iter()
    }

    /// Inserta un arquetipo ya construido (p. ej. restaurado desde disco). No fusiona: asume que el
    /// sujeto no está presente todavía. Úsese al rehidratar un store vacío.
    pub fn insert(&mut self, archetype: Archetype) {
        self.archetypes.push(archetype);
    }

    /// `IMPRINT`: consolida (ancla) el arquetipo de un sujeto si existe. Devuelve `false` si no hay
    /// ninguno —no se puede imprimir lo que no se ha destilado—. Ver [`Archetype::consolidate`].
    pub fn consolidate(&mut self, subject: &str, now: Tick, consolidation: f64) -> bool {
        match self.archetypes.iter_mut().find(|a| a.subject == subject) {
            Some(a) => {
                a.consolidate(now, consolidation);
                true
            }
            None => false,
        }
    }

    /// Búsqueda lineal Flat por **resonancia ponderada por vida** (L2): los `k` arquetipos que mejor
    /// puntúan con la consulta. La puntuación NO es el coseno crudo, sino `score = relevancia · vida`:
    ///
    /// ```text
    /// score = max(0, resonance(query)) · weight(now)
    /// ```
    ///
    /// donde `weight(now) = salience · e^(−λΔt) · (1 + reinforcement)` ya integra **recencia** (decay),
    /// **importancia** (salience) y **refuerzo**. Es la física nativa del motor usada para rankear, no
    /// un parche aditivo de coeficientes a mano: una memoria muy relevante pero desvanecida queda por
    /// debajo de otra igual de relevante pero viva. (Forma multiplicativa del retrieval de Generative
    /// Agents, sin pesos mágicos α/β/γ que afinar — coherente con la disciplina VERDAD 100%.)
    pub fn resonate(&self, query: &[f32], k: usize, now: Tick, theta_fade: f64) -> Vec<&Archetype> {
        let mut scored: Vec<(f64, &Archetype)> = self
            .archetypes
            .iter()
            .filter_map(|a| {
                // `weight()` incluye `e^x`: se evalúa **una sola vez** por arquetipo (era 2×).
                let life = a.trace.weight(now);
                if life < theta_fade {
                    return None; // arquetipo desvanecido
                }
                let relevance = a.resonance(query).max(0.0) as f64;
                Some((relevance * life, a))
            })
            .collect();
        scored.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, a)| a).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::IntentionVector;

    fn iv(subject: &str, c: Vec<f32>, absorbed: usize) -> IntentionVector {
        IntentionVector {
            subject: subject.to_string(),
            centroid: c,
            anomalies: vec![],
            core_label: format!("{subject}-core"),
            anomaly_labels: vec![],
            absorbed,
            redundant: 0,
            label_histogram: vec![(format!("{subject}-core"), absorbed)],
            modes: vec![],
        }
    }

    #[test]
    fn imprint_creates_then_evolves() {
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("user:X", vec![1.0, 0.0], 100), Resilience::High, 0.0);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("user:X").unwrap().represented, 100);

        // Segundo ciclo: el mismo sujeto evoluciona, no se duplica.
        store.imprint(&iv("user:X", vec![0.0, 1.0], 50), Resilience::High, 3600.0);
        assert_eq!(store.len(), 1);
        let a = store.get("user:X").unwrap();
        assert_eq!(a.represented, 150);
        // El núcleo se mueve hacia la nueva dirección **ponderado por volumen**: 100 eventos viejos
        // ([1,0]) + 50 nuevos ([0,1]) ⇒ (100·1, 50·1)/150 = (0.667, 0.333), no (0.5, 0.5).
        assert!(
            (a.core[0] - 2.0 / 3.0).abs() < 1e-6,
            "core[0] = {}",
            a.core[0]
        );
        assert!(
            (a.core[1] - 1.0 / 3.0).abs() < 1e-6,
            "core[1] = {}",
            a.core[1]
        );
    }

    #[test]
    fn evolve_weights_by_evidence_volume() {
        // Un quiebre minúsculo (1 evento) apenas mueve una identidad consolidada (10.000 eventos).
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("u", vec![1.0, 0.0], 10_000), Resilience::High, 0.0);
        store.imprint(&iv("u", vec![0.0, 1.0], 1), Resilience::High, 1.0);
        let core = &store.get("u").unwrap().core;
        assert!(core[0] > 0.999, "la identidad apenas se movió: {core:?}");
    }

    #[test]
    fn resonate_ranks_by_cosine() {
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("user:A", vec![1.0, 0.0], 10), Resilience::High, 0.0);
        store.imprint(&iv("user:B", vec![0.0, 1.0], 10), Resilience::High, 0.0);

        let top = store.resonate(&[0.9, 0.1], 1, 0.0, crate::entropy::DEFAULT_THETA_FADE);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].subject, "user:A");
    }

    #[test]
    fn resonate_weights_relevance_by_life_l2() {
        // Dos sujetos IGUAL de relevantes (misma dirección). Lo único que los distingue es la VIDA:
        // uno acaba de tocarse, el otro lleva una vida media decayendo. El físico debe rankear primero
        // al fresco aunque el coseno sea idéntico — el coseno crudo no podría distinguirlos.
        let half = Resilience::Low.halflife(); // 30 días
        let mut store = ArchetypeStore::new();
        store.imprint(&iv("user:stale", vec![1.0, 0.0], 10), Resilience::Low, 0.0);
        store.imprint(&iv("user:fresh", vec![1.0, 0.0], 10), Resilience::Low, half);

        let ranked = store.resonate(&[1.0, 0.0], 2, half, crate::entropy::DEFAULT_THETA_FADE);
        assert_eq!(ranked.len(), 2, "ambos siguen vivos");
        assert_eq!(
            ranked[0].subject, "user:fresh",
            "a igual relevancia, el más vivo primero"
        );
        assert_eq!(ranked[1].subject, "user:stale");
    }

    #[test]
    fn high_resilience_outlives_low() {
        assert!(Resilience::High.halflife() > Resilience::Low.halflife());
    }

    #[test]
    fn multimodal_resonance_recovers_the_right_mode_where_centroid_is_blind() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};

        // Un sujeto con DOS comportamientos opuestos: la media global es el vector NULO → el centroide
        // único es ciego (resonancia 0 para cualquier consulta). Es el caso patológico que destruía la
        // señal en datos multi-modales. Con modos, cada comportamiento se conserva nítido.
        let mut ps = Vec::new();
        for _ in 0..50 {
            ps.push(
                Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "left"),
            );
            ps.push(
                Perception::new("u", vec![-1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "right"),
            );
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        assert_eq!(
            iv.modes.len(),
            2,
            "dos comportamientos opuestos → dos modos"
        );

        let mut store = ArchetypeStore::new();
        store.imprint(&iv, Resilience::High, 0.0);
        let a = store.get("u").unwrap();

        // El núcleo global es ~nulo: el camino del centroide único NO puede recuperar nada.
        assert!(
            cosine(&a.core, &[1.0, 0.0]).abs() < 1e-3,
            "el centroide único es ciego: {:?}",
            a.core
        );
        // La resonancia multi-modal recupera el comportamiento que de verdad concierne a la consulta.
        assert!(
            (a.resonance(&[1.0, 0.0]) - 1.0).abs() < 1e-3,
            "el modo correcto resuena pleno"
        );
        assert!(
            (a.resonance(&[-1.0, 0.0]) - 1.0).abs() < 1e-3,
            "el otro modo también, según la consulta"
        );
    }

    #[test]
    fn resonant_mode_label_picks_the_aspect_that_matches_the_query() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};
        // Dos comportamientos ortogonales etiquetados → dos modos.
        let mut ps = Vec::new();
        for _ in 0..6 {
            ps.push(
                Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "noir"),
            );
            ps.push(
                Perception::new("u", vec![0.0, 1.0], 1.0, 3600.0, 0.0).with_trait("act", "docs"),
            );
        }
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        let mut store = ArchetypeStore::new();
        store.imprint(&iv, Resilience::High, 0.0);
        let a = store.get("u").unwrap();
        // Cada consulta recupera la etiqueta del modo que le concierne (base de RESONATING WITH).
        assert_eq!(a.resonant_mode_label(&[1.0, 0.0]).as_deref(), Some("noir"));
        assert_eq!(a.resonant_mode_label(&[0.0, 1.0]).as_deref(), Some("docs"));
    }

    #[test]
    fn consolidate_anchors_archetype_and_modes() {
        use crate::perception::Perception;
        use crate::synthesis::{distill, DistillConfig};
        let ps: Vec<Perception> = (0..4)
            .map(|_| Perception::new("u", vec![1.0, 0.0], 1.0, 3600.0, 0.0).with_trait("act", "x"))
            .collect();
        let refs: Vec<&Perception> = ps.iter().collect();
        let iv = distill("u", &refs, DistillConfig::default()).unwrap();
        let mut store = ArchetypeStore::new();
        store.imprint(&iv, Resilience::High, 0.0);
        let (r0, lambda0, mr0) = {
            let a = store.get("u").unwrap();
            (
                a.trace.reinforcement,
                a.trace.lambda,
                a.modes[0].trace.reinforcement,
            )
        };
        // IMPRINT real consolida: más refuerzo, λ menor (vida media mayor), modos también anclados.
        assert!(store.consolidate("u", 10.0, 0.2));
        let a = store.get("u").unwrap();
        assert!(a.trace.reinforcement > r0, "el arquetipo gana refuerzo");
        assert!(a.trace.lambda < lambda0, "λ baja → permanencia ganada");
        assert!(
            a.modes[0].trace.reinforcement > mr0,
            "los modos también se anclan"
        );
        // No se puede imprimir lo que no se ha destilado.
        assert!(!store.consolidate("ghost", 10.0, 0.2));
    }
}
