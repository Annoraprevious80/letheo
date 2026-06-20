//! Capa de Hechos — la memoria **episódica** (capa-1, el hipocampo).
//!
//! El arquetipo (capa-2, neocórtex) *generaliza*: comprime el comportamiento en modos y olvida lo
//! redundante. Pero hay cosas que un agente no puede permitirse promediar — "es alérgico a los
//! cacahuetes", "el coche es el rojo", "ya entregó el milestone 3". Eso es **conocimiento episódico
//! verbatim**: específico, sin pérdida, recuperable al pie de la letra.
//!
//! Hasta ahora esa capa vivía **fuera del motor**, como una lista en Python en la capa consumidora:
//! sin física de olvido, sin deduplicación, sin índice. Aquí entra al core bajo la **misma física**
//! que todo lo demás ([`EntropyTrace`]). Es el modelo de *Complementary Learning Systems* hecho
//! literal: dos representaciones (episódica rápida ↔ semántica lenta), **una sola física de
//! decaimiento**. Un hecho que no se vuelve a tocar decae y el GC semántico lo barre; uno que se
//! evoca o se repite se refuerza (spaced repetition). No hay dos sistemas pegados con cinta: hay un
//! motor con dos capas.

use crate::entropy::{EntropyTrace, Tick};
use crate::vector::{cosine, Vector};

/// Umbral de coseno por encima del cual dos hechos se consideran **el mismo hecho** (dedup → refuerzo
/// en vez de inserción). Alto a propósito: la capa-1 es verbatim, así que solo colapsa repeticiones
/// casi idénticas; una paráfrasis lejana se guarda como hecho distinto. Física declarada, ajustable
/// vía [`FactStore::with_dedup`] (no constante mágica — coherente con la disciplina VERDAD 100%).
pub const DEFAULT_FACT_DEDUP: f32 = 0.95;

/// Consolidación por refuerzo de un hecho (repetición o evocación): fracción en que se reduce λ, con
/// lo que la vida media crece (spaced repetition / FSRS). Pequeña a propósito: recordar un hecho lo
/// afianza, pero el olvido sigue siendo real si deja de tocarse.
pub const DEFAULT_FACT_CONSOLIDATION: f64 = 0.1;

/// Un hecho episódico: contenido **verbatim** + su embedding + su física de olvido.
#[derive(Debug, Clone)]
pub struct Fact {
    /// Sujeto al que pertenece el hecho (p. ej. "user:Xolotl", "agent:adder").
    pub subject: String,
    /// El contenido exacto, sin pérdida. Es la promesa de la capa-1: se devuelve tal cual se supo.
    pub text: String,
    /// Embedding semántico del hecho (del Provider de inferencia). Base de la recuperación y el dedup.
    pub embedding: Vector,
    /// Procedencia: qué agente/fuente aportó el hecho. Para auditoría y el mercado de memoria (L12).
    pub provenance: String,
    /// Tick en que se supo el hecho (creación). Inmutable; distinto de `trace.last_touch` (refuerzo).
    pub created_at: Tick,
    /// Física del olvido del hecho: decae salvo refuerzo/evocación. La misma [`EntropyTrace`] que rige
    /// percepciones, modos y arquetipos — una sola física para las dos capas.
    pub trace: EntropyTrace,
}

/// Resultado de `remember`: si el hecho era nuevo o si colapsó (refuerzo) sobre uno ya presente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Remember {
    /// Hecho nuevo: se insertó en el store.
    Inserted,
    /// Hecho ya conocido (≥ `theta_dedup`): no se duplicó, se reforzó el existente.
    Merged,
}

/// Un hecho recuperado por `recall`: su texto verbatim + procedencia + la puntuación física con que
/// salió. Es un clon desacoplado del store (el `recall` reforzó el hecho de origen al evocarlo).
#[derive(Debug, Clone)]
pub struct RecalledFact {
    pub text: String,
    pub provenance: String,
    /// `relevancia · vida` con que se rankeó (ver [`FactStore::recall`]).
    pub score: f64,
}

/// La memoria episódica: hechos verbatim con olvido, deduplicación e índice (Flat por ahora; el ANN
/// de L3 se enchufa aquí sin cambiar la semántica). Multi-sujeto: el dedup y el recall se aíslan por
/// sujeto, como [`crate::perception::PerceptionBuffer`].
#[derive(Debug, Clone)]
pub struct FactStore {
    facts: Vec<Fact>,
    theta_dedup: f32,
}

impl Default for FactStore {
    fn default() -> Self {
        Self {
            facts: Vec::new(),
            theta_dedup: DEFAULT_FACT_DEDUP,
        }
    }
}

impl FactStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Crea un store con un umbral de dedup explícito (ver [`DEFAULT_FACT_DEDUP`]).
    pub fn with_dedup(theta_dedup: f32) -> Self {
        Self {
            facts: Vec::new(),
            theta_dedup,
        }
    }

    pub fn theta_dedup(&self) -> f32 {
        self.theta_dedup
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Recorre los hechos en orden de inserción (para snapshots / persistencia).
    pub fn iter(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter()
    }

    /// Inserta un hecho ya construido **sin** dedup ni física nueva (p. ej. al rehidratar desde disco).
    /// Para registrar conocimiento nuevo úsese [`remember`](Self::remember).
    pub fn insert(&mut self, fact: Fact) {
        self.facts.push(fact);
    }

    /// Hechos vivos de un sujeto (peso ≥ θ_fade) en `now`. Lazy: evalúa el peso aquí, no por tic.
    pub fn alive_for<'a>(
        &'a self,
        subject: &'a str,
        now: Tick,
        theta_fade: f64,
    ) -> impl Iterator<Item = &'a Fact> + 'a {
        self.facts
            .iter()
            .filter(move |f| f.subject == subject && f.trace.weight(now) >= theta_fade)
    }

    /// Registra un hecho. Si ya existe uno del **mismo sujeto** cuya dirección resuena ≥ `theta_dedup`,
    /// no se duplica: se **refuerza** el existente (la repetición consolida, no infla el store). Si no,
    /// se inserta con su propia física de olvido (`salience`, `halflife`).
    #[allow(clippy::too_many_arguments)]
    pub fn remember(
        &mut self,
        subject: impl Into<String>,
        text: impl Into<String>,
        embedding: Vector,
        provenance: impl Into<String>,
        salience: f64,
        halflife: f64,
        now: Tick,
    ) -> Remember {
        let subject = subject.into();
        // Dedup dentro del sujeto: el hecho más parecido, si supera el umbral, es "el mismo hecho".
        let mut best = self.theta_dedup;
        let mut best_i: Option<usize> = None;
        for (i, f) in self.facts.iter().enumerate() {
            if f.subject != subject {
                continue;
            }
            let c = cosine(&f.embedding, &embedding);
            if c >= best {
                best = c;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => {
                self.facts[i]
                    .trace
                    .reinforce(now, DEFAULT_FACT_CONSOLIDATION);
                Remember::Merged
            }
            None => {
                self.facts.push(Fact {
                    subject,
                    text: text.into(),
                    embedding,
                    provenance: provenance.into(),
                    created_at: now,
                    trace: EntropyTrace::new(salience, halflife, now),
                });
                Remember::Inserted
            }
        }
    }

    /// **Recuperación dirigida** (capa-1): los `k` hechos del sujeto que mejor puntúan con la consulta,
    /// rankeados por la física nativa `score = max(0, relevancia) · weight(now)` — la misma forma que
    /// L2, no un parche aditivo. Evocar **refuerza** los hechos devueltos (spaced repetition: un hecho
    /// recordado resetea su decay y sobrevive; uno que nunca se evoca se desvanece). Devuelve copias
    /// desacopladas ([`RecalledFact`]) verbatim.
    pub fn recall(
        &mut self,
        subject: &str,
        query: &[f32],
        k: usize,
        now: Tick,
        theta_fade: f64,
    ) -> Vec<RecalledFact> {
        let mut scored: Vec<(f64, usize)> = self
            .facts
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if f.subject != subject {
                    return None;
                }
                let life = f.trace.weight(now); // `e^x` una sola vez por hecho (era 2×)
                if life < theta_fade {
                    return None;
                }
                let relevance = cosine(&f.embedding, query).max(0.0) as f64;
                Some((relevance * life, i))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(score, i)| {
                // Evocar es tocar: el hecho recuperado se refuerza (Δt→0) → su olvido se aplaza.
                self.facts[i]
                    .trace
                    .reinforce(now, DEFAULT_FACT_CONSOLIDATION);
                let f = &self.facts[i];
                RecalledFact {
                    text: f.text.clone(),
                    provenance: f.provenance.clone(),
                    score,
                }
            })
            .collect()
    }

    /// Búsqueda **read-only** por la misma física que `recall`, sin reforzar (para inspección/tests y
    /// para componer un EVOKE unificado sin efectos colaterales). Devuelve `(score, &Fact)` top-`k`.
    pub fn search<'a>(
        &'a self,
        subject: &str,
        query: &[f32],
        k: usize,
        now: Tick,
        theta_fade: f64,
    ) -> Vec<(f64, &'a Fact)> {
        let mut scored: Vec<(f64, &Fact)> = self
            .facts
            .iter()
            .filter_map(|f| {
                if f.subject != subject {
                    return None;
                }
                let life = f.trace.weight(now); // `e^x` una sola vez por hecho (era 2×)
                if life < theta_fade {
                    return None;
                }
                let relevance = cosine(&f.embedding, query).max(0.0) as f64;
                Some((relevance * life, f))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// `FADE` de la capa episódica: barre los hechos bajo umbral (su vida cayó por debajo de θ_fade) y
    /// devuelve cuántos se desvanecieron. El olvido es real también para los hechos exactos.
    pub fn fade(&mut self, now: Tick, theta_fade: f64) -> usize {
        let before = self.facts.len();
        self.facts.retain(|f| f.trace.weight(now) >= theta_fade);
        before - self.facts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::DEFAULT_THETA_FADE;

    const DAY: f64 = 86_400.0;

    #[test]
    fn dedup_collapses_near_identical_facts_and_reinforces() {
        let mut fs = FactStore::new();
        assert_eq!(
            fs.remember(
                "u",
                "bought running shoes",
                vec![1.0, 0.0],
                "agentA",
                1.0,
                DAY,
                0.0
            ),
            Remember::Inserted
        );
        // El mismo hecho otra vez (embedding idéntico): no se duplica, se refuerza.
        assert_eq!(
            fs.remember(
                "u",
                "bought running shoes",
                vec![1.0, 0.0],
                "agentA",
                1.0,
                DAY,
                1.0
            ),
            Remember::Merged
        );
        assert_eq!(fs.len(), 1, "el hecho repetido no infla el store");
        assert!(
            fs.iter().next().unwrap().trace.reinforcement > 0.0,
            "la repetición consolida"
        );
    }

    #[test]
    fn distinct_facts_are_kept_separate() {
        let mut fs = FactStore::new();
        fs.remember("u", "loves noir films", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        fs.remember(
            "u",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "a",
            1.0,
            DAY,
            0.0,
        ); // ortogonal
        assert_eq!(fs.len(), 2, "hechos distintos no se colapsan");
    }

    #[test]
    fn recall_returns_the_relevant_fact_verbatim() {
        let mut fs = FactStore::new();
        fs.remember(
            "u",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "a",
            1.0,
            DAY,
            0.0,
        );
        fs.remember("u", "drives a red car", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        let hits = fs.recall("u", &[0.0, 1.0], 1, 0.0, DEFAULT_THETA_FADE);
        assert_eq!(hits.len(), 1);
        // La promesa de la capa-1: el hecho exacto, palabra por palabra (no un gist).
        assert_eq!(hits[0].text, "allergic to peanuts");
    }

    #[test]
    fn recall_ranks_fresh_over_stale_at_equal_relevance() {
        let half = 30.0 * DAY;
        let mut fs = FactStore::new();
        // Direcciones casi iguales pero por debajo del umbral de dedup (cos≈0.92 < 0.95) → dos hechos.
        fs.remember("u", "stale fact", vec![1.0, 0.0], "a", 1.0, half, 0.0);
        fs.remember("u", "fresh fact", vec![0.92, 0.392], "a", 1.0, half, half);
        let q = [0.96, 0.2]; // a medio camino: relevancia comparable para ambos
        let hits = fs.search("u", &q, 2, half, DEFAULT_THETA_FADE);
        assert_eq!(hits.len(), 2, "ambos siguen vivos");
        assert_eq!(
            hits[0].1.text, "fresh fact",
            "a relevancia comparable, el más vivo primero"
        );
    }

    #[test]
    fn recalling_a_fact_resets_its_decay() {
        let half = 30.0 * DAY;
        let mut fs = FactStore::new();
        fs.remember("u", "recalled fact", vec![1.0, 0.0], "a", 1.0, half, 0.0);
        fs.remember("u", "forgotten fact", vec![0.0, 1.0], "a", 1.0, half, 0.0);
        // En t=half evocamos solo el primero (la consulta lo señala): se refuerza → Δt→0.
        let hits = fs.recall("u", &[1.0, 0.0], 1, half, DEFAULT_THETA_FADE);
        assert_eq!(hits[0].text, "recalled fact");
        // Mucho más tarde barremos: el evocado sobrevive (reinició su reloj), el otro se desvanece.
        let later = half * 5.0;
        let faded = fs.fade(later, DEFAULT_THETA_FADE);
        assert_eq!(faded, 1, "solo el hecho que nunca se evocó se desvanece");
        assert_eq!(fs.len(), 1);
        assert_eq!(fs.iter().next().unwrap().text, "recalled fact");
    }

    #[test]
    fn fade_sweeps_decayed_facts() {
        let mut fs = FactStore::new();
        fs.remember("u", "fleeting", vec![1.0, 0.0], "a", 0.2, DAY, 0.0); // baja salience, vida corta
        fs.remember("u", "durable", vec![0.0, 1.0], "a", 1.0, DAY * 100.0, 0.0);
        let faded = fs.fade(DAY * 5.0, DEFAULT_THETA_FADE);
        assert_eq!(faded, 1, "solo el hecho frágil se desvanece");
        assert_eq!(fs.iter().next().unwrap().text, "durable");
    }

    #[test]
    fn recall_isolates_subjects() {
        let mut fs = FactStore::new();
        fs.remember("alice", "alice secret", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        fs.remember("bob", "bob secret", vec![1.0, 0.0], "a", 1.0, DAY, 0.0);
        // Misma dirección pero distinto sujeto ⇒ NO se deduplican (el dedup es por sujeto).
        assert_eq!(fs.len(), 2);
        let hits = fs.recall("alice", &[1.0, 0.0], 5, 0.0, DEFAULT_THETA_FADE);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].text, "alice secret",
            "los hechos de un sujeto no se filtran a otro"
        );
    }
}
