//! Capa de Percepción — memoria volátil de corto plazo.
//!
//! Una `Perception` es un estímulo crudo recién asimilado (`PERCEIVE`). Nace decayendo: su
//! `EntropyTrace` determina cuánto pesa en cada instante. Deliberadamente frágil — si nada la
//! refuerza, cae bajo `θ_fade` y el GC semántico la barre. Ver `docs/03-engine-pipeline.md`.

use crate::entropy::{EntropyTrace, Tick};
use crate::vector::Vector;
use std::collections::HashMap;

/// Un estímulo percibido: el embedding semántico + sus rasgos + su rastro de entropía.
#[derive(Debug, Clone)]
pub struct Perception {
    /// Sujeto al que pertenece, p.ej. "user:Xolotl".
    pub subject: String,
    /// Embedding semántico del estímulo (del Provider de inferencia).
    pub embedding: Vector,
    /// Rasgos crudos (act, object, hue, urgency...). No es un esquema fijo.
    pub traits: HashMap<String, String>,
    /// Física del olvido de este estímulo.
    pub trace: EntropyTrace,
}

impl Perception {
    pub fn new(
        subject: impl Into<String>,
        embedding: Vector,
        salience: f64,
        halflife: f64,
        now: Tick,
    ) -> Self {
        Self {
            subject: subject.into(),
            embedding,
            traits: HashMap::new(),
            trace: EntropyTrace::new(salience, halflife, now),
        }
    }

    pub fn with_trait(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.traits.insert(key.into(), value.into());
        self
    }

    /// Peso actual (lazy). Atajo sobre el rastro de entropía.
    #[inline]
    pub fn weight(&self, now: Tick) -> f64 {
        self.trace.weight(now)
    }

    /// Texto representativo del estímulo: los **valores** de los rasgos en orden estable de clave.
    /// Es la etiqueta léxica que sobrevive a la destilación para que la prosa nombre el contenido
    /// (no solo vectores). P.ej. `{act: purchase, object: shoes}` → "purchase shoes".
    pub fn representative_text(&self) -> String {
        let mut keys: Vec<&String> = self.traits.keys().collect();
        keys.sort();
        keys.iter()
            .map(|k| self.traits[*k].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// La memoria sensorial de corto plazo: percepciones vivas, aún no barridas por FADE.
#[derive(Debug, Default)]
pub struct PerceptionBuffer {
    perceptions: Vec<Perception>,
}

impl PerceptionBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `PERCEIVE`: asimila un estímulo crudo.
    pub fn perceive(&mut self, p: Perception) {
        self.perceptions.push(p);
    }

    pub fn len(&self) -> usize {
        self.perceptions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perceptions.is_empty()
    }

    /// Percepciones vivas de un sujeto (peso ≥ θ_fade) en `now`. Lazy: evalúa el peso aquí, no por tic.
    pub fn alive_for<'a>(
        &'a self,
        subject: &'a str,
        now: Tick,
        theta_fade: f64,
    ) -> impl Iterator<Item = &'a Perception> + 'a {
        self.perceptions
            .iter()
            .filter(move |p| p.subject == subject && p.weight(now) >= theta_fade)
    }

    /// Como [`alive_for`](Self::alive_for) pero además exige que el predicado del usuario (cláusula
    /// `WHERE` de `DISTILL`) se cumpla. El predicado se evalúa fuera del core, manteniéndolo
    /// desacoplado de `letheo-mql`.
    pub fn alive_for_where<'a>(
        &'a self,
        subject: &'a str,
        now: Tick,
        theta_fade: f64,
        keep: impl Fn(&Perception) -> bool + 'a,
    ) -> impl Iterator<Item = &'a Perception> + 'a {
        self.perceptions
            .iter()
            .filter(move |p| p.subject == subject && p.weight(now) >= theta_fade && keep(p))
    }

    /// `FADE`: barrido del garbage collector semántico. Elimina las percepciones bajo umbral y
    /// devuelve cuántas se desvanecieron. Su contribución al arquetipo ya fue absorbida por DISTILL.
    pub fn fade_swept(&mut self, now: Tick, theta_fade: f64) -> usize {
        let before = self.perceptions.len();
        self.perceptions.retain(|p| p.weight(now) >= theta_fade);
        before - self.perceptions.len()
    }

    /// `FADE … WHERE`: desvanece las percepciones que satisfacen el predicado del usuario (la
    /// cláusula `WHERE` *es* la condición de olvido). Devuelve cuántas se barrieron.
    pub fn fade_swept_where(&mut self, drop_if: impl Fn(&Perception) -> bool) -> usize {
        let before = self.perceptions.len();
        self.perceptions.retain(|p| !drop_if(p));
        before - self.perceptions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_DAY: f64 = 12.0 * 3600.0;

    #[test]
    fn perceive_and_count() {
        let mut buf = PerceptionBuffer::new();
        buf.perceive(Perception::new(
            "user:X",
            vec![1.0, 0.0],
            0.5,
            HALF_DAY,
            0.0,
        ));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn fade_sweep_removes_decayed_noise() {
        let mut buf = PerceptionBuffer::new();
        // Ruido de baja salience y vida media corta.
        buf.perceive(Perception::new("user:X", vec![1.0], 0.2, HALF_DAY, 0.0));
        // Señal fuerte y persistente.
        buf.perceive(Perception::new(
            "user:X",
            vec![1.0],
            1.0,
            HALF_DAY * 100.0,
            0.0,
        ));

        let faded = buf.fade_swept(HALF_DAY * 5.0, crate::entropy::DEFAULT_THETA_FADE);
        assert_eq!(faded, 1, "solo el ruido se desvanece");
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn alive_filters_by_subject() {
        let mut buf = PerceptionBuffer::new();
        buf.perceive(Perception::new("user:X", vec![1.0], 1.0, HALF_DAY, 0.0));
        buf.perceive(Perception::new("user:Y", vec![1.0], 1.0, HALF_DAY, 0.0));
        let n = buf
            .alive_for("user:X", 0.0, crate::entropy::DEFAULT_THETA_FADE)
            .count();
        assert_eq!(n, 1);
    }
}
