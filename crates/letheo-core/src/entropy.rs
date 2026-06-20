//! La Física del Olvido — fuente de verdad: `docs/01-physics.md`.
//!
//! `weight(t) = salience · e^(−λ·Δt) · (1 + reinforcement)`,  `λ = ln2 / halflife`.
//!
//! **Lazy evaluation (crítico):** el peso NO se recalcula por tic de reloj. Cada recuerdo guarda
//! solo sus parámetros + `last_touch`; `weight()` es una función pura evaluada bajo demanda, solo
//! durante `DISTILL`, `EVOKE` o el barrido del garbage collector semántico.

/// Reloj lógico del runtime, en segundos. Abstraído para que los tests sean deterministas y para no
/// atar la física a un reloj de pared (el tiempo es un coeficiente de entropía, no un timestamp).
pub type Tick = f64;

/// `ln(2)`, usado para convertir vida media ↔ tasa de decaimiento.
pub const LN2: f64 = std::f64::consts::LN_2;

/// Vida media **máxima** de un recuerdo (segundos): ~10 años. La consolidación repetida reduce λ, pero
/// nunca por debajo del suelo derivado de aquí — así el olvido sigue siendo posible y **nada se vuelve
/// inmortal por mucho que se reviste**. Coherente con la tesis "el olvido es una feature".
pub const MAX_HALFLIFE_SECS: f64 = 10.0 * 365.0 * 86_400.0;
/// Suelo de λ derivado de [`MAX_HALFLIFE_SECS`]: la consolidación no puede llevar λ por debajo de esto.
pub const LAMBDA_FLOOR: f64 = LN2 / MAX_HALFLIFE_SECS;

/// Convierte una vida media (en segundos) a la tasa de decaimiento λ.
#[inline]
pub fn lambda_from_halflife(halflife: f64) -> f64 {
    assert!(halflife > 0.0, "halflife debe ser > 0");
    LN2 / halflife
}

/// Convierte λ de vuelta a vida media (segundos).
#[inline]
pub fn halflife_from_lambda(lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda debe ser > 0");
    LN2 / lambda
}

/// El rastro de entropía de un recuerdo: todo lo necesario para evaluar su peso perezosamente.
///
/// No almacena el peso; lo *deriva*. Esto es lo que permite tener millones de recuerdos sin
/// recalcular `e^x` por tic.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropyTrace {
    /// Carga inicial del estímulo al ser percibido, `(0, 1]` típico.
    pub salience: f64,
    /// Tasa de decaimiento λ. Puede reducirse con la repetición (consolidación).
    pub lambda: f64,
    /// Refuerzos acumulados (cada EVOKE/repetición suma).
    pub reinforcement: f64,
    /// Tick del último refuerzo/evocación. Δt = now − last_touch.
    pub last_touch: Tick,
}

impl EntropyTrace {
    /// Crea un rastro nuevo a partir de salience y vida media (segundos), tocado en `now`.
    pub fn new(salience: f64, halflife: f64, now: Tick) -> Self {
        Self {
            salience,
            lambda: lambda_from_halflife(halflife),
            reinforcement: 0.0,
            last_touch: now,
        }
    }

    /// Tiempo transcurrido desde el último contacto. Clampeado a `≥ 0` (un reloj que retrocede no
    /// "des-olvida").
    #[inline]
    pub fn delta_t(&self, now: Tick) -> f64 {
        (now - self.last_touch).max(0.0)
    }

    /// El peso del recuerdo en `now`. Función pura: no muta nada (lazy).
    ///
    /// `weight = salience · e^(−λ·Δt) · (1 + reinforcement)`
    pub fn weight(&self, now: Tick) -> f64 {
        let decay = (-self.lambda * self.delta_t(now)).exp();
        self.salience * decay * (1.0 + self.reinforcement)
    }

    /// Refuerzo (al ser evocado o repetido): resetea Δt a 0 y suma reinforcement con **rendimientos
    /// decrecientes** (`+= 1/(1+reinforcement)` → crece ~√n, no lineal: la frecuencia de uso ya no
    /// domina el ranking sin techo; el 1er refuerzo sigue sumando 1.0). Opcionalmente consolida
    /// reduciendo λ, pero **nunca por debajo de [`LAMBDA_FLOOR`]** (nada se vuelve inmortal).
    ///
    /// `consolidation` en `[0, 1)`: fracción en que se reduce λ por este refuerzo (0 = sin cambio).
    pub fn reinforce(&mut self, now: Tick, consolidation: f64) {
        debug_assert!((0.0..1.0).contains(&consolidation));
        self.last_touch = now;
        self.reinforcement += 1.0 / (1.0 + self.reinforcement);
        self.lambda = (self.lambda * (1.0 - consolidation)).max(LAMBDA_FLOOR);
    }

    /// ¿Está el recuerdo bajo el umbral de olvido? (candidato a FADE).
    #[inline]
    pub fn is_faded(&self, now: Tick, theta_fade: f64) -> bool {
        self.weight(now) < theta_fade
    }
}

/// Umbral por defecto bajo el cual un recuerdo se vuelve candidato a `FADE`
/// (ver `docs/01-physics.md` §3). Calibrable por dominio.
pub const DEFAULT_THETA_FADE: f64 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_DAY: f64 = 12.0 * 3600.0;

    #[test]
    fn lambda_halflife_roundtrip() {
        let l = lambda_from_halflife(HALF_DAY);
        assert!((halflife_from_lambda(l) - HALF_DAY).abs() < 1e-6);
    }

    #[test]
    fn weight_halves_after_one_halflife() {
        let t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        let w0 = t.weight(0.0);
        let w1 = t.weight(HALF_DAY);
        assert!((w0 - 1.0).abs() < 1e-9);
        assert!(
            (w1 - 0.5).abs() < 1e-6,
            "tras una vida media el peso se reduce a la mitad"
        );
    }

    #[test]
    fn weight_is_monotonic_decreasing_in_time() {
        let t = EntropyTrace::new(0.8, HALF_DAY, 0.0);
        let mut prev = f64::INFINITY;
        for step in 0..50 {
            let w = t.weight(step as f64 * 3600.0);
            assert!(w < prev, "el peso debe decrecer monótonamente con Δt");
            prev = w;
        }
    }

    #[test]
    fn weight_is_asymptotic_never_zero() {
        let t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        let w = t.weight(HALF_DAY * 1000.0);
        assert!(
            w > 0.0,
            "el decaimiento es asintótico: nunca llega exactamente a 0"
        );
        assert!(w < 1e-6);
    }

    #[test]
    fn reinforcement_resets_delta_t_and_raises_weight() {
        let mut t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        // Dejamos decaer una vida media: peso ~0.5.
        let decayed = t.weight(HALF_DAY);
        assert!((decayed - 0.5).abs() < 1e-6);
        // Evocar en ese momento: Δt→0 y reinforcement→1.
        t.reinforce(HALF_DAY, 0.0);
        let after = t.weight(HALF_DAY);
        assert!(after > decayed, "recordar refuerza el peso");
        // salience(1) · e^0 · (1 + 1) = 2.0
        assert!((after - 2.0).abs() < 1e-6);
    }

    #[test]
    fn consolidation_extends_halflife() {
        let mut t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        let lambda_before = t.lambda;
        t.reinforce(0.0, 0.5); // reduce λ a la mitad → vida media ×2
        assert!((t.lambda - lambda_before * 0.5).abs() < 1e-12);
        assert!(
            t.lambda < lambda_before,
            "la consolidación alarga la vida media"
        );
    }

    #[test]
    fn reinforcement_has_diminishing_returns_and_lambda_floor() {
        // El 1er refuerzo suma 1.0 (compatibilidad), los siguientes cada vez menos (~√n, no lineal):
        // la frecuencia de uso ya no domina el peso sin techo.
        let mut t = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        t.reinforce(0.0, 0.0);
        assert!((t.reinforcement - 1.0).abs() < 1e-9, "1er refuerzo = +1.0");
        for _ in 0..50 {
            t.reinforce(0.0, 0.0);
        }
        assert!(
            t.reinforcement < 12.0,
            "51 refuerzos ⇒ ≈√n, no 51: {}",
            t.reinforcement
        );
        assert!(t.reinforcement > 1.0, "pero sigue creciendo");

        // Suelo de λ: por mucho que consolides, la vida media no se vuelve infinita (no inmortalidad).
        let mut c = EntropyTrace::new(1.0, HALF_DAY, 0.0);
        for _ in 0..500 {
            c.reinforce(0.0, 0.5);
        }
        assert!(
            (c.lambda - LAMBDA_FLOOR).abs() < 1e-18,
            "λ se topa en el suelo, no en 0"
        );
    }

    #[test]
    fn fade_triggers_below_threshold() {
        let t = EntropyTrace::new(0.2, HALF_DAY, 0.0);
        assert!(
            !t.is_faded(0.0, DEFAULT_THETA_FADE),
            "recién percibido no se desvanece"
        );
        // Tras suficiente tiempo cae bajo θ_fade.
        assert!(t.is_faded(HALF_DAY * 5.0, DEFAULT_THETA_FADE));
    }
}
