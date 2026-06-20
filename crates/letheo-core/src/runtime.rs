//! El Cognitive Runtime — el organismo que "respira".
//!
//! Orquesta las tres capas: percepción (corto plazo) → síntesis (sueño) → arquetipo (largo plazo),
//! con FADE como barrido del garbage collector semántico.
//!
//! El "respirar" es síncrono y dirigido por ticks lógicos (determinista, testeable y offline); el
//! crate `letheo-async` lo monta sobre Tokio para correr de forma asíncrona y no bloquear la
//! percepción mientras el motor sueña. La física es **lazy** en cualquiera de los dos casos:
//! `breathe()` es el único punto donde se recalculan pesos en masa.

use crate::archetype::{ArchetypeStore, Resilience};
use crate::entropy::{Tick, DEFAULT_THETA_FADE};
use crate::evoke::{evoke, evoke_unified, CompressedContext, EvokeRequest, UnifiedContext};
use crate::factstore::{FactStore, RecalledFact, Remember};
use crate::perception::{Perception, PerceptionBuffer};
use crate::synthesis::{distill, DistillConfig};
use crate::vector::Vector;

/// Configuración del runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub theta_fade: f64,
    pub distill: DistillConfig,
    pub resilience: Resilience,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            theta_fade: DEFAULT_THETA_FADE,
            distill: DistillConfig::default(),
            resilience: Resilience::High,
        }
    }
}

/// Reporte de un ciclo de respiración (para observabilidad / sandbox).
#[derive(Debug, Default, Clone)]
pub struct BreathReport {
    pub distilled_subjects: usize,
    pub perceptions_absorbed: usize,
    pub faded: usize,
}

/// El runtime cognitivo: percibe, sueña, evoca, desvanece. Sostiene las **dos capas** bajo la misma
/// física: la semántica (`long_term`, arquetipos/modos — capa-2) y la episódica (`facts`, hechos
/// verbatim — capa-1). Una sola evocación las une (ver [`evoke_unified`](Self::evoke_unified)).
pub struct CognitiveRuntime {
    short_term: PerceptionBuffer,
    long_term: ArchetypeStore,
    facts: FactStore,
    cfg: RuntimeConfig,
}

impl CognitiveRuntime {
    pub fn new(cfg: RuntimeConfig) -> Self {
        Self {
            short_term: PerceptionBuffer::new(),
            long_term: ArchetypeStore::new(),
            facts: FactStore::new(),
            cfg,
        }
    }

    /// `PERCEIVE`: asimila un estímulo crudo en memoria de corto plazo.
    pub fn perceive(&mut self, p: Perception) {
        self.short_term.perceive(p);
    }

    /// Un ciclo de "sueño": para cada sujeto con percepciones vivas, `DISTILL` → `IMPRINT`, luego
    /// `FADE` barre el ruido ya absorbido. Este es el único punto de recálculo masivo de pesos.
    pub fn breathe(&mut self, subjects: &[&str], now: Tick) -> BreathReport {
        let mut report = BreathReport::default();

        for &subject in subjects {
            let alive: Vec<&Perception> = self
                .short_term
                .alive_for(subject, now, self.cfg.theta_fade)
                .collect();
            if alive.is_empty() {
                continue;
            }
            if let Some(iv) = distill(subject, &alive, self.cfg.distill) {
                report.perceptions_absorbed += iv.absorbed;
                self.long_term.imprint(&iv, self.cfg.resilience, now);
                report.distilled_subjects += 1;
            }
        }

        // FADE: el ruido cuyo voto ya vive en el arquetipo se desvanece.
        report.faded = self.short_term.fade_swept(now, self.cfg.theta_fade);
        report
    }

    /// Como [`breathe`](Self::breathe) pero solo destila las percepciones que cumplen el predicado
    /// `keep` (cláusula `WHERE` de `DISTILL`). El barrido `FADE` posterior sigue siendo global.
    pub fn breathe_where(
        &mut self,
        subjects: &[&str],
        now: Tick,
        keep: impl Fn(&Perception) -> bool,
    ) -> BreathReport {
        let mut report = BreathReport::default();

        for &subject in subjects {
            let alive: Vec<&Perception> = self
                .short_term
                .alive_for_where(subject, now, self.cfg.theta_fade, &keep)
                .collect();
            if alive.is_empty() {
                continue;
            }
            if let Some(iv) = distill(subject, &alive, self.cfg.distill) {
                report.perceptions_absorbed += iv.absorbed;
                self.long_term.imprint(&iv, self.cfg.resilience, now);
                report.distilled_subjects += 1;
            }
        }

        report.faded = self.short_term.fade_swept(now, self.cfg.theta_fade);
        report
    }

    /// `FADE … WHERE`: barrido explícito de las percepciones que satisfacen el predicado.
    pub fn fade_where(&mut self, drop_if: impl Fn(&Perception) -> bool) -> usize {
        self.short_term.fade_swept_where(drop_if)
    }

    /// `EVOKE`: resuelve la esencia de un sujeto dentro del presupuesto de tokens (solo capa-2).
    pub fn evoke(&self, req: &EvokeRequest, now: Tick) -> Option<CompressedContext> {
        evoke(&self.long_term, req, now)
    }

    /// Registra un **hecho episódico** (capa-1): contenido verbatim + embedding, bajo la física del
    /// olvido. Dedup semántico por sujeto: un hecho repetido se refuerza, no se duplica.
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
        self.facts.remember(
            subject, text, embedding, provenance, salience, halflife, now,
        )
    }

    /// `RECALL` (capa-1): recupera los hechos exactos más relevantes de un sujeto por física
    /// (`relevancia · vida`) y **los refuerza** (spaced repetition: evocar resetea su decay).
    pub fn recall(
        &mut self,
        subject: &str,
        query: &[f32],
        k: usize,
        now: Tick,
    ) -> Vec<RecalledFact> {
        self.facts
            .recall(subject, query, k, now, self.cfg.theta_fade)
    }

    /// `EVOKE` **unificado**: una sola evocación que responde carácter (capa-2) **y** nominal (capa-1)
    /// bajo UN presupuesto. `fact_budget` es la porción reservada a hechos; `fact_cost` es el tokenizer
    /// real inyectado. Read-only (ver [`evoke_unified`]).
    pub fn evoke_unified(
        &self,
        req: &EvokeRequest,
        query: &[f32],
        fact_budget: usize,
        now: Tick,
        fact_cost: impl Fn(&str) -> usize,
    ) -> UnifiedContext {
        evoke_unified(
            &self.long_term,
            &self.facts,
            req,
            query,
            fact_budget,
            now,
            fact_cost,
        )
    }

    /// `IMPRINT`: **consolida (ancla)** el arquetipo de un sujeto — refuerza su física y la de sus
    /// modos para ganar permanencia. Devuelve `false` si no hay arquetipo (no se puede imprimir lo
    /// que no se ha destilado). Ver [`crate::Archetype::consolidate`].
    pub fn consolidate(&mut self, subject: &str, now: Tick, consolidation: f64) -> bool {
        self.long_term.consolidate(subject, now, consolidation)
    }

    /// **Reflexión** (L8): insights de orden superior sobre la trayectoria del sujeto —transiciones
    /// dominantes y revivals— que no están en ningún evento individual. Vacío si no hay arquetipo.
    /// Ver [`crate::reflection::reflect`].
    pub fn reflect(&self, subject: &str) -> Vec<crate::reflection::Insight> {
        self.long_term
            .get(subject)
            .map(|a| crate::reflection::reflect(&a.arc))
            .unwrap_or_default()
    }

    /// **Sueño reflexivo** (L8): reflexiona sobre el arco del sujeto y **materializa** los insights
    /// como hechos de alta salience en la capa-1 (con embedding derivado de la geometría del arquetipo,
    /// sin provider). Así la sabiduría destilada del arco se vuelve recuperable por `RECALL`. Devuelve
    /// cuántos insights se guardaron. Pensado para invocarse en el ciclo de sueño tras `breathe`.
    pub fn dream_reflect(&mut self, subject: &str, now: Tick) -> usize {
        // Computamos los insights materializados con el arquetipo prestado, y soltamos el préstamo
        // antes de escribir en la capa-1 (campos distintos, préstamos disjuntos).
        let materials: Vec<(String, crate::vector::Vector)> = match self.long_term.get(subject) {
            Some(a) => crate::reflection::reflect(&a.arc)
                .iter()
                .filter_map(|ins| crate::reflection::materialize(a, ins))
                .collect(),
            None => return 0,
        };
        let halflife = 90.0 * 86_400.0; // 90 días: los insights son durables, pero no inmortales.
        let n = materials.len();
        for (text, embedding) in materials {
            self.facts.remember(
                subject,
                text,
                embedding,
                "reflection",
                crate::reflection::DEFAULT_INSIGHT_SALIENCE,
                halflife,
                now,
            );
        }
        n
    }

    /// `FADE` de la capa episódica: barre los hechos cuya vida cayó bajo θ_fade. Devuelve cuántos.
    pub fn fade_facts(&mut self, now: Tick) -> usize {
        self.facts.fade(now, self.cfg.theta_fade)
    }

    /// Acceso de solo lectura a la memoria episódica (para persistirla con `letheo-persist`).
    pub fn facts(&self) -> &FactStore {
        &self.facts
    }

    /// Acceso mutable a la memoria episódica (para rehidratarla desde disco).
    pub fn facts_mut(&mut self) -> &mut FactStore {
        &mut self.facts
    }

    /// `FADE` explícito: barrido del GC semántico sin un ciclo completo de sueño. Útil cuando el
    /// programa MQL quiere expresar olvido como acto sin disparar `DISTILL`.
    pub fn fade_only(&mut self, now: Tick, theta: f64) -> usize {
        self.short_term.fade_swept(now, theta)
    }

    /// Acceso de solo lectura a la memoria de largo plazo (para persistirla).
    pub fn long_term(&self) -> &ArchetypeStore {
        &self.long_term
    }

    /// Acceso mutable a la memoria de largo plazo (para rehidratarla desde disco).
    pub fn long_term_mut(&mut self) -> &mut ArchetypeStore {
        &mut self.long_term
    }

    pub fn short_term_len(&self) -> usize {
        self.short_term.len()
    }

    pub fn long_term_len(&self) -> usize {
        self.long_term.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HF: f64 = 3600.0;

    fn perception(subject: &str, e: Vec<f32>, salience: f64) -> Perception {
        Perception::new(subject, e, salience, HF, 0.0)
    }

    #[test]
    fn full_cycle_perceive_breathe_evoke() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());

        // 1000 percepciones casi idénticas (un hábito) + unas pocas anómalas.
        for _ in 0..1000 {
            rt.perceive(perception("user:Xolotl", vec![1.0, 0.0], 1.0));
        }
        for _ in 0..3 {
            rt.perceive(perception("user:Xolotl", vec![0.0, 1.0], 1.0));
        }
        assert_eq!(rt.short_term_len(), 1003);

        // El runtime sueña.
        let report = rt.breathe(&["user:Xolotl"], 0.0);
        assert_eq!(report.distilled_subjects, 1);
        assert_eq!(report.perceptions_absorbed, 1003);
        assert_eq!(rt.long_term_len(), 1, "una esencia consolidada");

        // EVOKE devuelve contexto ultra-comprimido dentro del presupuesto.
        let req = EvokeRequest::new("user:Xolotl", 800);
        let ctx = rt.evoke(&req, 0.0).unwrap();
        assert_eq!(ctx.represented, 1003);
        assert!(ctx.token_estimate <= 800);
        assert!(ctx.compression_ratio() > 100.0);
    }

    #[test]
    fn unified_runtime_evoke_spans_both_layers() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        // Capa-2: percibe un hábito y sueña → gist de carácter.
        for _ in 0..100 {
            rt.perceive(Perception::new(
                "user:X",
                vec![1.0, 0.0],
                1.0,
                86_400.0,
                0.0,
            ));
        }
        rt.breathe(&["user:X"], 0.0);
        // Capa-1: registra un hecho exacto (lo nominal).
        rt.remember(
            "user:X",
            "prefers window seat",
            vec![0.0, 1.0],
            "agent",
            1.0,
            86_400.0,
            0.0,
        );

        let req = EvokeRequest::new("user:X", 800);
        let u = rt.evoke_unified(
            &req,
            &[0.0, 1.0],
            100,
            0.0,
            crate::evoke::approx_token_count,
        );
        assert!(u.gist.is_some(), "carácter desde la capa-2");
        assert_eq!(u.facts.len(), 1, "nominal desde la capa-1");
        assert_eq!(u.facts[0].text, "prefers window seat");
        assert!(u.total_tokens <= 800);
    }

    #[test]
    fn runtime_reflect_surfaces_arc_transition() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        // Dos ciclos de sueño con comportamientos distintos → un arco con una transición trail→yoga.
        // El trail decae rápido (halflife 1s) para que no se cuele en el segundo ciclo (ver D14).
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![1.0, 0.0], 1.0, 1.0, 0.0).with_trait("act", "trail"),
            );
        }
        rt.breathe(&["u"], 0.0);
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![0.0, 1.0], 1.0, 86_400.0, 0.0).with_trait("act", "yoga"),
            );
        }
        rt.breathe(&["u"], 100.0);

        let insights = rt.reflect("u");
        assert!(
            insights.iter().any(
                |i| matches!(i, crate::reflection::Insight::Transition { from, to, .. }
                if from == "trail" && to == "yoga")
            ),
            "la reflexión sintetiza la transición del arco: {insights:?}"
        );
        assert!(
            rt.reflect("ghost").is_empty(),
            "sin arquetipo, sin insights"
        );
    }

    #[test]
    fn dream_reflect_materializes_insights_as_recallable_facts() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![1.0, 0.0], 1.0, 1.0, 0.0).with_trait("act", "trail"),
            );
        }
        rt.breathe(&["u"], 0.0);
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![0.0, 1.0], 1.0, 86_400.0, 0.0).with_trait("act", "yoga"),
            );
        }
        rt.breathe(&["u"], 100.0);

        // El sueño reflexivo materializa la transición como hecho de alta salience…
        let stored = rt.dream_reflect("u", 100.0);
        assert_eq!(stored, 1, "se guardó la transición trail→yoga como hecho");
        // …recuperable por resonancia con el comportamiento destino (capa-1).
        let hits = rt.recall("u", &[0.0, 1.0], 1, 100.0);
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].text.contains("trail → yoga"),
            "el insight se recupera: {}",
            hits[0].text
        );
    }

    #[test]
    fn noise_fades_after_breathing() {
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        rt.perceive(perception("user:X", vec![1.0, 0.0], 0.2)); // ruido débil
                                                                // Soñamos mucho después: el ruido ya cayó bajo θ_fade.
        let report = rt.breathe(&["user:X"], HF * 5.0);
        assert_eq!(report.faded, 1);
        assert_eq!(rt.short_term_len(), 0);
    }
}
