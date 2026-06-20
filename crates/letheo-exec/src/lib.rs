//! # letheo-exec · Ejecutor MQL
//!
//! Cierra el lazo del lenguaje: toma el AST que produce `letheo-mql` y lo traduce en operaciones
//! reales sobre `CognitiveRuntime`. La capa de orquestación (Python, CLI, agente) ya no necesita
//! conocer la API de Rust del core — habla MQL.
//!
//! Mapeo biológico:
//! - `PERCEIVE` → `Runtime::perceive(...)` con embedding del Provider (rasgos como texto).
//! - `DISTILL`  → `Runtime::breathe([subject])` (un ciclo de sueño para ese sujeto).
//! - `EVOKE`    → `Runtime::evoke(...)` con el token budget de la sentencia.
//! - `FADE`     → barrido del GC semántico (se realiza dentro del próximo `breathe`).
//! - `IMPRINT`  → consolida/ancla el arquetipo del sujeto (refuerza su física y la de sus modos).

use letheo_core::{
    ArcDetail, BreathReport, CognitiveRuntime, CompressedContext, EvokeRequest, Fact, Perception,
    RecalledFact, Tick,
};
use letheo_inference::Provider;
use letheo_mql::ast::{
    Distill, Evoke, Facts, Fade, Field, Imprint, Perceive, Predicate, Projection, Recall,
    Reinforce, Resilience, Resolution, Statement,
};

/// Resultado de ejecutar una sentencia MQL.
// `Evoked` lleva un `CompressedContext` grande frente a las demás variantes; es un tipo de resultado
// efímero (uno por sentencia), así que la diferencia de tamaño es inocua — no merece un `Box`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ExecResult {
    Perceived {
        subject: String,
    },
    Dreamed(BreathReport),
    Evoked(CompressedContext),
    Faded {
        swept: usize,
    },
    Imprinted {
        archetype: String,
        note: &'static str,
    },
    /// `RECALL`: hechos episódicos recuperados (verbatim), ordenados por física. Read-only.
    Recalled(Vec<RecalledFact>),
    /// `REINFORCE`: cuántos hechos se reforzaron (su decay se reseteó).
    Reinforced {
        count: usize,
    },
}

/// Errores de ejecución (separados de los de parseo).
#[derive(Debug, Clone)]
pub enum ExecError {
    NoSuchSubject(String),
    MissingBudget,
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecError::NoSuchSubject(s) => write!(f, "ningún arquetipo vivo para '{s}'"),
            ExecError::MissingBudget => write!(f, "EVOKE requiere WITHIN budget N tokens"),
        }
    }
}

impl std::error::Error for ExecError {}

/// El ejecutor lleva un runtime y un provider de embeddings. No es `Sync` por el provider.
pub struct Executor<P: Provider> {
    rt: CognitiveRuntime,
    provider: P,
}

impl<P: Provider> Executor<P> {
    pub fn new(rt: CognitiveRuntime, provider: P) -> Self {
        Self { rt, provider }
    }

    pub fn runtime(&self) -> &CognitiveRuntime {
        &self.rt
    }

    pub fn runtime_mut(&mut self) -> &mut CognitiveRuntime {
        &mut self.rt
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Ejecuta una sentencia MQL contra el runtime, en el tick lógico `now`.
    pub fn execute(&mut self, stmt: &Statement, now: Tick) -> Result<ExecResult, ExecError> {
        match stmt {
            Statement::Perceive(p) => self.exec_perceive(p, now),
            Statement::Distill(d) => self.exec_distill(d, now),
            Statement::Evoke(e) => self.exec_evoke(e, now),
            Statement::Fade(f) => self.exec_fade(f, now),
            Statement::Imprint(i) => self.exec_imprint(i, now),
            Statement::Recall(r) => self.exec_recall(r, now),
            Statement::Reinforce(r) => self.exec_reinforce(r, now),
        }
    }

    /// Ejecuta un programa completo (varias sentencias) y devuelve todos los resultados.
    pub fn execute_program(
        &mut self,
        stmts: &[Statement],
        now: Tick,
    ) -> Vec<Result<ExecResult, ExecError>> {
        stmts.iter().map(|s| self.execute(s, now)).collect()
    }

    fn exec_perceive(&mut self, p: &Perceive, now: Tick) -> Result<ExecResult, ExecError> {
        // El estímulo crudo: concatenamos los rasgos como texto, que el provider embebe.
        // Los rasgos *son* el estímulo (no hay payload binario en el AST).
        let text = traits_to_text(p);
        let embedding = self.provider.embed(&text);
        let salience = p.salience.unwrap_or(0.5);
        let halflife = p.halflife.map(|d| d.seconds).unwrap_or(86_400.0); // 1 día por defecto
        let mut perception = Perception::new(&p.subject, embedding, salience, halflife, now);
        for (k, v) in &p.traits {
            perception = perception.with_trait(k.clone(), v.clone());
        }
        self.rt.perceive(perception);
        Ok(ExecResult::Perceived {
            subject: p.subject.clone(),
        })
    }

    fn exec_distill(&mut self, d: &Distill, now: Tick) -> Result<ExecResult, ExecError> {
        let report = match &d.filter {
            None => self.rt.breathe(&[&d.subject], now),
            Some(pred) => self
                .rt
                .breathe_where(&[&d.subject], now, |p| eval_on(pred, p, now)),
        };
        Ok(ExecResult::Dreamed(report))
    }

    fn exec_evoke(&mut self, e: &Evoke, now: Tick) -> Result<ExecResult, ExecError> {
        let token_budget = e.token_budget.ok_or(ExecError::MissingBudget)?;

        // `ACROSS span D` → ventana temporal: solo hitos del arco con at ≥ now − D.
        let since = e.span.map(|d| (now - d.seconds).max(0.0));

        // `RESOLUTION` manda sobre `PROJECTING`; si no hay ninguno, arco completo.
        let arc_detail = match (e.resolution, e.projecting) {
            (Some(Resolution::Point), _) => ArcDetail::None,
            (Some(Resolution::Summary), _) => ArcDetail::Summary,
            (Some(Resolution::Arc), _) => ArcDetail::Full,
            (None, Some(Projection::Snapshot)) => ArcDetail::None,
            (None, Some(Projection::Trajectory)) => ArcDetail::Full,
            (None, None) => ArcDetail::Full,
        };

        let req = EvokeRequest {
            subject: e.subject.clone(),
            token_budget,
            since,
            arc_detail,
            ..EvokeRequest::new(e.subject.clone(), token_budget)
        };
        let mut ctx = self
            .rt
            .evoke(&req, now)
            .ok_or_else(|| ExecError::NoSuchSubject(e.subject.clone()))?;

        // `RESONATING WITH { rasgos }`: ya NO se ignora. Embebemos los rasgos con el provider real y
        // enfocamos la evocación en el **modo** del sujeto que resuena con ellos (su aspecto relevante,
        // no el comportamiento dominante global). Cierra la deuda #9 de VERDAD 100%.
        if !e.resonating_with.is_empty() {
            let query = self.provider.embed(&e.resonating_with.join(" "));
            ctx.resonating_mode = self
                .rt
                .long_term()
                .get(&e.subject)
                .and_then(|a| a.resonant_mode_label(&query));
        }
        Ok(ExecResult::Evoked(ctx))
    }

    fn exec_fade(&mut self, f: &Fade, now: Tick) -> Result<ExecResult, ExecError> {
        let swept = match &f.filter {
            // Sin WHERE: barrido por el umbral de olvido por defecto (la física decide qué cae).
            None => self
                .rt
                .fade_only(now, letheo_core::entropy::DEFAULT_THETA_FADE),
            // Con WHERE: el predicado del usuario *es* la condición de olvido.
            Some(pred) => self.rt.fade_where(|p| eval_on(pred, p, now)),
        };
        Ok(ExecResult::Faded { swept })
    }

    fn exec_imprint(&mut self, i: &Imprint, now: Tick) -> Result<ExecResult, ExecError> {
        // IMPRINT real (deuda #6 saldada): **consolida/ancla** el arquetipo del sujeto nombrado —
        // refuerza su física (y la de sus modos) para ganar permanencia. Ya no es un no-op. La
        // RESILIENCE pedida mapea cuánto se consolida (más resiliencia → más reducción de λ).
        let consolidation = match i.resilience {
            Some(Resilience::High) => 0.2,
            Some(Resilience::Medium) => 0.1,
            Some(Resilience::Low) | None => 0.05,
        };
        if self.rt.consolidate(&i.archetype, now, consolidation) {
            Ok(ExecResult::Imprinted {
                archetype: i.archetype.clone(),
                note: "esencia consolidada: Δt→0 y λ reducido (permanencia ganada)",
            })
        } else {
            Err(ExecError::NoSuchSubject(i.archetype.clone()))
        }
    }

    /// `RECALL` (capa-1): recupera los hechos episódicos del sujeto que resuenan con la consulta,
    /// rankeados por física (`relevancia · vida`) y recortados al top-k. **Read-only** (no refuerza —
    /// para eso está `REINFORCE`). El `WHERE` opcional filtra por `resonates`/`weight`/`age`/`salience`.
    fn exec_recall(&mut self, r: &Recall, now: Tick) -> Result<ExecResult, ExecError> {
        let query = self.provider.embed(&r.resonating_with.join(" "));
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        let mut scored: Vec<(f64, RecalledFact)> = self
            .rt
            .facts()
            .alive_for(&r.subject, now, theta)
            .filter(|f| match &r.filter {
                None => true,
                Some(pred) => pred.eval(&FactFacts {
                    f,
                    query: &query,
                    now,
                }),
            })
            .map(|f| {
                let relevance = letheo_core::vector::cosine(&f.embedding, &query).max(0.0) as f64;
                let score = relevance * f.trace.weight(now);
                (
                    score,
                    RecalledFact {
                        text: f.text.clone(),
                        provenance: f.provenance.clone(),
                        score,
                    },
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(r.k);
        Ok(ExecResult::Recalled(
            scored.into_iter().map(|(_, rf)| rf).collect(),
        ))
    }

    /// `REINFORCE` (capa-1): recupera los top-k hechos que resuenan con la consulta y **los refuerza**
    /// (resetea su decay → spaced repetition). Muta la capa-1. Devuelve cuántos se reforzaron.
    fn exec_reinforce(&mut self, r: &Reinforce, now: Tick) -> Result<ExecResult, ExecError> {
        let query = self.provider.embed(&r.resonating_with.join(" "));
        let reinforced = self.rt.recall(&r.subject, &query, r.k, now);
        Ok(ExecResult::Reinforced {
            count: reinforced.len(),
        })
    }
}

/// Puente entre el predicado MQL y una percepción concreta del runtime. Mantiene a `letheo-mql`
/// ignorante de `letheo-core`: la semántica del `WHERE` vive en el AST, los datos físicos aquí.
struct PerceptionFacts<'a> {
    p: &'a Perception,
    now: Tick,
}

impl Facts for PerceptionFacts<'_> {
    fn numeric(&self, field: &Field) -> Option<f64> {
        match field {
            Field::Weight => Some(self.p.weight(self.now)),
            Field::Salience => Some(self.p.trace.salience),
            Field::Age => Some(self.p.trace.delta_t(self.now)),
            // Sin consulta en el contexto de percepción (DISTILL/FADE) la resonancia no está disponible.
            Field::Resonance => None,
            Field::Trait(_) => None,
        }
    }

    fn text(&self, key: &str) -> Option<String> {
        self.p.traits.get(key).cloned()
    }
}

/// Evalúa un predicado de `WHERE` sobre una percepción en el tick `now`.
fn eval_on(pred: &Predicate, p: &Perception, now: Tick) -> bool {
    pred.eval(&PerceptionFacts { p, now })
}

/// Puente entre el predicado MQL y un **hecho episódico** (capa-1) + la consulta de la sentencia. Da
/// sentido a `WHERE resonates > θ`: la resonancia es el coseno del hecho con la consulta embebida.
/// Los hechos no tienen trait map → solo exponen física (`weight`/`salience`/`age`) y `resonance`.
struct FactFacts<'a> {
    f: &'a Fact,
    query: &'a [f32],
    now: Tick,
}

impl Facts for FactFacts<'_> {
    fn numeric(&self, field: &Field) -> Option<f64> {
        match field {
            Field::Weight => Some(self.f.trace.weight(self.now)),
            Field::Salience => Some(self.f.trace.salience),
            Field::Age => Some(self.f.trace.delta_t(self.now)),
            Field::Resonance => {
                Some(letheo_core::vector::cosine(&self.f.embedding, self.query) as f64)
            }
            Field::Trait(_) => None,
        }
    }

    fn text(&self, _key: &str) -> Option<String> {
        None
    }
}

/// Concatena los rasgos de un PERCEIVE en una sola línea de texto para embeber.
/// Estable: itera por orden alfabético de clave (BTreeMap ya lo hace).
fn traits_to_text(p: &Perceive) -> String {
    let parts: Vec<String> = p.traits.iter().map(|(k, v)| format!("{k} {v}")).collect();
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_core::RuntimeConfig;
    use letheo_inference::MockProvider;
    use letheo_mql::parse;

    fn fresh() -> Executor<MockProvider> {
        Executor::new(
            CognitiveRuntime::new(RuntimeConfig::default()),
            MockProvider::new(),
        )
    }

    /// Registra un hecho (capa-1) con embedding del texto `query_text` (para que una consulta con esos
    /// tokens lo recupere) y `halflife` dado. La capa-1 se escribe por API; MQL la consulta.
    fn remember_fact(
        ex: &mut Executor<MockProvider>,
        subject: &str,
        text: &str,
        query_text: &str,
        halflife: f64,
        now: f64,
    ) {
        let emb = ex.provider().embed(query_text);
        ex.runtime_mut()
            .remember(subject, text, emb, "test", 1.0, halflife, now);
    }

    #[test]
    fn full_program_perceive_distill_evoke() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: purchase, object: shoes }
            PERCEIVE interaction FROM subject "u:X" AS { act: purchase, object: shoes }
            PERCEIVE interaction FROM subject "u:X" AS { act: purchase, object: shoes }
            DISTILL subject "u:X" INTO intention_vector COMPRESSING BY semantic_variance
            EVOKE essence OF "u:X" WITHIN budget 800 tokens
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        assert_eq!(results.len(), 5);

        // 3 percepciones aceptadas
        for r in &results[..3] {
            assert!(matches!(r, Ok(ExecResult::Perceived { .. })));
        }
        // DISTILL → BreathReport con 1 sujeto consolidado
        match &results[3] {
            Ok(ExecResult::Dreamed(r)) => {
                assert_eq!(r.distilled_subjects, 1);
                assert_eq!(r.perceptions_absorbed, 3);
            }
            other => panic!("se esperaba Dreamed, fue {other:?}"),
        }
        // EVOKE → contexto con esos 3 eventos representados
        match &results[4] {
            Ok(ExecResult::Evoked(ctx)) => {
                assert_eq!(ctx.represented, 3);
                assert!(ctx.token_estimate <= 800);
            }
            other => panic!("se esperaba Evoked, fue {other:?}"),
        }
    }

    #[test]
    fn evoke_missing_budget_fails() {
        let stmts = parse(r#"EVOKE essence OF "u:X""#).unwrap();
        let mut ex = fresh();
        let r = &ex.execute_program(&stmts, 0.0)[0];
        assert!(matches!(r, Err(ExecError::MissingBudget)));
    }

    #[test]
    fn evoke_unknown_subject_fails() {
        let stmts = parse(r#"EVOKE essence OF "ghost" WITHIN budget 800 tokens"#).unwrap();
        let mut ex = fresh();
        let r = &ex.execute_program(&stmts, 0.0)[0];
        assert!(matches!(r, Err(ExecError::NoSuchSubject(_))));
    }

    #[test]
    fn fade_sweeps_decayed_noise() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 0.1 DECAYS halflife 1h
            FADE noise PRESERVING archetype_contribution
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        // Ejecutamos el PERCEIVE en t=0 y el FADE 10h después → el evento débil debe caer.
        ex.execute(&stmts[0], 0.0).unwrap();
        let r = ex.execute(&stmts[1], 3600.0 * 10.0).unwrap();
        match r {
            ExecResult::Faded { swept } => assert_eq!(swept, 1),
            other => panic!("se esperaba Faded, fue {other:?}"),
        }
    }

    #[test]
    fn distill_where_filters_by_trait() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: buy, domain: ecommerce }
            PERCEIVE interaction FROM subject "u:X" AS { act: buy, domain: ecommerce }
            PERCEIVE interaction FROM subject "u:X" AS { act: read, domain: news }
            DISTILL subject "u:X" FROM perceptions WHERE domain "ecommerce" INTO intention_vector
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        match &results[3] {
            Ok(ExecResult::Dreamed(r)) => {
                // Solo las 2 percepciones de dominio "ecommerce" se destilan; la de "news" no.
                assert_eq!(r.perceptions_absorbed, 2, "el WHERE filtró por rasgo");
            }
            other => panic!("se esperaba Dreamed, fue {other:?}"),
        }
    }

    #[test]
    fn fade_where_predicate_selects_what_to_forget() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 1.0 DECAYS halflife 100h
            PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 1.0 DECAYS halflife 100h
            FADE noise WHERE age > 3600 PRESERVING archetype_contribution
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        ex.execute(&stmts[0], 0.0).unwrap();
        ex.execute(&stmts[1], 0.0).unwrap();
        // 2h después: ambas tienen age = 7200 > 3600 ⇒ ambas se desvanecen pese a peso alto.
        match ex.execute(&stmts[2], 7200.0).unwrap() {
            ExecResult::Faded { swept } => assert_eq!(swept, 2, "el WHERE por edad las barrió"),
            other => panic!("se esperaba Faded, fue {other:?}"),
        }
    }

    #[test]
    fn evoke_resolution_point_drops_arc() {
        // Dos ciclos para tener arco; luego EVOKE con RESOLUTION point no debe devolver hitos.
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: a }
            DISTILL subject "u:X" INTO intention_vector
            PERCEIVE interaction FROM subject "u:X" AS { act: b }
            DISTILL subject "u:X" INTO intention_vector
            EVOKE essence OF "u:X" RESOLUTION point WITHIN budget 800 tokens
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        match results.last().unwrap() {
            Ok(ExecResult::Evoked(ctx)) => assert!(ctx.arc_points.is_empty(), "point ⇒ sin arco"),
            other => panic!("se esperaba Evoked, fue {other:?}"),
        }
    }

    #[test]
    fn imprint_consolidates_existing_archetype() {
        // IMPRINT real (deuda #6): primero se destila una esencia; luego IMPRINT la **ancla**.
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { a: b }
            PERCEIVE interaction FROM subject "u:X" AS { a: b }
            DISTILL subject "u:X" INTO intention_vector
            IMPRINT archetype "u:X" FROM intention_vector RESILIENCE high
        "#;
        let stmts = parse(src).unwrap();
        let mut ex = fresh();
        let results = ex.execute_program(&stmts, 0.0);
        match results.last().unwrap() {
            Ok(ExecResult::Imprinted { archetype, .. }) => assert_eq!(archetype, "u:X"),
            other => panic!("se esperaba Imprinted, fue {other:?}"),
        }
        // El IMPRINT cambió la física de verdad: la esencia ganó refuerzo (ya no es un no-op).
        let a = ex.runtime().long_term().get("u:X").unwrap();
        assert!(a.trace.reinforcement > 0.0, "IMPRINT consolidó la esencia");
    }

    #[test]
    fn imprint_unknown_subject_fails() {
        let stmts =
            parse(r#"IMPRINT archetype "ghost" FROM intention_vector RESILIENCE high"#).unwrap();
        let mut ex = fresh();
        let r = &ex.execute_program(&stmts, 0.0)[0];
        assert!(
            matches!(r, Err(ExecError::NoSuchSubject(_))),
            "no se imprime lo no destilado"
        );
    }

    #[test]
    fn evoke_resonating_with_focuses_on_the_matching_mode() {
        // Dos comportamientos sin tokens compartidos → dos modos nítidos.
        let src = r#"
            PERCEIVE interaction FROM subject "u" AS { topic: galaxies }
            PERCEIVE interaction FROM subject "u" AS { topic: galaxies }
            PERCEIVE interaction FROM subject "u" AS { flavor: cooking }
            PERCEIVE interaction FROM subject "u" AS { flavor: cooking }
            DISTILL subject "u" INTO intention_vector
        "#;
        let mut ex = fresh();
        ex.execute_program(&parse(src).unwrap(), 0.0);

        // RESONATING WITH ya NO se ignora (deuda #9): enfoca la evocación en el modo que resuena.
        let gal =
            parse(r#"EVOKE essence OF "u" RESONATING WITH { galaxies } WITHIN budget 800 tokens"#)
                .unwrap();
        match &ex.execute_program(&gal, 0.0)[0] {
            Ok(ExecResult::Evoked(ctx)) => {
                assert_eq!(ctx.resonating_mode.as_deref(), Some("galaxies"))
            }
            other => panic!("se esperaba Evoked, fue {other:?}"),
        }
        let cook =
            parse(r#"EVOKE essence OF "u" RESONATING WITH { cooking } WITHIN budget 800 tokens"#)
                .unwrap();
        match &ex.execute_program(&cook, 0.0)[0] {
            Ok(ExecResult::Evoked(ctx)) => {
                assert_eq!(ctx.resonating_mode.as_deref(), Some("cooking"))
            }
            other => panic!("se esperaba Evoked, fue {other:?}"),
        }
    }

    #[test]
    fn recall_returns_the_matching_fact_verbatim() {
        let day = 86_400.0;
        let mut ex = fresh();
        remember_fact(
            &mut ex,
            "u",
            "allergic to peanuts",
            "health allergy peanuts",
            day,
            0.0,
        );
        remember_fact(
            &mut ex,
            "u",
            "drives a red car",
            "vehicle car red",
            day,
            0.0,
        );
        let prog = parse(
            r#"RECALL facts FROM subject "u" RESONATING WITH { health, allergy } WITHIN k 1"#,
        )
        .unwrap();
        match &ex.execute_program(&prog, 0.0)[0] {
            Ok(ExecResult::Recalled(facts)) => {
                assert_eq!(facts.len(), 1);
                assert_eq!(
                    facts[0].text, "allergic to peanuts",
                    "capa-1 sin pérdida, verbatim"
                );
            }
            other => panic!("se esperaba Recalled, fue {other:?}"),
        }
    }

    #[test]
    fn reinforce_resets_decay_so_the_fact_survives() {
        let half = 30.0 * 86_400.0;
        let mut ex = fresh();
        remember_fact(&mut ex, "u", "fact alpha", "topic alpha", half, 0.0);
        remember_fact(&mut ex, "u", "fact beta", "topic beta", half, 0.0);
        // En t=half, REINFORCE solo el alpha → su decay se resetea.
        let prog =
            parse(r#"REINFORCE facts FROM subject "u" RESONATING WITH { alpha } WITHIN k 1"#)
                .unwrap();
        match ex.execute(&prog[0], half).unwrap() {
            ExecResult::Reinforced { count } => assert_eq!(count, 1),
            other => panic!("se esperaba Reinforced, fue {other:?}"),
        }
        // Mucho más tarde barremos: el reforzado sobrevive; el otro (nunca tocado) se desvanece.
        let swept = ex.runtime_mut().fade_facts(half * 5.0);
        assert_eq!(swept, 1, "solo el hecho no reforzado se desvanece");
    }

    #[test]
    fn recall_where_resonates_filters_by_threshold() {
        let day = 86_400.0;
        let mut ex = fresh();
        remember_fact(&mut ex, "u", "strong match", "query topic here", day, 0.0);
        remember_fact(&mut ex, "u", "weak match", "query", day, 0.0);
        // El débil resuena ~0.58 con la consulta (solo comparte "query"); el umbral 0.6 lo descarta.
        let prog = parse(r#"RECALL facts FROM subject "u" RESONATING WITH { query, topic, here } WHERE resonates > 0.6 WITHIN k 5"#).unwrap();
        match &ex.execute_program(&prog, 0.0)[0] {
            Ok(ExecResult::Recalled(facts)) => {
                assert_eq!(facts.len(), 1, "el predicado vectorial filtró al débil");
                assert_eq!(facts[0].text, "strong match");
            }
            other => panic!("se esperaba Recalled, fue {other:?}"),
        }
    }
}
