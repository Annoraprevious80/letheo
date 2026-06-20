//! # letheo-async · El organismo que respira solo
//!
//! En el core, `breathe()` es síncrono: el usuario decide *cuándo* el runtime sueña. Aquí cerramos
//! el último pilar arquitectónico del plan: montar ese bucle sobre **Tokio** para que el GC
//! semántico corra **de fondo**, sin bloquear la percepción y sin que nadie llame a `breathe`.
//!
//! Diseño (decisiones de diseño en `ROADMAP.md`):
//! - **Crate separado**, no dentro de `letheo-core`: el core permanece determinista y offline
//!   (`cargo test -p letheo-core` no arrastra Tokio). Aquí vive todo lo asíncrono.
//! - **Reloj lógico derivado del reloj real**: `Tick = segundos transcurridos · time_scale`. Con
//!   `tokio::time::pause()` los tests controlan el tiempo y son deterministas pese a ser async.
//! - **Dos disparadores de sueño**: un `interval` periódico (respiración basal) y *backpressure*
//!   (si se acumulan ≥ `pressure_watermark` percepciones sin consolidar, sueña ya — épica 6.1).
//!
//! El actor es dueño exclusivo del `CognitiveRuntime`; el mundo exterior habla con él por canales.
//! Eso evita locks y mantiene el core `!Sync` sin fricción.

use std::collections::HashSet;
use std::time::Duration;

use letheo_core::{
    BreathReport, CognitiveRuntime, CompressedContext, EvokeRequest, Perception, RuntimeConfig,
    Tick,
};
use letheo_inference::Provider;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{interval_at, Instant, MissedTickBehavior};

/// Configuración del runtime asíncrono.
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Núcleo cognitivo (umbrales, resiliencia, etc.).
    pub runtime: RuntimeConfig,
    /// Cada cuánto el organismo respira de forma basal.
    pub breath_interval: Duration,
    /// Cuántas percepciones sin consolidar disparan un sueño inmediato (backpressure). 0 = nunca.
    pub pressure_watermark: usize,
    /// Segundos lógicos (`Tick`) por segundo real. Permite acelerar la física frente al wall-clock.
    pub time_scale: f64,
    /// Capacidad del canal de comandos (cola de percepción acotada).
    pub channel_capacity: usize,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            breath_interval: Duration::from_secs(60),
            pressure_watermark: 256,
            time_scale: 1.0,
            channel_capacity: 1024,
        }
    }
}

/// Métricas acumuladas del organismo (observabilidad — épica 6.2).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Ciclos de sueño ejecutados (basales + por presión + a demanda).
    pub breaths: u64,
    /// Sueños disparados por el intervalo basal.
    pub breaths_basal: u64,
    /// Sueños disparados por backpressure.
    pub breaths_pressure: u64,
    /// Sueños disparados a demanda (`breathe(...)`).
    pub breaths_ondemand: u64,
    /// Sujetos consolidados en total.
    pub distilled_subjects: u64,
    /// Percepciones absorbidas en arquetipos.
    pub perceptions_absorbed: u64,
    /// Percepciones barridas por FADE.
    pub faded: u64,
    /// Percepciones recibidas desde el último sueño (presión actual).
    pub pending: usize,
    /// Tamaño de la memoria de corto plazo.
    pub short_term_len: usize,
    /// Número de arquetipos vivos.
    pub long_term_len: usize,
}

impl Stats {
    /// Renderiza las métricas en el formato de exposición de texto de Prometheus. El resultado se
    /// puede servir tal cual en un endpoint `/metrics` (sin dependencias ni servidor embebido aquí).
    pub fn render_prometheus(&self) -> String {
        let mut s = String::new();
        let mut metric = |name: &str, help: &str, kind: &str, value: u64| {
            s.push_str(&format!("# HELP letheo_{name} {help}\n"));
            s.push_str(&format!("# TYPE letheo_{name} {kind}\n"));
            s.push_str(&format!("letheo_{name} {value}\n"));
        };
        metric(
            "breaths_total",
            "Ciclos de sueño ejecutados.",
            "counter",
            self.breaths,
        );
        metric(
            "breaths_basal_total",
            "Sueños por intervalo basal.",
            "counter",
            self.breaths_basal,
        );
        metric(
            "breaths_pressure_total",
            "Sueños por backpressure.",
            "counter",
            self.breaths_pressure,
        );
        metric(
            "breaths_ondemand_total",
            "Sueños a demanda.",
            "counter",
            self.breaths_ondemand,
        );
        metric(
            "distilled_subjects_total",
            "Sujetos consolidados.",
            "counter",
            self.distilled_subjects,
        );
        metric(
            "perceptions_absorbed_total",
            "Percepciones absorbidas en arquetipos.",
            "counter",
            self.perceptions_absorbed,
        );
        metric(
            "faded_total",
            "Percepciones barridas por FADE.",
            "counter",
            self.faded,
        );
        metric(
            "pending",
            "Percepciones sin consolidar (presión actual).",
            "gauge",
            self.pending as u64,
        );
        metric(
            "short_term_len",
            "Tamaño de la memoria de corto plazo.",
            "gauge",
            self.short_term_len as u64,
        );
        metric(
            "long_term_len",
            "Arquetipos vivos.",
            "gauge",
            self.long_term_len as u64,
        );
        s
    }
}

/// Origen del último sueño — útil para tests y trazas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreathCause {
    Basal,
    Pressure,
    OnDemand,
}

/// Error de comunicación con el actor (el organismo dejó de respirar).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped;

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "el runtime asíncrono se ha detenido")
    }
}
impl std::error::Error for Stopped {}

type Reply<T> = oneshot::Sender<T>;

enum Cmd {
    Perceive(Perception),
    PerceiveText {
        subject: String,
        text: String,
        salience: f64,
        halflife: f64,
    },
    Breathe {
        subjects: Option<Vec<String>>,
        reply: Reply<BreathReport>,
    },
    Evoke {
        req: EvokeRequest,
        reply: Reply<Option<CompressedContext>>,
    },
    Stats(Reply<Stats>),
}

/// Mango (handle) clonable hacia el organismo. Todas las operaciones son asíncronas y no bloquean.
#[derive(Clone)]
pub struct AsyncRuntime {
    tx: mpsc::Sender<Cmd>,
}

impl AsyncRuntime {
    /// Arranca el organismo: crea el actor de fondo con un provider de embeddings y un núcleo nuevo.
    /// Devuelve el mango y el `JoinHandle` del actor (que termina cuando se sueltan todos los mangos).
    pub fn spawn<P>(provider: P, cfg: AsyncConfig) -> (Self, JoinHandle<()>)
    where
        P: Provider + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(cfg.channel_capacity);
        let actor = Actor::new(provider, cfg);
        let handle = tokio::spawn(actor.run(rx));
        (Self { tx }, handle)
    }

    /// `PERCEIVE`: asimila un estímulo ya embebido.
    pub async fn perceive(&self, p: Perception) -> Result<(), Stopped> {
        self.tx.send(Cmd::Perceive(p)).await.map_err(|_| Stopped)
    }

    /// `PERCEIVE` ergonómico: el actor embebe el texto con su provider (rasgos como estímulo).
    pub async fn perceive_text(
        &self,
        subject: impl Into<String>,
        text: impl Into<String>,
        salience: f64,
        halflife: f64,
    ) -> Result<(), Stopped> {
        self.tx
            .send(Cmd::PerceiveText {
                subject: subject.into(),
                text: text.into(),
                salience,
                halflife,
            })
            .await
            .map_err(|_| Stopped)
    }

    /// Fuerza un ciclo de sueño a demanda. `subjects = None` ⇒ todos los sujetos vistos.
    pub async fn breathe(&self, subjects: Option<Vec<String>>) -> Result<BreathReport, Stopped> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Cmd::Breathe { subjects, reply })
            .await
            .map_err(|_| Stopped)?;
        rx.await.map_err(|_| Stopped)
    }

    /// `EVOKE`: resuelve la esencia de un sujeto dentro del presupuesto de tokens.
    pub async fn evoke(&self, req: EvokeRequest) -> Result<Option<CompressedContext>, Stopped> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Cmd::Evoke { req, reply })
            .await
            .map_err(|_| Stopped)?;
        rx.await.map_err(|_| Stopped)
    }

    /// Instantánea de métricas del organismo.
    pub async fn stats(&self) -> Result<Stats, Stopped> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::Stats(reply)).await.map_err(|_| Stopped)?;
        rx.await.map_err(|_| Stopped)
    }

    /// Métricas en formato de exposición Prometheus, listas para un endpoint `/metrics`.
    pub async fn metrics_prometheus(&self) -> Result<String, Stopped> {
        Ok(self.stats().await?.render_prometheus())
    }
}

/// El actor: dueño exclusivo del runtime. Vive en su propia tarea Tokio.
struct Actor<P: Provider> {
    rt: CognitiveRuntime,
    provider: P,
    cfg: AsyncConfig,
    start: Instant,
    subjects: HashSet<String>,
    pending: usize,
    stats: Stats,
}

impl<P: Provider> Actor<P> {
    fn new(provider: P, cfg: AsyncConfig) -> Self {
        Self {
            rt: CognitiveRuntime::new(cfg.runtime.clone()),
            provider,
            cfg,
            start: Instant::now(),
            subjects: HashSet::new(),
            pending: 0,
            stats: Stats::default(),
        }
    }

    /// Tick lógico actual = segundos reales transcurridos · time_scale.
    fn now(&self) -> Tick {
        self.start.elapsed().as_secs_f64() * self.cfg.time_scale
    }

    async fn run(mut self, mut rx: mpsc::Receiver<Cmd>) {
        // `interval_at` con primer disparo a `breath_interval` (no inmediato): el recién nacido no
        // sueña antes de haber percibido nada.
        let first = Instant::now() + self.cfg.breath_interval;
        let mut ticker = interval_at(first, self.cfg.breath_interval);
        // Si el ejecutor se atrasa, no acumulamos sueños en ráfaga: saltamos los perdidos.
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                maybe_cmd = rx.recv() => {
                    match maybe_cmd {
                        Some(cmd) => self.handle(cmd),
                        None => break, // todos los mangos sueltos: el organismo muere en paz.
                    }
                }
                _ = ticker.tick() => {
                    self.breathe_all(BreathCause::Basal);
                }
            }
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Perceive(p) => self.ingest(p),
            Cmd::PerceiveText {
                subject,
                text,
                salience,
                halflife,
            } => {
                let now = self.now();
                let embedding = self.provider.embed(&text);
                self.ingest(Perception::new(subject, embedding, salience, halflife, now));
            }
            Cmd::Breathe { subjects, reply } => {
                let report = match subjects {
                    Some(s) => self.breathe_some(&s, BreathCause::OnDemand),
                    None => self.breathe_all(BreathCause::OnDemand),
                };
                let _ = reply.send(report);
            }
            Cmd::Evoke { req, reply } => {
                let ctx = self.rt.evoke(&req, self.now());
                let _ = reply.send(ctx);
            }
            Cmd::Stats(reply) => {
                let _ = reply.send(self.snapshot());
            }
        }
    }

    fn ingest(&mut self, p: Perception) {
        self.subjects.insert(p.subject.clone());
        self.rt.perceive(p);
        self.pending += 1;
        // Backpressure: demasiada percepción sin consolidar ⇒ sueña ahora, no esperes al intervalo.
        if self.cfg.pressure_watermark > 0 && self.pending >= self.cfg.pressure_watermark {
            self.breathe_all(BreathCause::Pressure);
        }
    }

    fn breathe_all(&mut self, cause: BreathCause) -> BreathReport {
        let subjects: Vec<String> = self.subjects.iter().cloned().collect();
        self.breathe_some(&subjects, cause)
    }

    fn breathe_some(&mut self, subjects: &[String], cause: BreathCause) -> BreathReport {
        let refs: Vec<&str> = subjects.iter().map(String::as_str).collect();
        let report = self.rt.breathe(&refs, self.now());

        self.stats.breaths += 1;
        match cause {
            BreathCause::Basal => self.stats.breaths_basal += 1,
            BreathCause::Pressure => self.stats.breaths_pressure += 1,
            BreathCause::OnDemand => self.stats.breaths_ondemand += 1,
        }
        self.stats.distilled_subjects += report.distilled_subjects as u64;
        self.stats.perceptions_absorbed += report.perceptions_absorbed as u64;
        self.stats.faded += report.faded as u64;
        self.pending = 0;
        report
    }

    fn snapshot(&self) -> Stats {
        Stats {
            pending: self.pending,
            short_term_len: self.rt.short_term_len(),
            long_term_len: self.rt.long_term_len(),
            ..self.stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_inference::MockProvider;

    fn cfg() -> AsyncConfig {
        AsyncConfig {
            breath_interval: Duration::from_secs(10),
            pressure_watermark: 0, // los tests basales y a-demanda no quieren presión de por medio
            ..Default::default()
        }
    }

    fn perception(subject: &str, e: Vec<f32>) -> Perception {
        // halflife enorme: en estos tests el tiempo lógico apenas avanza, nada debe caer por sí solo.
        Perception::new(subject, e, 1.0, 1.0e9, 0.0)
    }

    #[tokio::test]
    async fn on_demand_breathe_consolidates() {
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        for _ in 0..20 {
            rt.perceive(perception("u:X", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        let report = rt.breathe(None).await.unwrap();
        assert_eq!(report.distilled_subjects, 1);
        assert_eq!(report.perceptions_absorbed, 20);

        let ctx = rt
            .evoke(EvokeRequest::new("u:X", 800))
            .await
            .unwrap()
            .expect("hay esencia");
        assert_eq!(ctx.represented, 20);
    }

    #[tokio::test(start_paused = true)]
    async fn basal_breathing_runs_without_being_asked() {
        // Con el reloj pausado, controlamos el tiempo: nadie llama a breathe, pero el intervalo
        // dispara el sueño por su cuenta.
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        for _ in 0..5 {
            rt.perceive(perception("u:Y", vec![0.0, 1.0]))
                .await
                .unwrap();
        }
        // Antes del intervalo: nada consolidado.
        assert_eq!(rt.stats().await.unwrap().breaths, 0);

        // Avanzamos el reloj más allá del intervalo de respiración.
        tokio::time::advance(Duration::from_secs(11)).await;
        // Cedemos para que el actor procese el tick antes de consultar.
        tokio::task::yield_now().await;

        let s = rt.stats().await.unwrap();
        assert!(s.breaths >= 1, "el organismo respiró solo: {s:?}");
        assert_eq!(
            s.long_term_len, 1,
            "consolidó al sujeto sin que se lo pidieran"
        );
    }

    #[tokio::test]
    async fn backpressure_triggers_immediate_breath() {
        let pressured = AsyncConfig {
            pressure_watermark: 8,
            breath_interval: Duration::from_secs(3600), // lejísimos: el sueño NO es basal aquí
            ..cfg()
        };
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), pressured);
        for _ in 0..8 {
            rt.perceive(perception("u:Z", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        // La 8ª percepción cruza el watermark ⇒ sueño inmediato, sin esperar intervalo ni demanda.
        let s = rt.stats().await.unwrap();
        assert_eq!(s.breaths, 1, "la presión disparó un sueño: {s:?}");
        assert_eq!(s.long_term_len, 1);
        assert_eq!(s.pending, 0, "la presión se liberó");
    }

    #[tokio::test]
    async fn stops_cleanly_when_handle_dropped() {
        let (rt, handle) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        rt.perceive(perception("u:X", vec![1.0, 0.0]))
            .await
            .unwrap();
        drop(rt); // soltamos el único mango
                  // El actor debe terminar su tarea sin pánico.
        handle.await.expect("el actor terminó limpio");
    }

    #[tokio::test]
    async fn breath_causes_are_counted_separately() {
        let pressured = AsyncConfig {
            pressure_watermark: 4,
            breath_interval: Duration::from_secs(3600),
            ..cfg()
        };
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), pressured);
        for _ in 0..4 {
            rt.perceive(perception("u:A", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        rt.breathe(None).await.unwrap(); // a demanda
        let s = rt.stats().await.unwrap();
        assert_eq!(s.breaths_pressure, 1, "{s:?}");
        assert_eq!(s.breaths_ondemand, 1, "{s:?}");
        assert_eq!(s.breaths_basal, 0);
        assert_eq!(s.breaths, 2);
    }

    #[tokio::test]
    async fn prometheus_render_is_well_formed() {
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        for _ in 0..3 {
            rt.perceive(perception("u:M", vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        rt.breathe(None).await.unwrap();
        let text = rt.metrics_prometheus().await.unwrap();
        assert!(text.contains("# TYPE letheo_breaths_total counter"));
        assert!(text.contains("letheo_breaths_ondemand_total 1"));
        assert!(text.contains("# TYPE letheo_long_term_len gauge"));
        // Cada métrica trae HELP + TYPE + valor: 3 líneas por métrica, 10 métricas.
        assert_eq!(text.lines().count(), 30);
    }

    #[tokio::test]
    async fn evoke_on_unknown_subject_is_none() {
        let (rt, _h) = AsyncRuntime::spawn(MockProvider::new(), cfg());
        let ctx = rt.evoke(EvokeRequest::new("ghost", 800)).await.unwrap();
        assert!(ctx.is_none());
    }
}
