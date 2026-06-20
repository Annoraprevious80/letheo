//! # letheo (PyO3) — el Cognitive Runtime de Mnemosyne en Python.
//!
//! Dos modos de uso:
//!  1. **API biológica directa**: `rt.perceive(...)`, `rt.breathe(...)`, `rt.evoke(...)`.
//!  2. **MQL ejecutable**: `rt.execute_mql(src)` parsea y ejecuta un programa MQL completo.
//!
//! ```python
//! import letheo
//! rt = letheo.Runtime()
//! rt.execute_mql('''
//!     PERCEIVE interaction FROM subject "user:Xolotl" AS { act: purchase, object: shoes }
//!     DISTILL  subject "user:Xolotl" INTO intention_vector COMPRESSING BY semantic_variance
//!     EVOKE    essence OF "user:Xolotl" WITHIN budget 800 tokens
//! ''')
//! ```

use letheo_core::{
    approx_token_count, CognitiveRuntime, EvokeRequest, Insight, Perception, RuntimeConfig,
};
use letheo_exec::{ExecError, ExecResult, Executor};
use letheo_index::Retriever;
use letheo_inference::{CachingProvider, CandleProvider, Provider};

// El binding de producto es Candle-only: no existe build con Mock.
#[cfg(not(feature = "candle"))]
compile_error!("letheo-py requiere la feature `candle` (embeddings reales); no hay variante con Mock.");
use letheo_mql::{parse, validate};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "CompressedContext")]
#[derive(Clone)]
struct PyCompressedContext {
    #[pyo3(get)]
    subject: String,
    #[pyo3(get)]
    represented: usize,
    #[pyo3(get)]
    vectors_returned: usize,
    #[pyo3(get)]
    anomalies_included: usize,
    #[pyo3(get)]
    arc_points: Vec<(f64, f32)>,
    #[pyo3(get)]
    core_label: String,
    #[pyo3(get)]
    arc_labels: Vec<String>,
    #[pyo3(get)]
    anomaly_labels: Vec<String>,
    #[pyo3(get)]
    domain_arcs: Vec<(String, Vec<f32>)>,
    #[pyo3(get)]
    arc_label_histograms: Vec<Vec<(String, usize)>>,
    #[pyo3(get)]
    token_estimate: usize,
    #[pyo3(get)]
    compression_ratio: f64,
    /// Modo en que se enfocó la evocación por `RESONATING WITH` (capa-2), o `None`.
    #[pyo3(get)]
    resonating_mode: Option<String>,
    /// Trayectoria por modo vivo: `[(etiqueta, drift)]` — cuánto cambió cada comportamiento desde su origen.
    #[pyo3(get)]
    mode_drifts: Vec<(String, f32)>,
}

#[pymethods]
impl PyCompressedContext {
    fn __repr__(&self) -> String {
        format!(
            "CompressedContext(subject='{}', represented={}, arc_pts={}, token_estimate={}, ratio={:.1}:1)",
            self.subject,
            self.represented,
            self.arc_points.len(),
            self.token_estimate,
            self.compression_ratio
        )
    }
}

#[pyclass(name = "BreathReport")]
#[derive(Clone)]
struct PyBreathReport {
    #[pyo3(get)]
    distilled_subjects: usize,
    #[pyo3(get)]
    perceptions_absorbed: usize,
    #[pyo3(get)]
    faded: usize,
}

#[pymethods]
impl PyBreathReport {
    fn __repr__(&self) -> String {
        format!(
            "BreathReport(distilled_subjects={}, perceptions_absorbed={}, faded={})",
            self.distilled_subjects, self.perceptions_absorbed, self.faded
        )
    }
}

fn ctx_to_py(c: &letheo_core::CompressedContext) -> PyCompressedContext {
    PyCompressedContext {
        subject: c.subject.clone(),
        represented: c.represented,
        vectors_returned: c.vectors_returned,
        anomalies_included: c.anomalies_included,
        arc_points: c.arc_points.clone(),
        core_label: c.core_label.clone(),
        arc_labels: c.arc_labels.clone(),
        anomaly_labels: c.anomaly_labels.clone(),
        domain_arcs: c.domain_arcs.clone(),
        arc_label_histograms: c.arc_label_histograms.clone(),
        token_estimate: c.token_estimate,
        compression_ratio: c.compression_ratio(),
        resonating_mode: c.resonating_mode.clone(),
        mode_drifts: c.mode_drifts.clone(),
    }
}

// Provider del runtime vivo: Candle real (all-MiniLM-L6-v2). El core de crates no se toca; esto es
// cableado del puente PyO3. No hay variante con Mock — el Mock vive solo en los tests del core.
type RuntimeProvider = CachingProvider<CandleProvider>;

fn make_provider() -> PyResult<RuntimeProvider> {
    // Embeddings reales. Requiere LETHEO_MODEL_DIR apuntando al modelo en disco.
    let p = CandleProvider::load().map_err(|e| {
        PyOSError::new_err(format!(
            "no se pudo cargar el modelo Candle (define LETHEO_MODEL_DIR al dir de all-MiniLM-L6-v2): {e}"
        ))
    })?;
    Ok(CachingProvider::new(p))
}

/// El Cognitive Runtime. Internamente lleva un Executor con provider real (Candle) que sirve tanto a
/// la API directa como a la ejecución de MQL. La caché de embeddings embebe cada texto una sola vez.
#[pyclass(name = "Runtime")]
struct PyRuntime {
    exec: Executor<RuntimeProvider>,
    // Búsqueda por similitud a escala: Flat exacto bajo el umbral, HNSW por encima (cachea el índice).
    retriever: Retriever,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            exec: Executor::new(
                CognitiveRuntime::new(RuntimeConfig::default()),
                make_provider()?,
            ),
            retriever: Retriever::new(256),
        })
    }

    /// `PERCEIVE` directo: asimila un estímulo crudo (texto → embedding via provider local).
    #[pyo3(signature = (subject, text, salience=1.0, halflife_secs=64800.0, now=0.0))]
    fn perceive(
        &mut self,
        subject: &str,
        text: &str,
        salience: f64,
        halflife_secs: f64,
        now: f64,
    ) {
        let embedding = self.exec.provider().embed(text);
        // Guardamos el texto crudo como rasgo: es la etiqueta léxica que la destilación retiene para
        // que la prosa nombre el contenido (no solo vectores). Ver docs/06 §8.bis.
        let perception =
            Perception::new(subject, embedding, salience, halflife_secs, now).with_trait("text", text);
        self.exec.runtime_mut().perceive(perception);
    }

    /// `PERCEIVE` con un embedding **precalculado** (oráculo / Candle / sentence-transformers).
    /// Bypasea el provider interno: el embedding viene de fuera. `text` se guarda como etiqueta
    /// léxica para que la destilación pueda nombrar el contenido.
    #[pyo3(signature = (subject, embedding, text="", salience=1.0, halflife_secs=64800.0, now=0.0))]
    fn perceive_with_embedding(
        &mut self,
        subject: &str,
        embedding: Vec<f32>,
        text: &str,
        salience: f64,
        halflife_secs: f64,
        now: f64,
    ) {
        let p = Perception::new(subject, embedding, salience, halflife_secs, now)
            .with_trait("text", text);
        self.exec.runtime_mut().perceive(p);
    }

    /// Un ciclo de "sueño": DISTILL → IMPRINT para los sujetos dados, luego FADE del ruido.
    fn breathe(&mut self, subjects: Vec<String>, now: f64) -> PyBreathReport {
        let refs: Vec<&str> = subjects.iter().map(|s| s.as_str()).collect();
        let r = self.exec.runtime_mut().breathe(&refs, now);
        PyBreathReport {
            distilled_subjects: r.distilled_subjects,
            perceptions_absorbed: r.perceptions_absorbed,
            faded: r.faded,
        }
    }

    /// `EVOKE` directo: resuelve la esencia de un sujeto dentro del presupuesto de tokens.
    #[pyo3(signature = (subject, token_budget=800, now=0.0))]
    fn evoke(&self, subject: &str, token_budget: usize, now: f64) -> PyResult<PyCompressedContext> {
        let req = EvokeRequest::new(subject, token_budget);
        match self.exec.runtime().evoke(&req, now) {
            Some(c) => Ok(ctx_to_py(&c)),
            None => Err(PyValueError::new_err(format!("sin esencia viva para '{subject}'"))),
        }
    }

    /// **Capa-1** (`remember`): registra un hecho episódico verbatim (texto → embedding via provider),
    /// bajo la física del olvido. Dedup semántico por sujeto. La salience alta lo hace durable.
    #[pyo3(signature = (subject, text, provenance="agent", salience=0.9, halflife_secs=2592000.0, now=0.0))]
    fn remember(
        &mut self,
        subject: &str,
        text: &str,
        provenance: &str,
        salience: f64,
        halflife_secs: f64,
        now: f64,
    ) {
        let embedding = self.exec.provider().embed(text);
        self.exec
            .runtime_mut()
            .remember(subject, text, embedding, provenance, salience, halflife_secs, now);
    }

    /// **Capa-1** (`recall`): recupera los `k` hechos exactos más relevantes de un sujeto (por física)
    /// y los **refuerza** (spaced repetition). Devuelve `[(texto, procedencia, score)]`, verbatim.
    #[pyo3(signature = (subject, query, k=3, now=0.0))]
    fn recall(&mut self, subject: &str, query: &str, k: usize, now: f64) -> Vec<(String, String, f64)> {
        let q = self.exec.provider().embed(query);
        self.exec
            .runtime_mut()
            .recall(subject, &q, k, now)
            .into_iter()
            .map(|f| (f.text, f.provenance, f.score))
            .collect()
    }

    /// **EVOKE unificado**: una sola evocación que responde carácter (capa-2) **y** nominal (capa-1)
    /// bajo UN presupuesto. Devuelve `{gist: CompressedContext|None, facts: [(t,prov,score)],
    /// fact_tokens, total_tokens}`. El coste de hechos se mide con el estimador del core; la capa de
    /// orquestación puede inyectar tiktoken para el conteo exacto.
    #[pyo3(signature = (subject, query, token_budget=800, fact_budget=200, now=0.0))]
    fn evoke_unified<'py>(
        &self,
        py: Python<'py>,
        subject: &str,
        query: &str,
        token_budget: usize,
        fact_budget: usize,
        now: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let q = self.exec.provider().embed(query);
        let req = EvokeRequest::new(subject, token_budget);
        let u = self.exec.runtime().evoke_unified(&req, &q, fact_budget, now, approx_token_count);
        let d = PyDict::new(py);
        d.set_item("gist", u.gist.as_ref().map(ctx_to_py))?;
        let facts: Vec<(String, String, f64)> =
            u.facts.iter().map(|f| (f.text.clone(), f.provenance.clone(), f.score)).collect();
        d.set_item("facts", facts)?;
        d.set_item("fact_tokens", u.fact_tokens)?;
        d.set_item("total_tokens", u.total_tokens)?;
        Ok(d)
    }

    /// **Reflexión** (capa generativa): insights de orden superior sobre el arco del sujeto —
    /// transiciones y revivals que no están en ningún evento. Devuelve una lista de dicts.
    fn reflect<'py>(&self, py: Python<'py>, subject: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for ins in self.exec.runtime().reflect(subject) {
            let d = PyDict::new(py);
            match ins {
                Insight::Transition { from, to, support } => {
                    d.set_item("kind", "transition")?;
                    d.set_item("from", from)?;
                    d.set_item("to", to)?;
                    d.set_item("support", support)?;
                }
                Insight::Revival { domain } => {
                    d.set_item("kind", "revival")?;
                    d.set_item("domain", domain)?;
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// **Sueño reflexivo**: reflexiona y **materializa** los insights como hechos de alta salience en
    /// la capa-1 (recuperables por `recall`). Devuelve cuántos se guardaron.
    #[pyo3(signature = (subject, now=0.0))]
    fn dream_reflect(&mut self, subject: &str, now: f64) -> usize {
        self.exec.runtime_mut().dream_reflect(subject, now)
    }

    /// Búsqueda por **similitud** (no por id): los `k` sujetos cuya esencia más resuena con `query`.
    /// Usa el índice ANN (HNSW) por encima del umbral de tamaño, Flat exacto por debajo, **filtrando
    /// por vida**. Para enrutar una tarea al agente/sujeto más relevante (caso flota de Paideia).
    #[pyo3(signature = (query, k=5, now=0.0))]
    fn resonate(&mut self, query: &str, k: usize, now: f64) -> Vec<String> {
        let q = self.exec.provider().embed(query);
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        self.retriever
            .resonate_subjects(self.exec.runtime().long_term(), &q, k, now, theta)
    }

    /// Ejecuta un programa MQL completo. Devuelve una lista de dicts, uno por sentencia, con la
    /// forma `{"kind": "...", ...campos}`. Errores por sentencia van en `{"kind": "error", "message": "..."}`.
    #[pyo3(signature = (src, now=0.0))]
    fn execute_mql<'py>(
        &mut self,
        py: Python<'py>,
        src: &str,
        now: f64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let stmts = parse(src).map_err(|e| PyValueError::new_err(e.message))?;
        let mut out = Vec::with_capacity(stmts.len());
        for stmt in &stmts {
            let d = PyDict::new(py);
            match self.exec.execute(stmt, now) {
                Ok(ExecResult::Perceived { subject }) => {
                    d.set_item("kind", "perceived")?;
                    d.set_item("subject", subject)?;
                }
                Ok(ExecResult::Dreamed(r)) => {
                    d.set_item("kind", "dreamed")?;
                    d.set_item("distilled_subjects", r.distilled_subjects)?;
                    d.set_item("perceptions_absorbed", r.perceptions_absorbed)?;
                    d.set_item("faded", r.faded)?;
                }
                Ok(ExecResult::Evoked(c)) => {
                    d.set_item("kind", "evoked")?;
                    d.set_item("context", ctx_to_py(&c).into_pyobject(py)?)?;
                }
                Ok(ExecResult::Faded { swept }) => {
                    d.set_item("kind", "faded")?;
                    d.set_item("swept", swept)?;
                }
                Ok(ExecResult::Imprinted { archetype, note }) => {
                    d.set_item("kind", "imprinted")?;
                    d.set_item("archetype", archetype)?;
                    d.set_item("note", note)?;
                }
                Ok(ExecResult::Recalled(facts)) => {
                    d.set_item("kind", "recalled")?;
                    let items: Vec<(String, String, f64)> =
                        facts.iter().map(|f| (f.text.clone(), f.provenance.clone(), f.score)).collect();
                    d.set_item("facts", items)?;
                }
                Ok(ExecResult::Reinforced { count }) => {
                    d.set_item("kind", "reinforced")?;
                    d.set_item("count", count)?;
                }
                Err(e) => {
                    d.set_item("kind", "error")?;
                    d.set_item("message", e.to_string())?;
                    d.set_item("variant", match e {
                        ExecError::NoSuchSubject(_) => "no_such_subject",
                        ExecError::MissingBudget => "missing_budget",
                    })?;
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// Persiste **las dos capas** en `dir`: capa-2 (un snapshot JSON por arquetipo) y capa-1
    /// (`facts.json`). Devuelve cuántos arquetipos se guardaron. La memoria sobrevive al reinicio.
    fn save(&self, dir: &str) -> PyResult<usize> {
        let n = letheo_persist::save_store(dir, self.exec.runtime().long_term())
            .map_err(|e| PyOSError::new_err(format!("no se pudo guardar en '{dir}': {e}")))?;
        letheo_persist::save_facts(dir, self.exec.runtime().facts())
            .map_err(|e| PyOSError::new_err(format!("no se pudieron guardar los hechos en '{dir}': {e}")))?;
        Ok(n)
    }

    /// Rehidrata **las dos capas** desde `dir` (arquetipos + hechos). Reemplaza la memoria actual.
    /// Devuelve cuántos arquetipos se cargaron. Un directorio inexistente carga 0 (primer arranque).
    fn load(&mut self, dir: &str) -> PyResult<usize> {
        let store = letheo_persist::load_store(dir)
            .map_err(|e| PyOSError::new_err(format!("no se pudo cargar de '{dir}': {e}")))?;
        let facts = letheo_persist::load_facts(dir)
            .map_err(|e| PyOSError::new_err(format!("no se pudieron cargar los hechos de '{dir}': {e}")))?;
        let n = store.len();
        *self.exec.runtime_mut().long_term_mut() = store;
        *self.exec.runtime_mut().facts_mut() = facts;
        Ok(n)
    }

    /// Sujetos con esencia consolidada en memoria de largo plazo (p. ej. tras `load`).
    #[getter]
    fn subjects(&self) -> Vec<String> {
        self.exec.runtime().long_term().iter().map(|a| a.subject.clone()).collect()
    }

    #[getter]
    fn short_term_len(&self) -> usize {
        self.exec.runtime().short_term_len()
    }

    #[getter]
    fn long_term_len(&self) -> usize {
        self.exec.runtime().long_term_len()
    }

    /// Estadísticas de la caché de embeddings: `{hits, misses, entries, hit_rate}`.
    fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.exec.provider().stats();
        let d = PyDict::new(py);
        d.set_item("hits", s.hits)?;
        d.set_item("misses", s.misses)?;
        d.set_item("entries", s.entries)?;
        d.set_item("hit_rate", s.hit_rate())?;
        Ok(d)
    }
}

/// `Embedder` — embeddings reales con Candle (`all-MiniLM-L6-v2`, 384-dim), local y sin red.
///
/// Solo existe si el binding se compiló con `--features candle`. Calcula vectores en Python para
/// inyectarlos vía `Runtime.perceive_with_embedding` / `Session.perceive_vector` — el mismo enchufe
/// que usa el oráculo del arnés, pero con semántica real. Carga el modelo desde `LETHEO_MODEL_DIR`
/// (poblar con `python sandbox/fetch_model.py`).
#[cfg(feature = "candle")]
#[pyclass(name = "Embedder")]
struct PyEmbedder {
    inner: letheo_inference::CandleProvider,
}

#[cfg(feature = "candle")]
#[pymethods]
impl PyEmbedder {
    /// Carga el modelo desde el directorio en `LETHEO_MODEL_DIR`.
    #[new]
    fn new() -> PyResult<Self> {
        let inner = letheo_inference::CandleProvider::load()
            .map_err(|e| PyOSError::new_err(format!("no se pudo cargar el modelo Candle: {e}")))?;
        Ok(Self { inner })
    }

    /// Carga el modelo desde un directorio explícito (config.json, tokenizer.json, model.safetensors).
    #[staticmethod]
    fn from_dir(dir: &str) -> PyResult<Self> {
        let inner = letheo_inference::CandleProvider::from_dir(dir)
            .map_err(|e| PyOSError::new_err(format!("no se pudo cargar el modelo Candle: {e}")))?;
        Ok(Self { inner })
    }

    /// Dimensión de los embeddings (384).
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Embebe un texto → vector L2-normalizado de 384 dimensiones.
    fn embed(&self, text: &str) -> Vec<f32> {
        self.inner.embed(text)
    }
}

/// Parsea un programa MQL y devuelve el número de sentencias (sin ejecutar).
#[pyfunction]
fn parse_mql(src: &str) -> PyResult<usize> {
    parse(src).map(|s| s.len()).map_err(|e| PyValueError::new_err(e.message))
}

/// Valida semánticamente un programa MQL. Devuelve la lista de problemas (vacía ⇒ válido).
/// Errores de sintaxis se lanzan como `ValueError`.
#[pyfunction]
fn validate_mql(src: &str) -> PyResult<Vec<String>> {
    let stmts = parse(src).map_err(|e| PyValueError::new_err(e.message))?;
    Ok(validate(&stmts).iter().map(|p| p.to_string()).collect())
}

#[pymodule]
fn letheo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyCompressedContext>()?;
    m.add_class::<PyBreathReport>()?;
    #[cfg(feature = "candle")]
    m.add_class::<PyEmbedder>()?;
    m.add_function(wrap_pyfunction!(parse_mql, m)?)?;
    m.add_function(wrap_pyfunction!(validate_mql, m)?)?;
    Ok(())
}
