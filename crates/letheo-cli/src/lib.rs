//! # letheo-cli · REPL de MQL
//!
//! Cierra la experiencia de producto: *tocar* el lenguaje sin escribir Rust ni Python. Lee MQL,
//! lo ejecuta contra un `Executor<P: Provider>` (Candle real en el binario) y muestra el contexto
//! resuelto. Lleva un reloj lógico que se avanza con `:tick`, y persiste la memoria con `:save`/`:load`.
//!
//! La lógica vive aquí (testeable sin TTY); `main.rs` solo conecta stdin/stdout.

use std::path::PathBuf;

use letheo_core::{CognitiveRuntime, RuntimeConfig, Tick};
use letheo_exec::{ExecError, ExecResult, Executor};
use letheo_inference::{CandleProvider, Provider};
use letheo_mql::{parse, validate};

/// Resultado de evaluar una línea del REPL.
#[derive(Debug, PartialEq)]
pub enum Eval {
    /// Texto a mostrar al usuario.
    Output(String),
    /// El usuario pidió salir.
    Quit,
}

/// Estado del REPL: el runtime, el reloj lógico y la ruta de persistencia opcional.
/// Genérico sobre el `Provider`: el binario usa `CandleProvider` (real); los tests, `MockProvider`.
pub struct Repl<P: Provider> {
    exec: Executor<P>,
    now: Tick,
    persist: Option<PathBuf>,
}

/// REPL de producto: embeddings reales (Candle).
pub type RealRepl = Repl<CandleProvider>;

impl Repl<CandleProvider> {
    /// Crea el REPL de producto con provider Candle real. Requiere `LETHEO_MODEL_DIR`.
    pub fn real(persist: Option<PathBuf>) -> std::io::Result<Self> {
        let provider = CandleProvider::load().map_err(|e| std::io::Error::other(e.to_string()))?;
        Self::with_provider(provider, persist)
    }
}

impl<P: Provider> Repl<P> {
    /// Crea un REPL con el `provider` dado. Si `persist` apunta a snapshots, rehidrata la memoria.
    pub fn with_provider(provider: P, persist: Option<PathBuf>) -> std::io::Result<Self> {
        let mut exec = Executor::new(CognitiveRuntime::new(RuntimeConfig::default()), provider);
        if let Some(dir) = &persist {
            let store = letheo_persist::load_store(dir)?;
            *exec.runtime_mut().long_term_mut() = store;
        }
        Ok(Self {
            exec,
            now: 0.0,
            persist,
        })
    }

    pub fn now(&self) -> Tick {
        self.now
    }

    /// Evalúa una línea: meta-comando (`:`) o programa MQL.
    pub fn eval(&mut self, input: &str) -> Eval {
        let line = input.trim();
        if line.is_empty() {
            return Eval::Output(String::new());
        }
        if let Some(cmd) = line.strip_prefix(':') {
            return self.meta(cmd.trim());
        }
        self.run_mql(line)
    }

    fn meta(&mut self, cmd: &str) -> Eval {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let verb = parts.next().unwrap_or("");
        let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

        match verb {
            "q" | "quit" | "exit" => Eval::Quit,
            "help" | "h" | "?" => Eval::Output(HELP.to_string()),
            "now" => Eval::Output(format!("now = {:.0}s", self.now)),
            "tick" => match arg.and_then(|a| a.parse::<f64>().ok()) {
                Some(s) if s >= 0.0 => {
                    self.now += s;
                    Eval::Output(format!("⏱  now = {:.0}s", self.now))
                }
                _ => Eval::Output("uso: :tick <segundos ≥ 0>".into()),
            },
            "state" => Eval::Output(format!(
                "corto plazo: {} percepciones · largo plazo: {} arquetipos",
                self.exec.runtime().short_term_len(),
                self.exec.runtime().long_term_len(),
            )),
            "subjects" => {
                let subs: Vec<&str> = self
                    .exec
                    .runtime()
                    .long_term()
                    .iter()
                    .map(|a| a.subject.as_str())
                    .collect();
                Eval::Output(if subs.is_empty() {
                    "(sin arquetipos consolidados)".into()
                } else {
                    subs.join("\n")
                })
            }
            "save" => {
                let dir = arg.map(PathBuf::from).or_else(|| self.persist.clone());
                match dir {
                    None => Eval::Output("uso: :save <dir>  (o arranca con --persist)".into()),
                    Some(d) => {
                        match letheo_persist::save_store(&d, self.exec.runtime().long_term()) {
                            Ok(n) => Eval::Output(format!(
                                "💾 guardados {n} arquetipos en {}",
                                d.display()
                            )),
                            Err(e) => Eval::Output(format!("error al guardar: {e}")),
                        }
                    }
                }
            }
            "load" => {
                let dir = arg.map(PathBuf::from).or_else(|| self.persist.clone());
                match dir {
                    None => Eval::Output("uso: :load <dir>  (o arranca con --persist)".into()),
                    Some(d) => match letheo_persist::load_store(&d) {
                        Ok(store) => {
                            let n = store.len();
                            *self.exec.runtime_mut().long_term_mut() = store;
                            Eval::Output(format!("📂 cargados {n} arquetipos de {}", d.display()))
                        }
                        Err(e) => Eval::Output(format!("error al cargar: {e}")),
                    },
                }
            }
            other => Eval::Output(format!("comando desconocido: ':{other}' — prueba :help")),
        }
    }

    fn run_mql(&mut self, src: &str) -> Eval {
        let stmts = match parse(src) {
            Ok(s) => s,
            Err(e) => return Eval::Output(format!("⚠ error de sintaxis: {}", e.message)),
        };
        // Validación semántica: si el programa no tiene sentido, no lo ejecutamos a medias.
        let problems = validate(&stmts);
        if !problems.is_empty() {
            let msg: Vec<String> = problems.iter().map(|p| format!("⚠ {p}")).collect();
            return Eval::Output(msg.join("\n"));
        }
        let mut lines = Vec::new();
        for stmt in &stmts {
            lines.push(match self.exec.execute(stmt, self.now) {
                Ok(r) => format_result(&r),
                Err(e) => format!("⚠ {}", format_error(&e)),
            });
        }
        Eval::Output(lines.join("\n"))
    }

    /// Guarda la memoria si hay ruta de persistencia configurada (llamado al salir).
    pub fn autosave(&self) -> Option<std::io::Result<usize>> {
        self.persist
            .as_ref()
            .map(|dir| letheo_persist::save_store(dir, self.exec.runtime().long_term()))
    }
}

fn format_result(r: &ExecResult) -> String {
    match r {
        ExecResult::Perceived { subject } => format!("· percibido «{subject}»"),
        ExecResult::Dreamed(b) => format!(
            "· soñado: {} sujeto(s) consolidado(s), {} percepción(es) absorbida(s), {} desvanecida(s)",
            b.distilled_subjects, b.perceptions_absorbed, b.faded
        ),
        ExecResult::Evoked(c) => format!(
            "· evocado «{}»: {} eventos → {} vectores · ~{} tokens · compresión {:.1}:1",
            c.subject,
            c.represented,
            c.vectors_returned,
            c.token_estimate,
            c.compression_ratio()
        ),
        ExecResult::Faded { swept } => format!("· desvanecidas {swept} percepción(es)"),
        ExecResult::Imprinted { archetype, .. } => format!("· grabado «{archetype}» (esencia consolidada)"),
        ExecResult::Recalled(facts) => {
            if facts.is_empty() {
                "· recall: sin hechos que resuenen".to_string()
            } else {
                let items: Vec<String> = facts
                    .iter()
                    .map(|f| format!("«{}» ({:.2})", f.text, f.score))
                    .collect();
                format!("· recuperado(s) {} hecho(s): {}", facts.len(), items.join(", "))
            }
        }
        ExecResult::Reinforced { count } => format!("· reforzado(s) {count} hecho(s) (decay reseteado)"),
    }
}

fn format_error(e: &ExecError) -> String {
    match e {
        ExecError::NoSuchSubject(s) => format!("ningún arquetipo vivo para «{s}»"),
        ExecError::MissingBudget => "EVOKE requiere WITHIN budget N tokens".into(),
    }
}

pub const HELP: &str = "\
Verbos MQL — escríbelos directamente:
  PERCEIVE interaction FROM subject \"u:X\" AS { act: buy, object: shoes }
  DISTILL  subject \"u:X\" INTO intention_vector COMPRESSING BY semantic_variance
  EVOKE    essence OF \"u:X\" WITHIN budget 800 tokens
  EVOKE    essence OF \"u:X\" RESONATING WITH { nostalgia } WITHIN budget 800 tokens
  FADE     noise WHERE weight now < 0.05 PRESERVING archetype_contribution
  IMPRINT  archetype \"u:X\" FROM intention_vector RESILIENCE high
  RECALL   facts FROM subject \"u:X\" RESONATING WITH { allergy } WHERE resonates > 0.6 WITHIN k 3
  REINFORCE facts FROM subject \"u:X\" RESONATING WITH { allergy } WITHIN k 3

Meta-comandos:
  :tick <s>     avanza el reloj lógico <s> segundos
  :now          muestra el reloj
  :state        tamaño de memoria corto/largo plazo
  :subjects     arquetipos consolidados
  :save [dir]   persiste la memoria (un JSON por sujeto)
  :load [dir]   rehidrata la memoria desde disco
  :help         esta ayuda
  :quit         salir";

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_inference::MockProvider;

    fn repl() -> Repl<MockProvider> {
        Repl::with_provider(MockProvider::new(), None).unwrap()
    }

    fn out(e: Eval) -> String {
        match e {
            Eval::Output(s) => s,
            Eval::Quit => "<quit>".into(),
        }
    }

    #[test]
    fn full_session_perceive_distill_evoke() {
        let mut r = repl();
        out(r.eval(r#"PERCEIVE interaction FROM subject "u:X" AS { act: buy }"#));
        out(r.eval(r#"PERCEIVE interaction FROM subject "u:X" AS { act: buy }"#));
        let dreamed = out(r.eval(r#"DISTILL subject "u:X" INTO intention_vector"#));
        assert!(dreamed.contains("consolidado"), "{dreamed}");
        let evoked = out(r.eval(r#"EVOKE essence OF "u:X" WITHIN budget 800 tokens"#));
        assert!(
            evoked.contains("evocado") && evoked.contains("compresión"),
            "{evoked}"
        );
    }

    #[test]
    fn tick_advances_logical_clock() {
        let mut r = repl();
        assert_eq!(r.now(), 0.0);
        out(r.eval(":tick 3600"));
        assert_eq!(r.now(), 3600.0);
        out(r.eval(":tick 60"));
        assert_eq!(r.now(), 3660.0);
    }

    #[test]
    fn syntax_error_is_reported_not_panicked() {
        let mut r = repl();
        let o = out(r.eval("PERCEIVE wat"));
        assert!(o.contains("error de sintaxis"), "{o}");
    }

    #[test]
    fn quit_command() {
        let mut r = repl();
        assert_eq!(r.eval(":quit"), Eval::Quit);
        assert_eq!(r.eval(":q"), Eval::Quit);
    }

    #[test]
    fn save_then_load_via_meta_commands() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("letheo_cli_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap().to_string();

        // Sesión 1: consolida y guarda.
        let mut r = repl();
        out(r.eval(r#"PERCEIVE interaction FROM subject "u:X" AS { act: buy }"#));
        out(r.eval(r#"DISTILL subject "u:X" INTO intention_vector"#));
        let saved = out(r.eval(&format!(":save {path}")));
        assert!(saved.contains("guardados 1"), "{saved}");

        // Sesión 2: carga y evoca lo aprendido antes.
        let mut r2 = repl();
        let loaded = out(r2.eval(&format!(":load {path}")));
        assert!(loaded.contains("cargados 1"), "{loaded}");
        let subs = out(r2.eval(":subjects"));
        assert!(subs.contains("u:X"), "{subs}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn semantic_error_blocks_execution() {
        let mut r = repl();
        // budget 0 es sintácticamente válido pero semánticamente absurdo: no debe ejecutarse.
        let o = out(r.eval(r#"EVOKE essence OF "u:X" WITHIN budget 0 tokens"#));
        assert!(o.contains("presupuesto") && o.contains("> 0"), "{o}");
    }

    #[test]
    fn unknown_meta_command_is_friendly() {
        let mut r = repl();
        let o = out(r.eval(":wat"));
        assert!(o.contains("desconocido"), "{o}");
    }
}
