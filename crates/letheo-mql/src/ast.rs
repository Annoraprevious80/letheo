//! AST de MQL. Siete verbos: PERCEIVE · DISTILL · EVOKE · FADE · IMPRINT · RECALL · REINFORCE.
//! Gramática formal en `docs/02-mql-grammar.ebnf`.

use std::collections::BTreeMap;

/// Una sentencia MQL.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Perceive(Perceive),
    Distill(Distill),
    Evoke(Evoke),
    Fade(Fade),
    Imprint(Imprint),
    /// Capa-1: recuperación dirigida y sin pérdida de hechos episódicos por resonancia.
    Recall(Recall),
    /// Capa-1: refuerzo / spaced-repetition de hechos (resetea su decay).
    Reinforce(Reinforce),
}

/// Duración como coeficiente de entropía (no timestamp). Normalizada a segundos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Duration {
    pub seconds: f64,
}

impl Duration {
    pub fn from_value_unit(value: f64, unit: &str) -> Option<Self> {
        let mult = match unit {
            "m" | "min" => 60.0,
            "h" | "hour" => 3600.0,
            "d" | "day" => 86_400.0,
            "w" | "week" => 604_800.0,
            "month" => 2_592_000.0,
            "y" | "year" => 31_536_000.0,
            _ => return None,
        };
        Some(Duration {
            seconds: value * mult,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Perceive {
    pub subject: String,
    pub traits: BTreeMap<String, String>,
    pub salience: Option<f64>,
    pub halflife: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Distill {
    pub subject: String,
    pub compressing_by_variance: bool,
    pub retaining: Vec<String>,
    /// Filtro `WHERE` opcional sobre las percepciones a destilar.
    pub filter: Option<Predicate>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Predicados de WHERE (épica 9.0): el lenguaje deja de consumir el filtro de forma
// laxa y lo evalúa de verdad sobre cada percepción.
// ─────────────────────────────────────────────────────────────────────────────

/// Operador de comparación de un predicado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// El campo sobre el que se compara: una propiedad física del recuerdo o un rasgo arbitrario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field {
    /// Peso actual del recuerdo (`weight now`). Numérico, depende del tiempo.
    Weight,
    /// Carga inicial del estímulo. Numérico.
    Salience,
    /// Antigüedad en segundos desde el último contacto (`Δt`). Numérico.
    Age,
    /// Resonancia (coseno) del ítem con la consulta de la sentencia. Numérico, en `[-1, 1]`. Requiere
    /// que la sentencia aporte una consulta (`RESONATING WITH { … }`); sin consulta evalúa a falso.
    Resonance,
    /// Un rasgo cualquiera del trait map (p. ej. `domain`, `mood`).
    Trait(String),
}

/// Valor literal contra el que se compara.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Text(String),
}

/// Un predicado booleano de `WHERE`: comparaciones combinadas con AND/OR/NOT.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Cmp {
        field: Field,
        op: CmpOp,
        value: Value,
    },
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
}

/// Fuente de hechos sobre la que se evalúa un predicado. La implementa quien tenga los datos
/// (el ejecutor, sobre una `Perception`), manteniendo `letheo-mql` desacoplado de `letheo-core`.
pub trait Facts {
    /// Valor numérico de un campo físico (`Weight`/`Salience`/`Age`) en el contexto actual.
    fn numeric(&self, field: &Field) -> Option<f64>;
    /// Valor textual de un rasgo del trait map.
    fn text(&self, key: &str) -> Option<String>;
}

impl Predicate {
    /// Evalúa el predicado contra una fuente de hechos.
    pub fn eval(&self, f: &dyn Facts) -> bool {
        match self {
            Predicate::And(a, b) => a.eval(f) && b.eval(f),
            Predicate::Or(a, b) => a.eval(f) || b.eval(f),
            Predicate::Not(a) => !a.eval(f),
            Predicate::Cmp { field, op, value } => eval_cmp(field, *op, value, f),
        }
    }
}

fn eval_cmp(field: &Field, op: CmpOp, value: &Value, f: &dyn Facts) -> bool {
    match (field, value) {
        // Campo físico/resonancia vs. número → comparación numérica.
        (Field::Weight | Field::Salience | Field::Age | Field::Resonance, Value::Num(rhs)) => {
            match f.numeric(field) {
                Some(lhs) => cmp_num(lhs, op, *rhs),
                None => false,
            }
        }
        // Rasgo vs. número → intenta interpretar el rasgo como número.
        (Field::Trait(k), Value::Num(rhs)) => match f.text(k).and_then(|s| s.parse::<f64>().ok()) {
            Some(lhs) => cmp_num(lhs, op, *rhs),
            None => false,
        },
        // Rasgo vs. texto → comparación de cadenas (orden lexicográfico para </>).
        (Field::Trait(k), Value::Text(rhs)) => match f.text(k) {
            Some(lhs) => cmp_text(&lhs, op, rhs),
            None => false,
        },
        // Campo físico/resonancia vs. texto: sin sentido semántico → falso.
        (Field::Weight | Field::Salience | Field::Age | Field::Resonance, Value::Text(_)) => false,
    }
}

fn cmp_num(lhs: f64, op: CmpOp, rhs: f64) -> bool {
    match op {
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
    }
}

fn cmp_text(lhs: &str, op: CmpOp, rhs: &str) -> bool {
    match op {
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EssenceKind {
    Essence,
    Archetype,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resolution {
    Arc,
    Point,
    Summary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Trajectory,
    Snapshot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Evoke {
    pub kind: EssenceKind,
    pub subject: String,
    pub span: Option<Duration>,
    pub resonating_with: Vec<String>,
    pub resolution: Option<Resolution>,
    pub projecting: Option<Projection>,
    pub token_budget: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fade {
    pub target: String,
    pub preserving_archetype: bool,
    /// Filtro `WHERE` opcional: qué percepciones son candidatas a desvanecerse.
    pub filter: Option<Predicate>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resilience {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Imprint {
    pub archetype: String,
    pub resilience: Option<Resilience>,
}

/// `RECALL` — recuperación dirigida de hechos episódicos (capa-1), **sin pérdida** y **sin efectos**
/// (read-only). La consulta se forma con `RESONATING WITH { … }` (se embebe). `WHERE` opcional admite
/// `resonates`/`weight`/`age`/`salience`. `WITHIN k N` acota el top-k (default 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Recall {
    pub subject: String,
    pub resonating_with: Vec<String>,
    pub k: usize,
    pub filter: Option<Predicate>,
}

/// `REINFORCE` — refuerzo / spaced-repetition de los hechos que resuenan con la consulta (resetea su
/// decay → ganan permanencia). Muta la capa-1. `WITHIN k N` acota cuántos reforzar (default 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Reinforce {
    pub subject: String,
    pub resonating_with: Vec<String>,
    pub k: usize,
}
