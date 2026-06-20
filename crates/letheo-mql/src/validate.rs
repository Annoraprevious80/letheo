//! Validación semántica estática del AST (épica 9.1).
//!
//! El parser garantiza que el programa es *sintácticamente* válido; aquí comprobamos que además
//! tiene *sentido* antes de ejecutarlo: presupuestos positivos, sujetos no vacíos, percepciones con
//! al menos un rasgo, salience en rango. Son comprobaciones que no necesitan el runtime — fallan
//! temprano y con un mensaje claro, igual en la lib, el REPL y Python.

use crate::ast::{Statement, Value};

/// Tipo de problema semántico encontrado en una sentencia.
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticErrorKind {
    /// `EVOKE` sin `WITHIN budget N tokens`.
    MissingBudget,
    /// `EVOKE … WITHIN budget 0 tokens`: no cabe nada.
    ZeroBudget,
    /// Sujeto vacío (`""`).
    EmptySubject,
    /// `PERCEIVE … AS { }` sin ningún rasgo: no hay estímulo que embeber.
    EmptyPerception,
    /// `WITH salience S` fuera de `(0, 1]`.
    SalienceOutOfRange(f64),
    /// `FADE` con un predicado que compara un campo físico (`weight`/`salience`/`age`) con texto.
    PhysicalFieldVsText(String),
}

impl std::fmt::Display for SemanticErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticErrorKind::MissingBudget => {
                write!(f, "EVOKE necesita 'WITHIN budget N tokens'")
            }
            SemanticErrorKind::ZeroBudget => {
                write!(f, "el presupuesto de tokens debe ser > 0")
            }
            SemanticErrorKind::EmptySubject => write!(f, "el sujeto no puede estar vacío"),
            SemanticErrorKind::EmptyPerception => {
                write!(f, "PERCEIVE necesita al menos un rasgo en 'AS {{ ... }}'")
            }
            SemanticErrorKind::SalienceOutOfRange(s) => {
                write!(f, "salience {s} fuera de rango (0, 1]")
            }
            SemanticErrorKind::PhysicalFieldVsText(field) => {
                write!(
                    f,
                    "el campo físico '{field}' solo compara con números, no con texto"
                )
            }
        }
    }
}

/// Un error semántico localizado: a qué sentencia (índice 0-based) pertenece y qué pasa.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticError {
    /// Índice de la sentencia en el programa.
    pub stmt_index: usize,
    pub kind: SemanticErrorKind,
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sentencia #{}: {}", self.stmt_index + 1, self.kind)
    }
}

/// Valida un programa completo. Devuelve **todos** los errores (no se detiene en el primero) para
/// que el usuario los corrija de una pasada. Lista vacía ⇒ programa semánticamente válido.
pub fn validate(program: &[Statement]) -> Vec<SemanticError> {
    let mut errors = Vec::new();
    for (i, stmt) in program.iter().enumerate() {
        let mut push = |kind| {
            errors.push(SemanticError {
                stmt_index: i,
                kind,
            })
        };
        match stmt {
            Statement::Perceive(p) => {
                if p.subject.trim().is_empty() {
                    push(SemanticErrorKind::EmptySubject);
                }
                if p.traits.is_empty() {
                    push(SemanticErrorKind::EmptyPerception);
                }
                if let Some(s) = p.salience {
                    if !(s > 0.0 && s <= 1.0) {
                        push(SemanticErrorKind::SalienceOutOfRange(s));
                    }
                }
            }
            Statement::Distill(d) => {
                if d.subject.trim().is_empty() {
                    push(SemanticErrorKind::EmptySubject);
                }
            }
            Statement::Evoke(e) => {
                if e.subject.trim().is_empty() {
                    push(SemanticErrorKind::EmptySubject);
                }
                match e.token_budget {
                    None => push(SemanticErrorKind::MissingBudget),
                    Some(0) => push(SemanticErrorKind::ZeroBudget),
                    Some(_) => {}
                }
            }
            Statement::Fade(fade) => {
                if let Some(pred) = &fade.filter {
                    check_predicate(pred, &mut push);
                }
            }
            Statement::Imprint(im) => {
                if im.archetype.trim().is_empty() {
                    push(SemanticErrorKind::EmptySubject);
                }
            }
            Statement::Recall(r) => {
                if r.subject.trim().is_empty() {
                    push(SemanticErrorKind::EmptySubject);
                }
                if let Some(pred) = &r.filter {
                    check_predicate(pred, &mut push);
                }
            }
            Statement::Reinforce(r) => {
                if r.subject.trim().is_empty() {
                    push(SemanticErrorKind::EmptySubject);
                }
            }
        }
    }
    errors
}

/// Recorre un predicado buscando comparaciones sin sentido (campo físico vs. texto).
fn check_predicate(pred: &crate::ast::Predicate, push: &mut impl FnMut(SemanticErrorKind)) {
    use crate::ast::{Field, Predicate};
    match pred {
        Predicate::And(a, b) | Predicate::Or(a, b) => {
            check_predicate(a, push);
            check_predicate(b, push);
        }
        Predicate::Not(a) => check_predicate(a, push),
        Predicate::Cmp { field, value, .. } => {
            let physical = match field {
                Field::Weight => Some("weight"),
                Field::Salience => Some("salience"),
                Field::Age => Some("age"),
                Field::Resonance => Some("resonance"),
                Field::Trait(_) => None,
            };
            if let (Some(name), Value::Text(_)) = (physical, value) {
                push(SemanticErrorKind::PhysicalFieldVsText(name.to_string()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn validate_src(src: &str) -> Vec<SemanticError> {
        validate(&parse(src).unwrap())
    }

    #[test]
    fn valid_program_has_no_errors() {
        let src = r#"
            PERCEIVE interaction FROM subject "u:X" AS { act: buy } WITH salience 0.7
            DISTILL subject "u:X" INTO intention_vector
            EVOKE essence OF "u:X" WITHIN budget 800 tokens
        "#;
        assert!(validate_src(src).is_empty());
    }

    #[test]
    fn evoke_without_budget_is_flagged() {
        let errs = validate_src(r#"EVOKE essence OF "u:X""#);
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, SemanticErrorKind::MissingBudget);
    }

    #[test]
    fn zero_budget_is_flagged() {
        let errs = validate_src(r#"EVOKE essence OF "u:X" WITHIN budget 0 tokens"#);
        assert_eq!(errs[0].kind, SemanticErrorKind::ZeroBudget);
    }

    #[test]
    fn salience_out_of_range_is_flagged() {
        let errs = validate_src(
            r#"PERCEIVE interaction FROM subject "u:X" AS { a: b } WITH salience 1.5"#,
        );
        assert_eq!(errs[0].kind, SemanticErrorKind::SalienceOutOfRange(1.5));
    }

    #[test]
    fn empty_subject_is_flagged() {
        let errs = validate_src(r#"EVOKE essence OF "" WITHIN budget 800 tokens"#);
        assert_eq!(errs[0].kind, SemanticErrorKind::EmptySubject);
    }

    #[test]
    fn physical_field_vs_text_is_flagged() {
        let errs =
            validate_src(r#"FADE noise WHERE weight now "low" PRESERVING archetype_contribution"#);
        assert_eq!(errs.len(), 1);
        assert_eq!(
            errs[0].kind,
            SemanticErrorKind::PhysicalFieldVsText("weight".into())
        );
    }

    #[test]
    fn all_errors_reported_in_one_pass() {
        let src = r#"
            EVOKE essence OF "u:X"
            EVOKE essence OF "u:Y" WITHIN budget 0 tokens
        "#;
        let errs = validate_src(src);
        assert_eq!(errs.len(), 2);
        assert_eq!(errs[0].stmt_index, 0);
        assert_eq!(errs[1].stmt_index, 1);
    }
}
