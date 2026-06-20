//! # letheo-mql · Mnemonic Query Language
//!
//! Lexer + parser de los cinco verbos biológicos: PERCEIVE · DISTILL · EVOKE · FADE · IMPRINT.
//! **No existe** SELECT / INSERT / UPDATE / DELETE. Gramática en `docs/02-mql-grammar.ebnf`.
//!
//! ```
//! use letheo_mql::parse;
//! let prog = parse(r#"EVOKE essence OF "user:Xolotl" WITHIN budget 800 tokens"#).unwrap();
//! assert_eq!(prog.len(), 1);
//! ```

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod validate;

pub use ast::{CmpOp, Facts, Field, Predicate, Statement, Value};
pub use parser::{parse, ParseError};
pub use validate::{validate, SemanticError, SemanticErrorKind};
