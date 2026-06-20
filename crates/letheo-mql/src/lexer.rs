//! Lexer de MQL. Produce una secuencia de [`Token`] a partir del texto fuente.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Palabra clave o identificador (se distinguen en el parser, case-insensitive para keywords).
    Word(String),
    Str(String),
    Number(f64),
    /// Operador de comparación en predicados: `< <= > >= == !=`.
    Op(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    /// Comentarios de línea `--` ya se descartan; no se emiten.
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub pos: usize,
}

/// Tokeniza el código fuente MQL.
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();

    while i < chars.len() {
        let c = chars[i];

        // Espacios.
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Comentarios de línea: `-- ...`
        if c == '-' && i + 1 < chars.len() && chars[i + 1] == '-' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        match c {
            '<' | '>' | '=' | '!' => {
                let mut op = String::from(c);
                i += 1;
                if i < chars.len() && chars[i] == '=' {
                    op.push('=');
                    i += 1;
                }
                out.push(Token::Op(op));
            }
            '(' => {
                out.push(Token::LParen);
                i += 1;
            }
            ')' => {
                out.push(Token::RParen);
                i += 1;
            }
            '{' => {
                out.push(Token::LBrace);
                i += 1;
            }
            '}' => {
                out.push(Token::RBrace);
                i += 1;
            }
            ',' => {
                out.push(Token::Comma);
                i += 1;
            }
            ':' => {
                out.push(Token::Colon);
                i += 1;
            }
            '"' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '"' {
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(LexError {
                        message: "string sin cerrar".into(),
                        pos: start,
                    });
                }
                let s: String = chars[start..i].iter().collect();
                out.push(Token::Str(s));
                i += 1; // consume la comilla de cierre
            }
            _ if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num: String = chars[start..i].iter().collect();
                let val: f64 = num.parse().map_err(|_| LexError {
                    message: format!("número inválido: {num}"),
                    pos: start,
                })?;
                out.push(Token::Number(val));
            }
            _ if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let w: String = chars[start..i].iter().collect();
                out.push(Token::Word(w));
            }
            other => {
                return Err(LexError {
                    message: format!("carácter inesperado: {other:?}"),
                    pos: i,
                });
            }
        }
    }

    out.push(Token::Eof);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_basic() {
        let toks = lex(r#"PERCEIVE interaction FROM subject "user:X""#).unwrap();
        assert_eq!(toks[0], Token::Word("PERCEIVE".into()));
        assert_eq!(toks[3], Token::Word("subject".into()));
        assert_eq!(toks[4], Token::Str("user:X".into()));
    }

    #[test]
    fn lex_braces_and_numbers() {
        let toks = lex("{ salience: 0.2 }").unwrap();
        assert_eq!(toks[0], Token::LBrace);
        assert!(toks.contains(&Token::Number(0.2)));
    }

    #[test]
    fn lex_skips_comments() {
        let toks = lex("FADE noise -- olvida el ruido\n").unwrap();
        assert_eq!(toks[0], Token::Word("FADE".into()));
        assert_eq!(toks[1], Token::Word("noise".into()));
        assert_eq!(toks[2], Token::Eof);
    }
}
