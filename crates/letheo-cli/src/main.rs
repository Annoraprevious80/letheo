//! Binario `letheo`: REPL interactivo de MQL.
//!
//! Uso:
//!   letheo                      REPL interactivo
//!   letheo --persist ./mem      REPL que autocarga/autoguarda la memoria
//!   letheo --exec "<mql>"       ejecuta un programa y sale (no interactivo)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use letheo_cli::{Eval, RealRepl, HELP};

fn main() {
    let mut persist: Option<PathBuf> = None;
    let mut exec_src: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--persist" | "-p" => persist = args.next().map(PathBuf::from),
            "--exec" | "-e" => exec_src = args.next(),
            "--help" | "-h" => {
                println!("{HELP}");
                return;
            }
            other => {
                eprintln!("argumento desconocido: {other} (prueba --help)");
                std::process::exit(2);
            }
        }
    }

    let mut repl = match RealRepl::real(persist.clone()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no se pudo iniciar (¿LETHEO_MODEL_DIR al modelo all-MiniLM-L6-v2?): {e}");
            std::process::exit(1);
        }
    };

    // Modo no interactivo: ejecuta y sale.
    if let Some(src) = exec_src {
        if let Eval::Output(s) = repl.eval(&src) {
            if !s.is_empty() {
                println!("{s}");
            }
        }
        autosave(&repl);
        return;
    }

    // Modo interactivo.
    println!("Letheo · REPL de MQL — :help para ayuda, :quit para salir");
    if persist.is_some() {
        println!(
            "(persistencia activa: {} arquetipos cargados)",
            repl.eval(":subjects").describe_count()
        );
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("mql> ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl-D)
            Ok(_) => {}
            Err(e) => {
                eprintln!("error de lectura: {e}");
                break;
            }
        }

        match repl.eval(&line) {
            Eval::Quit => break,
            Eval::Output(s) => {
                if !s.is_empty() {
                    println!("{s}");
                }
            }
        }
    }

    autosave(&repl);
    println!("hasta luego.");
}

fn autosave(repl: &RealRepl) {
    if let Some(result) = repl.autosave() {
        match result {
            Ok(n) => eprintln!("💾 memoria guardada ({n} arquetipos)"),
            Err(e) => eprintln!("⚠ no se pudo autoguardar: {e}"),
        }
    }
}

/// Pequeña ayuda para el banner: cuenta líneas no vacías de una salida.
trait DescribeCount {
    fn describe_count(self) -> usize;
}
impl DescribeCount for Eval {
    fn describe_count(self) -> usize {
        match self {
            Eval::Output(s) if s.starts_with('(') => 0, // "(sin arquetipos...)"
            Eval::Output(s) => s.lines().filter(|l| !l.is_empty()).count(),
            Eval::Quit => 0,
        }
    }
}
