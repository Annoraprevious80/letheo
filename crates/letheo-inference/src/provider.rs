//! Abstracción de provider de inferencia — el "motor de pensamiento" desacoplado.
//!
//! Local-first: la inferencia ocurre in-process, sin red. El trait permite intercambiar el motor
//! (Mock determinista, Candle local) sin alterar la lógica del runtime.

/// Dimensión de embedding (all-MiniLM-L6-v2).
pub const EMBED_DIM: usize = 384;

/// Un motor de inferencia capaz de producir embeddings semánticos y resúmenes.
pub trait Provider {
    /// Dimensión de los embeddings que produce.
    fn dim(&self) -> usize;

    /// Convierte un texto crudo en un embedding denso normalizable.
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Resume/expresa en prosa un conjunto de rasgos (para el contexto que consume el LLM).
    /// El default concatena los fragmentos; un provider semántico (llama.cpp) puede sobreescribirlo.
    fn summarize(&self, fragments: &[&str]) -> String {
        fragments.join(" · ")
    }
}
