//! `MockProvider` — embeddings deterministas sin modelo.
//!
//! Permite validar la física y el parser (Fases 1–2) en CI offline. El mismo texto siempre produce
//! el mismo vector; textos parecidos producen vectores parecidos (hashing de tokens en buckets).

use crate::provider::{Provider, EMBED_DIM};

/// Provider determinista basado en hashing de tokens. Sin dependencias, sin red.
#[derive(Debug, Clone)]
pub struct MockProvider {
    dim: usize,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self { dim: EMBED_DIM }
    }
}

impl MockProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Permite una dimensión pequeña en tests.
    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}

/// Hash FNV-1a de 64 bits — determinista y sin dependencias.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Provider for MockProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        // Bag-of-tokens: cada token incrementa un bucket determinista. Textos similares → vectores
        // similares (comparten tokens), suficiente para probar resonancia y centroides.
        for token in text.split_whitespace() {
            let h = fnv1a(&token.to_lowercase());
            let bucket = (h % self.dim as u64) as usize;
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[bucket] += sign;
        }
        // Normaliza a vector unitario (estabiliza el coseno).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let p = MockProvider::new();
        assert_eq!(p.embed("zapatos de correr"), p.embed("zapatos de correr"));
    }

    #[test]
    fn similar_texts_more_similar_than_different() {
        let p = MockProvider::with_dim(64);
        let a = p.embed("zapatos de correr nocturno");
        let b = p.embed("zapatos de correr matutino");
        let c = p.embed("seguro de vida hipoteca banco");
        let cos = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(i, j)| i * j).sum::<f32>();
        assert!(cos(&a, &b) > cos(&a, &c), "textos afines resuenan más");
    }

    #[test]
    fn correct_dim() {
        assert_eq!(MockProvider::new().embed("x").len(), EMBED_DIM);
    }
}
