//! Caché de embeddings (épica 10.2): un decorador sobre cualquier [`Provider`].
//!
//! Embeber el mismo texto dos veces es desperdicio — con un modelo real (Candle/llama.cpp) es la
//! operación más cara del pipeline. `CachingProvider` memoiza `embed(text)` por el texto exacto, de
//! modo que estímulos repetidos (hábitos: el mismo `act/object` mil veces) se embeben una sola vez.
//!
//! Usa la clave de texto completa (no solo un hash) → cero colisiones. La interior mutability vía
//! `Mutex` lo mantiene `Sync`, apto tanto para el ejecutor síncrono como para el actor async.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::provider::Provider;

/// Envuelve un `Provider` y cachea sus embeddings por texto.
pub struct CachingProvider<P: Provider> {
    inner: P,
    cache: Mutex<HashMap<String, Vec<f32>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Estadísticas de la caché (observabilidad).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
}

impl CacheStats {
    /// Tasa de acierto en `[0, 1]`. 1.0 si no hubo consultas todavía.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl<P: Provider> CachingProvider<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// El provider envuelto.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            entries: self.cache.lock().unwrap().len(),
        }
    }

    /// Vacía la caché (no resetea los contadores acumulados).
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}

impl<P: Provider> Provider for CachingProvider<P> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // Camino caliente: ¿ya está cacheado?
        if let Some(v) = self.cache.lock().unwrap().get(text) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return v.clone();
        }
        // Fallo: computamos fuera del lock (el embedding real puede ser lento) y guardamos.
        self.misses.fetch_add(1, Ordering::Relaxed);
        let v = self.inner.embed(text);
        self.cache
            .lock()
            .unwrap()
            .insert(text.to_string(), v.clone());
        v
    }

    fn summarize(&self, fragments: &[&str]) -> String {
        self.inner.summarize(fragments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Provider que cuenta cuántas veces se le pidió embeber realmente.
    struct CountingProvider {
        calls: AtomicUsize,
    }

    impl CountingProvider {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl Provider for CountingProvider {
        fn dim(&self) -> usize {
            2
        }
        fn embed(&self, text: &str) -> Vec<f32> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            // Embedding determinista trivial: longitud y primer byte.
            vec![text.len() as f32, text.bytes().next().unwrap_or(0) as f32]
        }
    }

    #[test]
    fn repeated_text_is_embedded_once() {
        let p = CachingProvider::new(CountingProvider::new());
        let a = p.embed("purchase shoes");
        let b = p.embed("purchase shoes");
        let c = p.embed("purchase shoes");
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(
            p.inner().calls(),
            1,
            "solo se embebió una vez pese a 3 consultas"
        );

        let s = p.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 2);
        assert_eq!(s.entries, 1);
    }

    #[test]
    fn distinct_texts_are_separate_entries() {
        let p = CachingProvider::new(CountingProvider::new());
        let x = p.embed("alpha");
        let y = p.embed("beta");
        assert_ne!(x, y);
        assert_eq!(p.inner().calls(), 2);
        assert_eq!(p.stats().entries, 2);
    }

    #[test]
    fn hit_rate_reflects_usage() {
        let p = CachingProvider::new(CountingProvider::new());
        for _ in 0..9 {
            p.embed("same");
        }
        p.embed("other");
        // 10 consultas: 2 misses (same, other) + 8 hits.
        let s = p.stats();
        assert_eq!(s.hits, 8);
        assert_eq!(s.misses, 2);
        assert!((s.hit_rate() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn clear_empties_cache_but_keeps_counters() {
        let p = CachingProvider::new(CountingProvider::new());
        p.embed("x");
        p.clear();
        p.embed("x"); // recomputado tras limpiar
        assert_eq!(p.inner().calls(), 2);
        assert_eq!(p.stats().misses, 2);
        assert_eq!(p.stats().entries, 1);
    }

    #[test]
    fn dim_delegates_to_inner() {
        let p = CachingProvider::new(CountingProvider::new());
        assert_eq!(p.dim(), 2);
    }
}
