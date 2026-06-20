//! # letheo-inference · Kernel de inferencia local-first
//!
//! Abstracción [`Provider`] desacoplada del runtime. Providers:
//! - `CandleProvider` (feature `candle`) — `all-MiniLM-L6-v2` local, 384-dim. **Provider de producto.**
//! - `MockProvider` (feature `testing`, o bajo `cfg(test)`) — embeddings deterministas, sin modelo.
//!   Es un **doble de tests**, NO se compila en ningún binario de producto.

pub mod caching_provider;
pub mod provider;

// MockProvider solo existe para tests: bajo `cfg(test)` del propio crate, o vía la feature `testing`
// que activan las dev-dependencies de otros crates. Nunca entra en un build de producto.
#[cfg(any(test, feature = "testing"))]
pub mod mock_provider;
#[cfg(any(test, feature = "testing"))]
pub use mock_provider::MockProvider;

pub use caching_provider::{CacheStats, CachingProvider};
pub use provider::{Provider, EMBED_DIM};

#[cfg(feature = "candle")]
pub mod candle_provider;
#[cfg(feature = "candle")]
pub use candle_provider::CandleProvider;
