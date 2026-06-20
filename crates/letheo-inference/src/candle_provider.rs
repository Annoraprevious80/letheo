//! `CandleProvider` — inferencia local real con `all-MiniLM-L6-v2` (BERT), 384-dim.
//!
//! Coherente con "local-first": el modelo vive **en disco** y se ejecuta in-process, sin red ni
//! servicio externo. El descargador (huggingface_hub de Python o cualquier `git lfs`) se ejecuta
//! una vez fuera de banda; ver `sandbox/fetch_model.py`. Esto evita acoplar el runtime a un cliente
//! HTTP concreto.
//!
//! Solo se compila con `--features candle`.

use crate::provider::Provider;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Identificador del modelo (para documentación / descarga).
pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Variable de entorno que apunta al directorio del modelo en disco.
pub const MODEL_DIR_ENV: &str = "LETHEO_MODEL_DIR";

/// Provider de embeddings local basado en BERT (Candle). 384 dimensiones.
pub struct CandleProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    // El forward de BERT comparte tensores; serializamos el acceso para ser Sync-safe.
    lock: Mutex<()>,
}

impl CandleProvider {
    /// Carga el modelo desde el directorio indicado por `LETHEO_MODEL_DIR`.
    ///
    /// El directorio debe contener `config.json`, `tokenizer.json` y `model.safetensors`
    /// (ejecuta `python sandbox/fetch_model.py` una vez para poblarlo).
    pub fn load() -> Result<Self> {
        let dir = std::env::var(MODEL_DIR_ENV).map_err(|_| {
            anyhow::anyhow!(
                "define {MODEL_DIR_ENV} apuntando al directorio del modelo \
                 (corre `python sandbox/fetch_model.py` para descargarlo)"
            )
        })?;
        Self::from_dir(dir)
    }

    /// Carga `all-MiniLM-L6-v2` desde un directorio local. CPU por defecto.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let device = Device::Cpu;

        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");
        for p in [&config_path, &tokenizer_path, &weights_path] {
            anyhow::ensure!(
                p.exists(),
                "falta {} en el directorio del modelo",
                p.display()
            );
        }

        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .context("parsing config.json")?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            lock: Mutex::new(()),
        })
    }

    /// Embedding crudo (Result) — mean-pooling sobre tokens + normalización L2.
    fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
        let _guard = self.lock.lock().unwrap();

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        let n = ids.len();

        let token_ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::ones((1, n), DType::U8, &self.device)?;

        // forward → (1, seq_len, hidden=384)
        let out = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling sobre la dimensión de tokens → (384,)
        let (_b, seq, _h) = out.dims3()?;
        let pooled = (out.sum(1)? / seq as f64)?.squeeze(0)?;

        // Normalización L2.
        let norm = pooled.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let v: Vec<f32> = pooled.to_vec1()?;
        Ok(if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        })
    }
}

impl Provider for CandleProvider {
    fn dim(&self) -> usize {
        384
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // VERDAD 100% (deuda #7): un fallo de inferencia con el modelo YA cargado es excepcional y
        // **se falla ruidoso** — nunca se devuelve un embedding falso silencioso (un vector cero
        // contaminaría centroides y resonancias sin que nadie se entere). Los errores de *carga* se
        // exponen antes, vía `load()`/`from_dir()`.
        self.embed_inner(text).unwrap_or_else(|e| {
            panic!("CandleProvider: fallo de inferencia (modelo ya cargado): {e:#}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requiere el modelo en disco. Poblar con `python sandbox/fetch_model.py` y exportar
    // LETHEO_MODEL_DIR. Ignorado por defecto en CI sin modelo.
    #[test]
    #[ignore = "requiere LETHEO_MODEL_DIR con all-MiniLM-L6-v2; correr con --ignored"]
    fn loads_and_embeds_384() {
        let p = CandleProvider::load().expect("carga del modelo");
        let a = p.embed("running shoes at night");
        let b = p.embed("sneakers for nocturnal jogging");
        let c = p.embed("mortgage insurance bank loan");
        assert_eq!(a.len(), 384);

        let cos = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(i, j)| i * j).sum::<f32>();
        assert!(
            cos(&a, &b) > cos(&a, &c),
            "frases afines deben resonar más que las dispares"
        );
    }
}
