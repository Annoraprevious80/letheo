# 04 · Arquitectura (Rust Core + PyO3 + Inferencia Local)

## Capas

```
┌──────────────────────────────────────────────────────────┐
│ Orchestration (Python)  orchestration/letheo/             │
│  · API MQL de alto nivel · agentes · integración LLM      │
└───────────────────────────┬──────────────────────────────┘
                            │ PyO3 / FFI  (bindings/letheo-py)
┌───────────────────────────▼──────────────────────────────┐
│ Cognitive Runtime (Rust)                                  │
│  letheo-mql      lexer + parser → AST                     │
│  letheo-core     entropy · perception · synthesis ·       │
│                  archetype · evoke · runtime (Tokio)      │
│  letheo-inference  trait Provider                         │
│      ├─ MockProvider   (determinista, sin modelo, CI)     │
│      └─ CandleProvider (all-MiniLM-L6-v2, 384-dim, local) │
└──────────────────────────────────────────────────────────┘
```

## Decisiones

- **Rust sobre C++**: PyO3 (FFI), Candle (inferencia local Rust-native) y Tokio (bucle async que
  "respira") forman un stack coherente y memory-safe sin runtime de GC.
- **Lazy evaluation** de pesos (ver `01-physics.md` §2): el runtime agenda barridos, no aritmética
  por-tic.
- **Inferencia local-first**: in-process, latencia cero, sin dependencias de red. Modelo del MVP:
  `all-MiniLM-L6-v2` (~90MB, 384 dimensiones). `Provider` desacopla el "motor de pensamiento".
- **Índice vectorial**: búsqueda lineal **Flat (coseno) con SIMD** en el MVP. Un arquetipo son
  decenas de Vectores de Intención, no millones → lineal es más rápido y predecible que HNSW, que se
  difiere a v2.0.
- **MockProvider primero**: las Fases 1–2 validan física y parser con vectores deterministas, sin
  depender de un modelo. Candle entra en Fase 3.

## Verificación

```bash
cargo test --workspace                          # física, parser, síntesis (Mock), offline

# Inferencia local real (all-MiniLM-L6-v2). El modelo vive en disco (local-first):
python sandbox/fetch_model.py                   # descarga una vez (único paso con red)
export LETHEO_MODEL_DIR=$(pwd)/.models/all-MiniLM-L6-v2
cargo test -p letheo-inference --features candle -- --ignored   # carga BERT y embebe a 384-dim

# Bindings Python (PyO3):
python -m venv .venv && source .venv/Scripts/activate && pip install maturin pytest
cd bindings/letheo-py && maturin develop --release
pytest tests/                                   # smoke del runtime desde Python
```

> **Nota de descarga**: el runtime no incrusta un cliente HTTP. `CandleProvider::from_dir` carga el
> modelo desde un directorio local; `sandbox/fetch_model.py` (huggingface_hub) lo puebla una sola
> vez. Esto mantiene el core desacoplado de cualquier descargador y 100% offline en ejecución.
