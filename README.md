# Letheo — Cognitive Runtime de memoria para agentes

> **Letheo no es una base de datos. Es un *Cognitive Runtime*** — un organismo que respira
> (procesa / comprime) y olvida. No "almacena y consulta"; **percibe, sueña, evoca y desvanece**.

Cuando el historial de un agente crece, las memorias ingenuas fallan a presupuesto de tokens fijo:
meter todo el pasado al prompt (coste sin techo), resumir con un LLM en cada paso (coste **O(N)**), o
RAG — que recupera hechos puntuales pero es **ciego al tiempo**: no sabe que algo *cambió*. Letheo
destila el comportamiento en una estructura de **tamaño fijo**, leíble a **coste constante**, sea el
historial de 4.000 o 1.000.000 de eventos.

El **olvido estratégico es una feature**, no un bug: el peso de cada recuerdo decae por física
(entropía temporal) y solo el patrón sobrevive. El destino del motor es ser la **memoria de una flota
de super-agentes**: una sola física de decaimiento sobre **dos capas** — episódica (hechos exactos,
hipocampo) y semántica (identidad/trayectoria, neocórtex).

## Los verbos (MQL — *Mnemonic Query Language*)

No hay `SELECT / INSERT / UPDATE / DELETE`. El vocabulario es biológico:

| Verbo | Función |
|-------|---------|
| `PERCEIVE` | Asimila un estímulo crudo en memoria volátil de corto plazo. Nace decayendo. |
| `DISTILL`  | El "sueño": colapsa N percepciones en un *Vector de Intención* + sus **modos** (compresión multi-modal). |
| `EVOKE`    | Recuerda por **resonancia semántica** dentro de un *token budget*; `RESONATING WITH` enfoca un rasgo. |
| `FADE`     | Olvido estratégico modulado por entropía; preserva la contribución ya hecha al arquetipo. |
| `IMPRINT`  | Consolida/ancla un arquetipo resistente al olvido. |
| `RECALL`   | Capa-1: recuperación dirigida de **hechos exactos** (verbatim), read-only. |
| `REINFORCE`| Capa-1: spaced-repetition — recuerda y resetea el decay de un hecho. |

## El tiempo como coeficiente de entropía

El tiempo no es un timestamp; es un operador pasivo sobre el peso de cada recuerdo:

```
weight(t) = salience · e^(−λ · Δt) · (1 + reinforcement)        λ = ln2 / halflife
```

Δt se mide desde la **última evocación/refuerzo** (recordar resetea Δt → permanencia ganada). El peso
se evalúa **perezosamente** (lazy): solo en `DISTILL`, `EVOKE` o durante el barrido del GC semántico —
nunca por tic de reloj. El refuerzo tiene **rendimientos decrecientes** y la vida media un **suelo**:
nada se vuelve inmortal por mucho que se reviste.

## Las dos capas (Complementary Learning Systems)

Una sola física (`EntropyTrace`) gobierna las dos representaciones de la memoria:

- **Capa-2 · semántica** (`archetype` + `modes`): la identidad y la **trayectoria** del sujeto,
  descompuesta en **modos** de comportamiento (no una media ciega). Cada modo tiene su propia física de
  olvido **y su propio drift** (cuánto ha cambiado ese comportamiento desde que nació). Comprime, O(1).
- **Capa-1 · episódica** (`factstore`): hechos **verbatim** con embedding, dedup semántico y olvido.
  Responde lo nominal exacto que la capa-2 nunca guardaría.

`EVOKE` **unificado** responde **carácter Y nominal** en una sola evocación, repartiendo un único
presupuesto de tokens entre ambas capas.

## Uso (Python)

```python
from letheo_orchestration import Session

s = Session()

# Capa-2: percibe y "sueña" → la esencia (identidad + trayectoria, a coste fijo)
for _ in range(20):
    s.perceive("user:ada", act="reads sci-fi novels at night")
s.breathe()

# Capa-1: un hecho exacto, verbatim
s.remember("user:ada", "allergic to penicillin")

# Una sola evocación responde carácter (gist) Y nominal (hechos)
ctx = s.evoke_unified("user:ada", "what does ada read?")
print(s.recall("user:ada", "allergies", k=1))     # [('allergic to penicillin', ...)]

# Memoria generativa: insights del arco (transiciones, revivals)
print(s.reflect("user:ada"))

# Búsqueda por similitud entre sujetos (ANN a escala): enruta al más relevante
print(s.resonate("space opera fan", k=3))
```

…o el mismo motor como **MQL**:

```
PERCEIVE interaction FROM subject "user:ada" AS { act: reads, genre: scifi }
DISTILL  subject "user:ada" INTO intention_vector COMPRESSING BY semantic_variance
EVOKE    essence OF "user:ada" RESONATING WITH { nostalgia } WITHIN budget 800 tokens
RECALL   facts FROM subject "user:ada" RESONATING WITH { allergy } WHERE resonates > 0.6 WITHIN k 3
```

## Arquitectura

- **`crates/letheo-core`** (Rust): física del olvido, percepción, síntesis multi-modal, arquetipos, factstore, evoke unificado, reflexión, runtime.
- **`crates/letheo-inference`** (Rust): trait `Provider` + `CandleProvider` (`all-MiniLM-L6-v2`, local).
- **`crates/letheo-mql`** + **`crates/letheo-exec`** (Rust): lexer + parser de los verbos → AST → ejecutor.
- **`crates/letheo-index`** (Rust): índice ANN (HNSW) + `Retriever` Flat/HNSW con filtrado por vida.
- **`crates/letheo-{async,persist,calibration,cli}`** (Rust): runtime actor Tokio, persistencia (JSON + store embebido `redb`), calibración de umbrales, REPL MQL.
- **`bindings/letheo-py`** (PyO3) + **`orchestration/`** (Python): SDK de alto nivel (`Session`, prosa, tiktoken).

```
crates/ + bindings/   →  MOTOR (Rust)            percibe · sueña · evoca · olvida
orchestration/        →  SDK Python (Session)    capa consumidora del binding
```

## Instalación

```bash
# 1) Motor (offline, hermético) — sin red, sin modelo:
cargo test --workspace

# 2) Binding Python (requiere maturin + el modelo local en .models/):
maturin develop -m bindings/letheo-py/Cargo.toml --features candle
```

El `CandleProvider` carga `all-MiniLM-L6-v2` **desde disco** (local-first; no lo descarga en runtime).
Colócalo una vez y apunta `LETHEO_MODEL_DIR` ahí:

```bash
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 .models/all-MiniLM-L6-v2
export LETHEO_MODEL_DIR="$PWD/.models/all-MiniLM-L6-v2"
```

Candle lee la config, el tokenizer y los pesos en **safetensors**. El workspace de Rust
(`cargo test --workspace`) es **hermético**: no necesita el modelo — solo el binding Python lo requiere.

Ver [`docs/`](docs/) para la física, la gramática EBNF y el pipeline; el **porqué** del proyecto en
[`docs/10-thesis-agents-need-memory.md`](docs/10-thesis-agents-need-memory.md); y [`ROADMAP.md`](ROADMAP.md)
para el estado y lo que sigue.

## Estado

Motor (Rust) maduro y testeado offline: **`cargo test --workspace` → 144 passed, 0 failed, 2 ignored,
0 warnings**. Arquetipo multi-modal con trayectoria por-modo, retrieval físico, bicapa episódica
unificada, índice ANN a escala, memoria generativa, persistencia transaccional — bajo el invariante
**VERDAD 100%** (cero mock/fake/hardcode en el camino de producto; auditoría en
[`docs/05-honest-assessment.md`](docs/05-honest-assessment.md)).
