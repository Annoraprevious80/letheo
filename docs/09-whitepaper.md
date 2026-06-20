# Letheo — un Cognitive Runtime de memoria a coste constante

> Whitepaper técnico del **motor**. No es un folleto ni un benchmark comparativo: describe qué es el
> motor, su física y su arquitectura. Las fuentes de verdad son el código (`crates/`) y los docs
> [01-physics](01-physics.md), [02-mql-grammar](02-mql-grammar.ebnf), [03-engine-pipeline](03-engine-pipeline.md),
> [04-architecture](04-architecture.md).

---

## 1. Tesis

**Memoria = compresión con olvido, indexada y consultable bajo una sola física.** Un motor de memoria
no "guarda y consulta": **percibe**, **comprime en modos**, **olvida lo redundante** y **recupera por
resonancia ponderada por vida**. Cuando el historial de un sujeto crece, las memorias ingenuas fallan a
presupuesto de tokens fijo: o meten todo el pasado (coste sin techo) o solo ven el presente inmediato
(ciegas a la trayectoria). Letheo destila el comportamiento en una estructura de **tamaño fijo** cuyo
coste de lectura es **constante** sea el historial de 4.000 o 1.000.000 de eventos.

## 2. El tiempo como coeficiente de entropía

El peso de cada recuerdo decae por física, no por política:

```
weight(t) = salience · e^(−λ · Δt) · (1 + reinforcement)        λ = ln2 / halflife
```

Δt se mide desde la última evocación/refuerzo (recordar resetea Δt → permanencia ganada). El peso se
evalúa **perezosamente** (lazy): solo en `DISTILL`, `EVOKE` o durante el barrido del GC semántico —
nunca por tic de reloj. Esto permite millones de recuerdos sin recalcular `e^x` por tic. El **olvido
estratégico es una feature**: el ruido redundante decae y solo el patrón sobrevive.

## 3. Las dos capas bajo una sola física (Complementary Learning Systems)

La biología separa hipocampo (episódico, rápido, específico) y neocórtex (semántico, lento, general).
Letheo lo hace literal con **una sola** `EntropyTrace` gobernando ambas representaciones:

- **Capa-2 · semántica** (`archetype` + `modes`): la identidad y la **trayectoria** del sujeto. El
  comportamiento se descompone en **modos** coherentes (clustering determinista leader/DP-means), cada
  uno con su propia física de olvido. Un único centroide colapsaría comportamientos dispares en una
  media que no representa a ninguno; los modos los mantienen nítidos. Generaliza, comprime, O(1).
- **Capa-1 · episódica** (`factstore`): hechos **verbatim** con embedding, deduplicación semántica por
  sujeto y olvido. Responde lo nominal exacto que la capa-2 (una media) nunca podría guardar. Evocar un
  hecho lo refuerza (spaced repetition); uno que nunca se evoca se desvanece.

## 4. Los verbos (MQL)

El vocabulario es biológico, no SQL:

| Verbo | Función |
|-------|---------|
| `PERCEIVE` | Asimila un estímulo crudo en memoria volátil de corto plazo. Nace decayendo. |
| `DISTILL`  | El "sueño": colapsa N percepciones en un *Vector de Intención* + sus modos. |
| `EVOKE`    | Recupera por resonancia semántica dentro de un *token budget*. |
| `FADE`     | Olvido modulado por entropía; preserva la contribución ya hecha al arquetipo. |
| `IMPRINT`  | Consolida un arquetipo resistente al olvido (anclaje de evolución). |

## 5. EVOKE unificado

Una sola evocación reparte **un** presupuesto de tokens entre las dos capas: el **gist** caracterológico
(capa-2) y los **hechos exactos** top-k (capa-1), elegidos por score físico (`relevancia · weight(now)`)
con un knapsack greedy. Así una pregunta se responde en **carácter Y nominal** sin coser dos sistemas a
mano. El coste de tokens no se inventa: el de los hechos se **mide** con el tokenizer real inyectado, y
el del gist es un coste por vector **declarado y realimentable** desde tiktoken (ver
[05-honest-assessment](05-honest-assessment.md), deuda #1 saldada).

## 6. Recuperación física

`resonate` no ordena por coseno crudo: rankea por `score = max(0, relevancia) · weight(now)`, donde
`weight` integra recencia (decay), importancia (salience) y refuerzo. Una memoria muy relevante pero
desvanecida queda por debajo de otra igual de relevante pero viva — sin coeficientes α/β/γ a mano. Es
la forma multiplicativa del retrieval de *Generative Agents*, nativa de la física del motor.

## 7. Arquitectura

```
crates/letheo-core      física del olvido · percepción · síntesis multi-modal · arquetipo · factstore · evoke
crates/letheo-inference trait Provider + CandleProvider (all-MiniLM-L6-v2, local)
crates/letheo-mql/exec  lexer + parser de los verbos → AST → ejecutor
crates/letheo-async     runtime actor (Tokio), no bloqueante
crates/letheo-persist   persistencia JSON (arquetipos + hechos), round-trip sin pérdida
crates/letheo-calibration  calibración de umbrales contra ground-truth sintético
bindings/letheo-py      PyO3 → SDK Python (orchestration: Session, prosa, tiktoken)
```

## 8. Estado y dirección

El motor está bajo el invariante **VERDAD 100%**: nada en el camino de producto es mock, hardcode,
proxy o no-op (libro mayor de deudas en [05-honest-assessment](05-honest-assessment.md)). El roadmap
(ver [`ROADMAP.md`](../ROADMAP.md)) lo lleva más allá —arquetipo multi-modal, índice
ANN, storage embebido, memoria generativa— para que lo consuma una flota de super-agentes (Paideia)
como su substrato de memoria. El progreso se mide por **corrección del motor** (tests deterministas),
no por comparación con baselines.

---

*El núcleo, en una frase: un motor que destila la **evolución del comportamiento** y los **hechos
exactos** a **coste constante** sobre historiales ilimitados, bajo una sola física de olvido.*
