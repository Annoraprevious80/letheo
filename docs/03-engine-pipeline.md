# 03 · Flujo de Conciencia (Pipeline del Motor)

Trayecto de un estímulo —p. ej. *"user:Xolotl compró zapatos de correr a las 3 AM"*— a través de
las tres capas del Cognitive Runtime.

```
   estímulo crudo
        │  PERCEIVE
        ▼
┌────────────────────────────────────────────┐
│ CAPA DE PERCEPCIÓN  (corto plazo, volátil)  │  perception.rs
│ · alta resolución, baja vida media          │
│ · weight inicial bajo, halflife horas       │
└───────────────────┬────────────────────────┘
                    │  acumulación de miles de estímulos
                    ▼  DISTILL (ciclo de "sueño", async)
┌────────────────────────────────────────────┐
│ CAPA DE SÍNTESIS  (consolidación)           │  synthesis.rs
│ · centroide + semantic_variance             │
│ · descarta redundancia, retiene dirección   │
│ · produce un Vector de Intención            │
└───────────────────┬────────────────────────┘
                    │  IMPRINT (consistencia a través de ciclos)
                    ▼
┌────────────────────────────────────────────┐
│ CAPA DE ARQUETIPO  (largo plazo)            │  archetype.rs
│ · esencia del sujeto en pocos vectores      │
│ · anclaje de evolución (no inmortal)        │
│ · legible por una IA en una sola mirada     │
└────────────────────────────────────────────┘
                    ▲  EVOKE (resonancia semántica, token budget)
```

- **Percepción**: deliberadamente frágil. Si nada la refuerza, cae bajo `θ_fade` y la barre el GC
  semántico. Es el filtro que impide que el ruido suba.
- **Síntesis ("el sueño")**: corre en ciclos asíncronos sobre Tokio, no en tiempo real. Colapsa N
  percepciones en un Vector de Intención mediante el centroide y `semantic_variance`.
- **Arquetipo**: los Vectores de Intención consistentes a través de ciclos se `IMPRINT`-an. Disuelve
  la barrera de tokens: el arquetipo *ya es* el resumen pre-computado y mantenido vivo por la física.

`EVOKE` no recorre la historia: resuena con los arquetipos y reconstruye la esencia dentro de un
presupuesto de tokens, devolviendo un bloque de contexto ultra-comprimido (no una lista de eventos).
