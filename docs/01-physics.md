# 01 · Las Leyes de la Física (La Matemática del Olvido)

El éxito de Mnemosyne **no** depende de la velocidad de Rust, sino de la precisión de esta
matemática. Este documento es la fuente de verdad para `crates/letheo-core/src/entropy.rs` y
`synthesis.rs`.

## 1. La ecuación del olvido

```
weight(t) = salience · e^(−λ · Δt) · (1 + reinforcement)
```

| Símbolo | Significado | Rango |
|---------|-------------|-------|
| `salience` | Carga inicial del estímulo al ser percibido | `(0, 1]` típico |
| `λ` (lambda) | Tasa de decaimiento, derivada de la vida media | `λ = ln2 / halflife` |
| `Δt` | Tiempo desde el **último refuerzo/evocación** (NO desde la creación) | `≥ 0` |
| `reinforcement` | Refuerzos acumulados (recordar/repetir suma) | `≥ 0` |

### Propiedades exigidas
1. **Monótona decreciente en Δt** (a reinforcement fijo): a más tiempo sin tocar, menos peso.
2. **Asintótica a 0**: `e^(−λ·Δt) → 0` pero nunca alcanza 0. El recuerdo se vuelve candidato a
   `FADE` cuando cruza un umbral, no cuando "expira".
3. **Refuerzo = permanencia ganada**: cada `EVOKE`/repetición (a) resetea `Δt → 0`
   (`last_touch_t = now`) y (b) incrementa `reinforcement`, **y opcionalmente reduce λ**
   (la vida media crece con la repetición, como en la consolidación sináptica).

## 2. Lazy evaluation (corrección crítica de rendimiento)

Calcular `e^x` por cada vector en cada tic de un reloj asíncrono es prohibitivo a escala. **El
runtime NO recalcula pesos por tic.** Cada recuerdo persiste solo sus parámetros
(`salience, lambda, reinforcement, last_touch_t`). `weight(now)` es una **función pura** evaluada
bajo demanda, solo en tres momentos:

- (a) durante un ciclo **`DISTILL`** (el "sueño"),
- (b) durante un **`EVOKE`**,
- (c) durante el barrido del **garbage collector semántico** que busca candidatos a `FADE`.

El bucle Tokio **agenda barridos**, no aritmética por-tic.

## 3. Umbral de FADE

`FADE` se dispara cuando `weight(now) < θ_fade`. El evento se desvanece **solo después** de que su
contribución fue absorbida por un `DISTILL` previo (su "voto" ya vive en el centroide/arquetipo).
Se olvida el ladrillo, no la casa.

Valor inicial sugerido: `θ_fade = 0.05` (calibrable por dominio en el sandbox de Fase 4).

## 4. Curvas de entropía por dominio

La vida media no es global; se modula por el carácter del estímulo:

| Condición | Efecto sobre el decaimiento |
|-----------|------------------------------|
| `novelty` alta | decae lento (lo sorprendente perdura) → halflife ↑ |
| `repetition` alta | decae rápido y luego `IMPRINT` (se vuelve hábito) |
| `emotional_charge` alta | `halflife × N` (lo intenso resiste) |

## 5. `semantic_variance` (definición concreta, no caja negra)

Durante `DISTILL` el motor toma N vectores y calcula su **centroide** `c = mean(v_i)`. Para cada
evento mide la **similitud del coseno** `sim(v_i, c)`:

```
sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

- **Se DESVANECE (FADE)**: `sim(v_i, c) ≥ θ_redundancia` → ruido redundante/predecible.
- **Se RETIENE (IMPRINT)**: el **centroide** `c` (la nueva dirección del usuario) **y** los
  **outliers** con `sim(v_i, c) ≤ θ_anomalía` → *novelty* / cambio brusco de comportamiento.

Umbrales iniciales sugeridos: `θ_redundancia = 0.92`, `θ_anomalía = 0.30` (calibrar en Fase 4).
