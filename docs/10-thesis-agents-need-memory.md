# Agents don't need a database. They need memory — and a vault they can't corrupt.

> The thesis behind Letheo (and its sibling, the deterministic vault). Opinionated, and honest
> about its own limits.

## The interface mismatch

We are handing autonomous agents a 1974 interface. SQL — and the CRUD it underpins — was designed
for a *human or an application that already knows exactly which rows it wants*: `SELECT * FROM orders
WHERE id = 42`. It is declarative, relational, brilliant, and fifty years entrenched. It is also the
wrong primitive for an agent.

An agent does not think in rows. It operates on **meaning**, **relevance**, **recency**, and —
critically — **forgetting**. "What do I know about this user?" "What's relevant *now*?" "What can I
let go?" CRUD has no word for any of that. You can bolt a vector index onto a database and call it
"agent memory," but similarity search is *stateless retrieval* — it doesn't decay, it doesn't
reinforce what you used, it doesn't consolidate, it never forgets. It's a better `WHERE`, not a memory.

## Two failure modes of "just give the agent a database"

1. **Context rot.** Without a physics of forgetting, every interaction is kept at full weight forever.
   The store fills with noise; retrieval drowns the signal; the context window saturates with stale
   facts. Human memory solved this by *decaying* everything and consolidating only the patterns that
   recur. An agent's memory should too.

2. **Corruption.** The tempting shortcut is to let the LLM run migrations and write rows directly.
   That is the fastest path to a dropped table or a double-charge. A model that *hallucinates a
   mutation* is not a bug you can prompt away — it's a category error. The write path must not be
   something a probabilistic system owns.

## The two primitives agents actually need

**Memory — a language of `perceive · distill · evoke · forget`.** Not rows; *traces*. Each memory
carries an entropy trace and a weight that decays over time:

```
weight(t) = salience · e^(−λ·Δt) · (1 + reinforcement),   λ = ln2 / halflife
```

Evaluated lazily — never per clock tick — so millions of memories cost nothing between touches.
Recalling a memory *reinforces* it (spaced repetition: what you use survives, what you don't fades).
A sweep forgets what fell below threshold. And one *evoke* answers both classes of question under a
single token budget: the exact verbatim fact (episodic) **and** the distilled character (semantic),
the way Complementary Learning Systems split hippocampus and neocortex. **Forgetting is a feature,
not a bug.** That's Letheo's MQL — a *Mnemonic* Query Language.

**Determinism — a vault the LLM builds but never owns.** The model should reason and *author* at
build-time, and read at query-time, but every write must be deterministic, atomic, and appended to an
immutable, hash-chained ledger. *The LLM proposes; the vault disposes.* An AI can design your backend
and still be unable to corrupt it.

## What this is *not*

This is not "kill SQL." Two honest caveats:

- **Agents still need exact facts.** "The API key is `sk-…`", "the order shipped", "she's allergic to
  peanuts" — nominal truths a semantic gist would average away. The answer is **both layers**, not a
  replacement: verbatim episodic facts *and* a generalized gist, under one physics. A memory engine
  that can't return a fact verbatim is useless; one that can only return facts is just a database.

- **MQL sits *above* a deterministic store, it doesn't replace it.** Underneath the memory language is
  a SQL-grade, ACID vault. The contribution isn't a new database — it's the recognition that *the
  agent-facing interface is memory*, and that the write path beneath it must be deterministic. Frame
  it as a layer, not a coup.

## Why bother

Because the agent era inverts the assumption every datastore was built on. Supabase, Firebase, Neon,
Convex — all assume a *human* cures the schema and an app makes precise queries. The moment the
operator is an agent, you want the opposite: an interface that speaks meaning and forgetting, on a
substrate the operator cannot break. Pieces of this exist in isolation — memory libraries (Mem0, Zep,
Letta), verifiable/versioned databases (immudb, Dolt), deterministic workflow engines (Temporal). The
*combination* — a memory that forgets, on a vault an AI can build but never corrupt — does not exist
as one thing yet.

## Honest about the limits

This is early. The reference vault is single-box, SQLite-per-tenant, measured in megabytes today — it
has not been proven at volume, and horizontal scale is designed but unbuilt. A new query vocabulary
faces brutal adoption inertia; the "SQL killer" graveyard is large, and the only way to win is to be a
*complement* the agent era genuinely needs, not a replacement nobody asked for. The bet is that the
era is here.

**Letheo** is the first piece, open source: a cognitive runtime that perceives, distills, evokes — and
forgets. The memory an agent actually needs. The vault comes next.
