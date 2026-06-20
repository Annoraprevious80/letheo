//! # letheo-persist · Persistencia local-first de la memoria de largo plazo
//!
//! La persistencia de Letheo: hace que la esencia destilada **sobreviva al reinicio del proceso**. Guarda
//! un *snapshot por sujeto* (un archivo JSON por arquetipo) en un directorio, y lo rehidrata.
//!
//! Decisiones (ver `ROADMAP.md`, D28–D30):
//! - **Crate aparte con serde**, no en `letheo-core`: el core sigue sin dependencias de serialización
//!   (coherente con D6). Aquí viven DTOs espejo + conversiones desde/hacia los tipos del core.
//! - **Un archivo por sujeto** (`{sujeto-saneado}-{hash}.json`): snapshots independientes, fáciles
//!   de inspeccionar, versionar y migrar; el nombre lleva un hash para evitar colisiones de saneo.
//! - **JSON legible**: la memoria de un agente debe poder auditarse a mano (no es un blob opaco).

use std::fs;
use std::io;
use std::path::Path;

use letheo_core::archetype::ArcMilestone;
use letheo_core::entropy::EntropyTrace;
use letheo_core::factstore::{Fact, FactStore};
use letheo_core::modes::Mode;
use letheo_core::{Archetype, ArchetypeStore};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

/// Versión del formato en disco. Permite migraciones futuras sin romper snapshots viejos.
pub const SNAPSHOT_VERSION: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// DTOs espejo (serde) — desacoplan el formato en disco de los tipos del runtime.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct TraceDto {
    salience: f64,
    lambda: f64,
    reinforcement: f64,
    last_touch: f64,
}

#[derive(Serialize, Deserialize)]
struct MilestoneDto {
    at: f64,
    direction: Vec<f32>,
    absorbed: usize,
    #[serde(default)]
    label: String,
    #[serde(default)]
    label_histogram: Vec<(String, usize)>,
}

/// DTO de un modo del arquetipo (multi-modal). Su física (`trace`) se persiste igual que la del
/// arquetipo para que el olvido por modo sobreviva al reinicio — round-trip sin pérdida.
#[derive(Serialize, Deserialize)]
struct ModeDto {
    centroid: Vec<f32>,
    /// Dirección de nacimiento del modo (base del drift). `default` para snapshots previos: al cargar,
    /// si viene vacío se usa el centroide (drift 0 — honesto: no había nacimiento registrado).
    #[serde(default)]
    origin: Vec<f32>,
    #[serde(default)]
    label: String,
    #[serde(default)]
    label_histogram: Vec<(String, usize)>,
    absorbed: usize,
    trace: TraceDto,
}

#[derive(Serialize, Deserialize)]
struct ArchetypeDto {
    version: u32,
    subject: String,
    core: Vec<f32>,
    anomalies: Vec<Vec<f32>>,
    #[serde(default)]
    anomaly_labels: Vec<String>,
    #[serde(default)]
    core_label: String,
    represented: usize,
    arc: Vec<MilestoneDto>,
    trace: TraceDto,
    /// Modos del arquetipo. `default` para cargar snapshots previos a la era multi-modal sin error.
    #[serde(default)]
    modes: Vec<ModeDto>,
}

impl From<&Archetype> for ArchetypeDto {
    fn from(a: &Archetype) -> Self {
        ArchetypeDto {
            version: SNAPSHOT_VERSION,
            subject: a.subject.clone(),
            core: a.core.clone(),
            anomalies: a.anomalies.clone(),
            anomaly_labels: a.anomaly_labels.clone(),
            core_label: a.core_label.clone(),
            represented: a.represented,
            arc: a
                .arc
                .iter()
                .map(|m| MilestoneDto {
                    at: m.at,
                    direction: m.direction.clone(),
                    absorbed: m.absorbed,
                    label: m.label.clone(),
                    label_histogram: m.label_histogram.clone(),
                })
                .collect(),
            trace: TraceDto {
                salience: a.trace.salience,
                lambda: a.trace.lambda,
                reinforcement: a.trace.reinforcement,
                last_touch: a.trace.last_touch,
            },
            modes: a
                .modes
                .iter()
                .map(|m| ModeDto {
                    centroid: m.centroid.clone(),
                    origin: m.origin.clone(),
                    label: m.label.clone(),
                    label_histogram: m.label_histogram.clone(),
                    absorbed: m.absorbed,
                    trace: TraceDto {
                        salience: m.trace.salience,
                        lambda: m.trace.lambda,
                        reinforcement: m.trace.reinforcement,
                        last_touch: m.trace.last_touch,
                    },
                })
                .collect(),
        }
    }
}

impl From<ArchetypeDto> for Archetype {
    fn from(d: ArchetypeDto) -> Self {
        Archetype {
            subject: d.subject,
            core: d.core,
            modes: d
                .modes
                .into_iter()
                .map(|m| Mode {
                    // Legacy: snapshots sin `origin` → usar el centroide (drift 0, honesto).
                    origin: if m.origin.is_empty() {
                        m.centroid.clone()
                    } else {
                        m.origin
                    },
                    centroid: m.centroid,
                    label: m.label,
                    label_histogram: m.label_histogram,
                    absorbed: m.absorbed,
                    trace: EntropyTrace {
                        salience: m.trace.salience,
                        lambda: m.trace.lambda,
                        reinforcement: m.trace.reinforcement,
                        last_touch: m.trace.last_touch,
                    },
                })
                .collect(),
            anomalies: d.anomalies,
            anomaly_labels: d.anomaly_labels,
            core_label: d.core_label,
            represented: d.represented,
            arc: d
                .arc
                .into_iter()
                .map(|m| ArcMilestone {
                    at: m.at,
                    direction: m.direction,
                    absorbed: m.absorbed,
                    label: m.label,
                    label_histogram: m.label_histogram,
                })
                .collect(),
            trace: EntropyTrace {
                salience: d.trace.salience,
                lambda: d.trace.lambda,
                reinforcement: d.trace.reinforcement,
                last_touch: d.trace.last_touch,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DTOs de la capa episódica (capa-1): hechos verbatim con física de olvido.
// ─────────────────────────────────────────────────────────────────────────────

/// DTO de un hecho episódico. Su `trace` se persiste igual que la del arquetipo: el olvido (y el
/// refuerzo ganado por evocación/repetición) sobrevive al reinicio — round-trip sin pérdida.
#[derive(Serialize, Deserialize)]
struct FactDto {
    subject: String,
    text: String,
    embedding: Vec<f32>,
    #[serde(default)]
    provenance: String,
    created_at: f64,
    trace: TraceDto,
}

/// DTO del `FactStore` completo (un solo archivo `facts.json`). El sharding por sujeto y el índice en
/// disco llegan con el storage engine embebido (L4); aquí basta un snapshot legible y diffable.
#[derive(Serialize, Deserialize)]
struct FactStoreDto {
    version: u32,
    theta_dedup: f32,
    facts: Vec<FactDto>,
}

impl From<&FactStore> for FactStoreDto {
    fn from(s: &FactStore) -> Self {
        FactStoreDto {
            version: SNAPSHOT_VERSION,
            theta_dedup: s.theta_dedup(),
            facts: s
                .iter()
                .map(|f| FactDto {
                    subject: f.subject.clone(),
                    text: f.text.clone(),
                    embedding: f.embedding.clone(),
                    provenance: f.provenance.clone(),
                    created_at: f.created_at,
                    trace: TraceDto {
                        salience: f.trace.salience,
                        lambda: f.trace.lambda,
                        reinforcement: f.trace.reinforcement,
                        last_touch: f.trace.last_touch,
                    },
                })
                .collect(),
        }
    }
}

impl From<FactStoreDto> for FactStore {
    fn from(d: FactStoreDto) -> Self {
        let mut store = FactStore::with_dedup(d.theta_dedup);
        for f in d.facts {
            store.insert(Fact {
                subject: f.subject,
                text: f.text,
                embedding: f.embedding,
                provenance: f.provenance,
                created_at: f.created_at,
                trace: EntropyTrace {
                    salience: f.trace.salience,
                    lambda: f.trace.lambda,
                    reinforcement: f.trace.reinforcement,
                    last_touch: f.trace.last_touch,
                },
            });
        }
        store
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Nombre de archivo estable por sujeto
// ─────────────────────────────────────────────────────────────────────────────

/// Hash FNV-1a de 64 bits — determinista y sin dependencias. Solo para nombrar archivos.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Nombre de archivo del snapshot de un sujeto: legible + hash anticolisión.
pub fn snapshot_filename(subject: &str) -> String {
    let safe: String = subject
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    // Recorta el prefijo legible para no generar nombres gigantes.
    let safe: String = safe.chars().take(48).collect();
    format!("{safe}-{:016x}.json", fnv1a(subject))
}

fn json_err(e: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

// ─────────────────────────────────────────────────────────────────────────────
// API pública
// ─────────────────────────────────────────────────────────────────────────────

/// Guarda un único arquetipo como snapshot en `dir`. Crea el directorio si no existe.
pub fn save_archetype(dir: impl AsRef<Path>, a: &Archetype) -> io::Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let dto = ArchetypeDto::from(a);
    let json = serde_json::to_string_pretty(&dto).map_err(json_err)?;
    let path = dir.join(snapshot_filename(&a.subject));
    fs::write(path, json)
}

/// Persiste toda la memoria de largo plazo: un archivo por sujeto. Devuelve cuántos se guardaron.
pub fn save_store(dir: impl AsRef<Path>, store: &ArchetypeStore) -> io::Result<usize> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let mut n = 0;
    for a in store.iter() {
        save_archetype(dir, a)?;
        n += 1;
    }
    Ok(n)
}

/// Carga un único snapshot desde un archivo `.json`.
pub fn load_archetype(path: impl AsRef<Path>) -> io::Result<Archetype> {
    let bytes = fs::read(path)?;
    let dto: ArchetypeDto = serde_json::from_slice(&bytes).map_err(json_err)?;
    Ok(dto.into())
}

/// Rehidrata una `ArchetypeStore` desde un directorio de snapshots. Ignora archivos no `.json`.
/// Un directorio inexistente se trata como store vacía (primer arranque).
pub fn load_store(dir: impl AsRef<Path>) -> io::Result<ArchetypeStore> {
    let dir = dir.as_ref();
    let mut store = ArchetypeStore::new();
    if !dir.exists() {
        return Ok(store);
    }
    // Orden estable por nombre de archivo → restauración determinista. Se excluye `facts.json` (la
    // capa-1 vive en el mismo directorio pero NO es un arquetipo: la carga la `load_facts`).
    let mut paths: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .filter(|p| p.file_name().and_then(|n| n.to_str()) != Some(FACTS_FILENAME))
        .collect();
    paths.sort();
    for p in paths {
        store.insert(load_archetype(p)?);
    }
    Ok(store)
}

/// Nombre del archivo de la capa episódica dentro del directorio de memoria. No colisiona con los
/// snapshots por sujeto (`{prefijo}-{hash}.json`) porque no lleva sufijo de hash.
pub const FACTS_FILENAME: &str = "facts.json";

/// Persiste la memoria episódica completa (capa-1) como un único `facts.json`. Devuelve cuántos
/// hechos se guardaron. Crea el directorio si no existe.
pub fn save_facts(dir: impl AsRef<Path>, store: &FactStore) -> io::Result<usize> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let dto = FactStoreDto::from(store);
    let n = dto.facts.len();
    let json = serde_json::to_string_pretty(&dto).map_err(json_err)?;
    fs::write(dir.join(FACTS_FILENAME), json)?;
    Ok(n)
}

/// Rehidrata la memoria episódica desde `facts.json`. Un archivo ausente se trata como store vacía
/// (primer arranque), igual que [`load_store`].
pub fn load_facts(dir: impl AsRef<Path>) -> io::Result<FactStore> {
    let path = dir.as_ref().join(FACTS_FILENAME);
    if !path.exists() {
        return Ok(FactStore::new());
    }
    let bytes = fs::read(path)?;
    let dto: FactStoreDto = serde_json::from_slice(&bytes).map_err(json_err)?;
    Ok(dto.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Storage embebido (L4): redb — KV transaccional, single-file, ACID, pure-Rust.
//
// El JSON-por-sujeto (arriba) es inspeccionable y diffable, pero no transaccional ni multi-tenant a
// escala. Aquí el "DB" de verdad: un único archivo redb con los arquetipos **keyed por sujeto** (se
// actualiza uno sin reescribir los demás) y la capa-1 como blob. Cada escritura es una transacción
// atómica (ACID): un crash a mitad no deja estado corrupto. Se reusan los mismos DTOs serde; el export
// JSON se conserva para inspección.
// ─────────────────────────────────────────────────────────────────────────────

const ARCHETYPES: TableDefinition<&str, &[u8]> = TableDefinition::new("archetypes");
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");

fn db_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(format!("redb: {e}"))
}

/// Store embebido transaccional de la memoria (las dos capas) en un único archivo redb.
pub struct DbStore {
    db: Database,
}

impl DbStore {
    /// Abre (o crea) el store en `path` (un archivo, p. ej. `memory.redb`).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let db = Database::create(path).map_err(db_err)?;
        Ok(Self { db })
    }

    /// Upsert de **un solo sujeto** en una transacción atómica — multi-tenant: no toca a los demás.
    pub fn write_archetype(&self, a: &Archetype) -> io::Result<()> {
        let wtxn = self.db.begin_write().map_err(db_err)?;
        {
            let mut t = wtxn.open_table(ARCHETYPES).map_err(db_err)?;
            let bytes = serde_json::to_vec(&ArchetypeDto::from(a)).map_err(json_err)?;
            t.insert(a.subject.as_str(), bytes.as_slice())
                .map_err(db_err)?;
        }
        wtxn.commit().map_err(db_err)?;
        Ok(())
    }

    /// Upsert de **toda** la capa-2 en una sola transacción. Devuelve cuántos arquetipos se guardaron.
    pub fn write_store(&self, store: &ArchetypeStore) -> io::Result<usize> {
        let wtxn = self.db.begin_write().map_err(db_err)?;
        let mut n = 0;
        {
            let mut t = wtxn.open_table(ARCHETYPES).map_err(db_err)?;
            for a in store.iter() {
                let bytes = serde_json::to_vec(&ArchetypeDto::from(a)).map_err(json_err)?;
                t.insert(a.subject.as_str(), bytes.as_slice())
                    .map_err(db_err)?;
                n += 1;
            }
        }
        wtxn.commit().map_err(db_err)?;
        Ok(n)
    }

    /// Rehidrata la capa-2 completa desde el store.
    pub fn read_store(&self) -> io::Result<ArchetypeStore> {
        let mut store = ArchetypeStore::new();
        let rtxn = self.db.begin_read().map_err(db_err)?;
        let table = match rtxn.open_table(ARCHETYPES) {
            Ok(t) => t,
            Err(_) => return Ok(store), // tabla aún no creada → primer arranque
        };
        for row in table.iter().map_err(db_err)? {
            let (_k, v) = row.map_err(db_err)?;
            let dto: ArchetypeDto = serde_json::from_slice(v.value()).map_err(json_err)?;
            store.insert(dto.into());
        }
        Ok(store)
    }

    /// Persiste la capa-1 (hechos) como un blob transaccional. Devuelve cuántos hechos se guardaron.
    pub fn write_facts(&self, store: &FactStore) -> io::Result<usize> {
        let dto = FactStoreDto::from(store);
        let n = dto.facts.len();
        let bytes = serde_json::to_vec(&dto).map_err(json_err)?;
        let wtxn = self.db.begin_write().map_err(db_err)?;
        {
            let mut t = wtxn.open_table(META).map_err(db_err)?;
            t.insert("factstore", bytes.as_slice()).map_err(db_err)?;
        }
        wtxn.commit().map_err(db_err)?;
        Ok(n)
    }

    /// Rehidrata la capa-1 desde el store.
    pub fn read_facts(&self) -> io::Result<FactStore> {
        let rtxn = self.db.begin_read().map_err(db_err)?;
        let table = match rtxn.open_table(META) {
            Ok(t) => t,
            Err(_) => return Ok(FactStore::new()),
        };
        match table.get("factstore").map_err(db_err)? {
            Some(v) => {
                let dto: FactStoreDto = serde_json::from_slice(v.value()).map_err(json_err)?;
                Ok(dto.into())
            }
            None => Ok(FactStore::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_core::synthesis::IntentionVector;
    use letheo_core::Resilience;
    use std::env;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let mut d = env::temp_dir();
        d.push(format!("letheo_persist_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn iv(subject: &str, c: Vec<f32>, absorbed: usize) -> IntentionVector {
        IntentionVector {
            subject: subject.into(),
            centroid: c,
            anomalies: vec![vec![0.0, 1.0]],
            core_label: "core".into(),
            anomaly_labels: vec!["novelty".into()],
            absorbed,
            redundant: 0,
            label_histogram: vec![("core".into(), absorbed)],
            modes: vec![],
        }
    }

    fn sample_store() -> ArchetypeStore {
        let mut s = ArchetypeStore::new();
        s.imprint(
            &iv("user:Xolotl", vec![1.0, 0.0], 1000),
            Resilience::High,
            0.0,
        );
        s.imprint(
            &iv("user:Xolotl", vec![0.0, 1.0], 500),
            Resilience::High,
            3600.0,
        ); // evoluciona
        s.imprint(
            &iv("agent:Tlaloc", vec![0.3, 0.7], 42),
            Resilience::Medium,
            0.0,
        );
        s
    }

    #[test]
    fn filename_is_collision_resistant() {
        // Sujetos que sanean al mismo prefijo deben diferir por el hash.
        assert_ne!(snapshot_filename("user:X"), snapshot_filename("user_X"));
    }

    #[test]
    fn roundtrip_preserves_every_field() {
        let dir = tmp_dir("roundtrip");
        let original = sample_store();
        let saved = save_store(&dir, &original).unwrap();
        assert_eq!(saved, 2, "dos sujetos distintos → dos archivos");

        let restored = load_store(&dir).unwrap();
        assert_eq!(restored.len(), 2);

        let a = original.get("user:Xolotl").unwrap();
        let b = restored.get("user:Xolotl").unwrap();
        assert_eq!(a.subject, b.subject);
        assert_eq!(a.represented, b.represented);
        assert_eq!(a.core, b.core);
        assert_eq!(a.anomalies, b.anomalies);
        assert_eq!(a.arc.len(), b.arc.len(), "el arco evolutivo sobrevive");
        assert_eq!(a.arc[1].absorbed, b.arc[1].absorbed);
        // La física exacta del arquetipo se conserva (vida media ganada por consolidación incluida).
        assert_eq!(a.trace.lambda, b.trace.lambda);
        assert_eq!(a.trace.reinforcement, b.trace.reinforcement);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resonance_survives_restart() {
        let dir = tmp_dir("resonance");
        save_store(&dir, &sample_store()).unwrap();
        let restored = load_store(&dir).unwrap();
        // La memoria rehidratada sigue resonando: consulta cercana a Xolotl lo recupera.
        let top = restored.resonate(
            &[0.6, 0.4],
            1,
            7200.0,
            letheo_core::entropy::DEFAULT_THETA_FADE,
        );
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].subject, "user:Xolotl");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_long_term_survives_restart() {
        use letheo_core::{CognitiveRuntime, EvokeRequest, Perception, RuntimeConfig};
        let dir = tmp_dir("runtime");

        // Un runtime vive, sueña y consolida una esencia…
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        for _ in 0..50 {
            rt.perceive(Perception::new(
                "user:X",
                vec![1.0, 0.0],
                1.0,
                86_400.0,
                0.0,
            ));
        }
        rt.breathe(&["user:X"], 0.0);
        save_store(&dir, rt.long_term()).unwrap();
        drop(rt); // el proceso "se reinicia".

        // Un runtime nuevo rehidrata su memoria de largo plazo desde disco…
        let mut reborn = CognitiveRuntime::new(RuntimeConfig::default());
        assert_eq!(reborn.long_term_len(), 0);
        *reborn.long_term_mut() = load_store(&dir).unwrap();

        // …y puede evocar lo que aprendió en su "vida anterior".
        let ctx = reborn
            .evoke(&EvokeRequest::new("user:X", 800), 0.0)
            .expect("la esencia sobrevivió al reinicio");
        assert_eq!(ctx.represented, 50);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_dir_loads_empty() {
        let dir = tmp_dir("nonexistent_xyz");
        let store = load_store(&dir).unwrap();
        assert!(store.is_empty(), "primer arranque: store vacía, sin error");
    }

    #[test]
    fn factstore_roundtrip_preserves_facts_and_physics() {
        use letheo_core::FactStore;
        let dir = tmp_dir("facts");

        let mut fs = FactStore::new();
        fs.remember(
            "user:X",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "agentA",
            1.0,
            86_400.0,
            0.0,
        );
        fs.remember(
            "user:X",
            "drives a red car",
            vec![1.0, 0.0],
            "agentB",
            0.8,
            86_400.0,
            1.0,
        );
        // Evocar uno: gana refuerzo + λ consolidado → su física deja de ser trivial.
        let hits = fs.recall(
            "user:X",
            &[1.0, 0.0],
            1,
            2.0,
            letheo_core::entropy::DEFAULT_THETA_FADE,
        );
        assert_eq!(hits[0].text, "drives a red car");

        let n = save_facts(&dir, &fs).unwrap();
        assert_eq!(n, 2, "dos hechos distintos → dos entradas");

        let restored = load_facts(&dir).unwrap();
        assert_eq!(restored.len(), 2);

        // Round-trip sin pérdida: texto verbatim, embedding, procedencia y la física exacta (incluido
        // el refuerzo y la vida media consolidada por la evocación).
        let orig: Vec<_> = fs.iter().collect();
        let back: Vec<_> = restored.iter().collect();
        for (a, b) in orig.iter().zip(back.iter()) {
            assert_eq!(a.text, b.text, "el hecho exacto sobrevive verbatim");
            assert_eq!(a.subject, b.subject);
            assert_eq!(a.provenance, b.provenance);
            assert_eq!(a.embedding, b.embedding);
            assert_eq!(a.created_at, b.created_at);
            assert_eq!(
                a.trace.lambda, b.trace.lambda,
                "la vida media consolidada se conserva"
            );
            assert_eq!(a.trace.reinforcement, b.trace.reinforcement);
            assert_eq!(a.trace.last_touch, b.trace.last_touch);
        }

        // La capa-1 rehidratada sigue respondiendo nominal: el hecho exacto se recupera tras el reinicio.
        let mut restored = restored;
        let after = restored.recall(
            "user:X",
            &[0.0, 1.0],
            1,
            3.0,
            letheo_core::entropy::DEFAULT_THETA_FADE,
        );
        assert_eq!(
            after[0].text, "allergic to peanuts",
            "el hecho sobrevive al reinicio y se evoca"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn both_layers_coexist_in_one_dir() {
        use letheo_core::FactStore;
        // Las dos capas conviven en el MISMO directorio (como hace el binding al `save`).
        let dir = tmp_dir("both_layers");
        save_store(&dir, &sample_store()).unwrap();
        let mut facts = FactStore::new();
        facts.remember(
            "user:Xolotl",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "agent",
            1.0,
            86_400.0,
            0.0,
        );
        save_facts(&dir, &facts).unwrap();

        // `load_store` debe IGNORAR `facts.json` (no parsearlo como arquetipo) y los hechos cargan aparte.
        let restored = load_store(&dir).unwrap();
        assert_eq!(
            restored.len(),
            2,
            "los arquetipos cargan; facts.json se ignora"
        );
        assert_eq!(
            load_facts(&dir).unwrap().len(),
            1,
            "los hechos cargan por su lado"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    fn tmp_file(tag: &str) -> std::path::PathBuf {
        let mut p = env::temp_dir();
        p.push(format!("letheo_db_{tag}_{}.redb", std::process::id()));
        let _ = fs::remove_file(&p);
        p
    }

    #[test]
    fn db_roundtrips_both_layers_and_survives_reopen() {
        use letheo_core::FactStore;
        let path = tmp_file("roundtrip");
        {
            let db = DbStore::open(&path).unwrap();
            assert_eq!(db.write_store(&sample_store()).unwrap(), 2);
            let mut facts = FactStore::new();
            facts.remember(
                "user:Xolotl",
                "allergic to peanuts",
                vec![0.0, 1.0],
                "agent",
                1.0,
                86_400.0,
                0.0,
            );
            assert_eq!(db.write_facts(&facts).unwrap(), 1);
        } // se cierra el DB (libera el lock del archivo)

        // Reabrir: la memoria (las dos capas) sobrevive — durabilidad ACID.
        let db = DbStore::open(&path).unwrap();
        let store = db.read_store().unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("user:Xolotl").unwrap().represented, 1500);
        assert_eq!(db.read_facts().unwrap().len(), 1);
        drop(db);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn db_is_multi_tenant_per_subject() {
        let path = tmp_file("multitenant");
        let db = DbStore::open(&path).unwrap();
        db.write_store(&sample_store()).unwrap(); // user:Xolotl (1500) + agent:Tlaloc (42)

        // Actualizar SOLO un sujeto no debe tocar al otro (la clave es el sujeto).
        let mut one = ArchetypeStore::new();
        one.imprint(
            &iv("agent:Tlaloc", vec![1.0, 0.0], 999),
            Resilience::High,
            0.0,
        );
        db.write_archetype(one.get("agent:Tlaloc").unwrap())
            .unwrap();

        let store = db.read_store().unwrap();
        assert_eq!(store.len(), 2, "siguen los dos sujetos");
        assert_eq!(
            store.get("agent:Tlaloc").unwrap().represented,
            999,
            "el actualizado cambió"
        );
        assert_eq!(
            store.get("user:Xolotl").unwrap().represented,
            1500,
            "el otro quedó intacto"
        );
        drop(db);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn multimodal_archetype_survives_restart_with_physics() {
        use letheo_core::{CognitiveRuntime, Perception, RuntimeConfig};
        let dir = tmp_dir("multimodal");

        // Un sujeto con tres comportamientos distintos: el motor destila tres modos.
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        for _ in 0..10 {
            rt.perceive(
                Perception::new("u", vec![1.0, 0.0, 0.0], 1.0, 86_400.0, 0.0)
                    .with_trait("act", "noir"),
            );
            rt.perceive(
                Perception::new("u", vec![0.0, 1.0, 0.0], 1.0, 86_400.0, 0.0)
                    .with_trait("act", "docs"),
            );
            rt.perceive(
                Perception::new("u", vec![0.0, 0.0, 1.0], 1.0, 86_400.0, 0.0)
                    .with_trait("act", "scifi"),
            );
        }
        rt.breathe(&["u"], 0.0);
        let modes_before = rt.long_term().get("u").unwrap().modes.len();
        assert_eq!(modes_before, 3, "tres comportamientos → tres modos");
        save_store(&dir, rt.long_term()).unwrap();
        drop(rt);

        // Tras "reiniciar", los modos y su física (vida media, etiqueta) sobreviven sin pérdida.
        let restored = load_store(&dir).unwrap();
        let a = restored.get("u").unwrap();
        assert_eq!(a.modes.len(), 3, "los modos sobreviven al reinicio");
        let labels: Vec<&str> = a.modes.iter().map(|m| m.label.as_str()).collect();
        assert!(labels.contains(&"noir") && labels.contains(&"docs") && labels.contains(&"scifi"));
        // La resonancia multi-modal sigue funcionando tras rehidratar.
        assert!(
            (a.resonance(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-3,
            "el modo correcto resuena tras el reinicio"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mode_origin_and_drift_survive_restart() {
        use letheo_core::{CognitiveRuntime, Perception, RuntimeConfig};
        let dir = tmp_dir("mode_drift");
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        // Ciclo 1: un comportamiento en [1,0] (vida media corta para que no se cuele en el ciclo 2).
        for _ in 0..5 {
            rt.perceive(Perception::new("u", vec![1.0, 0.0], 1.0, 1.0, 0.0).with_trait("act", "x"));
        }
        rt.breathe(&["u"], 0.0);
        // Ciclo 2: el MISMO modo pero desplazado a [0.6,0.8] (cos 0.6 ≥ θ → funde) → el modo deriva.
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![0.6, 0.8], 1.0, 86_400.0, 0.0).with_trait("act", "x"),
            );
        }
        rt.breathe(&["u"], 100.0);
        let drift_before = rt.long_term().get("u").unwrap().modes[0].drift();
        assert!(
            drift_before > 0.0,
            "el modo derivó desde su origen: {drift_before}"
        );

        save_store(&dir, rt.long_term()).unwrap();
        drop(rt);
        let restored = load_store(&dir).unwrap();
        let drift_after = restored.get("u").unwrap().modes[0].drift();
        assert!(
            (drift_after - drift_before).abs() < 1e-6,
            "el origin (y el drift) sobreviven al reinicio"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
