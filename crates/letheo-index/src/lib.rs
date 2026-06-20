//! # letheo-index · Índice ANN (HNSW) sobre el motor
//!
//! El core hace búsqueda lineal Flat (exacta, O(n)) — perfecta a decenas de modos/hechos, inviable a
//! millones. Aquí vive la aceleración: un índice **HNSW** sobre las dos capas —`(sujeto × modo)`
//! (capa-2) y los hechos (capa-1)— que el motor consulta en O(log n). Crate **aparte** para que
//! `letheo-core` siga hermético/offline (igual que `letheo-async` aísla Tokio): la dependencia ANN
//! externa no entra en el núcleo.
//!
//! **Truco de métrica**: HNSW asume una distancia métrica; el motor rankea por **coseno**. Sobre
//! vectores normalizados a norma unidad, la distancia euclídea es monótona con `(1 − coseno)`
//! (`‖a−b‖² = 2 − 2·cos`), así que el vecino más cercano euclídeo *es* el de mayor coseno. Por eso el
//! índice normaliza centroides y consulta. El Flat del core se conserva como **oráculo exacto**.
//!
//! El [`Retriever`] cablea ambos: Flat exacto por debajo de un umbral de tamaño, HNSW por encima, con
//! **filtrado por vida** (filtered-ANN) integrado.

use instant_distance::{Builder, HnswMap, Point, Search};
use letheo_core::{ArchetypeStore, FactStore, Tick};
use std::collections::HashSet;

/// Seed fijo del constructor HNSW → índice **reproducible** bit a bit (sin azar de sistema), coherente
/// con la disciplina de determinismo del motor.
const INDEX_SEED: u64 = 0x1E7E0;

/// Un punto del índice: un embedding **normalizado a norma unidad**.
#[derive(Clone, Debug)]
struct UnitPoint(Vec<f32>);

impl Point for UnitPoint {
    /// Distancia euclídea. Sobre vectores unidad es monótona con `(1 − coseno)`, así que el ranking
    /// ANN coincide con el del coseno del motor.
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(&other.0)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// A qué `(sujeto, modo)` corresponde un punto de capa-2.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModeRef {
    pub subject: String,
    /// Índice del modo dentro del arquetipo (`0` si el arquetipo no tiene modos: su núcleo global).
    pub mode: usize,
}

/// A qué hecho corresponde un punto de capa-1: el sujeto y la posición del hecho al construir el índice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FactRef {
    pub subject: String,
    pub position: usize,
}

/// Índice ANN genérico sobre puntos `(embedding, valor)`. `ModeIndex`/`FactIndex` lo especializan.
pub struct AnnIndex<V> {
    map: Option<HnswMap<UnitPoint, V>>,
    len: usize,
}

impl<V: Clone> AnnIndex<V> {
    /// Construye el índice desde puntos `(embedding, valor)` crudos.
    pub fn from_points(points: Vec<(Vec<f32>, V)>) -> Self {
        let len = points.len();
        if len == 0 {
            return Self { map: None, len: 0 };
        }
        let (pts, vals): (Vec<UnitPoint>, Vec<V>) = points
            .into_iter()
            .map(|(c, v)| (UnitPoint(normalize(&c)), v))
            .unzip();
        let map = Builder::default().seed(INDEX_SEED).build(pts, vals);
        Self {
            map: Some(map),
            len,
        }
    }

    /// Los `k` valores cuyo embedding más resuena con la consulta, vía HNSW (≈ O(log n)). Equivale al
    /// top-k por coseno del Flat, pero sin recorrer todos los puntos.
    pub fn resonate(&self, query: &[f32], k: usize) -> Vec<V> {
        let map = match &self.map {
            Some(m) => m,
            None => return Vec::new(),
        };
        let q = UnitPoint(normalize(query));
        let mut search = Search::default();
        map.search(&q, &mut search)
            .take(k)
            .map(|item| item.value.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Índice de la **capa-2**: un punto por cada `(sujeto × modo)` de una `ArchetypeStore`.
pub type ModeIndex = AnnIndex<ModeRef>;

impl ModeIndex {
    /// Construye el índice desde un store: un punto por cada modo de cada arquetipo (centroide). Un
    /// arquetipo sin modos (legado) aporta su núcleo global como un único punto (`mode = 0`).
    pub fn build(store: &ArchetypeStore) -> Self {
        let mut points = Vec::new();
        for a in store.iter() {
            if a.modes.is_empty() {
                points.push((
                    a.core.clone(),
                    ModeRef {
                        subject: a.subject.clone(),
                        mode: 0,
                    },
                ));
            } else {
                for (i, m) in a.modes.iter().enumerate() {
                    points.push((
                        m.centroid.clone(),
                        ModeRef {
                            subject: a.subject.clone(),
                            mode: i,
                        },
                    ));
                }
            }
        }
        Self::from_points(points)
    }

    /// Los `k` **sujetos** más resonantes (deduplicando modos del mismo sujeto, conservando el orden).
    pub fn resonate_subjects(&self, query: &[f32], k: usize) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for r in self.resonate(query, k.saturating_mul(4).max(k)) {
            if seen.insert(r.subject.clone()) {
                out.push(r.subject);
                if out.len() == k {
                    break;
                }
            }
        }
        out
    }
}

/// Índice de la **capa-1**: un punto por cada hecho episódico de una `FactStore`.
pub type FactIndex = AnnIndex<FactRef>;

impl FactIndex {
    /// Construye el índice desde la memoria episódica: un punto por hecho (su embedding).
    pub fn build_facts(store: &FactStore) -> Self {
        let points = store
            .iter()
            .enumerate()
            .map(|(i, f)| {
                (
                    f.embedding.clone(),
                    FactRef {
                        subject: f.subject.clone(),
                        position: i,
                    },
                )
            })
            .collect();
        Self::from_points(points)
    }
}

/// Recuperador que combina el **Flat exacto** del core (rápido a poca escala, rankea por
/// relevancia·vida) con el **HNSW** (a gran escala), eligiendo según un umbral de tamaño. Cachea el
/// índice y lo reconstruye cuando cambia el nº de arquetipos. Filtra los resultados HNSW por **vida**
/// (filtered-ANN), de modo que un arquetipo desvanecido no se devuelve aunque resuene.
pub struct Retriever {
    threshold: usize,
    cache: Option<(usize, ModeIndex)>,
}

impl Retriever {
    /// Crea un recuperador que usa Flat hasta `threshold` arquetipos y HNSW por encima.
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            cache: None,
        }
    }

    /// Top-`k` sujetos resonantes y **vivos**. Por debajo del umbral usa el Flat exacto del core (que
    /// ya rankea por relevancia·vida); por encima usa el HNSW (reconstruido si cambió el tamaño del
    /// store) y filtra por vida. `&mut self` porque cachea el índice.
    pub fn resonate_subjects(
        &mut self,
        store: &ArchetypeStore,
        query: &[f32],
        k: usize,
        now: Tick,
        theta_fade: f64,
    ) -> Vec<String> {
        if store.len() <= self.threshold {
            return store
                .resonate(query, k, now, theta_fade)
                .into_iter()
                .map(|a| a.subject.clone())
                .collect();
        }
        // HNSW: (re)construir si cambió el nº de arquetipos.
        if !matches!(&self.cache, Some((n, _)) if *n == store.len()) {
            self.cache = Some((store.len(), ModeIndex::build(store)));
        }
        let index = &self.cache.as_ref().unwrap().1;

        // Margen amplio de candidatos → filtrar por vida → deduplicar a k sujetos.
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for r in index.resonate(query, k.saturating_mul(8).max(k)) {
            if let Some(a) = store.get(&r.subject) {
                if a.trace.weight(now) >= theta_fade && seen.insert(r.subject.clone()) {
                    out.push(r.subject);
                    if out.len() == k {
                        break;
                    }
                }
            }
        }
        out
    }

    /// Invalida el índice cacheado (úsese tras mutar los modos sin cambiar el nº de arquetipos).
    pub fn invalidate(&mut self) {
        self.cache = None;
    }
}

/// Normaliza a norma unidad. Vector nulo → se devuelve igual (distancia bien definida).
fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_core::synthesis::IntentionVector;
    use letheo_core::{ArchetypeStore, FactStore, Resilience};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Vector pseudo-aleatorio determinista (LCG) de dimensión `dim`.
    fn pseudo(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// `recall@k` del índice contra el Flat exacto, promediado sobre `queries` consultas.
    fn measure_recall(points: &[(Vec<f32>, ModeRef)], dim: usize, k: usize, queries: u64) -> f64 {
        let index = AnnIndex::from_points(points.to_vec());
        let (mut hits, mut total) = (0usize, 0usize);
        for q in 0..queries {
            let query = pseudo(dim, 1_000_000 + q);
            let mut exact: Vec<(f32, &ModeRef)> =
                points.iter().map(|(c, r)| (cosine(c, &query), r)).collect();
            exact.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let exact_set: HashSet<&ModeRef> = exact.iter().take(k).map(|(_, r)| *r).collect();
            hits += index
                .resonate(&query, k)
                .iter()
                .filter(|r| exact_set.contains(r))
                .count();
            total += k;
        }
        hits as f64 / total as f64
    }

    #[test]
    fn ann_recall_at_k_matches_flat_oracle() {
        let dim = 48;
        let points: Vec<(Vec<f32>, ModeRef)> = (0..800)
            .map(|i| {
                (
                    pseudo(dim, i as u64),
                    ModeRef {
                        subject: format!("s{i}"),
                        mode: 0,
                    },
                )
            })
            .collect();
        let recall = measure_recall(&points, dim, 10, 60);
        assert!(
            recall >= 0.99,
            "recall@10 = {recall:.4} (objetivo ≥ 0.99 vs Flat exacto)"
        );
    }

    #[test]
    #[ignore = "escala (20k modos): el build HNSW en debug tarda; correr con --ignored o --release"]
    fn ann_holds_recall_at_scale() {
        // A 20.000 modos el índice se construye y mantiene recall alto (el motor a escala).
        let dim = 48;
        let points: Vec<(Vec<f32>, ModeRef)> = (0..20_000)
            .map(|i| {
                (
                    pseudo(dim, i as u64),
                    ModeRef {
                        subject: format!("s{i}"),
                        mode: 0,
                    },
                )
            })
            .collect();
        let recall = measure_recall(&points, dim, 10, 30);
        assert!(recall >= 0.95, "recall@10 a escala = {recall:.4} (≥ 0.95)");
    }

    #[test]
    fn fact_index_recovers_the_relevant_fact() {
        // Capa-1: hechos con direcciones distintas → el índice recupera el correcto por consulta.
        let mut fs = FactStore::new();
        fs.remember(
            "u0",
            "peanuts allergy",
            vec![1.0, 0.0, 0.0],
            "t",
            1.0,
            86_400.0,
            0.0,
        );
        fs.remember(
            "u1",
            "red car",
            vec![0.0, 1.0, 0.0],
            "t",
            1.0,
            86_400.0,
            0.0,
        );
        fs.remember(
            "u2",
            "window seat",
            vec![0.0, 0.0, 1.0],
            "t",
            1.0,
            86_400.0,
            0.0,
        );
        let index = FactIndex::build_facts(&fs);
        assert_eq!(index.len(), 3);
        let top = index.resonate(&[0.1, 0.9, 0.0], 1);
        assert_eq!(
            top[0],
            FactRef {
                subject: "u1".into(),
                position: 1
            }
        );
    }

    fn store_of(n: usize, dim: usize) -> ArchetypeStore {
        let mut store = ArchetypeStore::new();
        for i in 0..n {
            store.imprint(
                &IntentionVector {
                    subject: format!("user:{i}"),
                    centroid: pseudo(dim, i as u64),
                    anomalies: vec![],
                    core_label: "x".into(),
                    anomaly_labels: vec![],
                    absorbed: 10,
                    redundant: 0,
                    label_histogram: vec![("x".into(), 10)],
                    modes: vec![],
                },
                Resilience::High,
                0.0,
            );
        }
        store
    }

    #[test]
    fn retriever_flat_and_hnsw_paths_agree_on_top_subject() {
        let dim = 48;
        let store = store_of(120, dim);
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        let mut flat = Retriever::new(10_000); // 120 ≤ 10000 → Flat
        let mut hnsw = Retriever::new(0); //               120 > 0     → HNSW
        for q in 0..20 {
            let query = pseudo(dim, 7_000 + q);
            let f = flat.resonate_subjects(&store, &query, 1, 0.0, theta);
            let h = hnsw.resonate_subjects(&store, &query, 1, 0.0, theta);
            assert_eq!(f, h, "Flat y HNSW coinciden en el sujeto top (q={q})");
        }
    }

    #[test]
    fn retriever_hnsw_filters_out_faded_subjects() {
        // Un sujeto muy relevante pero desvanecido NO se devuelve por la ruta HNSW (filtered-ANN).
        let dim = 8;
        let mut store = store_of(60, dim);
        // Insertamos un sujeto alineado con la consulta pero con vida media corta → desvanecido en T.
        let target = pseudo(dim, 999);
        store.imprint(
            &IntentionVector {
                subject: "user:faded".into(),
                centroid: target.clone(),
                anomalies: vec![],
                core_label: "x".into(),
                anomaly_labels: vec![],
                absorbed: 10,
                redundant: 0,
                label_histogram: vec![("x".into(), 10)],
                modes: vec![],
            },
            Resilience::Low,
            0.0,
        );
        let theta = letheo_core::entropy::DEFAULT_THETA_FADE;
        let mut hnsw = Retriever::new(0);
        // A 200 días el sujeto Low (vida media 30d) ya se desvaneció, mientras los High (720d) siguen
        // vivos. El desvanecido no aparece pese a resonar perfecto con la consulta (filtered-ANN).
        let later = 200.0 * 86_400.0;
        let hits = hnsw.resonate_subjects(&store, &target, 5, later, theta);
        assert!(!hits.is_empty(), "hay sujetos vivos que devolver");
        assert!(
            !hits.contains(&"user:faded".to_string()),
            "el desvanecido se filtró: {hits:?}"
        );
    }

    #[test]
    fn empty_index_is_well_defined() {
        let index: AnnIndex<ModeRef> = AnnIndex::from_points(Vec::new());
        assert!(index.is_empty());
        assert!(index.resonate(&[1.0, 0.0], 5).is_empty());
    }
}
