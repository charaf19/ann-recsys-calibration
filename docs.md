Below is a complete, production‑grade **Markdown** document you can drop into your repo as `README.md`. It spells out the project’s aims, architecture, pipeline, file layout, interfaces, metrics, evaluation design, ablation/sweep flow, and troubleshooting—the “boring details” reviewers and collaborators need.

---

# Tiny Index, Big Impact — End‑to‑End ANN for Recommender Retrieval

> **Goal in one line:** Evaluate fast, memory‑bounded ANN indexes (HNSW, IVF‑Flat, IVF‑PQ, Flat‑PQ, and Flat exact) for top‑K candidate generation on real recommendation datasets, with rigorous **end‑to‑end** metrics (leave‑one‑out user relevance, coverage/fairness) and device‑level latency.

---

## 1) Why this project exists

Most ANN papers report recall vs. latency on synthetic vectors or on nearest‑neighbor agreement against exact indices. That misses what matters in recommenders: **does the index recover items the user actually likes**, under realistic memory/latency budgets?

This project:

* Produces **item embeddings** from interaction data.
* Builds **budget‑aware** ANN indices.
* Measures **device‑level latency & memory**.
* Evaluates **end‑to‑end** retrieval relevance using **leave‑one‑out** from interactions (NDCG@k, Recall@k, HR@k, Precision@k, MAP@k, MRR@k).
* Tracks **coverage**, **Gini exposure**, and **long‑tail uplift**.
* Generates **publication‑ready tables/plots**.

Supported datasets (as used in the current study):

* **MovieLens‑1M** (`ml-1m`)
* **MovieLens‑20M** (`ml-20m`)
* **Goodbooks‑10k** (`goodbooks`)
* **Amazon‑Books** (`amazon-books`)

---

## 2) Repository layout (high‑level)

```
.
├─ src/
│  ├─ prepare_dataset.py          # canonical entry for dataset preparation → interactions CSV
│  ├─ train_embeddings.py         # trains item embeddings → item_vecs.npy + item_ids.npy
│  ├─ build_index.py              # builds FAISS/HNSW indices under a memory budget
│  ├─ run_device.py               # measures device-level latency, writes latency_{method}.json
│  ├─ eval_end2end.py             # end-to-end relevance + coverage/fairness metrics (LOO)
│  ├─ run_grid.py                 # runs full grid across datasets/methods, writes summary CSVs
│  ├─ figures.py                  # makes plots from summary_all.csv → results/figures/*.png
│  └─ datasets/
│     ├─ ml1m.py, ml20m.py        # dataset-specific prep helpers (used by prepare_dataset.py)
│     ├─ goodbooks.py             # robust loader (+timestamp synthesis if missing)
│     └─ amazon_books.py          # Amazon Books preparation (core or subset)
│
├─ data/                          # prepared CSVs + learned embeddings + built indices (by default)
│  ├─ ml1m.csv, ml20m.csv, goodbooks.csv, amazon_books.csv
│  ├─ emb_<dataset>/{item_vecs.npy, item_ids.npy}
│  └─ index_<dataset>_<method>/*  # index artifacts (see §5)
│
├─ results/
│  ├─ summary_<dataset>.csv       # per-dataset summary from run_grid.py
│  ├─ summary_all.csv             # concatenation of all datasets’ summaries
│  ├─ ablations/ablation_<dataset>.csv  # optional sweeps
│  └─ figures/*.png               # figures.py outputs
│
└─ README.md (this file)
```

---

## 3) Data contract: interactions CSV

All downstream steps assume **one canonical schema**:

| column      | dtype         | meaning                                              |
| ----------- | ------------- | ---------------------------------------------------- |
| `user_id`   | string or int | user identifier (will be normalized internally)      |
| `item_id`   | string or int | item identifier (mapped to vector row indices)       |
| `timestamp` | int64         | interaction time; used to build leave‑one‑out splits |

Notes:

* If a source lacks timestamps (e.g., **Goodbooks‑10k** `ratings.csv`), we **synthesize** a per‑user chronological timestamp using `groupby(user).cumcount()+1`. This preserves interaction order without claiming absolute time.
* Preparation scripts coerce IDs to string, and `timestamp` to `int64` with missing→0 fallback.

---

## 4) Dataset preparation

**Entry point:** `src/prepare_dataset.py`

Usage:

```bash
python src/prepare_dataset.py --dataset <ml-1m|ml-20m|goodbooks|amazon-books> --out data/<name>.csv
```

### 4.1 MovieLens (1M / 20M)

* Reads ratings, renames columns to the canonical schema,
* drops fields not needed for retrieval,
* writes `data/ml1m.csv` or `data/ml20m.csv`.

### 4.2 Goodbooks‑10k

* If available locally at one of:

  * `data/goodbooks/ratings.csv`,
  * `data/raw/goodbooks/ratings.csv`,
  * `data/goodbooks-10k/ratings.csv`,
  * or any `data/**/ratings.csv` with expected columns,
    it uses that; otherwise the helper can download from the **public ratings CSV** you referenced earlier.
* Normalizes columns to `user_id, book_id[, rating][, timestamp]`.
* If `timestamp` missing → synthesizes per‑user order as timestamp.
* Renames `book_id` → `item_id`.
* Writes `data/goodbooks.csv`.

### 4.3 Amazon‑Books

* Prepares the Books subset (IDs → canonical `user_id,item_id,timestamp`).
* Writes `data/amazon_books.csv`.

> **Tip:** Very large datasets can stress RAM during CSV load; prefer `engine="c"`, `usecols`, and dtype down‑casting in your local variant if needed.

---

## 5) Embeddings

**Entry point:** `src/train_embeddings.py`

Usage:

```bash
python src/train_embeddings.py --interactions data/<dataset>.csv --dim 128 --out_dir data/emb_<dataset>
```

Outputs:

* `item_vecs.npy` — shape `(N_items, dim)` (`float32`)
* `item_ids.npy` — array of original item IDs (aligned to rows)

**Remarks**

* The current training is deliberately lightweight; the project focuses on **indexing & retrieval**, not embedding SOTA.
* Seeds are fixed for reproducibility.

---

## 6) Index builders (budget‑aware)

**Entry point:** `src/build_index.py`

Usage (representative):

```bash
# Flat exact (L2)
python src/build_index.py --method flat --item_vecs data/emb_X/item_vecs.npy \
  --item_ids data/emb_X/item_ids.npy --out_dir data/index_X_flat --budget_mb 100

# HNSW (distance = L2): construction efC=200, graph degree M=24
python src/build_index.py --method hnsw --item_vecs ... --item_ids ... \
  --out_dir data/index_X_hnsw --budget_mb 100 --M 24 --efc 200

# IVF-Flat (nlist auto-heurstic)
python src/build_index.py --method ivfflat --item_vecs ... --item_ids ... \
  --out_dir data/index_X_ivfflat --budget_mb 100 --nlist auto

# IVF-PQ (with optional OPQ)
python src/build_index.py --method ivfpq --item_vecs ... --item_ids ... \
  --out_dir data/index_X_ivfpq --budget_mb 100 --nlist auto --m 32 --bits 8 --opq

# Flat-PQ (no coarse quantizer)
python src/build_index.py --method flatpq --item_vecs ... --item_ids ... \
  --out_dir data/index_X_flatpq --budget_mb 100 --m 32 --bits 8
```

**Files produced per index (out_dir):**

* `faiss_flat.index` (Flat exact), `faiss_ivfflat.index`, `faiss_ivfpq.index`, `faiss_flatpq.index`, or `hnsw_index.bin`
* Aux files: pretransform params if OPQ, and a **size report** in the build logs
* Some builders log “INFO: tiny catalog; disabling OPQ …” if the sample is too small/fragile.

**Heuristics & budget**

* `--budget_mb` is a soft target; builders pick parameters (e.g., `nlist`) to roughly respect it.
* Small catalogs yield smaller `nlist` and vice versa (e.g., `nlist≈√N` cap with a training sample, as seen in logs like `nlist=59/151/606`).
* IVF training samples are capped (e.g., 2000, 9984, 38848 in logs) to keep training fast; FAISS will warn if k‑means is data limited.

---

## 7) Device‑level latency

**Entry point:** `src/run_device.py`

What it does:

* **Discovers** the index file from a directory (`_resolve_index_path` tries common names like `faiss_ivfflat.index`, `index.faiss`, `hnsw_index.bin`, or the largest `.faiss` file).
* **Identifies** the method (`flat`, `flatpq`, `ivfflat`, `ivfpq`, `hnsw`) via FAISS downcast / HNSW path.
* Sets **query‑time knobs**:

  * `--ef` for HNSW (search breadth),
  * `--nprobe` for IVF (number of cells probed).
* Runs warm‑ups, then `--queries` random queries, measures latencies, and reports **p50/p95/p99/mean** and **RSS memory** deltas.
* Writes `latency_{method}.json` in the index directory and prints the same JSON to stdout.

Usage (examples):

```bash
# Flat exact
python src/run_device.py --index data/index_ml-1m_flat --item_vecs data/emb_ml-1m/item_vecs.npy \
  --queries 2000 --topk 100

# HNSW with ef=128
python src/run_device.py --index data/index_ml-20m_hnsw --item_vecs data/emb_ml-20m/item_vecs.npy \
  --queries 2000 --topk 100 --ef 128

# IVF-Flat with nprobe=16
python src/run_device.py --index data/index_ml-20m_ivfflat --item_vecs data/emb_ml-20m/item_vecs.npy \
  --queries 2000 --topk 100 --nprobe 16
```

---

## 8) End‑to‑end evaluation (leave‑one‑out)

**Entry point:** `src/eval_end2end.py` (updated)

**Key change:** now requires `--interactions` (the ground truth source) and evaluates **user relevance** using **leave‑one‑out (LOO)**:

* For each user with ≥2 interactions, hold out the **last** item by `timestamp` as the **anchor** query; the remaining items are positives.
* Map `item_id → row index` using `item_ids.npy`.
* Query ANN to get top‑K candidates; compute metrics vs. the **positives** of that user.

It also computes **ANN agreement** with an exact `IndexFlatL2` (optional to report), plus **coverage**, **Gini exposure**, and **long‑tail uplift** (tail defined from **training** popularity distribution).

**Metrics reported (means over queries):**

* `ndcg_at_k_mean`, `recall_at_topk_mean`, `hr_at_k_mean`, `precision_at_k_mean`, `map_at_k_mean`, `mrr_at_k_mean`
* `coverage_at_k` (fraction of catalog ever exposed within top‑k),
* `gini_exposure` (inequality of exposures),
* `long_tail_uplift` (tail share in exposures minus tail share in training popularity),
* `ann_recall_vs_exact_at_k_mean` (agreement; not a user relevance metric).

**Usage (examples):**

```bash
# Flat
python src/eval_end2end.py --interactions data/ml1m.csv \
  --item_vecs data/emb_ml-1m/item_vecs.npy --index data/index_ml-1m_flat \
  --ann_method flat --queries 2000 --topk 100 --metric_topk 10

# HNSW
python src/eval_end2end.py --interactions data/ml20m.csv \
  --item_vecs data/emb_ml-20m/item_vecs.npy --index data/index_ml-20m_hnsw \
  --ann_method hnsw --queries 2000 --topk 100 --metric_topk 10 --ef 128

# IVF-Flat
python src/eval_end2end.py --interactions data/goodbooks.csv \
  --item_vecs data/emb_goodbooks/item_vecs.npy --index data/index_goodbooks_ivfflat \
  --ann_method ivfflat --queries 2000 --topk 100 --metric_topk 10 --nprobe 16
```

> **Implementation notes (boring but important):**
>
> * `_resolve_index_path` and `_prepare_ann` hide index file naming and wrapper extraction (`faiss.extract_index_ivf`) so you can point `--index` at a **directory**.
> * Exact baseline uses `IndexFlatL2` over **the same** item vectors; the **ANN agreement** metric is reported separately from user relevance.
> * Tail mask is computed from **training** interactions (per‑user all‑but‑last), not from test exposures.

---

## 9) Grid runner (multi‑dataset automation)

**Entry point:** `src/run_grid.py` (updated to pass `--interactions`)

Core responsibilities:

* **Prepares** dataset → `data/<dataset>.csv`.
* **Trains** embeddings → `data/emb_<dataset>/{item_vecs,item_ids}.npy`.
* **Builds** each index family in `METHODS = ["flat","hnsw","ivfflat","ivfpq","flatpq"]`.
* **Measures latency** via `run_device.py`.
* **Evaluates end‑to‑end** via `eval_end2end.py` (**now passing `--interactions`**).
* **Aggregates** per‑dataset summaries → `results/summary_<dataset>.csv`, then concatenates → `results/summary_all.csv`.

Run everything (default 128‑dim, 2k queries, topK=100, metric@10, budget 100 MB):

```bash
python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 100
```

Subset datasets:

```bash
python src/run_grid.py --datasets ml-1m goodbooks --queries 2000 --topk 100 --metric_topk 10
```

> **Note:** `run_device.py` writes `latency_{method}.json` into each index directory.
> For end‑to‑end results, use the updated `eval_end2end.py`; if you maintain your own variant, ensure it either (a) writes `end2end_{method}.json` into the index directory **or** (b) you capture stdout in `run_grid.py`. The provided `run_grid.py` expects files as in your latest working runs.

---

## 10) Ablations & sweeps (robustness)

**Recommended minimal sweep** (for the paper):

* HNSW: `ef ∈ {64, 128, 256}`
* IVF‑Flat: `nprobe ∈ {4, 16, 32}`
* IVF‑PQ: OPQ toggle (`--opq` on/off) at `nprobe=16`

A ready‑to‑run **PowerShell** script (`ablation.ps1`) is provided separately in our conversation. It builds once per family, varies **query‑time knobs**, captures latency & end‑to‑end JSON, and writes `results/ablations/ablation_<dataset>.csv`.

---

## 11) Figures

**Entry point:** `src/figures.py`

* Consumes `results/summary_all.csv`
* Creates: `results/figures/coverage_vs_ndcg.png`, `latency_vs_ndcg.png`, `recall_vs_latency.png`
* Use these in your paper (filenames/descriptions were already aligned with your templates).

Run:

```bash
python src/figures.py --summary results/summary_all.csv --out_dir results/figures
```

---

## 12) Metrics (definitions)

Let `ranked[0..K-1]` be top‑K ANN results for a query, `positives` be the user’s training positives (from LOO), `k = metric_topk`.

* **Recall@k** = |ranked[:k] ∩ positives[:k]| / min(k, |positives|)
* **Precision@k** = |ranked[:k] ∩ positives[:k]| / k
* **HR@k** = 1 if any positive in ranked[:k], else 0
* **AP@k / MAP@k** = average precision up to rank k; mean over queries
* **MRR@k** = reciprocal rank of first hit in top‑k
* **NDCG@k** with binary gains: `g_i∈{0,1}` and log discount, normalized by ideal DCG
* **Coverage@k** = fraction of items with non‑zero exposure within top‑k across all queries
* **Gini exposure** = Lorenz‑based inequality of exposure counts
* **Long‑tail uplift** = (tail share in exposures) − (tail share in training popularity),
  where **tail** = items with popularity ≤ 20th percentile among non‑zero‑pop items.

**ANN agreement** (diagnostic only):

* **ann_recall_vs_exact_at_k** = |ANN[:k] ∩ Exact[:k]| / min(k, |Exact[:k]|)

---

## 13) Reproducibility & environment

* **Python:** 3.10–3.11
* **Core deps:** `numpy`, `pandas`, `faiss-cpu`, `psutil`, `matplotlib`
* **Windows:** HNSW uses FAISS, so no local C++ build tools are required.
* **Virtual env:**

  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install -r requirements.txt  # or pip install numpy pandas faiss-cpu psutil matplotlib
  ```

**Performance knobs:**

* Set `OMP_NUM_THREADS` if you want to control FAISS threading.
* Run on consistent hardware for fair latency comparisons.
* For very large datasets (Amazon‑Books), ensure enough RAM for embeddings and index builds (HNSW graphs can exceed your `--budget_mb` if M is too high or if you change defaults).

---

## 14) Extending the project

### Add a new dataset

1. Implement `src/datasets/<name>.py` with a `prepare_<name>(out_csv)` function that writes the canonical schema.
2. Register it in `prepare_dataset.py`.
3. Add an entry to `DATASETS` in `src/run_grid.py`:

   ```python
   "new-ds": {"prepare": ["python","src/prepare_dataset.py","--dataset","new-ds","--out","data/newds.csv"],
              "csv":"data/newds.csv"},
   ```

### Add a new ANN method

1. Extend `src/build_index.py` with a new `--method`.
2. Update `src/run_device.py`:

   * Recognize the artifact (file naming).
   * Return `(method, search_fn)` in `_prepare_ann(...)`.
3. Add the method name to `METHODS` in `src/run_grid.py`.

---

## 15) Troubleshooting (common failures)

* **`FileNotFoundError: ratings.csv` (Goodbooks):**
  Use the robust `goodbooks.py` that searches common paths or downloads the CSV. Verify the path:
  `data/goodbooks-10k/ratings.csv` or run the downloader variant.

* **`eval_end2end.py: error: --interactions required`:**
  You’re using the updated evaluator; ensure `run_grid.py` passes `--interactions <csv>` (the provided `run_grid.py` does).

* **`faiss::FileIOReader ... could not open ... faiss_ivfpq.index`**
  Point `--index` to the **directory**, not an assumed filename. `_resolve_index_path` will find `faiss_ivfpq.index`. Rebuild if the file is missing.

* **HNSW memory too large** on huge datasets:
  Reduce `--M` or construction `--efc`, or lower `--dim` during embedding to reduce item vector memory.

---

## 16) What gets written where (artifact map)

* **Embeddings:** `data/emb_<dataset>/{item_vecs.npy, item_ids.npy}`
* **Indices:** `data/index_<dataset>_<method>/*`
* **Latency JSON (device):** `data/index_<dataset>_<method>/latency_<method>.json`
* **(If enabled) End‑to‑end JSON:** `data/index_<dataset>_<method>/end2end_<method>.json`
* **Per‑dataset summary:** `results/summary_<dataset>.csv`
* **All datasets summary:** `results/summary_all.csv`
* **Ablations:** `results/ablations/ablation_<dataset>.csv`
* **Figures:** `results/figures/*.png`

---

## 17) Command cheatsheet

**Full grid (all datasets, default settings)**

```bash
python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 100
```

**Single dataset only**

```bash
python src/run_grid.py --datasets ml-1m --queries 2000 --topk 100 --metric_topk 10
```

**End‑to‑end (one index)**

```bash
python src/eval_end2end.py --interactions data/ml1m.csv \
  --item_vecs data/emb_ml-1m/item_vecs.npy --index data/index_ml-1m_ivfflat \
  --ann_method ivfflat --queries 2000 --topk 100 --metric_topk 10 --nprobe 16
```

**Device latency (one index)**

```bash
python src/run_device.py --index data/index_ml-20m_hnsw \
  --item_vecs data/emb_ml-20m/item_vecs.npy --queries 2000 --topk 100 --ef 128
```

**Figures**

```bash
python src/figures.py --summary results/summary_all.csv --out_dir results/figures
```

**Ablation sweep (PowerShell, example)**

```powershell
.\ablation.ps1 -Dataset ml-20m -Queries 2000 -TopK 100 -MetricTopK 10 -BudgetMB 100
```

---

## 18) Design decisions (for maintainers)

* **Directory‑first index resolution** makes scripts robust to file naming variations and FAISS pretransform wrappers.
* **User‑relevance first**: end‑to‑end LOO metrics are primary; ANN agreement is reported but never used as a surrogate for user relevance.
* **Budget awareness**: building defaults aim to keep within `--budget_mb`; adjust `nlist`, (M,efC), (m,bits) per dataset.
* **Separation of concerns**: each stage is independently runnable and emits artifacts other stages can consume (good for ablations).

---

## 19) Known limitations & easy next steps

* **Embedding simplicity**: You can drop in stronger item encoders if desired; the indexing harness won’t change.
* **OPQ heuristics**: On very small catalogs, OPQ is auto‑disabled to avoid unstable k‑means training; use `--force_opq` if you want to study it explicitly.
* **Large‑scale memory**: For very large catalogs, you may want GPU FAISS or product‑quantized residual pipelines; add them as new methods.

---

## 20) Maintainer quick start (developer cadence)

1. Verify environment (venv, FAISS/HNSW available).
2. `python src/run_grid.py --datasets ml-1m ml-20m goodbooks amazon-books`
3. Inspect `results/summary_all.csv`, run `python src/figures.py ...`.
4. Optional: `.\ablation.ps1 -Dataset ml-20m ...`
5. Commit: README + results CSVs + figures.

---

### Appendix A — Internals by file (names only)

* **`prepare_dataset.py`** → calls dataset‑specific `prepare_*` to write canonical CSV.
* **`train_embeddings.py`** → reads CSV, produces `item_vecs.npy`, `item_ids.npy`.
* **`build_index.py`** → `build_flat`, `build_hnsw`, `build_ivfflat`, `build_ivfpq`, `build_flatpq`.
* **`run_device.py`** → `_resolve_index_path`, `_prepare_ann`, latency percentiles, `rss_mb`.
* **`eval_end2end.py`** → `_load_id_map`, `_make_loo_queries`, `_prepare_ann`, metrics implementations.
* **`run_grid.py`** → `DATASETS`, `METHODS`, orchestration + CSV aggregation.
* **`figures.py`** → reads `summary_all.csv`, produces standard plots.

---

### Appendix B — Metric formulas (explicit)

* `DCG@k = Σ_{i=1..k} (2^{rel_i} − 1) / log2(i+1)`, `NDCG@k = DCG@k / IDCG@k` with binary `rel_i`.
* `Precision@k = |hits| / k`; `Recall@k = |hits| / min(k, |positives|)`.
* `HR@k = 1[|hits| > 0]`.
* `AP@k = (1/k) Σ_{i∈hits} Precision@i`; `MAP@k` = mean over queries.
* `MRR@k = 1 / rank(first_hit)` or 0 if none ≤ k.
* **Gini exposure** on exposure counts `x` uses Lorenz curve formula:
  `G = (n + 1 − 2 * Σ cumsum(x) / Σ x) / n`.

---

