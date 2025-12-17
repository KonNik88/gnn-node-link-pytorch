# Graph Neural Networks: from Demo Graphs to a Real Recommender System
![License](https://img.shields.io/github/license/KonNik88/gnn-node-link-pytorch)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-green)

This repository contains **two tiers** of work with Graph Neural Networks (PyTorch Geometric):

1) **Demo / Sandbox tier (Planetoid: Cora/PubMed)**  
   Small graphs used to validate training loops, metrics, and core GNN workflow.

2) **Real-world tier: Graph-based Recommender System (Goodbooks-10k)**  
   A recommendation pipeline on a large user–item graph with **leave-one-out (LOO)** evaluation, sampling-based training, and architecture benchmarks.

The demo tier is useful as a minimal reference. The main value of the repo is the **real-world recommender system** in `graph_recsys/`.

---

## Repository layout

```text
.
├─ notebooks/                 # demo notebooks (Planetoid sandbox)
├─ data/                      # demo data (optional)
├─ artifacts/                 # demo outputs (optional)
├─ archive/                   # old / deprecated files (not tracked)
├─ graph_recsys/              # ✅ main project: book recommender (Goodbooks-10k)
│  ├─ notebooks/              # experiments: Graph3, sampling models, eval
│  ├─ data_raw/               # raw Goodbooks-10k files (local)
│  ├─ data_processed/         # processed splits, mappings, graph bundles (local)
│  └─ artifacts/              # runs, histories, checkpoints, summaries (local)
└─ README.md
```

> Note: this repo is intentionally **not package-first**. It is a **research notebook workflow** with artifacts saved to disk.

---

## Tier 1 — Demo / Sandbox (Planetoid)

### Goals
- Validate PyG setup and core GNN workflow
- Node classification and basic link prediction on small citation graphs
- Quick sanity checks before scaling to a real graph

This tier lives in the repo root (`notebooks/`, `data/`, `artifacts/`).

---

## Tier 2 — Real-world project: `graph_recsys/` (Goodbooks-10k)

### Problem
Build and compare GNN recommenders for **book recommendation** on Goodbooks-10k.

### Graph (Graph3 reference)
We use an augmented graph where core signal is:
- **user–book interactions** (train only)

and we add book metadata relations:
- book–tag
- book–author
- book–language
- book–year_bin
- optional book–book similarity edges (e.g., TF-IDF cosine)

Graph3 is treated as a **reference pipeline**: same splits, same evaluation protocol, fair comparison.

### Evaluation
- **Leave-one-out (LOO)** split (one positive per user for validation and one for test)
- Candidate ranking with:
  - C = 1000 (10k users sampled)
  - C = 2000 (10k users sampled)
- Metrics:
  - Hit@10/20/50
  - NDCG@10/20/50

### Models benchmarked (this cycle)
- GraphSAGE + neighbor sampling + BPR
- GAT + neighbor sampling + BPR
- TransformerConv + neighbor sampling + BPR
- PinSAGE-style random-walk sampling + BPR
- R-GCN (tested as a negative baseline due to objective mismatch)

Key research finding:
> Ranking objective alignment (BPR) + sampling strategy matters more than architectural complexity alone.

---

## Results snapshot
Best models achieved approximately:
- **NDCG@10 ~ 0.04** (candidate ranking, C=1000, 10k users sampled)

Exact results and discussion are in the final report notebook.

---

## How to run
This repo is notebook-driven.

1) Open notebooks:
- `notebooks/` (demo tier)
- `graph_recsys/notebooks/` (real-world tier)

2) In each notebook, update `PROJECT_ROOT` / `BUNDLE_DIR` in the first cell if needed.

---

## Final report
The project conclusion and research summary:
- `graph_recsys/notebooks/09_final_eval_and_report.ipynb`

It contains:
- unified comparison table (manual, research-grade)
- key findings and failure modes
- limitations and future work roadmap

---

## Future work (recorded for v2)
- relation ablation study (which edges help vs add noise)
- true heterogeneous GNNs: HeteroConv / HGT
- text-augmented node features for books (SBERT/MiniLM embeddings)
- self-supervised pretraining (masked edges / contrastive)

---

## Comparison with a Hybrid Recommender System
In previous work, we built a production-oriented hybrid recommender system (ALS + SBERT + CatBoost).
It differs fundamentally from the GNN-based models explored here:

| Aspect | Hybrid Recommender | GNN-based Recommender |
|------|-------------------|-----------------------|
| Core idea | Feature-based ranking | Representation learning |
| Candidate generation | ALS + vector search | Implicit via graph |
| Features | Explicit (TF-IDF, SBERT, metadata) | Structural only (this project) |
| Model | CatBoost (tree ensemble) | GraphSAGE / GAT / PinSAGE |
| Training objective | Classification / ranking | Pairwise ranking (BPR) |
| Cold start | Strong (content features) | Weak without text features |
| Scalability | High (modular) | High with sampling |
| Interpretability | High | Low–medium |

Absolute metric values are not directly comparable due to different candidate pools and feature availability.
However, both systems operate in a candidate-ranking setting:
- the hybrid system is expected to achieve higher ranking quality due to explicit content features and a supervised ranker,
- the GNN system demonstrates competitive ranking using only interaction structure and graph connectivity.

Hybrid systems win today through feature richness and engineering.
GNN-based systems win through representation learning and long-term extensibility.

---

## License
MIT — see [LICENSE](LICENSE)
