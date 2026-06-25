# Netflix AI Recommender Case Study

[![CI](https://github.com/jkinarthur/netflix-ai-recommender-case-study/actions/workflows/ci.yml/badge.svg)](https://github.com/jkinarthur/netflix-ai-recommender-case-study/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-quality educational repository for a Netflix-style recommendation system case study, built with strong analytics framing, reproducible evaluation, and presentation-ready outputs.

## Why this repository exists

This project demonstrates how to connect recommender-system engineering to business value and decision quality, not just offline ranking scores.

It was designed around a peer-reviewed foundation and upgraded with:

- hybrid recommendation architecture,
- explicit cold-start handling,
- diversity-aware reranking,
- PSO-based parameter optimization,
- multi-metric analytics and executive-level visual reporting.

## Research basis

Li, Y., Liu, K., Satapathy, R., Wang, S., & Cambria, E. (2024). Recent developments in recommender systems: A survey. IEEE Computational Intelligence Magazine, 19(2), 78-95. https://doi.org/10.1109/MCI.2024.3363984

## Core methods implemented

- Popularity baseline
- Item-based collaborative filtering (cosine similarity)
- Content-based scoring (TF-IDF genre representation)
- Hybrid Lite ranker (CF + content fusion)
- Cold-start adaptation (dynamic weighting + popularity prior)
- MMR-style diversity reranking
- Particle Swarm Optimization (PSO) for key hyperparameters

## Evaluation metrics

- Precision@10
- Recall@10
- NDCG@10
- Coverage@10
- Intra-list Diversity
- Novelty
- Inference Latency (ms)

## Project structure

```text
.
|-- app.py                         # Streamlit application
|-- run_demo.py                    # End-to-end pipeline entrypoint
|-- regenerate_charts.py           # Deterministic chart generation from metrics
|-- src/
|   `-- recommender_pipeline.py    # Recommender methods + evaluation + optimization
|-- outputs/                       # Generated artifacts (ignored in git)
|-- data/                          # Downloaded dataset cache (ignored in git)
|-- build_v4_pptx.py               # Slide build utility
|-- build_v6_logical_flow.py       # Slide flow reordering utility
`-- add_coldstart_pso_slide.py     # Adds cold-start + PSO evidence slide
```

## Quickstart

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run pipeline and generate metrics/charts:

```bash
python run_demo.py
```

3. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Generated artifacts

`outputs/` includes:

- `metrics.csv` and `metrics.json`
- `summary.json` (includes tuned PSO parameters)
- `metric_overview.png` (executive scorecard)
- `accuracy_diversity_tradeoff.png` (Pareto-style analytics view)
- `artifacts.pkl` (serialized runtime artifacts)

## Reproducibility notes

- Dataset is auto-downloaded from GroupLens (MovieLens Latest Small).
- The pipeline uses deterministic seeding for PSO where applicable.
- Generated assets are excluded from git to keep repository clean and code-first.

## Governance

- License: [MIT](LICENSE)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security: [SECURITY.md](SECURITY.md)

## Intended audience

- AI/ML students building case-study quality projects
- Data analytics instructors and reviewers
- Teams needing a concise blueprint for business-aligned recommender demos
