# Lightweight Recommendation Engine Demo

This project builds a lightweight, presentation-ready recommender system with measurable analytics.

## Peer-Reviewed Basis (Current)

Li, Y., Liu, K., Satapathy, R., Wang, S., & Cambria, E. (2024). Recent developments in recommender systems: A survey. *IEEE Computational Intelligence Magazine, 19*(2), 78-95. https://doi.org/10.1109/MCI.2024.3363984

Implementation concept used here:
- lightweight hybrid recommendation (collaborative + content signals),
- multi-metric evaluation (not only accuracy),
- diversity-aware reranking for responsible recommendations.

## What This Demo Includes

- Baseline 1: popularity recommender
- Baseline 2: item-based collaborative filtering
- Proposed model: `hybrid_lite` (CF + content + diversity rerank)
- Evaluation metrics: Precision@10, Recall@10, NDCG@10, Coverage@10, Intra-list Diversity, Novelty, Latency
- Interactive system demo via Streamlit

## Dataset

- MovieLens Latest Small (downloaded automatically)
- Source: GroupLens (public benchmark)

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python run_demo.py
streamlit run app.py
```

## Outputs

Generated under `outputs/`:
- `metrics.csv`
- `metrics.json`
- `summary.json`
- `metric_overview.png`
- `accuracy_diversity_tradeoff.png`
- `artifacts.pkl`

## Suggested Slide Integration

- Replace dense text with 1 metrics table + 2 charts from `outputs/`.
- Emphasize business-aligned KPIs:
  - ranking quality (NDCG@10),
  - recommendation breadth (Coverage@10),
  - user experience safeguards (Diversity, Novelty),
  - practicality (Latency).
