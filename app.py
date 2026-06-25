from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender_pipeline import recommend, recommend_cold_start_by_genres

st.set_page_config(page_title="Netflix-Style Recommender Demo", layout="wide")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ART_PATH = OUT / "artifacts.pkl"
METRICS_PATH = OUT / "metrics.csv"

st.title("Lightweight Recommender Demo (Paper-based)")
st.caption(
    "Based on Li et al. (2024), IEEE Computational Intelligence Magazine, DOI: 10.1109/MCI.2024.3363984"
)

if not ART_PATH.exists() or not METRICS_PATH.exists():
    st.warning("Artifacts missing. Run: python run_demo.py")
    st.stop()

with ART_PATH.open("rb") as f:
    bundle = pickle.load(f)

art = bundle["artifacts"]
hybrid_params = bundle.get("hybrid_params", {})
metrics = pd.read_csv(METRICS_PATH)

movies = art.movies[["movieId", "title", "genres"]].copy()
movies_lookup = movies.set_index("movieId")

st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&display=swap');

        .stApp {
            background: radial-gradient(1100px 650px at 20% -10%, #2a0005 0%, #0c0c0f 50%, #08080b 100%);
            color: #f5f5f7;
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.8px;
        }

        .hero {
            border: 1px solid rgba(229, 9, 20, 0.45);
            background: linear-gradient(140deg, rgba(229,9,20,0.22), rgba(18,18,24,0.8));
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 10px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.35);
        }

        .kpi-card {
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }

        .kpi-label {
            color: #b9bac7;
            font-size: 0.82rem;
            margin-bottom: 2px;
        }

        .kpi-value {
            color: #ffffff;
            font-size: 1.28rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .accent {
            color: #e50914;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
)

st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0; color:#ffffff;">Netflix-Themed Recommender Lab</h1>
            <div style="color:#cfd3df; margin-top:4px;">
                Cold-start-first hybrid ranking with diversity-aware reranking and PSO-tuned parameters.
            </div>
            <div style="color:#8f95a6; font-size:0.86rem; margin-top:6px;">
                Paper basis: Li et al. (2024), IEEE Computational Intelligence Magazine, DOI: 10.1109/MCI.2024.3363984
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1.45])

with left:
    st.subheader("Demo Controls")
    persona = st.radio(
        "Persona",
        options=["Existing user", "Cold-start user"],
        index=1,
        horizontal=True,
        help="Cold-start user has no history. Existing user has watch history.",
    )

    model = st.selectbox("Model", options=["hybrid_lite", "item_cf", "popularity"], index=0)
    top_k = st.slider("Top-K", min_value=5, max_value=20, value=10, step=1)

    if persona == "Existing user":
        user_id = st.selectbox("User ID", options=list(map(int, art.users.tolist())), index=0)
        recs = recommend(
            art,
            int(user_id),
            model=model,
            k=int(top_k),
            **(hybrid_params if model == "hybrid_lite" else {}),
        )
        st.caption(
            f"History size: {art.user_interaction_count.get(int(user_id), 0)} interactions. "
            "Hybrid automatically increases content/popularity priors for colder profiles."
        )
    else:
        genre_options = art.all_genres if hasattr(art, "all_genres") else []
        preferred = st.multiselect(
            "Preferred genres (for new user profile)",
            options=genre_options,
            default=["action", "adventure"] if "action" in genre_options and "adventure" in genre_options else genre_options[:2],
        )
        # In cold-start mode, emphasize hybrid + genre profile matching.
        recs = recommend_cold_start_by_genres(
            art,
            preferred_genres=preferred,
            k=int(top_k),
            diversity_weight=float(hybrid_params.get("diversity_weight", 0.30)),
            top_pool_size=int(hybrid_params.get("top_pool_size", 140)),
            popularity_blend=float(hybrid_params.get("popularity_prior_max", 0.35)),
        )
        st.caption(
            "No watch history available. Recommendations are built from genre profile + popularity prior + diversity reranking."
        )

    rows = []
    for rank, movie_id in enumerate(recs, start=1):
        title = movies_lookup.at[movie_id, "title"] if movie_id in movies_lookup.index else f"Movie {movie_id}"
        genres = movies_lookup.at[movie_id, "genres"] if movie_id in movies_lookup.index else "Unknown"
        rows.append({"rank": rank, "movieId": movie_id, "title": title, "genres": genres})

    st.subheader("Recommended Titles")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("Evaluation Metrics (Leave-One-Out, K=10)")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    best_ndcg = metrics.sort_values("ndcg_at_k", ascending=False).iloc[0]
    best_cov = metrics.sort_values("coverage_at_k", ascending=False).iloc[0]
    fastest = metrics.sort_values("latency_ms", ascending=True).iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Best NDCG@10</div>
              <div class="kpi-value">{best_ndcg['ndcg_at_k']:.4f}</div>
              <div class="kpi-label"><span class="accent">{best_ndcg['model']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Best Coverage@10</div>
              <div class="kpi-value">{best_cov['coverage_at_k']:.4f}</div>
              <div class="kpi-label"><span class="accent">{best_cov['model']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Fastest Model</div>
              <div class="kpi-value">{fastest['latency_ms']:.3f} ms</div>
              <div class="kpi-label"><span class="accent">{fastest['model']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_chart = OUT / "metric_overview.png"
    tradeoff_chart = OUT / "accuracy_diversity_tradeoff.png"
    if metric_chart.exists():
        st.image(str(metric_chart), caption="Model comparison")
    if tradeoff_chart.exists():
        st.image(str(tradeoff_chart), caption="Accuracy-diversity tradeoff")

st.markdown("---")
st.write(
    "Presentation note: This system now emphasizes cold-start resolution and PSO-tuned hybrid ranking, "
    "with measurable outcomes (accuracy, diversity, coverage, novelty, latency) instead of descriptive-only architecture."
)

if hybrid_params:
    st.caption(
        "PSO-tuned hybrid parameters: "
        f"alpha={hybrid_params.get('alpha', 0):.3f}, "
        f"diversity_weight={hybrid_params.get('diversity_weight', 0):.3f}, "
        f"top_pool_size={hybrid_params.get('top_pool_size', 0)}, "
        f"cold_start_threshold={hybrid_params.get('cold_start_threshold', 0)}, "
        f"popularity_prior_max={hybrid_params.get('popularity_prior_max', 0):.3f}"
    )
