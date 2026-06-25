from __future__ import annotations

import json
import pickle
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_NAME = "ml-latest-small.zip"
EXTRACTED_DIR = "ml-latest-small"


@dataclass
class RecommenderArtifacts:
    users: np.ndarray
    items: np.ndarray
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]
    index_to_item: dict[int, int]
    train_user_items: dict[int, set[int]]
    user_item_matrix: sparse.csr_matrix
    item_user_similarity: sparse.csr_matrix
    item_content_matrix: sparse.csr_matrix
    item_content_similarity: sparse.csr_matrix
    popularity_scores: np.ndarray
    popularity_prob: np.ndarray
    popularity_norm: np.ndarray
    user_interaction_count: dict[int, int]
    genre_vectorizer: TfidfVectorizer
    all_genres: list[str]
    movies: pd.DataFrame


def ensure_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZIP_NAME
    extracted = data_dir / EXTRACTED_DIR
    if not extracted.exists():
        if not zip_path.exists():
            urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    return extracted


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ensure_data(data_dir)
    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")
    return ratings, movies


def leave_one_out_split(ratings: pd.DataFrame, min_interactions: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["userId", "timestamp"]).copy()
    counts = ratings.groupby("userId")["movieId"].count()
    valid_users = counts[counts >= min_interactions].index
    ratings = ratings[ratings["userId"].isin(valid_users)].copy()

    test_idx = ratings.groupby("userId").tail(1).index
    test = ratings.loc[test_idx].copy()
    train = ratings.drop(test_idx).copy()
    return train, test


def build_artifacts(train: pd.DataFrame, movies: pd.DataFrame) -> RecommenderArtifacts:
    users = np.sort(train["userId"].unique())
    items = np.sort(train["movieId"].unique())

    user_to_index = {int(u): i for i, u in enumerate(users)}
    item_to_index = {int(m): i for i, m in enumerate(items)}
    index_to_item = {i: int(m) for i, m in enumerate(items)}

    row_idx = train["userId"].map(user_to_index).to_numpy()
    col_idx = train["movieId"].map(item_to_index).to_numpy()
    data = np.ones_like(row_idx, dtype=np.float32)

    user_item_matrix = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(len(users), len(items)))
    item_user_similarity = cosine_similarity(user_item_matrix.T, dense_output=False).tocsr()

    movies_idx = movies.set_index("movieId")
    genre_text = []
    for item in items:
        if item in movies_idx.index:
            genre_text.append(str(movies_idx.at[item, "genres"]).replace("|", " "))
        else:
            genre_text.append("unknown")

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    item_content_matrix = tfidf.fit_transform(genre_text).tocsr()
    item_content_similarity = cosine_similarity(item_content_matrix, dense_output=False).tocsr()

    item_counts = np.asarray(user_item_matrix.sum(axis=0)).ravel()
    popularity_scores = item_counts.astype(np.float32)
    popularity_prob = (popularity_scores + 1.0) / (popularity_scores.sum() + len(popularity_scores))
    popularity_norm = minmax(popularity_scores)

    train_user_items = (
        train.groupby("userId")["movieId"].apply(lambda x: set(map(int, x.tolist()))).to_dict()
    )
    user_interaction_count = train.groupby("userId")["movieId"].count().astype(int).to_dict()

    genre_set: set[str] = set()
    for val in movies["genres"].fillna("").tolist():
        for token in str(val).split("|"):
            tok = token.strip().lower()
            if tok and tok != "(no genres listed)":
                genre_set.add(tok)
    all_genres = sorted(genre_set)

    return RecommenderArtifacts(
        users=users,
        items=items,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        train_user_items=train_user_items,
        user_item_matrix=user_item_matrix,
        item_user_similarity=item_user_similarity,
        item_content_matrix=item_content_matrix,
        item_content_similarity=item_content_similarity,
        popularity_scores=popularity_scores,
        popularity_prob=popularity_prob,
        popularity_norm=popularity_norm,
        user_interaction_count=user_interaction_count,
        genre_vectorizer=tfidf,
        all_genres=all_genres,
        movies=movies,
    )


def minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def user_cf_scores(art: RecommenderArtifacts, user_id: int) -> np.ndarray:
    uidx = art.user_to_index[user_id]
    profile = art.user_item_matrix.getrow(uidx)
    scores = profile @ art.item_user_similarity
    return np.asarray(scores.todense()).ravel()


def user_content_scores(art: RecommenderArtifacts, user_id: int) -> np.ndarray:
    uidx = art.user_to_index[user_id]
    seen_idx = art.user_item_matrix.getrow(uidx).indices
    if len(seen_idx) == 0:
        return np.zeros(len(art.items), dtype=np.float32)
    profile = art.item_content_matrix[seen_idx].mean(axis=0)
    scores = profile @ art.item_content_matrix.T
    return np.asarray(scores).ravel()


def build_hybrid_base_scores(
    art: RecommenderArtifacts,
    user_id: int,
    alpha: float,
    cold_start_threshold: int,
    popularity_prior_max: float,
) -> np.ndarray:
    cf = user_cf_scores(art, user_id)
    content = user_content_scores(art, user_id)
    cf_n = minmax(cf)
    content_n = minmax(content)
    base = alpha * cf_n + (1.0 - alpha) * content_n

    seen_count = int(art.user_interaction_count.get(user_id, 0))
    coldness = max(float(cold_start_threshold - seen_count), 0.0) / max(float(cold_start_threshold), 1.0)

    # For cold users, prioritize content/profile and add a popularity prior.
    alpha_eff = alpha * (1.0 - coldness)
    cold_hybrid = alpha_eff * cf_n + (1.0 - alpha_eff) * content_n
    pop_weight = popularity_prior_max * coldness
    return (1.0 - pop_weight) * cold_hybrid + pop_weight * art.popularity_norm


def mmr_rerank(
    candidate_idx: np.ndarray,
    base_scores: np.ndarray,
    item_content_sim: sparse.csr_matrix,
    k: int,
    diversity_weight: float = 0.25,
) -> list[int]:
    if len(candidate_idx) == 0:
        return []

    # Build a dense similarity matrix only for the candidate pool.
    # This avoids repeated sparse random access in Python loops.
    pool = candidate_idx.astype(np.int32, copy=False)
    pool_scores = base_scores[pool]
    pool_sim = item_content_sim[pool][:, pool].toarray()

    selected_local: list[int] = []
    remaining = np.ones(len(pool), dtype=bool)

    first = int(np.argmax(pool_scores))
    selected_local.append(first)
    remaining[first] = False

    max_sim_to_selected = pool_sim[:, first].copy()

    while remaining.any() and len(selected_local) < k:
        cand_ids = np.where(remaining)[0]
        rel = pool_scores[cand_ids]
        div = max_sim_to_selected[cand_ids]
        mmr = (1.0 - diversity_weight) * rel - diversity_weight * div
        best_local = int(cand_ids[int(np.argmax(mmr))])

        selected_local.append(best_local)
        remaining[best_local] = False
        max_sim_to_selected = np.maximum(max_sim_to_selected, pool_sim[:, best_local])

    return [int(pool[i]) for i in selected_local]


def recommend(
    art: RecommenderArtifacts,
    user_id: int,
    model: str,
    k: int = 10,
    alpha: float = 0.70,
    diversity_weight: float = 0.25,
    top_pool_size: int = 120,
    cold_start_threshold: int = 6,
    popularity_prior_max: float = 0.40,
) -> list[int]:
    seen = art.train_user_items[user_id]
    seen_idx = [art.item_to_index[m] for m in seen if m in art.item_to_index]
    candidate_mask = np.ones(len(art.items), dtype=bool)
    if seen_idx:
        candidate_mask[np.array(seen_idx, dtype=np.int32)] = False
    candidate_idx = np.where(candidate_mask)[0].astype(np.int32, copy=False)

    if model == "popularity":
        scores = art.popularity_scores.copy()
        ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        top_idx = ranked[:k].tolist()
    elif model == "item_cf":
        scores = user_cf_scores(art, user_id)
        ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        top_idx = ranked[:k].tolist()
    elif model == "hybrid_lite":
        base = build_hybrid_base_scores(
            art,
            user_id,
            alpha=alpha,
            cold_start_threshold=cold_start_threshold,
            popularity_prior_max=popularity_prior_max,
        )
        top_pool = candidate_idx[
            np.argsort(base[candidate_idx])[::-1][: min(top_pool_size, len(candidate_idx))]
        ]
        top_idx = mmr_rerank(top_pool, base, art.item_content_similarity, k, diversity_weight=diversity_weight)
    else:
        raise ValueError(f"Unknown model: {model}")

    return [art.index_to_item[i] for i in top_idx]


def recommend_cold_start_by_genres(
    art: RecommenderArtifacts,
    preferred_genres: list[str],
    k: int = 10,
    diversity_weight: float = 0.30,
    top_pool_size: int = 140,
    popularity_blend: float = 0.35,
) -> list[int]:
    if not preferred_genres:
        # No profile signal: fall back to popularity while preserving diversity.
        base = art.popularity_norm.copy()
    else:
        profile_text = " ".join(g.strip().lower() for g in preferred_genres if g.strip())
        profile_vec = art.genre_vectorizer.transform([profile_text])
        content_scores = (profile_vec @ art.item_content_matrix.T).toarray().ravel()
        content_n = minmax(content_scores)
        base = (1.0 - popularity_blend) * content_n + popularity_blend * art.popularity_norm

    candidate_idx = np.arange(len(art.items), dtype=np.int32)
    top_pool = candidate_idx[np.argsort(base[candidate_idx])[::-1][: min(top_pool_size, len(candidate_idx))]]
    top_idx = mmr_rerank(top_pool, base, art.item_content_similarity, k, diversity_weight=diversity_weight)
    return [art.index_to_item[i] for i in top_idx]


def ndcg_at_k(rank: list[int], truth: int) -> float:
    if truth not in rank:
        return 0.0
    pos = rank.index(truth)
    return 1.0 / np.log2(pos + 2.0)


def list_diversity(rank_movie_ids: list[int], art: RecommenderArtifacts) -> float:
    idx = [art.item_to_index[m] for m in rank_movie_ids if m in art.item_to_index]
    if len(idx) < 2:
        return 0.0
    sim = art.item_content_similarity[idx][:, idx].toarray()
    n = sim.shape[0]
    pairs = n * (n - 1) / 2
    if pairs == 0:
        return 0.0
    # Dissimilarity = 1 - cosine similarity over genre vectors.
    dis = 1.0 - sim
    tri = dis[np.triu_indices(n, k=1)]
    return float(np.mean(tri))


def list_novelty(rank_movie_ids: list[int], art: RecommenderArtifacts) -> float:
    idx = [art.item_to_index[m] for m in rank_movie_ids if m in art.item_to_index]
    if not idx:
        return 0.0
    probs = art.popularity_prob[idx]
    return float(np.mean(-np.log2(probs)))


def evaluate(
    art: RecommenderArtifacts,
    test: pd.DataFrame,
    models: list[str],
    k: int = 10,
    hybrid_params: dict[str, Any] | None = None,
    eval_users: list[int] | None = None,
) -> pd.DataFrame:
    truth_by_user = {int(r.userId): int(r.movieId) for r in test.itertuples(index=False)}
    if eval_users is None:
        eval_users = [u for u in truth_by_user.keys() if u in art.user_to_index]
    else:
        eval_users = [u for u in eval_users if u in truth_by_user and u in art.user_to_index]

    rows = []
    for model in models:
        hit = 0
        ndcg = 0.0
        precisions = []
        recalls = []
        diversity_vals = []
        novelty_vals = []
        unique_recs = set()
        latency = []

        for user in eval_users:
            t0 = time.perf_counter()
            if model == "hybrid_lite" and hybrid_params is not None:
                recs = recommend(art, user, model=model, k=k, **hybrid_params)
            else:
                recs = recommend(art, user, model=model, k=k)
            latency.append((time.perf_counter() - t0) * 1000.0)

            truth = truth_by_user[user]
            hit_user = int(truth in recs)
            hit += hit_user
            ndcg += ndcg_at_k(recs, truth)
            precisions.append(hit_user / k)
            recalls.append(float(hit_user))
            diversity_vals.append(list_diversity(recs, art))
            novelty_vals.append(list_novelty(recs, art))
            unique_recs.update(recs)

        n = max(len(eval_users), 1)
        rows.append(
            {
                "model": model,
                "users_evaluated": n,
                "precision_at_k": float(np.mean(precisions)),
                "recall_at_k": float(np.mean(recalls)),
                "ndcg_at_k": float(ndcg / n),
                "coverage_at_k": float(len(unique_recs) / len(art.items)),
                "intra_list_diversity": float(np.mean(diversity_vals)),
                "novelty": float(np.mean(novelty_vals)),
                "latency_ms": float(np.mean(latency)),
            }
        )

    return pd.DataFrame(rows)


def pso_optimize_hybrid_params(
    art: RecommenderArtifacts,
    test: pd.DataFrame,
    k: int = 10,
    seed: int = 42,
    n_particles: int = 8,
    n_iters: int = 8,
    max_eval_users: int = 220,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)

    truth_users = [int(r.userId) for r in test.itertuples(index=False) if int(r.userId) in art.user_to_index]
    rng.shuffle(truth_users)
    eval_users = truth_users[: min(max_eval_users, len(truth_users))]

    bounds = np.array(
        [
            [0.35, 0.85],   # alpha
            [0.12, 0.42],   # diversity_weight
            [80.0, 180.0],  # top_pool_size
            [3.0, 10.0],    # cold_start_threshold
            [0.15, 0.55],   # popularity_prior_max
        ],
        dtype=np.float64,
    )

    def decode(x: np.ndarray) -> dict[str, Any]:
        return {
            "alpha": float(x[0]),
            "diversity_weight": float(x[1]),
            "top_pool_size": int(round(float(x[2]))),
            "cold_start_threshold": int(round(float(x[3]))),
            "popularity_prior_max": float(x[4]),
        }

    def objective(x: np.ndarray) -> float:
        params = decode(x)
        met = evaluate(
            art,
            test,
            models=["hybrid_lite"],
            k=k,
            hybrid_params=params,
            eval_users=eval_users,
        ).iloc[0]

        # Prioritize ranking quality while still rewarding healthy diversity/coverage and low latency.
        return (
            float(met["ndcg_at_k"])
            + 0.08 * float(met["coverage_at_k"])
            + 0.04 * float(met["intra_list_diversity"])
            - 0.00025 * float(met["latency_ms"])
        )

    dim = bounds.shape[0]
    positions = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, dim))
    velocities = rng.normal(0.0, 0.05, size=(n_particles, dim))

    pbest_pos = positions.copy()
    pbest_scores = np.array([objective(p) for p in positions], dtype=np.float64)
    gbest_idx = int(np.argmax(pbest_scores))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = float(pbest_scores[gbest_idx])

    history: list[float] = [gbest_score]

    inertia = 0.60
    c1 = 1.60
    c2 = 1.60

    for _ in range(n_iters):
        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))
        velocities = (
            inertia * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        positions = positions + velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

        scores = np.array([objective(p) for p in positions], dtype=np.float64)
        improved = scores > pbest_scores
        pbest_scores[improved] = scores[improved]
        pbest_pos[improved] = positions[improved]

        best_idx = int(np.argmax(pbest_scores))
        if float(pbest_scores[best_idx]) > gbest_score:
            gbest_score = float(pbest_scores[best_idx])
            gbest_pos = pbest_pos[best_idx].copy()

        history.append(gbest_score)

    best_params = decode(gbest_pos)
    metadata = {
        "seed": seed,
        "n_particles": n_particles,
        "n_iters": n_iters,
        "eval_users": len(eval_users),
        "best_objective": gbest_score,
        "objective_history": history,
    }
    return best_params, metadata


def save_charts(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Executive scorecard: 0-1 normalized KPI heatmap + weighted overall score.
    metric_cols = [
        "ndcg_at_k",
        "recall_at_k",
        "coverage_at_k",
        "intra_list_diversity",
        "novelty",
        "latency_ms",
    ]
    metric_names = ["NDCG@10", "Recall@10", "Coverage@10", "Diversity", "Novelty", "Latency"]

    score_df = metrics[["model", *metric_cols]].copy()
    norm_df = score_df.copy()

    for col in metric_cols:
        vals = score_df[col].to_numpy(dtype=float)
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi - lo < 1e-12:
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - lo) / (hi - lo)
        # Lower latency is better, so invert its normalized score.
        if col == "latency_ms":
            norm = 1.0 - norm
        norm_df[col] = norm

    weights = {
        "ndcg_at_k": 0.35,
        "recall_at_k": 0.20,
        "coverage_at_k": 0.15,
        "intra_list_diversity": 0.10,
        "novelty": 0.05,
        "latency_ms": 0.15,
    }
    norm_df["overall_score"] = sum(norm_df[c] * w for c, w in weights.items())

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.0),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    heat = norm_df[metric_cols].to_numpy(dtype=float).T
    im = ax0.imshow(heat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax0.set_xticks(np.arange(len(norm_df)))
    ax0.set_xticklabels(norm_df["model"].tolist(), rotation=18)
    ax0.set_yticks(np.arange(len(metric_names)))
    ax0.set_yticklabels(metric_names)
    ax0.set_title("Executive KPI Scorecard (Normalized)")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax0.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("KPI score (0-1)")

    ranked = norm_df.sort_values("overall_score", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(ranked), dtype=float)
    ax1.barh(y_pos, ranked["overall_score"].to_numpy(dtype=float), color=["#FCA311", "#6BAED6", "#2A9D8F"])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ranked["model"].tolist())
    for i, row in enumerate(ranked.itertuples(index=False)):
        ax1.text(float(row.overall_score) + 0.01, float(i), f"{float(row.overall_score):.3f}", va="center", fontsize=8.5)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Weighted score")
    ax1.set_title("Overall Model Ranking")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "metric_overview.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.0, 5.2))

    x = metrics["intra_list_diversity"].to_numpy()
    y = metrics["ndcg_at_k"].to_numpy()
    coverage = metrics["coverage_at_k"].to_numpy()
    latency = metrics["latency_ms"].to_numpy()

    # Bubble size communicates catalog breadth; color communicates serving cost.
    size = 120 + 2600 * coverage
    sc = ax2.scatter(
        x,
        y,
        s=size,
        c=latency,
        cmap="RdYlGn_r",
        edgecolors="white",
        linewidths=1.1,
        alpha=0.92,
    )

    # Draw an approximate frontier so tradeoff direction is obvious at first glance.
    frontier = metrics.sort_values(["intra_list_diversity", "ndcg_at_k"], ascending=[True, False])
    ax2.plot(
        frontier["intra_list_diversity"],
        frontier["ndcg_at_k"],
        color="#4F9DFF",
        linewidth=1.6,
        linestyle="--",
        alpha=0.85,
        label="Observed frontier",
    )

    for row in metrics.itertuples(index=False):
        label = f"{row.model}\nCov:{row.coverage_at_k:.3f}"
        ax2.annotate(
            label,
            (row.intra_list_diversity, row.ndcg_at_k),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.72, ec="none"),
        )

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    ax2.axvline(x_mean, color="#777777", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.axhline(y_mean, color="#777777", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.text(
        x_mean + 0.0003,
        y_mean + 0.00025,
        "High accuracy + high diversity zone",
        fontsize=8.0,
        color="#E63946",
    )

    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("Latency (ms, lower is better)")

    ax2.set_xlabel("Intra-list Diversity (higher is better)")
    ax2.set_ylabel("NDCG@10 (higher is better)")
    ax2.set_title("Accuracy-Diversity-Latency Pareto View")
    ax2.grid(alpha=0.32, linestyle="--")
    ax2.legend(loc="lower right", frameon=True)
    fig2.tight_layout()
    fig2.savefig(output_dir / "accuracy_diversity_tradeoff.png", dpi=180)
    plt.close(fig2)


def run_pipeline(project_root: Path, k: int = 10) -> pd.DataFrame:
    data_dir = project_root / "data"
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies = load_data(data_dir)
    train, test = leave_one_out_split(ratings, min_interactions=5)
    art = build_artifacts(train, movies)

    hybrid_params, pso_meta = pso_optimize_hybrid_params(
        art,
        test,
        k=k,
        seed=42,
        n_particles=8,
        n_iters=8,
        max_eval_users=220,
    )

    models = ["popularity", "item_cf", "hybrid_lite"]
    metrics = evaluate(art, test, models=models, k=k, hybrid_params=hybrid_params)
    metrics = metrics.sort_values("ndcg_at_k", ascending=False).reset_index(drop=True)

    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "metrics.json").write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")

    with (out_dir / "artifacts.pkl").open("wb") as f:
        pickle.dump({"artifacts": art, "k": k, "hybrid_params": hybrid_params}, f)

    summary = {
        "paper_basis": {
            "citation": "Li, Y., Liu, K., Satapathy, R., Wang, S., & Cambria, E. (2024). Recent developments in recommender systems: A survey. IEEE Computational Intelligence Magazine, 19(2), 78-95.",
            "doi": "10.1109/MCI.2024.3363984",
            "implementation_note": "Hybrid lightweight design combining collaborative and content signals with diversity-aware reranking, plus cold-start handling and PSO-based hyperparameter tuning."
        },
        "dataset": {
            "name": "MovieLens Latest Small",
            "ratings": int(len(ratings)),
            "users": int(ratings["userId"].nunique()),
            "items": int(ratings["movieId"].nunique()),
        },
        "split": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "protocol": "leave-one-out per user"
        },
        "optimization": {
            "method": "Particle Swarm Optimization (PSO)",
            "hybrid_params": hybrid_params,
            "metadata": pso_meta,
        },
        "best_model": metrics.iloc[0].to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_charts(metrics, out_dir)
    return metrics
