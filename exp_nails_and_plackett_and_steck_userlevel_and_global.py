# %%

from pathlib import Path
from tqdm import tqdm
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from utils._ebrec._constants import *
from utils._ebrec._python import (
    rank_predictions_by_score,
    write_submission_file,
    create_lookup_dict,
)

from utils._python import (
    compute_nails_adjustment_factors,
    compute_normalized_distribution,
    compute_smoothed_kl_divergence,
    compute_nails,
    greedy_steck_rerank,
    softmax,
)

from utils._utils import * 
from arguments.args_nails import args


# %%
# -----------------------------
# Args / paths
# -----------------------------
distribution_type = args.distribution_type          # "Uniform" / "Editorial"
act_func = args.act_func                            # "softmax"
article_selection = args.article_selection          # "stoch" or "der" (used for submission ranking only)
with_replacement = args.with_replacement            # False
lambda_values = args.lambda_values                  # [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
filter_auto = args.filter_auto                      # True
make_submission_file = args.make_submission_file    # True/False

alpha_steck_select = args.alpha_steck_select        # Steck selection alpha
unknown_cat = args.unknown_cat                      # fallback weight (e.g., 1e-6)
alpha = args.alpha                                  # KL smoothing alpha
topN = args.topN                                    # 10
n_samples_test = args.n_samples_test                # optionally subsample behaviors
show_plot = args.show_plot

PROJECT_ROOT = Path(args.project_root)
PATH = Path(args.data_dir)
file = Path(args.score_dir)                         # parquet with scores + inview + impression_id
PATH_EMB = Path(args.emb_dir)

PATH_DUMP = PROJECT_ROOT.joinpath("data", "ebnerd_submissions")
PATH_DUMP.mkdir(parents=True, exist_ok=True)

np.random.seed(123)

PLOT_TAG = "compare_user_paper"
article_selection_name = article_selection
plot_dir = PROJECT_ROOT.joinpath(
    "plot",
    f"nails_{PLOT_TAG}_{distribution_type}_{article_selection_name}",
)
plot_dir.mkdir(parents=True, exist_ok=True)

print(f"Running: {plot_dir} ({distribution_type})")


# %%
# -----------------------------
# Load article metadata + category lookup
# -----------------------------
df_articles = (
    pl.read_parquet(PATH.joinpath("articles.parquet"))
    .join(pl.read_parquet(PATH_EMB), on=DEFAULT_ARTICLE_ID_COL)
    .with_columns(pl.col("category_str"))
)

lookup_cat_str = create_lookup_dict(
    df_articles, key=DEFAULT_ARTICLE_ID_COL, value="category_str"
)

lookup_attr = lookup_cat_str  # dict: article_id -> category_str


# %%
# -----------------------------
# Behaviors (impression_id > 0)  === USER-LEVEL EVAL DATA ===
# -----------------------------
df = pl.scan_parquet(file).filter(pl.col(DEFAULT_IMPRESSION_ID_COL) > 0).collect()
if n_samples_test:
    df = df.sample(n=n_samples_test, seed=123)

# Add per-row category list aligned with DEFAULT_INVIEW_ARTICLES_COL and the score vector
df = df.with_columns(
    pl.col(DEFAULT_INVIEW_ARTICLES_COL)
    .list.eval(pl.element().replace_strict(lookup_cat_str, default="unknown"))
    .alias("inview_categories")
)

U = df.height
print(f"Loaded behaviors rows (users/impressions): U={U}")


# %%
# -----------------------------
# Define target distribution P*(c) and category key-set
# -----------------------------
df_universe = (
    df.select(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL).alias("aid"),
        pl.col("inview_categories").alias("cat"),
    )
    .explode(["aid", "cat"])
)

if filter_auto:
    df_universe = df_universe.filter(pl.col("cat") != "auto")

all_cats = df_universe["cat"].to_list()
unique_cats = sorted(set(all_cats))

if distribution_type == "Uniform":
    p_star_ei = compute_normalized_distribution(set(unique_cats))
elif distribution_type == "Editorial":
    # editorial prior over categories from the evaluation universe
    p_star_ei = compute_normalized_distribution(all_cats)
else:
    raise ValueError(f"{distribution_type} not defined")

# Ensure stable key order + values list
nails_keys = list(p_star_ei.keys())
nails_values = [float(p_star_ei[k]) for k in nails_keys]

if not np.isclose(sum(nails_values), 1.0):
    raise ValueError(f"P*(c) sums to {sum(nails_values)} (expected 1.0)")

print(f"n_categories={len(nails_keys)}  keys={nails_keys}")
print(f"P*(c)={p_star_ei}")


# %%
# -----------------------------
# Helpers (user-level)
# -----------------------------
THRESHOLDS = (1.00, 1.50, 2.00, 2.50, 3.00)


def _normalize_dict(d, keys, eps=1e-12):
    vals = np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float64)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())

    if not np.isfinite(s) or s <= 0.0:
        vals[:] = 1.0 / len(keys)
    else:
        vals /= s

    vals = np.clip(vals, eps, None)
    vals /= float(vals.sum())
    return {k: float(v) for k, v in zip(keys, vals)}


def _kl_from_dist(dist_dict, keys, p_values, alpha, eps=1e-12) -> float:
    q = np.array([float(dist_dict.get(k, 0.0)) for k in keys], dtype=np.float64)
    s = float(q.sum())
    if not np.isfinite(s) or s <= 0.0:
        q = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    else:
        q /= s
    q = np.clip(q, eps, None)
    q /= float(q.sum())
    return float(compute_smoothed_kl_divergence(p=p_values, q=q.tolist(), alpha=alpha))


def user_expected_cat_mass(p_i_u: np.ndarray, cats: np.ndarray, keys: list) -> dict:
    """Mass per category for a single user: sum_i p_i_u for items in category."""
    out = {k: 0.0 for k in keys}
    for c in np.unique(cats):
        if c in out:
            out[c] = float(p_i_u[cats == c].sum())
    return out


def update_count_hist(counts: dict, cat_list: list, keys: list):
    for c in cat_list:
        if c in counts:
            counts[c] += 1


def plackett_luce_permutation(worth: np.ndarray, rng: np.random.Generator, eps: float = 1e-12) -> np.ndarray:
    """
    Sample a permutation according to the Plackett–Luce model (without replacement).

    worth: positive worth parameters (aligned probabilities are fine).
    Returns indices in chosen order (best first).
    """
    worth = np.asarray(worth, dtype=np.float64)
    worth = np.clip(worth, eps, None)

    remaining = np.arange(worth.shape[0])
    perm = np.empty_like(remaining)

    for m in range(worth.shape[0]):
        w = worth[remaining]
        p = w / np.sum(w)
        choice_pos = rng.choice(len(remaining), p=p)
        perm[m] = remaining[choice_pos]
        remaining = np.delete(remaining, choice_pos)

    return perm


def summarize_user_distances(dist_list, thresholds=THRESHOLDS):
    arr = np.asarray(dist_list, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out = {
        "mean": float(arr.mean()) if len(arr) else np.nan,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.quantile(arr, 0.5)) if len(arr) else np.nan,
        "p90": float(np.quantile(arr, 0.9)) if len(arr) else np.nan,
        "p95": float(np.quantile(arr, 0.95)) if len(arr) else np.nan,
    }
    for t in thresholds:
        out[f"pct_le_{t}"] = float((arr <= t).mean()) if len(arr) else np.nan
    return out

# -----------------------------
# NEW helpers: JS divergence + exposure + utility trade-off
# -----------------------------

def _to_prob_array(dist_dict, keys, eps=1e-12) -> np.ndarray:
    q = np.array([float(dist_dict.get(k, 0.0)) for k in keys], dtype=np.float64)
    s = float(q.sum())
    if not np.isfinite(s) or s <= 0.0:
        q = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    else:
        q /= s
    q = np.clip(q, eps, None)
    q /= float(q.sum())
    return q

def compute_js_divergence(p_values: list[float], q_values: np.ndarray, alpha=1e-4, eps=1e-12) -> float:
    """
    Jensen–Shannon divergence using your smoothed KL implementation.
    p_values: target distribution values (list in nails_keys order)
    q_values: candidate distribution as numpy array (already same length)
    """
    p = np.asarray(p_values, dtype=np.float64)
    p = np.clip(p, eps, None)
    p /= float(p.sum())

    q = np.asarray(q_values, dtype=np.float64)
    q = np.clip(q, eps, None)
    q /= float(q.sum())

    m = 0.5 * (p + q)
    # Use the same smoothed KL for consistency with your KL metrics.
    kl_pm = float(compute_smoothed_kl_divergence(p=p.tolist(), q=m.tolist(), alpha=alpha))
    kl_qm = float(compute_smoothed_kl_divergence(p=q.tolist(), q=m.tolist(), alpha=alpha))
    return 0.5 * (kl_pm + kl_qm)

def _js_from_dist(dist_dict, keys, p_values, alpha, eps=1e-12) -> float:
    q = _to_prob_array(dist_dict, keys, eps=eps)
    return float(compute_js_divergence(p_values=p_values, q_values=q, alpha=alpha, eps=eps))

def update_weighted_hist(mass: dict, cat_list: list, weights: np.ndarray, keys: list):
    # cat_list length == len(weights) == topN
    for c, w in zip(cat_list, weights):
        if c in mass:
            mass[c] += float(w)

def score_sum(scores: np.ndarray, idx: np.ndarray) -> float:
    # idx is indices into scores
    if idx.size == 0:
        return 0.0
    return float(np.sum(scores[idx]))

def jaccard_at_n(a_ids: np.ndarray, b_ids: np.ndarray) -> float:
    # a_ids and b_ids are item-id arrays (length N)
    A = set(map(int, a_ids.tolist()))
    B = set(map(int, b_ids.tolist()))
    if not A and not B:
        return 1.0
    return float(len(A & B) / len(A | B))


def plot_topN_histogram_all_methods(
    keys,
    p_star: dict,

    # user-level
    det_user: dict,
    pl_user: dict,
    steck_user: dict,        # NEW
    kl_det_user: float,
    kl_pl_user: float,
    kl_steck_user: float,    # NEW

    # global/paper method
    det_global: dict,
    pl_global: dict,
    kl_det_global: float,
    kl_pl_global: float,

    topN: int,
    outpath,
    title_prefix: str = "",
):
    cats = list(keys)

    tgt = np.array([p_star.get(c, 0.0) for c in cats], dtype=float)

    detU   = np.array([det_user.get(c, 0.0) for c in cats], dtype=float)
    plU    = np.array([pl_user.get(c, 0.0) for c in cats], dtype=float)
    steckU = np.array([steck_user.get(c, 0.0) for c in cats], dtype=float)  # NEW

    detG = np.array([det_global.get(c, 0.0) for c in cats], dtype=float)
    plG  = np.array([pl_global.get(c, 0.0) for c in cats], dtype=float)

    x = np.arange(len(cats))
    width = 0.13  # 6 bars per category

    fig, ax = plt.subplots(figsize=(14, 6))

    # 6 bars: user det, user PL, user STECK, target, global det, global PL
    ax.bar(x - 2.5*width, detU,   width, label=f"User Det Top@{topN}")
    ax.bar(x - 1.5*width, plU,    width, label=f"User PL Top@{topN}")
    ax.bar(x - 0.5*width, steckU, width, label=f"User STECK Top@{topN}")   # NEW
    ax.bar(x + 0.5*width, tgt,    width, label="Target")
    ax.bar(x + 1.5*width, detG,   width, label=f"Global Det Top@{topN}")
    ax.bar(x + 2.5*width, plG,    width, label=f"Global PL Top@{topN}")

    ax.set_ylabel("Probability")
    ax.set_title(
        f"{title_prefix}Top@{topN} category histogram over users.\n"
        f"User: KL(det)={kl_det_user:.4f}, KL(PL)={kl_pl_user:.4f}, KL(STECK)={kl_steck_user:.4f}    "
        f"Global: KL(det)={kl_det_global:.4f}, KL(PL)={kl_pl_global:.4f}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# %%
# -----------------------------
# Coverage universe (unique candidate IDs across evaluation df)
# -----------------------------
UNIVERSE_IDS = set(df_universe["aid"].to_list())
UNIVERSE_SIZE = len(UNIVERSE_IDS)
print(f"Universe size for coverage: {UNIVERSE_SIZE}")

# %% ---------------------------------------------------------
# Precompute GLOBAL induced category distribution P(c) from base probs
# and derive GLOBAL NAILS weights w_global(c) = P*(c) / P(c)
# ------------------------------------------------------------
print("Computing global induced P(c) from base probabilities (for global NAILS)...")

global_base_mass = {k: 0.0 for k in nails_keys}

for row_i in tqdm(df.iter_rows(named=True), total=U, ncols=80):
    cats = np.asarray(row_i["inview_categories"], dtype=object)
    pred_scores = np.asarray(row_i["scores"], dtype=np.float64)

    if filter_auto:
        mask_keep = (cats != "auto")
        cats_f = cats[mask_keep]
        pred_scores_f = pred_scores[mask_keep]
    else:
        cats_f = cats
        pred_scores_f = pred_scores

    if pred_scores_f.size == 0:
        continue

    if act_func == "softmax":
        p_i_u = softmax(pred_scores_f, axis=0)
    else:
        p_i_u = np.clip(pred_scores_f, 0.0, None)
        s = float(p_i_u.sum())
        p_i_u = p_i_u / s if s > 0 else np.full_like(p_i_u, 1.0 / len(p_i_u))

    u_mass = user_expected_cat_mass(p_i_u, cats_f, nails_keys)
    for k in nails_keys:
        global_base_mass[k] += float(u_mass.get(k, 0.0))

# Normalize to P(c)
p_c_global = _normalize_dict(global_base_mass, nails_keys)

# Global weights for "paper/global" NAILS
p_w_global = compute_nails_adjustment_factors(p_star_ei, p_c_global)

print("Done.")
print(f"P(c) global (base-induced): {p_c_global}")
print(f"w_global(c)=P*(c)/P(c): {p_w_global}")


# %%
# -----------------------------
# Runs (only needed for PL sampling variability)
# -----------------------------
runs = 1  # keep 5 by default; set to 1 if you want speed and PL variance is negligible
base_seed = 123

eval_rows = []

for run in tqdm(range(runs), ncols=80):
    rng = np.random.default_rng(base_seed + run)

    for lambda_val in lambda_values:
        # Aggregators across users
        user_kl_expected_base = []
        user_kl_expected_aligned = []
        user_kl_topN_det = []
        user_kl_topN_pl = []
        user_kl_topN_steck = []

        # Global expected mass accumulators
        exp_base_mass_global = {k: 0.0 for k in nails_keys}
        exp_aligned_mass_global = {k: 0.0 for k in nails_keys}

        # Global Top@N histogram accumulators
        topN_det_counts = {k: 0 for k in nails_keys}
        topN_pl_counts = {k: 0 for k in nails_keys}
        topN_steck_counts = {k: 0 for k in nails_keys}

        # Coverage accumulators
        det_rec_ids = []
        pl_rec_ids = []
        steck_rec_ids = []

        # For optional submission writing (aligned score vectors per row)
        aligned_score_rows = []

        # For global NAILS (paper)
        user_kl_expected_aligned_global = []
        user_kl_topN_det_global = []
        user_kl_topN_pl_global = []

        exp_aligned_mass_globalMethod = {k: 0.0 for k in nails_keys}

        topN_det_counts_globalMethod = {k: 0 for k in nails_keys}
        topN_pl_counts_globalMethod = {k: 0 for k in nails_keys}

        det_rec_ids_globalMethod = []
        pl_rec_ids_globalMethod = []

        # -----------------------------
        # Exposure-weighted (position-discounted) category mass
        # -----------------------------
        expw_det_mass_global = {k: 0.0 for k in nails_keys}
        expw_pl_mass_global  = {k: 0.0 for k in nails_keys}
        expw_steck_mass_global = {k: 0.0 for k in nails_keys}

        expw_det_mass_globalMethod = {k: 0.0 for k in nails_keys}
        expw_pl_mass_globalMethod  = {k: 0.0 for k in nails_keys}

        # per-user divergences for exposure-weighted distributions
        user_js_expw_det = []
        user_js_expw_pl  = []
        user_js_expw_steck = []

        user_kl_expw_det = []
        user_kl_expw_pl  = []
        user_kl_expw_steck = []

        user_js_expw_det_globalMethod = []
        user_js_expw_pl_globalMethod  = []
        user_kl_expw_det_globalMethod = []
        user_kl_expw_pl_globalMethod  = []

        # Define rank weights once (DCG-style), normalized so each user contributes total mass 1
        r = np.arange(1, topN + 1)
        rank_w = 1.0 / np.log2(r + 1.0)
        rank_w = rank_w / float(rank_w.sum())

        # -----------------------------
        # Utility / disruption vs baseline (base model)
        # -----------------------------
        user_regret_det = []
        user_regret_pl = []
        user_regret_steck = []

        user_jaccard_det = []
        user_jaccard_pl = []
        user_jaccard_steck = []

        user_regret_det_globalMethod = []
        user_regret_pl_globalMethod = []
        user_jaccard_det_globalMethod = []
        user_jaccard_pl_globalMethod = []


        for row_i in df.iter_rows(named=True):
            # Candidate article IDs and categories
            inview_ids = np.asarray(row_i[DEFAULT_INVIEW_ARTICLES_COL], dtype=np.int64)
            cats = np.asarray(row_i["inview_categories"], dtype=object)

            # Scores vector (same length as inview_ids)
            pred_scores = np.asarray(row_i["scores"], dtype=np.float64)

            # Filter out "auto" candidates consistently if requested
            if filter_auto:
                mask_keep = (cats != "auto")
                inview_ids_f = inview_ids[mask_keep]
                cats_f = cats[mask_keep]
                pred_scores_f = pred_scores[mask_keep]
            else:
                inview_ids_f = inview_ids
                cats_f = cats
                pred_scores_f = pred_scores

            if pred_scores_f.size == 0:
                # Degenerate row, skip (or record NaNs)
                aligned_score_rows.append(pred_scores.tolist())
                continue
            
            # -----------------------------
            # baseline Top@N from base model scores (utility reference)
            # -----------------------------
            base_top_idx = np.argsort(-pred_scores_f)[:topN]
            base_top_ids = inview_ids_f[base_top_idx]
            base_sum = score_sum(pred_scores_f, base_top_idx)

            # Convert to probabilities if requested
            if act_func == "softmax":
                p_i_u = softmax(pred_scores_f, axis=0)
            else:
                # If already probabilities, ensure nonneg and normalize
                p_i_u = np.clip(pred_scores_f, 0.0, None)
                s = float(p_i_u.sum())
                p_i_u = p_i_u / s if s > 0 else np.full_like(p_i_u, 1.0 / len(p_i_u))

            # -----------------------------
            # Expected base distribution (per-user)
            # -----------------------------
            u_base_mass = user_expected_cat_mass(p_i_u, cats_f, nails_keys)
            u_base_dist = _normalize_dict(u_base_mass, nails_keys)
            user_kl_expected_base.append(_kl_from_dist(u_base_dist, nails_keys, nails_values, alpha))

            # Update global expected base mass
            for k in nails_keys:
                exp_base_mass_global[k] += float(u_base_dist.get(k, 0.0))

            # -----------------------------
            # USER-LEVEL correction weights
            # -----------------------------
            # Induced category distribution from base scores: P(c|u)
            p_c_u = {c: max(u_base_mass.get(c, 0.0), 1e-12) for c in nails_keys}

            # w(c|u) = P*(c) / P(c|u)
            p_w_u = compute_nails_adjustment_factors(p_star_ei, p_c_u)

            # -----------------------------
            # USER-LEVEL aligned distribution
            # -----------------------------
            w_items_user = np.array([float(p_w_u.get(c, unknown_cat)) for c in cats_f], dtype=np.float64)

            aligned_user = compute_nails(
                p_omega=p_i_u,
                p_star_ei=w_items_user,
                p_ei=np.ones_like(p_i_u),
                lambda_=lambda_val,
            )
            aligned_user = np.asarray(aligned_user, dtype=np.float64)
            aligned_user = np.clip(aligned_user, 0.0, None)
            s_user = float(aligned_user.sum())
            aligned_user = (aligned_user / s_user) if s_user > 0 else p_i_u


            # -----------------------------
            # GLOBAL (paper) aligned distribution
            # -----------------------------
            w_items_global = np.array([float(p_w_global.get(c, unknown_cat)) for c in cats_f], dtype=np.float64)

            aligned_global = compute_nails(
                p_omega=p_i_u,
                p_star_ei=w_items_global,
                p_ei=np.ones_like(p_i_u),
                lambda_=lambda_val,
            )
            aligned_global = np.asarray(aligned_global, dtype=np.float64)
            aligned_global = np.clip(aligned_global, 0.0, None)
            s_glob = float(aligned_global.sum())
            aligned_global = (aligned_global / s_glob) if s_glob > 0 else p_i_u
            
            u_aligned_mass_glob = user_expected_cat_mass(aligned_global, cats_f, nails_keys)
            u_aligned_dist_glob = _normalize_dict(u_aligned_mass_glob, nails_keys)
            user_kl_expected_aligned_global.append(_kl_from_dist(u_aligned_dist_glob, nails_keys, nails_values, alpha))

            for k in nails_keys:
                exp_aligned_mass_globalMethod[k] += float(u_aligned_dist_glob.get(k, 0.0))

            # -----------------------------
            # Expected aligned distribution (per-user)
            # -----------------------------
            u_aligned_mass = user_expected_cat_mass(aligned_user, cats_f, nails_keys)
            u_aligned_dist = _normalize_dict(u_aligned_mass, nails_keys)
            user_kl_expected_aligned.append(_kl_from_dist(u_aligned_dist, nails_keys, nails_values, alpha))

            # Update global expected aligned mass
            for k in nails_keys:
                exp_aligned_mass_global[k] += float(u_aligned_dist.get(k, 0.0))

            # -----------------------------
            # Deterministic Top@N from aligned
            # -----------------------------
            det_top_idx = np.argsort(-aligned_user)[:topN]
            det_top_cats = cats_f[det_top_idx].tolist()
            
            # Exposure-weighted category dist (user-level DET)
            det_expw_mass_u = {k: 0.0 for k in nails_keys}
            update_weighted_hist(det_expw_mass_u, det_top_cats, rank_w, nails_keys)
            det_expw_dist_u = _normalize_dict(det_expw_mass_u, nails_keys)

            user_kl_expw_det.append(_kl_from_dist(det_expw_dist_u, nails_keys, nails_values, alpha))
            user_js_expw_det.append(_js_from_dist(det_expw_dist_u, nails_keys, nails_values, alpha))

            # accumulate global exposure mass
            for k in nails_keys:
                expw_det_mass_global[k] += float(det_expw_dist_u.get(k, 0.0))

            update_count_hist(topN_det_counts, det_top_cats, nails_keys)
            det_rec_ids.extend(inview_ids_f[det_top_idx].tolist())

            # Utility trade-off and disruption
            det_top_ids = inview_ids_f[det_top_idx]
            det_sum = score_sum(pred_scores_f, det_top_idx)
            user_regret_det.append(base_sum - det_sum)
            user_jaccard_det.append(jaccard_at_n(base_top_ids, det_top_ids))

            det_user_dist = compute_normalized_distribution(det_top_cats)
            det_user_dist = {k: det_user_dist.get(k, 0.0) for k in nails_keys}
            user_kl_topN_det.append(_kl_from_dist(det_user_dist, nails_keys, nails_values, alpha))

            # -----------------------------
            # Plackett–Luce Top@N from aligned
            # -----------------------------
            perm = plackett_luce_permutation(aligned_user, rng=rng, eps=1e-12)
            pl_top_idx = perm[:topN]
            pl_top_ids = inview_ids_f[pl_top_idx]
            pl_sum = score_sum(pred_scores_f, pl_top_idx)
            user_regret_pl.append(base_sum - pl_sum)
            user_jaccard_pl.append(jaccard_at_n(base_top_ids, pl_top_ids))
            pl_top_cats = cats_f[pl_top_idx].tolist()
            # Exposure-weighted category dist (user-level PL)
            pl_expw_mass_u = {k: 0.0 for k in nails_keys}
            update_weighted_hist(pl_expw_mass_u, pl_top_cats, rank_w, nails_keys)
            pl_expw_dist_u = _normalize_dict(pl_expw_mass_u, nails_keys)

            user_kl_expw_pl.append(_kl_from_dist(pl_expw_dist_u, nails_keys, nails_values, alpha))
            user_js_expw_pl.append(_js_from_dist(pl_expw_dist_u, nails_keys, nails_values, alpha))

            for k in nails_keys:
                expw_pl_mass_global[k] += float(pl_expw_dist_u.get(k, 0.0))

            update_count_hist(topN_pl_counts, pl_top_cats, nails_keys)
            pl_rec_ids.extend(inview_ids_f[pl_top_idx].tolist())

            pl_user_dist = compute_normalized_distribution(pl_top_cats)
            pl_user_dist = {k: pl_user_dist.get(k, 0.0) for k in nails_keys}
            user_kl_topN_pl.append(_kl_from_dist(pl_user_dist, nails_keys, nails_values, alpha))

            # -----------------------------
            # STECK greedy Top@N from aligned_user (deterministic reranker)
            # -----------------------------
            steck_top_ids = greedy_steck_rerank(
                ids=inview_ids_f,
                lookup_attr=lookup_cat_str,   # article_id -> category_str
                scores=aligned_user,          # use aligned distribution as relevance proxy
                p_target=p_star_ei,           # target category distribution
                lambda_=lambda_val,
                k=topN,
                alpha=alpha_steck_select,     # make sure this exists
            )

            steck_top_ids_arr = np.asarray(steck_top_ids, dtype=np.int64)
            # Map STECK ids -> indices in filtered inview to get pred_scores_f utility
            id2pos = {int(id_): j for j, id_ in enumerate(inview_ids_f.tolist())}
            steck_top_idx = np.array([id2pos.get(int(i), -1) for i in steck_top_ids_arr], dtype=int)
            steck_top_idx = steck_top_idx[steck_top_idx >= 0]

            steck_sum = float(np.sum(pred_scores_f[steck_top_idx])) if steck_top_idx.size else 0.0
            user_regret_steck.append(base_sum - steck_sum)
            user_jaccard_steck.append(jaccard_at_n(base_top_ids, steck_top_ids_arr[:topN]))

            # categories for selected ids
            steck_top_cats = [lookup_cat_str.get(int(i), "unknown") for i in steck_top_ids]
            
            # Exposure-weighted category dist (user-level STECK)
            steck_expw_mass_u = {k: 0.0 for k in nails_keys}
            update_weighted_hist(steck_expw_mass_u, steck_top_cats[:topN], rank_w, nails_keys)
            steck_expw_dist_u = _normalize_dict(steck_expw_mass_u, nails_keys)

            user_kl_expw_steck.append(_kl_from_dist(steck_expw_dist_u, nails_keys, nails_values, alpha))
            user_js_expw_steck.append(_js_from_dist(steck_expw_dist_u, nails_keys, nails_values, alpha))

            for k in nails_keys:
                expw_steck_mass_global[k] += float(steck_expw_dist_u.get(k, 0.0))


            update_count_hist(topN_steck_counts, steck_top_cats, nails_keys)
            steck_rec_ids.extend(list(steck_top_ids))

            steck_user_dist = compute_normalized_distribution(steck_top_cats)
            steck_user_dist = {k: steck_user_dist.get(k, 0.0) for k in nails_keys}
            user_kl_topN_steck.append(_kl_from_dist(steck_user_dist, nails_keys, nails_values, alpha))


            # =============================
            # GLOBAL METHOD (paper): Top@N from aligned_global
            # =============================

            # Deterministic Top@N from aligned_global
            det_top_idx_glob = np.argsort(-aligned_global)[:topN]
            det_top_cats_glob = cats_f[det_top_idx_glob].tolist()
            update_count_hist(topN_det_counts_globalMethod, det_top_cats_glob, nails_keys)
            det_rec_ids_globalMethod.extend(inview_ids_f[det_top_idx_glob].tolist())

            det_user_dist_glob = compute_normalized_distribution(det_top_cats_glob)
            det_user_dist_glob = {k: det_user_dist_glob.get(k, 0.0) for k in nails_keys}
            user_kl_topN_det_global.append(_kl_from_dist(det_user_dist_glob, nails_keys, nails_values, alpha))

            # Plackett–Luce Top@N from aligned_global
            perm_glob = plackett_luce_permutation(aligned_global, rng=rng, eps=1e-12)
            pl_top_idx_glob = perm_glob[:topN]
            pl_top_cats_glob = cats_f[pl_top_idx_glob].tolist()
            update_count_hist(topN_pl_counts_globalMethod, pl_top_cats_glob, nails_keys)
            pl_rec_ids_globalMethod.extend(inview_ids_f[pl_top_idx_glob].tolist())

            pl_user_dist_glob = compute_normalized_distribution(pl_top_cats_glob)
            pl_user_dist_glob = {k: pl_user_dist_glob.get(k, 0.0) for k in nails_keys}
            user_kl_topN_pl_global.append(_kl_from_dist(pl_user_dist_glob, nails_keys, nails_values, alpha))

            # Regret and Jaccard for global method
            det_top_ids_glob = inview_ids_f[det_top_idx_glob]
            det_sum_glob = score_sum(pred_scores_f, det_top_idx_glob)
            user_regret_det_globalMethod.append(base_sum - det_sum_glob)
            user_jaccard_det_globalMethod.append(jaccard_at_n(base_top_ids, det_top_ids_glob))

            pl_top_ids_glob = inview_ids_f[pl_top_idx_glob]
            pl_sum_glob = score_sum(pred_scores_f, pl_top_idx_glob)
            user_regret_pl_globalMethod.append(base_sum - pl_sum_glob)
            user_jaccard_pl_globalMethod.append(jaccard_at_n(base_top_ids, pl_top_ids_glob))

            # Exposure-weighted dist for global/paper method DET
            det_expw_mass_u_gm = {k: 0.0 for k in nails_keys}
            update_weighted_hist(det_expw_mass_u_gm, det_top_cats_glob, rank_w, nails_keys)
            det_expw_dist_u_gm = _normalize_dict(det_expw_mass_u_gm, nails_keys)

            user_kl_expw_det_globalMethod.append(_kl_from_dist(det_expw_dist_u_gm, nails_keys, nails_values, alpha))
            user_js_expw_det_globalMethod.append(_js_from_dist(det_expw_dist_u_gm, nails_keys, nails_values, alpha))
            for k in nails_keys:
                expw_det_mass_globalMethod[k] += float(det_expw_dist_u_gm.get(k, 0.0))

            # Exposure-weighted dist for global/paper method PL
            pl_expw_mass_u_gm = {k: 0.0 for k in nails_keys}
            update_weighted_hist(pl_expw_mass_u_gm, pl_top_cats_glob, rank_w, nails_keys)
            pl_expw_dist_u_gm = _normalize_dict(pl_expw_mass_u_gm, nails_keys)

            user_kl_expw_pl_globalMethod.append(_kl_from_dist(pl_expw_dist_u_gm, nails_keys, nails_values, alpha))
            user_js_expw_pl_globalMethod.append(_js_from_dist(pl_expw_dist_u_gm, nails_keys, nails_values, alpha))
            for k in nails_keys:
                expw_pl_mass_globalMethod[k] += float(pl_expw_dist_u_gm.get(k, 0.0))


            # -----------------------------
            # Store aligned scores (for submission writing)
            # -----------------------------
            # Re-expand to the original candidate length so rank_predictions_by_score aligns with the row shape
            if filter_auto:
                final_scores = np.zeros_like(pred_scores)
                final_scores[np.where(mask_keep)[0]] = aligned_user
            else:
                final_scores = aligned_user

            aligned_score_rows.append(final_scores)

        # =========================
        # Global metrics for this (run, lambda)
        # =========================
        # Global expected dists (average over users)
        exp_base_dist_global = {k: exp_base_mass_global[k] / U for k in nails_keys}
        exp_aligned_dist_global = {k: exp_aligned_mass_global[k] / U for k in nails_keys}

        kl_expected_base_global = _kl_from_dist(exp_base_dist_global, nails_keys, nails_values, alpha)
        kl_expected_aligned_global = _kl_from_dist(exp_aligned_dist_global, nails_keys, nails_values, alpha)

        # Global Top@N histogram dists
        det_hist_dist = _normalize_dict(topN_det_counts, nails_keys)
        pl_hist_dist = _normalize_dict(topN_pl_counts, nails_keys)
        steck_hist_dist = _normalize_dict(topN_steck_counts, nails_keys)

        kl_topN_det_global = _kl_from_dist(det_hist_dist, nails_keys, nails_values, alpha)
        kl_topN_pl_global = _kl_from_dist(pl_hist_dist, nails_keys, nails_values, alpha)
        kl_topN_steck_global = _kl_from_dist(steck_hist_dist, nails_keys, nails_values, alpha)

        # -----------------------------
        # Global exposure-weighted distributions (average over users)
        # -----------------------------
        expw_det_dist_global = {k: expw_det_mass_global[k] / U for k in nails_keys}
        expw_pl_dist_global  = {k: expw_pl_mass_global[k] / U for k in nails_keys}
        expw_steck_dist_global = {k: expw_steck_mass_global[k] / U for k in nails_keys}

        KL_expw_det_global = _kl_from_dist(expw_det_dist_global, nails_keys, nails_values, alpha)
        KL_expw_pl_global  = _kl_from_dist(expw_pl_dist_global, nails_keys, nails_values, alpha)
        KL_expw_steck_global = _kl_from_dist(expw_steck_dist_global, nails_keys, nails_values, alpha)

        JS_expw_det_global = _js_from_dist(expw_det_dist_global, nails_keys, nails_values, alpha)
        JS_expw_pl_global  = _js_from_dist(expw_pl_dist_global, nails_keys, nails_values, alpha)
        JS_expw_steck_global = _js_from_dist(expw_steck_dist_global, nails_keys, nails_values, alpha)

        # -----------------------------
        # Mean over users (exposure-weighted)
        # -----------------------------
        KL_expw_det_user_mean = float(np.mean(user_kl_expw_det)) if user_kl_expw_det else np.nan
        KL_expw_pl_user_mean  = float(np.mean(user_kl_expw_pl)) if user_kl_expw_pl else np.nan
        KL_expw_steck_user_mean = float(np.mean(user_kl_expw_steck)) if user_kl_expw_steck else np.nan

        JS_expw_det_user_mean = float(np.mean(user_js_expw_det)) if user_js_expw_det else np.nan
        JS_expw_pl_user_mean  = float(np.mean(user_js_expw_pl)) if user_js_expw_pl else np.nan
        JS_expw_steck_user_mean = float(np.mean(user_js_expw_steck)) if user_js_expw_steck else np.nan

        # -----------------------------
        # Global/paper method exposure-weighted
        # -----------------------------
        expw_det_dist_globalMethod = {k: expw_det_mass_globalMethod[k] / U for k in nails_keys}
        expw_pl_dist_globalMethod  = {k: expw_pl_mass_globalMethod[k] / U for k in nails_keys}

        KL_expw_det_globalMethod_global = _kl_from_dist(expw_det_dist_globalMethod, nails_keys, nails_values, alpha)
        KL_expw_pl_globalMethod_global  = _kl_from_dist(expw_pl_dist_globalMethod, nails_keys, nails_values, alpha)
        JS_expw_det_globalMethod_global = _js_from_dist(expw_det_dist_globalMethod, nails_keys, nails_values, alpha)
        JS_expw_pl_globalMethod_global  = _js_from_dist(expw_pl_dist_globalMethod, nails_keys, nails_values, alpha)

        KL_expw_det_globalMethod_user_mean = float(np.mean(user_kl_expw_det_globalMethod)) if user_kl_expw_det_globalMethod else np.nan
        KL_expw_pl_globalMethod_user_mean  = float(np.mean(user_kl_expw_pl_globalMethod)) if user_kl_expw_pl_globalMethod else np.nan
        JS_expw_det_globalMethod_user_mean = float(np.mean(user_js_expw_det_globalMethod)) if user_js_expw_det_globalMethod else np.nan
        JS_expw_pl_globalMethod_user_mean  = float(np.mean(user_js_expw_pl_globalMethod)) if user_js_expw_pl_globalMethod else np.nan

        # -----------------------------
        # Utility / disruption summaries
        # -----------------------------
        regret_det_mean = float(np.mean(user_regret_det)) if user_regret_det else np.nan
        regret_pl_mean  = float(np.mean(user_regret_pl)) if user_regret_pl else np.nan
        regret_steck_mean = float(np.mean(user_regret_steck)) if user_regret_steck else np.nan

        jaccard_det_mean = float(np.mean(user_jaccard_det)) if user_jaccard_det else np.nan
        jaccard_pl_mean  = float(np.mean(user_jaccard_pl)) if user_jaccard_pl else np.nan
        jaccard_steck_mean = float(np.mean(user_jaccard_steck)) if user_jaccard_steck else np.nan

        regret_det_gm_mean = float(np.mean(user_regret_det_globalMethod)) if user_regret_det_globalMethod else np.nan
        regret_pl_gm_mean  = float(np.mean(user_regret_pl_globalMethod)) if user_regret_pl_globalMethod else np.nan
        jaccard_det_gm_mean = float(np.mean(user_jaccard_det_globalMethod)) if user_jaccard_det_globalMethod else np.nan
        jaccard_pl_gm_mean  = float(np.mean(user_jaccard_pl_globalMethod)) if user_jaccard_pl_globalMethod else np.nan


        # Mean user KLs
        kl_expected_base_user_mean = float(np.mean(user_kl_expected_base)) if user_kl_expected_base else np.nan
        kl_expected_aligned_user_mean = float(np.mean(user_kl_expected_aligned)) if user_kl_expected_aligned else np.nan
        kl_topN_det_user_mean = float(np.mean(user_kl_topN_det)) if user_kl_topN_det else np.nan
        kl_topN_pl_user_mean = float(np.mean(user_kl_topN_pl)) if user_kl_topN_pl else np.nan
        kl_topN_steck_user_mean = float(np.mean(user_kl_topN_steck)) if user_kl_topN_steck else np.nan

        # Coverage@N (over candidate universe)
        cov_det = (len(set(det_rec_ids)) / UNIVERSE_SIZE) if UNIVERSE_SIZE > 0 else 0.0
        cov_pl = (len(set(pl_rec_ids)) / UNIVERSE_SIZE) if UNIVERSE_SIZE > 0 else 0.0
        cov_steck = (len(set(steck_rec_ids)) / UNIVERSE_SIZE) if UNIVERSE_SIZE > 0 else 0.0

        # Threshold summaries (“how close are users”)
        exp_base_close = summarize_user_distances(user_kl_expected_base, THRESHOLDS)
        exp_aligned_close = summarize_user_distances(user_kl_expected_aligned, THRESHOLDS)
        top_det_close = summarize_user_distances(user_kl_topN_det, THRESHOLDS)
        top_pl_close = summarize_user_distances(user_kl_topN_pl, THRESHOLDS)
        top_steck_close = summarize_user_distances(user_kl_topN_steck, THRESHOLDS)

        # Flatten closeness fields into columns (match your naming scheme)
        user_closeness_cols = {}
        for prefix, summary in [
            ("exp_base", exp_base_close),
            ("exp_aligned", exp_aligned_close),
            ("topN_det", top_det_close),
            ("topN_pl", top_pl_close),
            ("topN_steck", top_steck_close),
        ]:
            for t in THRESHOLDS:
                user_closeness_cols[f"{prefix}_pct_le_{float(t)}"] = summary[f"pct_le_{t}"]
        
        # =========================
        # GLOBAL METHOD (paper): metrics for this (run, lambda)
        # =========================

        # Global expected dist (paper/global method)
        exp_aligned_dist_globalMethod = {k: exp_aligned_mass_globalMethod[k] / U for k in nails_keys}
        kl_expected_aligned_globalMethod_global = _kl_from_dist(
            exp_aligned_dist_globalMethod, nails_keys, nails_values, alpha
        )

        # Global Top@N histogram KLs (paper/global method)
        det_hist_dist_globalMethod = _normalize_dict(topN_det_counts_globalMethod, nails_keys)
        pl_hist_dist_globalMethod = _normalize_dict(topN_pl_counts_globalMethod, nails_keys)

        kl_topN_det_globalMethod_global = _kl_from_dist(det_hist_dist_globalMethod, nails_keys, nails_values, alpha)
        kl_topN_pl_globalMethod_global  = _kl_from_dist(pl_hist_dist_globalMethod, nails_keys, nails_values, alpha)

        # Mean user KLs (paper/global method)
        kl_expected_aligned_globalMethod_user_mean = float(np.mean(user_kl_expected_aligned_global)) if user_kl_expected_aligned_global else np.nan
        kl_topN_det_globalMethod_user_mean = float(np.mean(user_kl_topN_det_global)) if user_kl_topN_det_global else np.nan
        kl_topN_pl_globalMethod_user_mean  = float(np.mean(user_kl_topN_pl_global)) if user_kl_topN_pl_global else np.nan

        # Coverage@N (paper/global method)
        cov_det_globalMethod = (len(set(det_rec_ids_globalMethod)) / UNIVERSE_SIZE) if UNIVERSE_SIZE > 0 else 0.0
        cov_pl_globalMethod  = (len(set(pl_rec_ids_globalMethod)) / UNIVERSE_SIZE) if UNIVERSE_SIZE > 0 else 0.0

        # Threshold summaries (paper/global method)
        exp_aligned_global_close = summarize_user_distances(user_kl_expected_aligned_global, THRESHOLDS)
        top_det_global_close = summarize_user_distances(user_kl_topN_det_global, THRESHOLDS)
        top_pl_global_close  = summarize_user_distances(user_kl_topN_pl_global, THRESHOLDS)

        global_method_closeness_cols = {}
        for prefix, summary in [
            ("exp_aligned_global", exp_aligned_global_close),
            ("topN_det_global", top_det_global_close),
            ("topN_pl_global", top_pl_global_close),
        ]:
            for t in THRESHOLDS:
                global_method_closeness_cols[f"{prefix}_pct_le_{float(t)}"] = summary[f"pct_le_{t}"]


        # Record summary row
        eval_rows.append(
            {
                "run": run,
                "lambda": lambda_val,
                "U": U,
                "KL_expected_base_global": kl_expected_base_global,
                "KL_expected_aligned_global": kl_expected_aligned_global,
                "KL_expected_base_user_mean": kl_expected_base_user_mean,
                "KL_expected_aligned_user_mean": kl_expected_aligned_user_mean,
                "KL_topN_hist_det_global": kl_topN_det_global,
                "KL_topN_hist_pl_global": kl_topN_pl_global,
                "KL_topN_hist_steck_global": kl_topN_steck_global,
                "KL_topN_user_det_mean": kl_topN_det_user_mean,
                "KL_topN_user_pl_mean": kl_topN_pl_user_mean,
                "KL_topN_user_steck_mean": kl_topN_steck_user_mean,
                "coverage_det": cov_det,
                "coverage_pl": cov_pl,
                # --- global/paper method metrics ---
                "KL_expected_aligned_globalMethod_global": kl_expected_aligned_globalMethod_global,
                "KL_expected_aligned_globalMethod_user_mean": kl_expected_aligned_globalMethod_user_mean,
                "KL_topN_hist_det_globalMethod_global": kl_topN_det_globalMethod_global,
                "KL_topN_hist_pl_globalMethod_global": kl_topN_pl_globalMethod_global,
                "KL_topN_user_det_globalMethod_mean": kl_topN_det_globalMethod_user_mean,
                "KL_topN_user_pl_globalMethod_mean": kl_topN_pl_globalMethod_user_mean,
                "coverage_det_globalMethod": cov_det_globalMethod,
                "coverage_pl_globalMethod": cov_pl_globalMethod,
                "coverage_steck": cov_steck,

                # --- Exposure-weighted alignment (user-level methods) ---
                "KL_expw_det_global": KL_expw_det_global,
                "KL_expw_pl_global": KL_expw_pl_global,
                "KL_expw_steck_global": KL_expw_steck_global,
                "KL_expw_det_user_mean": KL_expw_det_user_mean,
                "KL_expw_pl_user_mean": KL_expw_pl_user_mean,
                "KL_expw_steck_user_mean": KL_expw_steck_user_mean,

                "JS_expw_det_global": JS_expw_det_global,
                "JS_expw_pl_global": JS_expw_pl_global,
                "JS_expw_steck_global": JS_expw_steck_global,
                "JS_expw_det_user_mean": JS_expw_det_user_mean,
                "JS_expw_pl_user_mean": JS_expw_pl_user_mean,
                "JS_expw_steck_user_mean": JS_expw_steck_user_mean,

                # --- Exposure-weighted alignment (global/paper method) ---
                "KL_expw_det_globalMethod_global": KL_expw_det_globalMethod_global,
                "KL_expw_pl_globalMethod_global": KL_expw_pl_globalMethod_global,
                "JS_expw_det_globalMethod_global": JS_expw_det_globalMethod_global,
                "JS_expw_pl_globalMethod_global": JS_expw_pl_globalMethod_global,

                "KL_expw_det_globalMethod_user_mean": KL_expw_det_globalMethod_user_mean,
                "KL_expw_pl_globalMethod_user_mean": KL_expw_pl_globalMethod_user_mean,
                "JS_expw_det_globalMethod_user_mean": JS_expw_det_globalMethod_user_mean,
                "JS_expw_pl_globalMethod_user_mean": JS_expw_pl_globalMethod_user_mean,

                # --- Utility / disruption ---
                "regret_det_user_mean": regret_det_mean,
                "regret_pl_user_mean": regret_pl_mean,
                "regret_steck_user_mean": regret_steck_mean,
                "jaccard_det_user_mean": jaccard_det_mean,
                "jaccard_pl_user_mean": jaccard_pl_mean,
                "jaccard_steck_user_mean": jaccard_steck_mean,

                "regret_det_globalMethod_user_mean": regret_det_gm_mean,
                "regret_pl_globalMethod_user_mean": regret_pl_gm_mean,
                "jaccard_det_globalMethod_user_mean": jaccard_det_gm_mean,
                "jaccard_pl_globalMethod_user_mean": jaccard_pl_gm_mean,

                **global_method_closeness_cols,
                **user_closeness_cols,
            }
        )

        if run == 0:
            out_png = plot_dir.joinpath(f"top{topN}_hist_ALLMETHODS_lambda{lambda_val}.png")
            plot_topN_histogram_all_methods(
                keys=nails_keys,
                p_star=p_star_ei,

                det_user=det_hist_dist,
                pl_user=pl_hist_dist,
                steck_user=steck_hist_dist,
                kl_det_user=kl_topN_det_global,
                kl_pl_user=kl_topN_pl_global,
                kl_steck_user=kl_topN_steck_global,

                det_global=det_hist_dist_globalMethod,
                pl_global=pl_hist_dist_globalMethod,
                kl_det_global=kl_topN_det_globalMethod_global,
                kl_pl_global=kl_topN_pl_globalMethod_global,

                topN=topN,
                outpath=out_png,
                title_prefix="NAILS comparison — ",
            )


        # Write submission files (one per run + lambda)
        if make_submission_file:
            df_out = df.with_columns(pl.Series(f"scores_{lambda_val}", aligned_score_rows))
            df_out = df_out.with_columns(
                pl.col(f"scores_{lambda_val}")
                .map_elements(lambda x: list(rank_predictions_by_score(x)))
                .alias("ranked_scores")
            )

            write_submission_file(
                impression_ids=df_out[DEFAULT_IMPRESSION_ID_COL],
                prediction_scores=df_out["ranked_scores"],
                path=PATH_DUMP.joinpath(
                    f"nails_{PLOT_TAG}_predictions_run{run}_lambda{lambda_val}.txt"
                ),
                filename_zip=f"nails_{PLOT_TAG}_predictions_run{run}_lambda{lambda_val}.zip",
            )

# %%
# -----------------------------
# Save evaluation summaries (per-run and aggregated over runs)
# -----------------------------
df_eval = pl.DataFrame(eval_rows)

out_xlsx = plot_dir.joinpath(f"eval_top{topN}_{PLOT_TAG}_user_vs_paper.xlsx")
df_eval.write_excel(out_xlsx)
print(f"Wrote per-run summary to: {out_xlsx}")

# Aggregate over runs (mean±std over runs per lambda)
base_metric_cols = [
    "KL_expected_base_global",
    "KL_expected_aligned_global",
    "KL_expected_base_user_mean",
    "KL_expected_aligned_user_mean",
    "KL_topN_hist_det_global",
    "KL_topN_hist_pl_global",
    "KL_topN_hist_steck_global",
    "KL_topN_user_det_mean",
    "KL_topN_user_pl_mean",
    "KL_topN_user_steck_mean",
    "coverage_det",
    "coverage_pl",
    "coverage_steck",
    "KL_expected_aligned_globalMethod_global",
    "KL_expected_aligned_globalMethod_user_mean",
    "KL_topN_hist_det_globalMethod_global",
    "KL_topN_hist_pl_globalMethod_global",
    "KL_topN_user_det_globalMethod_mean",
    "KL_topN_user_pl_globalMethod_mean",
    "coverage_det_globalMethod",
    "coverage_pl_globalMethod",

    "KL_expw_det_global",
    "KL_expw_pl_global",
    "KL_expw_steck_global",
    "KL_expw_det_user_mean",
    "KL_expw_pl_user_mean",
    "KL_expw_steck_user_mean",
    "JS_expw_det_global",
    "JS_expw_pl_global",
    "JS_expw_steck_global",
    "JS_expw_det_user_mean",
    "JS_expw_pl_user_mean",
    "JS_expw_steck_user_mean",

    "KL_expw_det_globalMethod_global",
    "KL_expw_pl_globalMethod_global",
    "JS_expw_det_globalMethod_global",
    "JS_expw_pl_globalMethod_global",
    "KL_expw_det_globalMethod_user_mean",
    "KL_expw_pl_globalMethod_user_mean",
    "JS_expw_det_globalMethod_user_mean",
    "JS_expw_pl_globalMethod_user_mean",

    "regret_det_user_mean",
    "regret_pl_user_mean",
    "regret_steck_user_mean",
    "jaccard_det_user_mean",
    "jaccard_pl_user_mean",
    "jaccard_steck_user_mean",
    "regret_det_globalMethod_user_mean",
    "regret_pl_globalMethod_user_mean",
    "jaccard_det_globalMethod_user_mean",
    "jaccard_pl_globalMethod_user_mean",

]
user_closeness_cols = []
for prefix in ["exp_base", "exp_aligned", "topN_det", "topN_pl", "topN_steck"]:
    user_closeness_cols.extend([f"{prefix}_pct_le_{float(t)}" for t in THRESHOLDS])

for prefix in ["exp_aligned_global", "topN_det_global", "topN_pl_global"]:
    user_closeness_cols.extend([f"{prefix}_pct_le_{float(t)}" for t in THRESHOLDS])


all_cols = base_metric_cols + user_closeness_cols

missing = [c for c in all_cols if c not in df_eval.columns]
if missing:
    print("WARNING: Missing columns in df_eval:", missing)

df_eval_agg = (
    df_eval.group_by("lambda")
    .agg(
        [
            pl.len().alias("n_runs"),
            pl.col("U").mean().alias("U_mean"),
            *[pl.col(c).mean().alias(f"{c}_mean_over_runs") for c in all_cols],
            *[pl.col(c).std(ddof=1).alias(f"{c}_std_over_runs") for c in all_cols],
        ]
    )
    .sort("lambda")
    .with_columns([pl.col(f"{c}_std_over_runs").fill_null(0.0) for c in all_cols])
)

out_xlsx_agg = plot_dir.joinpath(f"eval_top{topN}_{PLOT_TAG}_user_vs_paper_agg.xlsx")
df_eval_agg.write_excel(out_xlsx_agg)
print(f"Wrote aggregated (mean±std) summary to: {out_xlsx_agg}")


# ------------------------------------------------------------
print("Plotting summary curves over lambda (single run)...")

def _get_col(df: pl.DataFrame, name: str):
    if name not in df.columns:
        raise KeyError(f"Missing column: {name}")
    return np.asarray(df[name].to_list(), dtype=float)

def make_lambda_summary_plot_mean_only(
    df_agg: pl.DataFrame,
    x_col: str,
    series: list,   # [{"mean": "..._mean_over_runs", "label": "..."}]
    title: str,
    ylabel: str,
    outpath: Path,
):
    x = _get_col(df_agg, x_col)
    fig, ax = plt.subplots(figsize=(10, 5))

    for s in series:
        y = _get_col(df_agg, s["mean"])
        ax.plot(x, y, label=s["label"])

    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

LAM = "lambda"

# 1) Expected KL (GLOBAL dist): user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "KL_expected_aligned_global_mean_over_runs",
         "label": "User-level NAILS: expected KL (global dist)"},
        {"mean": "KL_expected_aligned_globalMethod_global_mean_over_runs",
         "label": "Global/paper NAILS: expected KL (global dist)"},
    ],
    title="Expected category KL vs λ (global distribution)",
    ylabel="Smoothed KL",
    outpath=plot_dir.joinpath(f"summary_expectedKL_global_{PLOT_TAG}.png"),
)

# 2) Expected KL (mean over users): user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "KL_expected_aligned_user_mean_mean_over_runs",
         "label": "User-level NAILS: mean_u expected KL"},
        {"mean": "KL_expected_aligned_globalMethod_user_mean_mean_over_runs",
         "label": "Global/paper NAILS: mean_u expected KL"},
    ],
    title="Expected category KL vs λ (mean over users)",
    ylabel="Smoothed KL",
    outpath=plot_dir.joinpath(f"summary_expectedKL_userMean_{PLOT_TAG}.png"),
)

# 3) Top@N histogram KL (global): det + PL, user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "KL_topN_hist_det_global_mean_over_runs",
         "label": f"User-level: Det Top@{topN} hist KL"},
        {"mean": "KL_topN_hist_det_globalMethod_global_mean_over_runs",
         "label": f"Global/paper: Det Top@{topN} hist KL"},
        {"mean": "KL_topN_hist_pl_global_mean_over_runs",
         "label": f"User-level: PL Top@{topN} hist KL"},
        {"mean": "KL_topN_hist_pl_globalMethod_global_mean_over_runs",
         "label": f"Global/paper: PL Top@{topN} hist KL"},
         {"mean": "KL_topN_hist_steck_global_mean_over_runs",
         "label": f"User-level: STECK Top@{topN} hist KL"},
    ],
    title=f"Top@{topN} histogram KL vs λ (global)",
    ylabel="Smoothed KL",
    outpath=plot_dir.joinpath(f"summary_top{topN}_histKL_{PLOT_TAG}.png"),
)

# 4) Top@N user-mean KL: det + PL, user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "KL_topN_user_det_mean_mean_over_runs",
         "label": f"User-level: mean_u Det Top@{topN} KL"},
        {"mean": "KL_topN_user_det_globalMethod_mean_mean_over_runs",
         "label": f"Global/paper: mean_u Det Top@{topN} KL"},
        {"mean": "KL_topN_user_pl_mean_mean_over_runs",
         "label": f"User-level: mean_u PL Top@{topN} KL"},
        {"mean": "KL_topN_user_pl_globalMethod_mean_mean_over_runs",
         "label": f"Global/paper: mean_u PL Top@{topN} KL"},
        {"mean": "KL_topN_user_steck_mean_mean_over_runs",
         "label": f"User-level: mean_u STECK Top@{topN} KL"},
    ],
    title=f"Top@{topN} KL vs λ (mean over users)",
    ylabel="Smoothed KL",
    outpath=plot_dir.joinpath(f"summary_top{topN}_userMeanKL_{PLOT_TAG}.png"),
)

# 5) Coverage@N: det + PL + steck, user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "coverage_det_mean_over_runs",
         "label": f"User-level: Det coverage@{topN}"},
        {"mean": "coverage_det_globalMethod_mean_over_runs",
         "label": f"Global/paper: Det coverage@{topN}"},
        {"mean": "coverage_pl_mean_over_runs",
         "label": f"User-level: PL coverage@{topN}"},
        {"mean": "coverage_pl_globalMethod_mean_over_runs",
         "label": f"Global/paper: PL coverage@{topN}"},
        {"mean": "coverage_steck_mean_over_runs",
         "label": f"User-level: STECK coverage@{topN}"},
    ],
    title=f"Coverage@{topN} vs λ",
    ylabel="Coverage (unique recs / universe)",
    outpath=plot_dir.joinpath(f"summary_coverage_top{topN}_{PLOT_TAG}.png"),
)

# 6) Exposure-weighted JS (global): user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "JS_expw_det_global_mean_over_runs",
         "label": f"User-level: DET exposure-JS@{topN}"},
        {"mean": "JS_expw_pl_global_mean_over_runs",
         "label": f"User-level: PL exposure-JS@{topN}"},
        {"mean": "JS_expw_steck_global_mean_over_runs",
         "label": f"User-level: STECK exposure-JS@{topN}"},
        {"mean": "JS_expw_det_globalMethod_global_mean_over_runs",
         "label": f"Global/paper: DET exposure-JS@{topN}"},
        {"mean": "JS_expw_pl_globalMethod_global_mean_over_runs",
         "label": f"Global/paper: PL exposure-JS@{topN}"},
    ],
    title=f"Exposure-weighted JS vs λ (global, Top@{topN})",
    ylabel="JS (smoothed)",
    outpath=plot_dir.joinpath(f"summary_expwJS_global_top{topN}_{PLOT_TAG}.png"),
)

# 7) Exposure-weighted JS (mean over users): user-level vs globalMethod
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "JS_expw_det_user_mean_mean_over_runs",
         "label": f"User-level: mean_u DET exposure-JS@{topN}"},
        {"mean": "JS_expw_pl_user_mean_mean_over_runs",
         "label": f"User-level: mean_u PL exposure-JS@{topN}"},
        {"mean": "JS_expw_steck_user_mean_mean_over_runs",
         "label": f"User-level: mean_u STECK exposure-JS@{topN}"},
        {"mean": "JS_expw_det_globalMethod_user_mean_mean_over_runs",
         "label": f"Global/paper: mean_u DET exposure-JS@{topN}"},
        {"mean": "JS_expw_pl_globalMethod_user_mean_mean_over_runs",
         "label": f"Global/paper: mean_u PL exposure-JS@{topN}"},
    ],
    title=f"Exposure-weighted JS vs λ (mean over users, Top@{topN})",
    ylabel="JS (smoothed)",
    outpath=plot_dir.joinpath(f"summary_expwJS_userMean_top{topN}_{PLOT_TAG}.png"),
)

# 8) Utility cost: mean regret vs λ
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "regret_det_user_mean_mean_over_runs", "label": f"User-level: DET regret@{topN}"},
        {"mean": "regret_pl_user_mean_mean_over_runs",  "label": f"User-level: PL regret@{topN}"},
        {"mean": "regret_steck_user_mean_mean_over_runs", "label": f"User-level: STECK regret@{topN}"},
        {"mean": "regret_det_globalMethod_user_mean_mean_over_runs", "label": f"Global/paper: DET regret@{topN}"},
        {"mean": "regret_pl_globalMethod_user_mean_mean_over_runs",  "label": f"Global/paper: PL regret@{topN}"},
    ],
    title=f"Utility cost vs λ (mean regret@{topN} vs baseline)",
    ylabel="Regret (baseline sum - method sum)",
    outpath=plot_dir.joinpath(f"summary_regret_top{topN}_{PLOT_TAG}.png"),
)

# 9) Disruption: mean Jaccard@N vs λ (higher is less disruption)
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "jaccard_det_user_mean_mean_over_runs", "label": f"User-level: DET Jaccard@{topN}"},
        {"mean": "jaccard_pl_user_mean_mean_over_runs",  "label": f"User-level: PL Jaccard@{topN}"},
        {"mean": "jaccard_steck_user_mean_mean_over_runs", "label": f"User-level: STECK Jaccard@{topN}"},
        {"mean": "jaccard_det_globalMethod_user_mean_mean_over_runs", "label": f"Global/paper: DET Jaccard@{topN}"},
        {"mean": "jaccard_pl_globalMethod_user_mean_mean_over_runs",  "label": f"Global/paper: PL Jaccard@{topN}"},
    ],
    title=f"Disruption vs λ (mean Jaccard@{topN} vs baseline)",
    ylabel="Jaccard overlap",
    outpath=plot_dir.joinpath(f"summary_jaccard_top{topN}_{PLOT_TAG}.png"),
)

def make_pareto_scatter(
    df_agg: pl.DataFrame,
    x_col: str,          # column name in df_agg (e.g. "JS_expw_det_global_mean_over_runs")
    y_col: str,          # e.g. "regret_det_user_mean_mean_over_runs"
    lam_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    annotate: bool = True,
):
    x = _get_col(df_agg, x_col)
    y = _get_col(df_agg, y_col)
    lam = _get_col(df_agg, lam_col)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y)

    if annotate:
        for xi, yi, li in zip(x, y, lam):
            ax.annotate(f"{li:g}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

# 10) Pareto: exposure alignment vs utility cost (label points by λ)

make_pareto_scatter(
    df_eval_agg,
    x_col="JS_expw_det_global_mean_over_runs",
    y_col="regret_det_user_mean_mean_over_runs",
    lam_col=LAM,
    title=f"Pareto trade-off (User-level DET): exposure-JS vs regret (Top@{topN})",
    xlabel="Exposure-weighted JS (global) ↓",
    ylabel="Mean regret ↑",
    outpath=plot_dir.joinpath(f"pareto_userDET_expwJS_vs_regret_top{topN}_{PLOT_TAG}.png"),
)

make_pareto_scatter(
    df_eval_agg,
    x_col="JS_expw_steck_global_mean_over_runs",
    y_col="regret_steck_user_mean_mean_over_runs",
    lam_col=LAM,
    title=f"Pareto trade-off (User-level STECK): exposure-JS vs regret (Top@{topN})",
    xlabel="Exposure-weighted JS (global) ↓",
    ylabel="Mean regret ↑",
    outpath=plot_dir.joinpath(f"pareto_userSTECK_expwJS_vs_regret_top{topN}_{PLOT_TAG}.png"),
)

make_pareto_scatter(
    df_eval_agg,
    x_col="JS_expw_det_globalMethod_global_mean_over_runs",
    y_col="regret_det_globalMethod_user_mean_mean_over_runs",
    lam_col=LAM,
    title=f"Pareto trade-off (Global/paper DET): exposure-JS vs regret (Top@{topN})",
    xlabel="Exposure-weighted JS (global) ↓",
    ylabel="Mean regret ↑",
    outpath=plot_dir.joinpath(f"pareto_globalDET_expwJS_vs_regret_top{topN}_{PLOT_TAG}.png"),
)

# 11) Flat Top@N hist KL vs Exposure-weighted KL (global) — shows why exposure matters
make_lambda_summary_plot_mean_only(
    df_eval_agg,
    x_col=LAM,
    series=[
        {"mean": "KL_topN_hist_det_global_mean_over_runs", "label": f"Flat hist KL (DET)"},
        {"mean": "KL_expw_det_global_mean_over_runs", "label": f"Exposure KL (DET)"},
        {"mean": "KL_topN_hist_pl_global_mean_over_runs", "label": f"Flat hist KL (PL)"},
        {"mean": "KL_expw_pl_global_mean_over_runs", "label": f"Exposure KL (PL)"},
        {"mean": "KL_topN_hist_steck_global_mean_over_runs", "label": f"Flat hist KL (STECK)"},
        {"mean": "KL_expw_steck_global_mean_over_runs", "label": f"Exposure KL (STECK)"},
    ],
    title=f"Why exposure weighting matters: flat vs exposure KL (global, Top@{topN})",
    ylabel="Smoothed KL",
    outpath=plot_dir.joinpath(f"summary_flat_vs_expw_KL_global_top{topN}_{PLOT_TAG}.png"),
)


print("Done: wrote summary plots into:", plot_dir)