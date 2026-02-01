# User-level NAILS on EB-NeRD — evaluation + extensions

This repository contains my bachelor thesis implementation extending NAILS (Normative Alignment via Internal Label Shift; Kruse et al., 2025) from a global correction to a user-level correction. The implementation runs on EB-NeRD (Kruse et al., 2024) evaluation data and compares:

- User-level NAILS (my extension): compute weights per impression and align per user
- Global/paper NAILS (baseline): compute a single global correction and apply it to all users
- Ranking strategies on aligned scores:
  - DET: deterministic Top@N
  - PL: Plackett–Luce sampling Top@N
  - STECK: greedy calibrated reranking (Steck, 2018) applied on aligned scores

In addition to the original KL-based calibration metrics, I added exposure-weighted alignment, utility/regret trade-offs, and disruption metrics.

## 1. Starting point (baseline script)

The starting script (provided by the original codebase) implemented:

1. Load a test candidate matrix with `impression_id == 0` to estimate a global induced distribution \(P(c)\) from the model probabilities.
2. Compute global weights:
   \[
   w(c) = \frac{P^\*(c)}{P(c)}
   \]
3. Apply the NAILS transformation category-wise and normalize per user.
4. Evaluate:
   - User-level KL distribution (per-user KL between Top@N histogram and target)
   - Global KL on summed scores and Top@N histograms
   - Coverage@N
5. Create submission files.

## 2. What I changed

### A) Data and evaluation universe
**Change:** evaluate user-level alignment on real recommendation impressions.

- Baseline: used `impression_id == 0` “test candidate matrix” primarily to estimate global category priors and to run some global/top-N evaluation.
- Now: evaluation is performed on behaviors/impressions where `impression_id > 0` (i.e., user-level rows), which is the correct unit for “user-level alignment”.
- For each impression row, I construct `inview_categories`, aligned index-wise with `inview_articles` and the corresponding `scores` vector.
- I define the evaluation universe as the set of all candidate article IDs that appear across the evaluation impressions, and use that universe as the denominator for coverage:
  \[
  \text{Coverage@}N = \frac{|\{\text{unique recommended article IDs}\}|}{|\{\text{unique candidate IDs in evaluation universe}\}|}.
  \]

### B) Target distribution \(P^\*(c)\) and category ordering
Change: make the target distribution and key ordering explicit and stable.

- I compute \(P^\*(c)\) using:
  - Editorial: empirical category frequencies in the evaluation universe.
- I explicitly store:
  - `nails_keys = list(p_star_ei.keys())`
  - `nails_values = [p_star_ei[k] for k in nails_keys]`
- This ensures that all divergences compare distributions under a consistent category ordering.

### C) Global/paper NAILS retained as baseline
Change: keep the original “paper/global” method in the same script for a direct comparison.

### D) Main contribution: user-level NAILS (per-impression weights)
Change: compute correction weights per impression \(u\).

For each impression \(u\):

1. Convert base scores into probabilities \(p(i\mid u)\).
2. Compute the user-level category mass:
   \[
   P(c\mid u)=\sum_{i \in I_u: c(i)=c} p(i\mid u).
   \]
3. Compute user-specific weights:
   \[
   w(c\mid u) = \frac{P^\*(c)}{P(c\mid u)}.
   \]
4. Apply NAILS on items using the weight of their category:
   \[
   \tilde{p}(i\mid u;\lambda) = \text{NAILS}\Big(p(i\mid u), \; w(c(i)\mid u), \; \lambda\Big),
   \]
   followed by normalization.

Difference between global-level and user-level: the global method corrects "global-level bias", while user-level NAILS corrects "per-user label shift", so each impression is aligned toward the same normative target distribution without forcing every user to share the same correction.

### E) Ranking policies on the aligned distribution
Change: evaluate multiple ranking strategies using the aligned probabilities \(\tilde{p}(i\mid u;\lambda)\).

After producing aligned probabilities for each impression:

1. DET (deterministic)  
   Select Top@N by sorting \(\tilde{p}(i\mid u;\lambda)\) descending.

2. PL (Plackett–Luce, without replacement)  
   Sample a full permutation according to Plackett–Luce “worth” parameters proportional to \(\tilde{p}(i\mid u;\lambda)\), then take the first N items.
   - This ranker is stochastic, but the differences over runs were negligible, so I ended up only doing one run (also due to computation restraints mainly caused by lack of time to evaluate results).

3. STECK (greedy calibrated reranking)  
   Apply a greedy reranker to select Top@N to better match \(P^\*(c)\), using aligned scores.

This yields three user-level recommendation lists per \(\lambda\), plus two lists from the global/paper method (DET and PL).

### F) Evaluation expansions (beyond the baseline metrics)
Change: add metrics that are more in line with “what users see”.

#### 1) Expected category alignment (distributional, before Top@N)
For each impression, compute expected category mass under aligned probabilities:
\[
\hat{P}(c\mid u)=\sum_{i\in I_u: c(i)=c}\tilde{p}(i\mid u;\lambda).
\]
Which gives metrics that shows:
- Global expected distribution: average \(\hat{P}(c\mid u)\) over users
- Mean-over-users KL: average per-user divergence

#### 2) Top@N alignment (flat histograms)
For each method and impression, build the Top@N category histogram. Which gives:
- Global histogram KL: KL histogram aggregated over all users
- Mean-over-users KL: average per-user KL

#### 3) Exposure-weighted alignment (position-discounted)
Top@N “flat” histograms treat rank 1 and rank N equally, but exposure is higher at the top ranks.

- I define a normalized exposure vector \(w_r\) (discounted cumulative gain-style):
  \[
  w_r \propto \frac{1}{\log_2(r+1)}, \quad r=1,\dots,N, \quad \sum_r w_r=1.
  \]
- For a Top@N list, accumulate category mass with these weights to obtain an exposure-weighted distribution \(Q^{\text{exp}}_u(c)\).

I then compute both:
- Exposure-weighted KL (smoothed KL)
- Exposure-weighted JS divergence (Jensen–Shannon), implemented via two smoothed KL calls:
  \[
  JS(P^\*\parallel Q)=\frac{1}{2}KL(P^\*\parallel M)+\frac{1}{2}KL(Q\parallel M), \;\; M=\frac{1}{2}(P^\*+Q).
  \]

#### 4) Utility cost (regret vs baseline)
Change: measure the utility lost by alignment/reranking compared to the original base ranking.

For each impression:
- Compute baseline Top@N by sorting original scores \(s(i\mid u)\).
- Compute baseline score-sum:
  \[
  S_{\text{base}}(u)=\sum_{i \in \text{TopN}_{\text{base}}} s(i\mid u).
  \]
- Compute method score-sum \(S_{\text{method}}(u)\) from the same original scores but evaluated on the method-selected indices.
- Define regret:
  \[
  \text{Regret}(u)=S_{\text{base}}(u)-S_{\text{method}}(u).
  \]
Report mean regret over users.

This quantifies “alignment cost” in the native scoring scale of the base model.

#### 5) Disruption (Jaccard overlap)
Change: measure how much the recommended set changes relative to the baseline Top@N.

For each impression:
\[
\text{Jaccard@}N(u)=\frac{|A_u\cap B_u|}{|A_u\cup B_u|}
\]
where \(A_u\) is baseline Top@N and \(B_u\) is method Top@N.

Higher Jaccard means less disruption (more similar to baseline).

### G) Aggregation and plotting over \(\lambda\)
Change: export both per-run and aggregated summaries, and produce consistent λ-sweep plots.

- The script writes:
  - Per-run summary: `eval_top{N}_...xlsx`
  - Aggregated summary (mean ± std over runs): `..._agg.xlsx`
- Plots include:
  - Expected alignment KL vs \(\lambda\) (global + mean-over-users)
  - Top@N histogram KL vs \(\lambda\) (global + mean-over-users)
  - Coverage@N vs \(\lambda\)
  - Exposure-weighted JS vs \(\lambda\) (global + mean-over-users)
  - Regret vs \(\lambda\)
  - Jaccard vs \(\lambda\)
  - Flat vs exposure-weighted KL vs \(\lambda\)

## 3. Implementation notes (what to look for in the code)

### User-level alignment
- `p_c_u` computed from `u_base_mass`
- `p_w_u = compute_nails_adjustment_factors(p_star_ei, p_c_u)`
- `aligned_user = compute_nails(p_omega=p_i_u, p_star_ei=w_items_user, p_ei=ones, lambda_=lambda_val)`

### Global/paper alignment
- `p_c_global` computed once across all impressions
- `p_w_global = compute_nails_adjustment_factors(p_star_ei, p_c_global)`
- `aligned_global` computed per impression using `w_items_global`

### Ranking policies
- DET: `np.argsort(-aligned_user)[:topN]`
- PL: `plackett_luce_permutation(aligned_user, rng)[:topN]`
- STECK: `greedy_steck_rerank(..., scores=aligned_user, p_target=p_star_ei, lambda_=lambda_val, k=topN, alpha=alpha_steck_select)`

### Exposure weighting
- `rank_w = 1/log2(r+1)` normalized to sum to 1
- `update_weighted_hist(...)` accumulates weighted category exposure mass

### Utility and disruption
- baseline: Top@N from `pred_scores_f`
- regret: `base_sum - method_sum`
- jaccard: overlap of baseline IDs vs method IDs


## 4. How to run
NOTE: Data is missing from this, as the files are too large to export to github. 

1. Install dependencies from requirements.txt
2. Ensure data (.parquet files) is located in nails-main/data, whereas final per lambda nails predictions are stored correctly. My setup is a bit messy, so a parent folder to nails-main stores final predictions in another folder named data.
3. Run the script:
   - It produces:
     - submission zips in `data/ebnerd_submissions/`
     - plots in `plot/nails_{tag}_{distribution}_{strategy}/`
     - Excel summaries in the same plot directory

