# Hybrid Evaluation — Brinc Model + Innovation Agent

Date: 2026-08-29

> Scale note: the agent's innovation score is now the official 5-criteria
> /25 rubric. The hybrid numbers below used the earlier /100 innovation score;
> ranking-based AUC/Spearman conclusions are unchanged because the metrics are
> rank-based.

## Setup

- Model: Brinc CatBoost (118 features, target `outcome_clean == "Accepted"`),
  refit with `artifacts/brinc_model_full/config.json` params on the **same
  engineered features**, validated by 5-fold StratifiedGroupKFold grouped on
  project + date-of-birth.
- Probabilities are **out-of-fold** (honest); the in-sample
  `predictions.csv` was deliberately not used (it gives leaky AUC = 1.0).
- Agent: `deepseek-v4-flash` + Tavily innovation score (0-100).
- Data: `dashboard_ready.csv`, 2024-2025 only (1,027 rows). **Outcome
  columns (`outcome`, `outcome_clean`, `is_winner`) were removed from both the
  model feature matrix and the agent's dashboard input**; labels were kept only
  for scoring.
- Sample: 30 rows, stratified 15 Accepted + 15 Rejected, fixed seed 42.

Operating point note: the bundled cutoff is now `0.125` (was `0.0625`), which
keeps the same OOF F2 while improving precision on the 2024-25 subset.

## Results (same 30 scored rows)

| Scorer | ROC AUC | PR AUC | Spearman |
| --- | --- | --- | --- |
| Brinc model only (on the 30 rows) | **0.791** | **0.794** | **+0.505** |
| Innovation agent only | 0.438 | 0.530 | -0.108 |
| Hybrid, model weight 0.6 | 0.689 | 0.707 | +0.327 |

Model-only on the full 1,027-row subset: ROC AUC 0.811, PR AUC 0.538,
Spearman +0.439.

Weight sweep (AUC on the 30 rows): agent only 0.44 → w=0.25: 0.52 →
w=0.5: 0.64 → w=0.75: 0.75 → model only: **0.79**. Every positive agent
weight reduces AUC versus the model alone.

Score split: accepted ideas averaged 52.7 agent-innovation and 0.439 model
probability; rejected ideas averaged 56.2 innovation and 0.138 model
probability. Top agent-scored ideas include rejected "برنامج تدريبي" (68.0)
and accepted "EINK" (65.5); the model's top picks include accepted "Hajz"
(0.98) and rejected "LaundriX" (0.86).

## Interpretation

- The Brinc model **is** the strong predictor of acceptance, as expected
  (≈0.79-0.81 AUC vs ≈0.44 for the agent).
- The agent's innovation score is **not a useful predictor of acceptance** —
  it is slightly **anti-correlated** (Spearman -0.11): the most innovative
  ideas in this applicant pool are often not the accepted ones.
- Consequently, **blending the agent into the model makes the hybrid worse,
  not better** (0.689 vs 0.791). The best hybrid weight is 1.0, i.e., use the
  model alone for acceptance ranking.
- Recommendation: keep the model as the acceptance/selection scorer and keep
  the agent's score as a separate **innovation/quality** axis for human review;
  do not add it into the acceptance probability.

## Model-first cascade (primary filter -> agent evaluation)

Better architecture: the model is the **primary** ranker, then the agent
searches and evaluates only the shortlist. Measured on the same 2024-25 data,
with outcomes removed and honest OOF probabilities:

| Cascade step | Result |
| --- | --- |
| Model top-30 shortlist | **73.3% accepted** (22/30) vs 21.0% baseline positive rate |
| Model recall at top-30 | 10.2% of all accepted applicants in scope |
| Model primary top-15 precision | **73.3%** (11/15) |
| Agent secondary re-rank top-15 | 66.7% (10/15) |

Within the shortlist, model AUC = 0.545, agent AUC = 0.449, hybrid AUC =
0.540. The agent's innovation re-rank does **not** improve accepted-precision
inside the model shortlist (66.7% vs 73.3%).

What this means in practice:

- Use the model to produce the primary candidate list (e.g., top-30)
  -- precision jumps from 21% to 73%.
- Use the agent on that shortlist as a **second opinion and annotation** (web
  evidence, risks, innovation rationale), not as an acceptance re-ranker.
- If you must combine, keep the model rank primary and treat the agent score
  as a tie-breaker/curation signal for human review.

Run it:

```bash
~/Desktop/filter_3/venv/bin/python hybrid_evaluate.py --mode cascade --stage agent --top-k 30
~/Desktop/filter_3/venv/bin/python hybrid_evaluate.py --mode cascade --stage metrics --top-k 30
```

## Reproduce

```bash
~/Desktop/filter_3/venv/bin/python hybrid_evaluate.py --stage model
~/Desktop/filter_3/venv/bin/python hybrid_evaluate.py --stage agent
~/Desktop/filter_3/venv/bin/python hybrid_evaluate.py --stage metrics
```

Requires the agent packages (`openai`, `tavily-python`) in the `filter_3`
venv, plus the DeepSeek/Tavily keys in `filter_2/.streamlit/secrets.toml`
(or exported environment variables).
