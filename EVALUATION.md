# Agent Evaluation Report

Date: 2026-08-30

## Rubric

The agent scores ideas with the official 5-criteria selection rubric:

- Problem / Need (0-5)
- Solution / Idea (0-5)
- Innovation / Differentiation (0-5)
- Market Potential (0-5)
- Feasibility (0-5)

Each criterion has equal weight (20%), producing a total score out of 25.
Verdict thresholds are **Strong >= 19**, **Promising >= 13**, and
**Weak < 13**.

## Setup

- Agent backend: `deepseek-v4-flash` via the DeepSeek OpenAI-compatible
  endpoint, with Tavily for web research.
- Data: `~/Desktop/filter_3/dashboard_ready.csv`, 1,483 rows total.
- Evaluation scope: **2024-2025 only (1,027 rows); 2023 (456 rows) excluded.
  Dashboard analysis runs only on the 2024-2025 subset.**
- Samples: 10 real submitted ideas, fixed seed 7 (8 English, 2 Arabic from the
  original `problem`/`solution` fields).
- Run:

```bash
python -u evaluate_agent.py --samples 10 --seed 7 --sleep 0 \
  --max-retries 2 --out /tmp/mashroo3i_eval_current.json
```

## Random-sample results (current /25 run)

All 10 real-idea runs passed every functional check:

| Criterion | Pass rate |
| --- | --- |
| Report parses (score + dashboard) | 10 / 10 |
| All 5 rubric dimensions present with rationale | 10 / 10 |
| Dimension scores within 0-5 | 10 / 10 |
| Weighted /25 total is internally consistent | 10 / 10 |
| At least one valid web source per idea | 10 / 10 |
| Dashboard summary + insights generated | 10 / 10 |
| Dashboard KPIs match ground truth | 10 / 10 |
| Dashboard narrative cites real values | 10 / 10 |
| No agent errors | 10 / 10 |

Score distribution: **6.0 - 19.0 (mean 13.5)**. Verdicts were **1 Strong,
6 Promising, and 3 Weak**. Sources per idea: 6-17.

End-to-end latency: **mean 60.19s, p50 60.30s, max 69.13s**. Each run issues
up to six web-search queries; the number of agent calls depends on the
tool-use path.

## Control cases (bad vs. good ideas)

The curated suite was also rerun live on the current /25 rubric:

| Case | Expected | Score | Verdict | Direction |
| --- | --- | --- | --- | --- |
| generic food delivery app | Weak | 14.0 | Promising | Fail |
| another photo social network | Weak | 10.0 | Weak | Pass |
| ordinary coffee shop | Weak | 14.0 | Promising | Fail |
| generic online marketplace | Weak | 12.0 | Weak | Pass |
| commodity water delivery | Weak | 15.0 | Promising | Fail |
| Arabic generic restaurant delivery | Weak | 10.0 | Weak | Pass |
| vague / lacking substance | Edge | 3.0 | Weak | Pass |
| circular-construction waste tech | Strong | 18.0 | Promising | Fail |
| agri-IoT freshness chain | Strong | 17.0 | Promising | Fail |

Direction check: **4 / 9**. All structural, source, dashboard, and error checks
passed **9 / 9**.

## Interpretation

The current agent is reliable at producing a complete, evidence-backed,
grounded report for every input tested. Its score discrimination is weaker on
the curated controls: three generic, weak ideas were labelled **Promising**,
and both differentiated ideas fell just below the **Strong** threshold.

So on this run the /25 score behaves better as a **structured second opinion
and annotation** than as a standalone good-idea/bad-idea ranker.

## Notes

- The evaluation harness was aligned to the current /25 rubric: dimension
  scores are checked against 0-5, and the total is recomputed as
  `5 * sum(score * weight)`.
- The deterministic dashboard KPIs are still injected into the report, so
  dashboard accuracy does not depend on hallucination-free narrative.

## Reproduce

```bash
python evaluate_agent.py --samples 10 --seed 7 --sleep 0 --max-retries 2 \
  --out /tmp/mashroo3i_eval.json
python evaluate_agent.py --suite cases --sleep 0 --max-retries 2 \
  --out /tmp/mashroo3i_cases.json
python evaluate_agent.py --offline   # smoke check, no live calls
```
