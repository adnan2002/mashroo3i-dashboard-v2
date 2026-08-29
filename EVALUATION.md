# Agent Evaluation Report

Date: 2026-08-29

> Scale note: the agent now uses the official 5-criteria /25 selection rubric.
> The numbers below were produced with the earlier 6-dimension /100 rubric;
> rerun `evaluate_agent.py` to refresh them on the /25 scale.

## Setup

- Model: `deepseek-v4-flash` via the `DEEPSEEK_API_KEY` secret (DeepSeek
  OpenAI-compatible endpoint), Tavily for web search.
- Data: `~/Desktop/filter_3/dashboard_ready.csv`, 1,483 rows total.
- Evaluation scope: **2024-2025 only (1,027 rows); 2023 (456 rows) excluded.
  Dashboard analysis runs only on the 2024-2025 subset.**
- Samples: 10 real submitted ideas, fixed seed 7 (8 English, 2 Arabic from the
  original `problem`/`solution` fields).
- Run: `python -u evaluate_agent.py --samples 10 --sleep 2.0 --max-retries 3`

## Results

| Criterion | Pass rate |
| --- | --- |
| Report parses (score + dashboard) | 10 / 10 |
| All 6 rubric dimensions present with rationale | 10 / 10 |
| Dimension scores within 0-10 | 10 / 10 |
| Weighted 0-100 total is internally consistent | 10 / 10 |
| At least one valid web source per idea | 10 / 10 |
| Dashboard summary + insights generated | 10 / 10 |
| Dashboard KPIs match ground truth | 10 / 10 |
| Dashboard narrative cites real values | 10 / 10 |
| No agent errors | 10 / 10 |

Score distribution: 37.0 - 63.5 (mean ~51.8). 7 ideas scored Promising,
3 scored Weak. Sources per idea: 10-15. End-to-end latency: mean 53s,
p50 54s, max 74s (5 Tavily searches + 2 DeepSeek calls per run).

## Notes

- DeepSeek `deepseek-v4-flash` supports native tool calls; the JSON tool-request
  fallback remains for models that do not.
- Two reliability fixes were validated during evaluation: incomplete final-model
  answers no longer overwrite complete tool results, and scoring/dashboard JSON
  calls retry with a larger output budget when the response is truncated.
- The deterministic dashboard KPIs are injected into the report regardless of
  model wording, so dashboard accuracy does not depend on hallucination-free
  narrative.

## Control cases (bad vs. good ideas)

To test discrimination -- not just formatting -- a curated suite of 9
controlled ideas was run live:

| Case | Expected | Score | Verdict |
| --- | --- | --- | --- |
| generic food delivery app | Weak | 34.0 | Weak |
| another photo social network | Weak | 22.5 | Weak |
| ordinary coffee shop | Weak | 45.5 | Weak |
| generic online marketplace | Weak | 49.5 | Weak |
| commodity water delivery | Weak | 46.0 | Weak |
| Arabic generic restaurant delivery | Weak | 41.5 | Weak |
| vague / lacking substance | Weak | 23.0 | Weak |
| circular-construction waste tech | Strong | 69.0 | Promising |
| agri-IoT freshness chain | Strong | 68.0 | Promising |

Direction check: 9 / 9 (all generic ideas <= 50; both differentiated ideas
>= 60). All other criteria also passed 9 / 9, including valid sources on every
case (no crash on the vague input, Arabic input handled correctly).

## Reproduce

```bash
python evaluate_agent.py --samples 10 --out /tmp/mashroo3i_eval.json
python evaluate_agent.py --suite cases --out /tmp/mashroo3i_cases.json
python evaluate_agent.py --offline   # smoke check, no live calls
```
