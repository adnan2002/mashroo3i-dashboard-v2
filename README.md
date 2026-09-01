# Mashroo3i Dashboard

Analytics dashboard for Mashroo3i applicant data. This project is a plain
Python conversion of `Dashboard_Mashroo3i.ipynb`, so it can be run, deployed,
and tested without a notebook kernel.

## Project layout

- `app.py` - the Dash application (layout, callbacks, entrypoint)
- `streamlit_app.py` - Streamlit deployment entrypoint (reuses the same chart logic)
- `selection_criteria.json` - editable selection rubric loaded by the AI pages
- `.streamlit/config.toml` - Streamlit theme/server configuration
- `model/` - bundled Brinc acceptance model (joblib, threshold, feature engineering)
- `tests/` - smoke tests plus committed `fixtures/` CSV data used by them

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open <http://localhost:8050> and upload an applications CSV or Excel file
(`tests/fixtures/applications.csv` is a small sample).

Streamlit (recommended for deployment):

```bash
streamlit run streamlit_app.py
```

Then open <http://localhost:8501> and upload a CSV or Excel file.

## Production

Run behind a WSGI server using the exported Flask app:

```bash
gunicorn --bind 0.0.0.0:8050 app:server
```

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open <https://share.streamlit.io> and sign in with GitHub.
3. Choose **New app** and select this repository.
   Use `adnan2002/mashroo3i-streamlit-dashboard`, branch `main`.
   The older public `mashroo3i-dashboard` repository is a different app
   and will fail with a missing local CSV file.
4. Set main file path to `streamlit_app.py`, **not** `app.py`.
5. Streamlit will install `requirements.txt` and start the app automatically.

The Streamlit build uses the same charts, filters, and warm orange/peach theme
as the Dash version.

## Idea Validator page

The Streamlit app includes an **Idea Validator** page that does exactly two
things:

1. **Selection scoring** - searches the web (Tavily) and scores the idea
   against the official selection rubric: Problem / Need, Solution / Idea,
   Innovation / Differentiation, Market Potential, and Feasibility - each
   /5, total /25. Market Potential and Feasibility explicitly weigh the
   idea's **impact on Bahrain** (local jobs, market, regulations, national
   alignment), and the report includes a short Bahrain-impact note.

The page shows a live staged status ("Evaluating / Searching the web (i/6) /
Generating your /25 selection score") while working, then renders the
structured score card, dashboard insights, and sources when finished.

The sidebar is split into two sections: **Dashboard** (the analytics pages)
and **AI** (Idea Validator, Similarity & Search, Selection Advisor).

The agent uses the already-uploaded CSVs; no separate upload is needed. Add
these keys to Streamlit secrets:

Locally, create `.streamlit/secrets.toml` (gitignored):

```toml
DEEPSEEK_API_KEY = "sk-<your-key>"
TAVILY_API_KEY = "tvly-..."
AGENT_MODEL = "deepseek-v4-flash"   # optional; defaults to deepseek-v4-flash
```

On Streamlit Community Cloud, enter the same keys under
**Settings → Secrets** (e.g. `DEEPSEEK_API_KEY = "sk-..."`). Keys are read
from secrets first, then from environment variables, so CLI tools with
exported keys still work.

The default model is DeepSeek `deepseek-v4-flash` (free-tier alternatives can
be selected with `AGENT_MODEL`). The model is invoked through DeepSeek's
OpenAI-compatible endpoint; the agent automatically falls back to a JSON
tool-request flow when the model does not emit native tool calls.

## Evaluate the agent

Run the evaluation harness on real data (2024-2025 only, 2023 excluded):

```bash
python evaluate_agent.py
python evaluate_agent.py --samples 6 --out /tmp/mashroo3i_eval.json
python evaluate_agent.py --offline   # no live API calls
```

### Selection criteria

The current five-criterion /25 rubric is loaded from `selection_criteria.json`
in the project root. Both **Idea Validator** and **Selection Advisor** show
the same editable criteria table: name, guidance text, and weight.

- Add/remove rows with the table editor. Weights are normalised to 100% when
  generating a score, so the total stays /25.
- Clicking **Validate & Analyze** or **Run agent** uses the in-memory criteria
  only; the criteria file is not changed.
- Clicking **Save criteria** overwrites `selection_criteria.json`. The
  Bahrain-impact scoring instruction and `bahrain_impact` response field stay
  fixed in the backend prompt.

### Selection Advisor page in Streamlit

The app has a **Selection Advisor** page that runs the same cascade
interactively:

1. Upload the full Brinc applicant CSV once in the dashboard sidebar. The
   Selection Advisor uses exactly that dashboard-loaded file, with no separate
   upload control. Then **Rank candidates with the Brinc model** -- it loads
   the bundled `final_model.joblib` from the project `model/` folder and shows
   the primary shortlist.
2. Pick how many of the top candidates to review, then **Run agent on top N** --
   the agent searches the web and scores each one on the /25 rubric, **streaming
   each candidate's progress live** (status + result as it completes), then a
   summary table appears. Results are cached per candidate.
3. Review the combined table (model rank + /25 selection score/verdict) and
   expand each candidate for evidence, risks, and recommendations.

The sidebar filters (Years, Cohorts, Outcomes, Sectors, Applicant Type) also
apply to the dashboard-loaded applicant rows: the shortlist is re-ranked on
the filtered population.

The `model/` folder is self-contained: `final_model.joblib`,
`final_threshold.json` (current operating point: 0.125), and vendored
`features_build.py`. Set `BRINC_MODEL_DIR` to override the bundle location.
Edit `model/final_threshold.json` to raise/lower the predicted-accepted cutoff
(lower = more recall; higher = fewer, more precise candidates).

### Similarity & Search page and similarity notices

The **Similarity & Search** page (in the **AI** section) has two tabs:

- **Similar ideas** compares every past idea with every other past idea and
  groups the related ones into clusters. Each group is shown once as a member
  list (name · cohort · sector; no duplicated pairs), and its "See full
  descriptions" expander shows, for every idea, the closest match inside the
  group as a percentage, e.g. "Most similar to Kazizo Cafe: 88% (Similar)",
  along with the full problem and solution. Relationships of 92% or higher are
  labelled "very similar (likely the same idea)"; anything from the chosen
  threshold up to 92% is labelled "similar (different idea, same concept)".
  The similarity threshold is adjustable from 60% to 95% (default 70%), and
  each idea appears in only one group of up to 12 members (larger similarity
  neighborhoods are split across groups). It works fully offline from the
  bundled index files in the project root: `idea_embeddings.npy`,
  `ideas_metadata.csv`, and `ideas_index.json` (plus `idea_search.py`, the
  builder used to refresh them). Identical, same-name, and near-identical
  repeats are collapsed.
- **Search** lets you type any query and shows up to 10 matching past ideas
  in a table with a "Score (%)" column. Pressing Enter or clicking **Search**
  runs the query. It uses brute-force keyword matches first and falls back to
  semantic matches (Voyage) when no keyword match is found.

The **Idea Validator** also shows a brief automatic notice once an idea and
description are entered, comparing the new idea with its closest past
submissions (name, problem, and solution). This query-time check embeds the
new text with the Voyage API, so add the key to Streamlit secrets:

```toml
VOYAGE_AI_API_KEY = "pa-<your-key>"
```

On Streamlit Community Cloud, add the same key under **Settings → Secrets**.
The flagging page itself needs no key; search falls back to keywords without
one.

## Test

```bash
python tests/test_dashboard.py
```
