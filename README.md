# Mashroo3i Dashboard

Analytics dashboard for Mashroo3i applicant data. This project is a plain
Python conversion of `Dashboard_Mashroo3i.ipynb`, so it can be run, deployed,
and tested without a notebook kernel.

## Project layout

- `app.py` - the Dash application (layout, callbacks, entrypoint)
- `streamlit_app.py` - Streamlit deployment entrypoint (reuses the same chart logic)
- `.streamlit/config.toml` - Streamlit theme/server configuration
- `make_dummy_mashroo3i.py` - generates a synthetic CSV for development/testing
- `mashroo3i_dtypes.py` - pandas dtype map for the cleaned source CSV
- `tests/test_dashboard.py` - smoke tests (no browser required)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Local only:

```bash
python app.py --no-ngrok
```

Then open <http://localhost:8050> and upload a CSV (for example
`dummy_mashroo3i_2023_2026.csv`, generated with
`python make_dummy_mashroo3i.py`).

Streamlit (recommended for deployment):

```bash
streamlit run streamlit_app.py
```

Then open <http://localhost:8501> and upload a CSV or Excel file.

Publicly via ngrok:

```bash
cp .env.example .env   # then put your ngrok authtoken in .env
python app.py
```

The public URL is printed on startup. If the token is missing or ngrok fails,
the dashboard still starts locally instead of crashing.

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

## Test

```bash
python tests/test_dashboard.py
```
