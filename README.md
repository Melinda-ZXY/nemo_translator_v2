# Nemo Translator V2

A separate Streamlit demo for the second Nemo lexicon.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

For Streamlit Community Cloud, create a new app from this repository and set the entry point to:

```text
app.py
```

Orthography/glyph output uses SVG assets from `assets/version2`. Tokens without a matching SVG keep a placeholder in the UI.
