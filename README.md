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

## Fish Audio TTS

Set these in Streamlit Community Cloud `Settings -> Secrets`:

```toml
FISH_API_KEY = "your_fish_audio_api_key"
FISH_REFERENCE_ID = "your_speaker_or_reference_id"
FISH_MODEL = "s2-pro"
```

`FISH_SPEAKER_ID` is also accepted as an alias for `FISH_REFERENCE_ID`.

The app also has a `Fish Audio 设置` panel where API key and speaker/reference ID can be entered temporarily in the browser. Streamlit Secrets are still recommended for deployment.

## Emotional Audio Post-Processing

The app requests WAV audio from Fish Audio, then applies text-independent waveform post-processing in `audio_emotion.py`.

Supported modes:

- `normal`
- `urgent`
- `angry`
- `angry_urgent`

The module does not use the source sentence or syllable labels. It analyzes RMS energy, pauses, phrase boundaries, local peaks, onsets, and final voiced tails directly from the waveform.

Standalone use:

```python
from audio_emotion import process_wav_file

process_wav_file("input.wav", "output_angry.wav", mode="angry")
```

Preset parameters are exposed through `EMOTION_PRESETS` and `get_emotion_preset()`.
