# Mirror Mind

Mirror Mind is a Gradio application for building rich AI personas and scenarios. It connects to LM Studio on `localhost:1234` and lets you craft characters, worlds and prompt packs that generate detailed system prompts.

## Features
- **Chat Interface** powered by LM Studio with adjustable generation parameters.
- **Enhanced Character Creator** with relationship and mood fields, custom traits and backstory preview.
- **Enhanced Scenario Designer** supporting environmental tone, cultural influences, story hooks and editable world rules.
- **Prompt Packs** combining one or two characters with a scenario using Jinja2 templates. Packs can be auto‑optimized via the OpenAI API.
- **JSON Storage** in the `data/` directory for characters, scenarios and prompt packs.

## Requirements
- Python 3.10+
- Packages listed in `requirements.txt`
- LM Studio running locally if you want to chat with a model.

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the App
Launch the Gradio UI:
```bash
python app.py
```
The interface will be available at `http://localhost:7861`.

## Tests
Basic test scripts are provided:
```bash
pytest -q
```
Tests may fail if required packages or network access for the OpenAI client are missing.

## Repository Layout
- `app.py` – Gradio interface definition.
- `ui_callbacks.py` – Callback functions for UI actions.
- `models/` – Pydantic models for characters, scenarios and prompt packs.
- `utils/` – Storage layer, prompt generation and LM Studio chat wrapper.
- `data/` – Saved JSON files for your characters, scenarios and prompt packs.
- `test_*.py` – Simple scripts to exercise prompt generation and storage.
