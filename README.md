# Mirror Mind

This project provides a simple interface for experimenting with character and scenario prompts. It uses [Gradio](https://gradio.app/) and other Python libraries.

## Requirements

- **Python**: 3.10 or newer

## Installation

Install the required dependencies with `pip`:

```bash
pip install -r requirements.txt
```

## Running the application

Start the application by executing `app.py`:

```bash
python app.py
```

The script launches a local Gradio server on port `7861`. Ensure that LM Studio is running on `http://localhost:1234` as printed when the app starts.

## Running the tests

The test suite uses `pytest`. After installing the dependencies, run:

```bash
pytest
```

