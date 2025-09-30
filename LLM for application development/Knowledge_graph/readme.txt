# Knowledge Graph Extraction with OpenAI

This project shows how to build a simple knowledge graph from unstructured text
using a large language model via the OpenAI API. It focuses on a clean and
minimal codebase and provides both a command‑line tool and a small web UI
powered by Streamlit.

Unlike earlier versions of this demo, **there is no heuristic entity
extraction**. All graphs are extracted by the model itself. The API key is
read from the `OPENAI_API_KEY` environment variable; you never pass it on the
command line.

## Features

- 🔁 **LLM‑powered extraction** – Uses an OpenAI chat model (default
  `gpt‑3.5‑turbo`) to return a JSON object containing nodes and edges. The
  prompt instructs the model to capture entities and their relationships
  present in the text.
- 🧱 **Clean codebase** – The logic lives in a single file (`app.py`) and is
  easy to read and extend. There are no unit tests or heuristic fallbacks.
- 📄 **Command‑line interface** – Extract a knowledge graph from a file,
  inline text, or interactively from stdin. After extraction it prints
  nodes, edges and a simple adjacency list to demonstrate how to query the
  graph.
- 🌐 **Streamlit UI** – A minimal web interface (`streamlit_app.py`) lets
  users paste text, choose a model, extract a graph, and interact with the
  resulting nodes and neighbors. Ideal for demonstrations and educational
  purposes.
- 🧠 **Graph exploration** – Builds an in‑memory graph using NetworkX (if
  installed) and displays neighbors for selected nodes. Falls back to a
  simple adjacency dictionary if NetworkX is unavailable.

## Requirements

Install the dependencies listed in `requirements.txt`. At minimum you
need the `openai` package. To run the Streamlit app and build graphs with
NetworkX you also need `streamlit` and `networkx`.

## Usage

1. **Set your API key** (required):

```bash
export OPENAI_API_KEY="sk-..."  # your OpenAI API key
```

2. **Command‑line extraction**:

Process a text file:

```bash
python app.py --file data/sample.txt
```

Process inline text:

```bash
python app.py --text "Ada Lovelace collaborates with Charles Babbage."
```

Run in interactive mode (paste text, then Ctrl‑D/Ctrl‑Z):

```bash
python app.py --interactive
```

The script prints the list of nodes and edges extracted by the model,
followed by an adjacency list showing which entities are connected.

3. **Web UI with Streamlit**:

Launch the app:

```bash
streamlit run streamlit_app.py
```

Paste text into the editor, select a model, click “Extract graph” and explore
the neighbors of any node via the drop‑down.

## Data

A sample text file (`data/sample.txt`) is included for experimentation.
Replace it or paste your own text into the CLI or web app.

## Extending the demo

To adapt this project for your own needs you can:

- Modify the prompt in `extract_with_openai` to enforce different schemas or
  include additional attributes such as types and timestamps.
- Swap `gpt‑3.5‑turbo` for `gpt‑4` via the `--model` option or the UI.
- Load the extracted graph into Neo4j or another graph database for
  persistent storage and complex queries.
