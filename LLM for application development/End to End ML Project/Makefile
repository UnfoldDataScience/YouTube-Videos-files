PYTHON := python
VENV := .venv
ACTIVATE := $(VENV)/bin/activate

# Default configuration file
CONFIG := config/config.yaml

install: setup

.PHONY: setup
setup:
	# Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV)
	. $(ACTIVATE) && pip install --upgrade pip
	. $(ACTIVATE) && pip install -r requirements.txt
	. $(ACTIVATE) && pip install pre-commit
	. $(ACTIVATE) && pre-commit install

.PHONY: train
train:
	. $(ACTIVATE) && $(PYTHON) -m src.train

.PHONY: eval
eval:
	. $(ACTIVATE) && $(PYTHON) -m src.evaluate

.PHONY: app
app:
	. $(ACTIVATE) && streamlit run app.py

.PHONY: test
test:
	. $(ACTIVATE) && pytest -q

.PHONY: lint
lint:
	. $(ACTIVATE) && ruff src tests
	. $(ACTIVATE) && black --check src tests

.PHONY: format
format:
	. $(ACTIVATE) && isort src tests
	. $(ACTIVATE) && black src tests

.PHONY: all
all: train eval

.PHONY: clean
clean:
	rm -rf $(VENV)
	rm -rf artifacts
	rm -rf logs