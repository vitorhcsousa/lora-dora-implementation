.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel pipenv && \
	python -m pip install -e .


.PHONY: format
format:
	black dora_implementation
	ruff check --fix dora_implementation

.PHONY: lint
lint:
	poetry run pylint dora_implementation

.PHONY: reset-env
reset-env:
	rm -rf venv
	make venv