.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel pipenv && \
	python -m pip install -e .


.PHONY: format
format:
	black dora-implementation
	ruff check --fix dora-implementation
