.PHONY: sync format lint type test check

sync:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy src tests

test:
	uv run pytest

check: format lint type test

