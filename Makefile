.PHONY: sync format format-check lint lint-fix type test check fix

sync:
	uv sync

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

type:
	uv run mypy src

test:
	uv run pytest

check: format-check lint type test

fix: lint-fix format
