.PHONY: install format lint typecheck test check train predict

install:
	poetry install
	poetry run pre-commit install

format:
	poetry run ruff format src tests
	poetry run ruff check src tests --fix

lint:
	poetry run ruff check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest

check:
	poetry check
	poetry run ruff format --check src tests
	poetry run ruff check src tests
	poetry run mypy src
	poetry run pytest

train:
	poetry run python -m loan_approval.train

predict:
	poetry run python -m loan_approval.predict
