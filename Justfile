# https://github.com/casey/just

# Don't show the recipe name when running
set quiet

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

setup:
	uv venv --python 3.13
	uv sync --all-extras --cache-dir .uv_cache
	uv run pre-commit install

# Sync dev dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

# Sync production dependencies (excludes dev dependencies)
prod-sync:
	uv sync --all-extras --no-dev --cache-dir .uv_cache

# Install pre commit hooks
install-hooks:
	uv run pre-commit install

# Run ruff formatting
format:
	uv run ruff format

# Run ruff linting and mypy type checking
lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package warghame_rl

# Run tests using pytest
test:
	uv run pytest --verbose --color=yes tests

# Run all checks: format, lint, and test
validate: format lint test

# Build docker image
dockerize:
	docker build -t warghame-rl .

# Initiate RL model training
run-training:
	uv run main.py --train

# Run the model in inference mode, rendering the environment
run-inference:
	uv run main.py --render

run-env-test:
	uv run main.py --env_test