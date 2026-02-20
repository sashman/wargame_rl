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
	uv run mypy --ignore-missing-imports --install-types --non-interactive wargame_rl/ tests/

# Run tests using pytest
test:
	uv run pytest --verbose --color=yes tests

# Run all checks: format, lint, and test
validate: format lint test

# Build docker image
dockerize:
	docker build -t warghame-rl .

# Use it like: just train path/to/config.yaml
# Or with network type: just train path/to/config.yaml mlp
train env_config_path model='':
	@if [ -z "{{model}}" ]; then \
		uv run train.py --record-during-training --env-config-path {{env_config_path}}; \
	else \
		uv run train.py --record-during-training --env-config-path {{env_config_path}} --network-type {{model}}; \
	fi

simulate-latest network_type='':
	@if [ -z "{{network_type}}" ]; then \
		uv run simulate.py; \
	else \
		uv run simulate.py --network-type {{network_type}}; \
	fi

simulate checkpoint env_config_path network_type='':
	@if [ -z "{{network_type}}" ]; then \
		uv run simulate.py --checkpoint-path {{checkpoint}} --env-config-path {{env_config_path}}; \
	else \
		uv run simulate.py --checkpoint-path {{checkpoint}} --env-config-path {{env_config_path}} --network-type {{network_type}}; \
	fi

clean-checkpoints:
	rm -rf checkpoints/

clean-wandb:
	rm -rf wandb/

clean: clean-checkpoints clean-wandb

# Profile training with pyinstrument (HTML output, no recording)
# Use it like: just profile path/to/config.yaml
# Or with network type: just profile path/to/config.yaml mlp
# Or with max epochs: just profile path/to/config.yaml '' 10
profile env_config_path model='' max_epochs='':
	uv run pyinstrument -r html -o profile.html train.py \
		--env-config-path {{env_config_path}} \
		{{ if model != "" { "--network-type " + model } else { "" } }} \
		{{ if max_epochs != "" { "--max-epochs " + max_epochs } else { "" } }}

# Run a test env in isolation with random action
test-env:
	uv run main.py --env_test
