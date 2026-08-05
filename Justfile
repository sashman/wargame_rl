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
	uv run mypy --ignore-missing-imports --install-types --non-interactive wargame_rl/ tests/ scripts/

# Run tests using pytest with coverage (parallel via xdist)
test:
	uv run pytest -n auto --verbose --color=yes --cov=wargame_rl --cov-report=term-missing --cov-report=xml:coverage.xml tests

# Run all checks: format, lint, and test
validate: format lint test

# Build docker image
dockerize:
	docker build -t warghame-rl .

# Use it like: just train path/to/config.yaml
# Or with algorithm: just train path/to/config.yaml ppo
# Or with algorithm and network: just train path/to/config.yaml dqn transformer
# Or with an epoch cap: just train path/to/config.yaml ppo transformer 800
# Or with a match event log for analysis: just train path/to/config.yaml ppo transformer 800 true
train env_config_path='examples/env_config/4v4_scripted_opponent_fixed_objectives_2_reward_phases.yaml' algorithm='ppo' model='transformer' max_epochs='' record_events='' *extra='':
	@uv run train.py --record-during-training \
		--env-config-path {{env_config_path}} \
		--algorithm {{algorithm}} \
		{{ if model != "" { "--network-type " + model } else { "" } }} \
		{{ if max_epochs != "" { "--max-epochs " + max_epochs } else { "" } }} \
		{{ if record_events != "" { "--record-events" } else { "" } }} \
		{{extra}}

# Like train-multi, but caps every arm at the same epoch count so the arms stay
# comparable and the experiment ends on its own.
# Use: just train-multi-epochs 1000 config1.yaml config2.yaml
train-multi-epochs max_epochs *configs:
	@trap 'kill 0' INT TERM && \
	group="train-multi-$(date +%Y-%m-%d-%H-%M-%S)" && \
	i=1 && \
	for c in {{configs}}; do \
		uv run train.py --record-during-training --env-config-path "$c" --algorithm ppo --network-type transformer --max-epochs {{max_epochs}} --run-suffix "$i" --wandb-group "$group" & \
		i=$((i+1)); \
	done && \
	wait

# Like train-multi-epochs, but runs every config at each of N seeds, so an arm
# carries an error bar. Needed because the dice alone contribute more outcome
# spread than the scenario does (`just measure-noise-floor`), which makes any
# single-seed difference under ~10pp unreadable.
# Use: just train-multi-seeds 1000 2 config1.yaml config2.yaml   (-> 4 runs)
train-multi-seeds max_epochs n_seeds *configs:
	@trap 'kill 0' INT TERM && \
	group="train-multi-$(date +%Y-%m-%d-%H-%M-%S)" && \
	for s in $(seq 1 {{n_seeds}}); do \
		for c in {{configs}}; do \
			uv run train.py --record-during-training --env-config-path "$c" --algorithm ppo --network-type transformer --max-epochs {{max_epochs}} --seed "$s" --run-suffix "s$s" --wandb-group "$group" & \
		done; \
	done && \
	wait

# Run multiple env configs in parallel. Each run gets a unique --run-suffix and shared --wandb-group.
# Uses PPO + transformer. Use: just train-multi config1.yaml config2.yaml
# Trap INT/TERM so Ctrl+C kills all background train.py processes.
train-multi *configs:
	@trap 'kill 0' INT TERM && \
	group="train-multi-$(date +%Y-%m-%d-%H-%M-%S)" && \
	i=1 && \
	for c in {{configs}}; do \
		uv run train.py --record-during-training --env-config-path "$c" --algorithm ppo --network-type transformer --run-suffix "$i" --wandb-group "$group" & \
		i=$((i+1)); \
	done && \
	wait

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

# Record a match event log from a trained checkpoint (no rendering) for analysis.
# Use it like: just record-sim checkpoints/<run>/best.ckpt examples/env_config/foo.yaml
record-sim checkpoint env_config_path num_episodes='1' network_type='transformer':
	@uv run simulate.py \
		--checkpoint-path {{checkpoint}} \
		--env-config-path {{env_config_path}} \
		--network-type {{network_type}} \
		--num-episodes {{num_episodes}} \
		--no-render \
		--record-events

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

# Record a short training run with event logging (E2E demo: 1 epoch, no wandb)
record env_config_path='examples/env_config/4_models_2_objectives_fixed.yaml':
	uv run train.py --record-events --max-epochs 1 --no-wandb --env-config-path {{env_config_path}}

# Replay a recorded match event log (narrate all steps)
replay file:
	uv run replay_events.py narrate {{file}}

# Show summary of a recorded match event log
replay-summary file:
	uv run replay_events.py summary {{file}}

# Compact rolling-mean summary of a Wandb training run. Use: just run-summary <run_id> [bucket]
run-summary run_id bucket='50':
	@uv run python -m scripts.run_summary {{run_id}} {{bucket}}

# Measure every reward phase's criteria against a checkpoint, plus the min_fraction curve
measure-phase-gates checkpoint env_config n_episodes='30':
	@uv run python -m scripts.measure_phase_gates {{checkpoint}} {{env_config}} {{n_episodes}}

# Scripted baseline scores for an env config -- the floor and bar for any learned policy.
# Pass `record` as the third argument to also write reference traces to recordings/.
# Pass `seed_base` (e.g. 700000) to score on the same layouts as measure-checkpoint.
measure-baselines env_config n_episodes='25' record='' seed_base='':
	@uv run python -m scripts.measure_baselines {{env_config}} {{n_episodes}} "{{record}}" "{{seed_base}}"

# Score a checkpoint on held-out seeds through the same code path as the baselines,
# so the two are directly comparable. Pass `record` as the fourth argument for a trace.
measure-checkpoint checkpoint env_config n_episodes='30' record='':
	@uv run python -m scripts.measure_checkpoint {{checkpoint}} {{env_config}} {{n_episodes}} {{record}}

# How much of a config's outcome spread is dice rather than policy. Holds the
# layouts fixed and varies only the combat seed, so the within-layout spread is
# the noise floor any arm-to-arm difference has to clear.
measure-noise-floor env_config n_layouts='10' n_combat_seeds='10' policy='':
	@uv run python -m scripts.measure_noise_floor {{env_config}} {{n_layouts}} {{n_combat_seeds}} "{{policy}}"

# Terrain-layout statistics for a random_terrain config: coverage, how often a
# sightline is blocked, and how much of the board is genuinely out of sight.
# Tune a terrain profile here, not after a thousand epochs of training.
measure-terrain env_config n_layouts='200':
	@uv run python -m scripts.measure_terrain {{env_config}} {{n_layouts}}

# Analyze a recorded match for training evaluation
analyze file:
	uv run analyze_events.py report {{file}}

# Analyze a recorded match (JSON output for programmatic use)
analyze-json file:
	uv run analyze_events.py report {{file}} --json

# Compare multiple recorded matches side-by-side
analyze-compare +files:
	uv run analyze_events.py compare {{files}}

# Run a test env in isolation with random action
test-env:
	uv run main.py --env_test

# One-shot: create branch from main, commit, push, open PR. Use after staging changes.
# Always branches from main; if not on main, checks out main and pulls first.
# Example: just ship feature/my-feature "Add reward shaping for distance"
# PR title and body are filled from the commit message (gh pr create --fill).
# Commit uses title + body so --fill gets a PR description (body = same as title).
ship branch commit_message:
	git stash -u
	git checkout main
	git pull
	@git checkout -b {{branch}} 2>/dev/null || git checkout {{branch}}
	git stash pop
	git add -A
	git commit -m "{{commit_message}}" -m "{{commit_message}}"
	git push -u origin {{branch}}
	gh pr create --fill
