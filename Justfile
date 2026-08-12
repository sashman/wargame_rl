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
# Or with an epoch cap: just train path/to/config.yaml 800
# Or with a match event log for analysis: just train path/to/config.yaml 800 true
train env_config_path='configs/dev/4v4_two_phases.yaml' max_epochs='' record_events='' *extra='':
	@uv run train.py --record-during-training \
		--env-config-path {{env_config_path}} \
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
		uv run train.py --record-during-training --env-config-path "$c" --max-epochs {{max_epochs}} --run-suffix "$i" --wandb-group "$group" & \
		i=$((i+1)); \
	done && \
	wait

# Like train-multi-epochs, but runs every config at each of N seeds, so an arm
# carries an error bar. Needed because the dice alone contribute more outcome
# spread than the scenario does (`just measure-noise-floor`), which makes any
# single-seed difference under ~10pp unreadable.
#
# One seed group at a time. A PPO transformer run holds ~3.8 GB of VRAM, so
# eight at once overflows a 24 GB card and the losers die on a 12 MiB
# allocation partway through training -- not at startup, where it would be
# obvious. Keep the concurrent count equal to the number of configs.
#
# Use: just train-multi-seeds 1000 2 config1.yaml config2.yaml   (-> 4 runs)
train-multi-seeds max_epochs n_seeds *configs:
	@trap 'kill 0' INT TERM && \
	group="train-multi-$(date +%Y-%m-%d-%H-%M-%S)" && \
	for s in $(seq 1 {{n_seeds}}); do \
		echo "=== seed $s of {{n_seeds}} ===" && \
		for c in {{configs}}; do \
			uv run train.py --record-during-training --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed "$s" --run-suffix "s$s" --wandb-group "$group" & \
		done; \
		wait; \
	done

# Run every config once at a single seed, in an existing wandb group. For
# re-running one seed group of a train-multi-seeds batch that died partway --
# passing the original group keeps the reruns next to their siblings in the UI,
# which train-multi-seeds cannot do because it always mints a fresh group and
# always starts from seed 1.
# Use: just train-seed 1000 2 train-multi-2026-08-05-12-06-29 config1.yaml config2.yaml
train-seed max_epochs seed group *configs:
	@trap 'kill 0' INT TERM && \
	for c in {{configs}}; do \
		uv run train.py --record-during-training --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed {{seed}} --run-suffix "s{{seed}}" --wandb-group "{{group}}" & \
	done; \
	wait

# One arm of a screen: every config at each of N seeds, with an explicit run-name
# tag and extra train.py flags appended. `train-multi-seeds` cannot do this --
# it mints its own group and passes no flags, so a 2x2 over (config x flag)
# needs the flag axis driven from outside.
#
# `tag` is appended to the run suffix, so arms sharing a config still land in
# distinct checkpoint directories and are distinguishable in the Wandb UI.
# Pass an empty `flags` for the no-flag arm; keep `group` the same across arms
# so the whole screen groups together.
#
# One seed group at a time, same as train-multi-seeds: a PPO transformer run
# holds ~3.8 GB of VRAM, so keep the concurrent count equal to the config count.
#
# Use: just train-arm 1000 2 my-screen tag "--some-flag" a.yaml b.yaml
train-arm max_epochs n_seeds group tag flags *configs:
	@trap 'kill 0' INT TERM && \
	for s in $(seq 1 {{n_seeds}}); do \
		echo "=== seed $s of {{n_seeds}} ({{tag}}) ===" && \
		for c in {{configs}}; do \
			uv run train.py --record-during-training --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed "$s" --run-suffix "s$s{{tag}}" --wandb-group "{{group}}" {{flags}} & \
		done; \
		wait; \
	done

# One seed of each config, with an explicit tag and extra train.py flags, in an
# existing wandb group. `train-arm` covers the same ground but runs its seeds
# sequentially, so a two-seed arm costs two full training windows; invoking this
# twice in parallel puts both seeds in one. Use it when the seeds must land
# together -- a control measured against an existing run, where waiting out a
# second window buys nothing.
#
# Keep the concurrent count equal to the number of runs: each holds ~3.8 GB.
# Use: just train-seed-flags 1000 1 my-group -notf32 "--no-tf32" a.yaml
train-seed-flags max_epochs seed group tag flags *configs:
	@trap 'kill 0' INT TERM && \
	for c in {{configs}}; do \
		uv run train.py --record-during-training --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed {{seed}} --run-suffix "s{{seed}}{{tag}}" --wandb-group "{{group}}" {{flags}} & \
	done; \
	wait

# Run multiple env configs in parallel. Each run gets a unique --run-suffix and shared --wandb-group.
# Uses PPO + transformer. Use: just train-multi config1.yaml config2.yaml
# Trap INT/TERM so Ctrl+C kills all background train.py processes.
train-multi *configs:
	@trap 'kill 0' INT TERM && \
	group="train-multi-$(date +%Y-%m-%d-%H-%M-%S)" && \
	i=1 && \
	for c in {{configs}}; do \
		uv run train.py --record-during-training --env-config-path "$c" --run-suffix "$i" --wandb-group "$group" & \
		i=$((i+1)); \
	done && \
	wait

simulate-latest:
	uv run simulate.py

simulate checkpoint env_config_path:
	uv run simulate.py --checkpoint-path {{checkpoint}} --env-config-path {{env_config_path}}

# Record a match event log from a trained checkpoint (no rendering) for analysis.
# Use it like: just record-sim checkpoints/<run>/best.ckpt configs/golden/25v25_shooting_opponent.yaml
record-sim checkpoint env_config_path num_episodes='1':
	@uv run simulate.py \
		--checkpoint-path {{checkpoint}} \
		--env-config-path {{env_config_path}} \
		--num-episodes {{num_episodes}} \
		--no-render \
		--record-events

clean-checkpoints:
	rm -rf checkpoints/

clean-wandb:
	rm -rf wandb/

clean: clean-checkpoints clean-wandb

# Profile training with pyinstrument (HTML output, no recording)
# --no-wandb because profiling should not open a live run and log a fake experiment.
# Use it like: just profile path/to/config.yaml
# Or with max epochs: just profile path/to/config.yaml 10
profile env_config_path max_epochs='5':
	uv run pyinstrument -r html -o profile.html train.py \
		--no-wandb \
		--env-config-path {{env_config_path}} \
		{{ if max_epochs != "" { "--max-epochs " + max_epochs } else { "" } }}

# Where an epoch's wall-clock goes: per-section and per-reward-calculator env.step cost.
# Pass `engaged` as the third argument to force full engagement and quote the
# line-of-sight ceiling rather than the cost under a policy that never closes.
# Use it like: just measure-throughput configs/golden/25v25_single_phase.yaml
measure-throughput env_config n_steps='400' engaged='':
	@uv run python -m scripts.measure_throughput {{env_config}} {{n_steps}} {{engaged}}

# Record a short training run with event logging (E2E demo: 1 epoch, no wandb)
record env_config_path='configs/dev/tiny.yaml':
	uv run train.py --record-events --max-epochs 1 --no-wandb --env-config-path {{env_config_path}}

# Replay a recorded match event log (narrate all steps)
replay file:
	uv run replay_events.py narrate {{file}}

# Show summary of a recorded match event log
replay-summary file:
	uv run replay_events.py summary {{file}}

# Replay a recording visually: an interactive window (play/pause/step/scrub) or,
# with an out path, an MP4. Reads terrain from schema-2.1 recordings.
replay-render file out='':
	uv run replay_events.py render {{file}} {{ if out != '' { '--out ' + out } else { '' } }}

# Compact rolling-mean summary of a Wandb training run. Use: just run-summary <run_id> [bucket]
run-summary run_id bucket='50':
	@uv run python -m scripts.run_summary {{run_id}} {{bucket}}

# Measure every reward phase's criteria against a checkpoint, plus the min_fraction curve
measure-phase-gates checkpoint env_config n_episodes='30':
	@uv run python -m scripts.measure_phase_gates {{checkpoint}} {{env_config}} {{n_episodes}}

# Scripted baseline scores for an env config -- the floor and bar for any learned policy.
# Pass `record` as the third argument to also write reference traces to recordings/.
# Pass `seed_base` (e.g. 700000) to score on the same layouts as measure-checkpoint.
#
# n=100 to match measure-checkpoint's default: an agent row and a baseline row
# must be drawn from the same layout set *and* the same number of episodes, or
# the comparison inherits the larger of the two error bars.
measure-baselines env_config n_episodes='100' record='' seed_base='':
	@uv run python -m scripts.measure_baselines {{env_config}} {{n_episodes}} "{{record}}" "{{seed_base}}"

# Score a checkpoint on held-out seeds through the same code path as the baselines,
# so the two are directly comparable. Pass `record` as the fourth argument for a trace.
#
# n=100, not 30. Per-episode vp_margin sd is ~45-50 on the 25v25 configs, so the
# standard error on the mean is ~8-9 at n=30 -- larger than most arm differences
# ever measured here (4-10 vp), which made the authoritative measurement unable
# to resolve what it was measuring. n=100 halves that to ~4.5 and costs minutes
# against the hours a training run costs. Scoring was the cheap half being
# under-sampled while the expensive half was over-sampled.
measure-checkpoint checkpoint env_config n_episodes='100' record='':
	@uv run python -m scripts.measure_checkpoint {{checkpoint}} {{env_config}} {{n_episodes}} "{{record}}"

# Final evaluation: score a policy on the real table layouts, one row per map.
# Training uses `random_terrain`, so this is the only thing that asks how the
# policy does on the boards the game is actually played on. It runs the golden
# scenario unchanged and swaps only `terrain`, so evaluation cannot drift from
# what was trained. Takes a baseline name or a checkpoint, like
# measure-objective-split.
# Use it like: just measure-maps <ckpt> configs/golden/25v25_shooting_opponent.yaml
measure-maps policy env_config n_episodes='100' maps_dir='':
	@uv run python -m scripts.measure_maps {{policy}} {{env_config}} {{n_episodes}} {{maps_dir}}

# Re-render the preview PNG beside every evaluation map
# Use it like: just render-maps
render-maps env_config='' maps_dir='':
	@uv run python -m scripts.render_maps {{env_config}} {{maps_dir}}

# Which reward calculator actually pays a policy, and how much of the ledger is
# global. Weights are not shares: a small global term paid to every model on
# every step is a floor a movement term has to compete with, and a large weight
# that rarely fires is cheap. Run it before tuning a term, to check the term is
# a meaningful share at all -- `model_kills` looked like the driver of a
# range-managing policy and measured 4.5% of income. NOTE the inference this
# supports is narrow: a share of MEAN income rules a term out as the largest
# income stream, and does NOT rule it out as the driver of a behaviour. What
# moves a policy gradient is a term's variation across the actions being
# compared -- a kill is a lumpy 2.0 in one model's own row against
# objective_hold's 0.15-0.30/step. Takes a baseline name or a checkpoint,
# like measure-objective-split.
measure-income-share policy env_config n_episodes='30':
	@uv run python -m scripts.measure_income_share {{policy}} {{env_config}} {{n_episodes}}

# Two scripted policies over the SAME seed list, differenced per episode.
# `measure-baselines` prints one aggregate row each, and on 25v25 the
# per-episode vp_margin sd is ~45-90, so two such rows cannot resolve anything
# under ~10-18 vp at n=100 -- larger than most effects measured here. Pairing
# removes the layout variance those rows are made of. Written after a
# target-selection comparison read +8.0 unpaired at n=60 and +1.7 +/- 5.7 paired
# at n=100; the first number was noise. Read the win count beside the mean -- a
# positive mean with a losing win count is a heavy tail, not an improvement.
# Use: just measure-paired squad_march_shoot contest_and_spread <config> 100
measure-paired policy_a policy_b env_config n_episodes='100' seed_base='700000':
	@uv run python -m scripts.measure_paired_policies {{policy_a}} {{policy_b}} {{env_config}} {{n_episodes}} {{seed_base}}

# Why an objective was not held: abandoned, narrowly lost, or lost by a mile.
# `held` alone cannot separate those, and they call for different fixes. Also
# reports the redistribution ceiling -- what any pure re-allocation lever could
# buy at best -- so a reward-shaping idea can be ruled out before it is trained.
# Takes a baseline name or a checkpoint path.
measure-objective-split policy env_config n_episodes='100':
	@uv run python -m scripts.measure_objective_split {{policy}} {{env_config}} {{n_episodes}}

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

# Compare the v2 render backends (pygame / pygame_aa / pillow): one Scene through
# each, PNGs + contact sheets + ms/frame, so the default is picked by evidence.
render-bakeoff out_dir='bakeoff_out' n_timing='20':
	@uv run python -m scripts.render_bakeoff {{out_dir}} {{n_timing}}

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

# Watch a scripted policy play in a window — no checkpoint needed. [Tab] lists the keys.
# Use it like: just play · just play configs/dev/tiny.yaml random · just play <cfg> squad_march tabletop
play env_config_path='configs/golden/25v25_shooting_opponent.yaml' policy='squad_march_shoot' theme='default':
	uv run play.py {{env_config_path}} {{policy}} {{theme}}

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
