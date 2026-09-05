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
	@uv run train.py --record-during-training --record-threat-range --record-engagement-range \
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
		uv run train.py --record-during-training --record-threat-range --record-engagement-range --env-config-path "$c" --max-epochs {{max_epochs}} --run-suffix "$i" --wandb-group "$group" & \
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
			uv run train.py --record-during-training --record-threat-range --record-engagement-range --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed "$s" --run-suffix "s$s" --wandb-group "$group" & \
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
		uv run train.py --record-during-training --record-threat-range --record-engagement-range --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed {{seed}} --run-suffix "s{{seed}}" --wandb-group "{{group}}" & \
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
			uv run train.py --record-during-training --record-threat-range --record-engagement-range --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed "$s" --run-suffix "s$s{{tag}}" --wandb-group "{{group}}" {{flags}} & \
		done; \
		wait; \
	done

# The self-play MECHANISM SCREEN, pre-registered in
# reports/2026-08-31-self-play-preregistration.md. Launches the ARM and its
# CONTROL for one seed with BYTE-IDENTICAL flags apart from `--self-play`.
#
# ⚠ **Run both sides through THIS recipe, never one here and one through
# `train-coherency-baseline`.** That recipe adds `--record-every-n-epochs 10`,
# and a differing recording cadence is not something to assume is free -- the
# pair is a paired estimator only while the flags match. `--self-play` off
# constructs no scheduler at all, so the two runs start from bit-identical
# weights (`train.py` seeds before it builds the model).
#
# ⚠ **This stage returns NO VERDICT.** It passes if snapshots write and load
# back, the pool spans the run rather than collapsing onto its newest member,
# and nothing raises. Three seeds cannot resolve a difference under ~28 vp, so
# a score from it is not quoted. The deciding run is 6 seeds at 1000 epochs.
#
# ⚠ SIGKILL writes no snapshot -- the pool hook is a Lightning callback, same
# trap as `last.ckpt`. Score a killed run from its highest `ppo-NNN-*.ckpt`.
#
# Two concurrent runs per seed, ~3.8 GB of VRAM each.
#
# Use: just train-self-play-screen 300 1
# With melee: just train-self-play-screen 300 1 melee-screen configs/experiments/25v25_maps_melee_approach.yaml squad_march_take_charge
# ⚠ The ANCHOR must be able to use the features the config enables. It is the
# pool's permanent floor, never evicted, so on a melee config the default
# `squad_march_take` -- whose `select_charge` returns STAY -- pins the floor to
# a policy that never charges, and the learner can top the ladder without ever
# meeting a charge. Pass `squad_march_take_charge` there instead.
# ⚠ Run seeds SEQUENTIALLY: two runs per seed is ~7 GB of VRAM.
train-self-play-screen max_epochs='300' seed='1' group='self-play-screen' env_config='configs/golden/25v25_maps_two_mode.yaml' anchor='squad_march_take':
	@trap 'kill 0' INT TERM && \
	uv run train.py --record-during-training --record-threat-range --record-engagement-range \
		--env-config-path {{env_config}} --max-epochs {{max_epochs}} --n-eval-episodes 30 \
		--seed {{seed}} --ent-coef 0.003 --no-tf32 \
		--self-play --pfsp-mode uniform --snapshot-every-n-epochs 25 \
		--pool-capacity 8 --pool-anchor {{anchor}} \
		--run-suffix "s{{seed}}-selfplay" --wandb-group "{{group}}" & \
	uv run train.py --record-during-training --record-threat-range --record-engagement-range \
		--env-config-path {{env_config}} --max-epochs {{max_epochs}} --n-eval-episodes 30 \
		--seed {{seed}} --ent-coef 0.003 --no-tf32 \
		--run-suffix "s{{seed}}-control" --wandb-group "{{group}}" & \
	wait

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
		uv run train.py --record-during-training --record-threat-range --record-engagement-range --env-config-path "$c" --max-epochs {{max_epochs}} --n-eval-episodes 30 --seed {{seed}} --run-suffix "s{{seed}}{{tag}}" --wandb-group "{{group}}" {{flags}} & \
	done; \
	wait

# Recordings carry the THREAT and ENGAGEMENT overlays. Threat is range ∩ line
# of sight -- the same predicate the shooting mask uses -- so the picture
# cannot disagree with the rule, and it is the only way to see WHY a model
# did not shoot. A video has no keyboard, so these have to be set at launch.
# ⚠ They change what the VIDEO looks like, never what the run trains.
#
# THE COHERENCY BASELINE OF RECORD. Use this, not `just train`, for
# `configs/golden/25v25_maps_two_mode.yaml`: `ent_coef` is not an env-config
# field and the PPO default is 0.03, while the baseline is the 0.003 arm --
# worth +5.9 +/- 2.5 vp read paired on seed. Plain `just train` silently trains
# the worse arm.
#
# `first_seed` and `tag` exist to EXTEND an existing set of seeds rather than
# retrain it. Seed spread on this config is 11 vp, and on the four-opponent
# table it is larger than the differences being read, so the answer to a
# marginal result is more seeds -- which must not re-run the seeds already
# trained. `tag` keeps the new runs in the same checkpoint lineage as the old.
# Use: just train-coherency-baseline 300 3
# Use: just train-coherency-baseline 300 3 4 -newmaps   # seeds 4,5,6
#
# `env_config` exists so an ARM can be trained with byte-identical flags to the
# control. Pairing needs the same seed AND the same flags -- `train-seed-flags`
# omits `--record-every-n-epochs`, and a differing recording cadence is not
# something to assume is free. Defaulted, so every existing invocation is
# unchanged.
train-coherency-baseline max_epochs='300' n_seeds='3' first_seed='1' tag='-baseline' env_config='configs/golden/25v25_maps_two_mode.yaml':
	@trap 'kill 0' INT TERM && \
	for s in $(seq {{first_seed}} $(( {{first_seed}} + {{n_seeds}} - 1 ))); do \
		uv run train.py --record-during-training --record-every-n-epochs 10 \
			--record-threat-range --record-engagement-range \
			--env-config-path {{env_config}} \
			--max-epochs {{max_epochs}} --n-eval-episodes 30 --seed "$s" \
			--ent-coef 0.003 --run-suffix "s$s{{tag}}" \
			--wandb-group coherency-baseline & \
	done; \
	wait

# Can the policy explore WHERE a squad goes, or only how fast? A product policy
# pays p^k for DIRECTIONAL disagreement inside a squad and nothing for speed
# disagreement, so the surviving entropy is predicted to be one shared angle.
# Circular variance 0 means the squad has exactly one direction available to it.
#
# What a policy buys with the advance move and what it pays. Separates the three
# encoding defects: DOMINATED advances (cost the unit's shooting for a distance a
# normal move reaches), who pulled the trigger (one model forfeits all five), and
# advancing from INSIDE an objective (pays in full for no payable progress).
# Use: just measure-advance-use squad_march_take configs/experiments/25v25_maps_advance.yaml 10
measure-advance-use policy env_config n_episodes='10' decode_topk='1' *overrides:
	@uv run python -m scripts.measure_advance_use {{policy}} {{env_config}} {{n_episodes}} {{decode_topk}} {{overrides}}

# Use: just measure-angle-collapse squad_march_take configs/experiments/24v24_maps_spare_squads.yaml 20
measure-angle-collapse policy env_config n_episodes='20' decode_topk='1':
	@uv run python -m scripts.measure_angle_collapse {{policy}} {{env_config}} {{n_episodes}} {{decode_topk}}

# Do squads converge, or take separate objectives? The discriminating number is
# SQUADS PER OCCUPIED OBJECTIVE -- 1.00 means each squad has a point to itself.
# Squads move under a 2" chain and `objective_hold` requires coherence, so the
# squad is the allocation quantum; if squads bunch, no reward weight fixes it.
#
# Use: just measure-squad-dispersion squad_march_take configs/experiments/24v24_maps_spare_squads.yaml 20
measure-squad-dispersion policy env_config n_episodes='20' decode_topk='1':
	@uv run python -m scripts.measure_squad_dispersion {{policy}} {{env_config}} {{n_episodes}} {{decode_topk}}

# What standing on an objective earns a model, against what it costs it. Reports
# the income differential, the excess death hazard, and the hazard at which the
# two break even -- i.e. whether hiding is correct play under this reward.
#
# Use: just measure-hold-hazard squad_march_take configs/experiments/24v24_maps_spare_squads.yaml 30
measure-hold-hazard policy env_config n_episodes='30' decode_topk='1':
	@uv run python -m scripts.measure_hold_hazard {{policy}} {{env_config}} {{n_episodes}} {{decode_topk}}

# How often the mission's 15 VP per-turn cap binds, and what it discards. The
# fourth objective a side controls pays ZERO while the tables carry five or six,
# so `held` can rise without a single extra point being paid. `held` cannot see
# that; this can.
#
# Use: just measure-vp-cap squad_march_take configs/golden/25v25_maps_two_mode.yaml 20
measure-vp-cap policy env_config n_episodes='20' decode_topk='1':
	@uv run python -m scripts.measure_vp_cap {{policy}} {{env_config}} {{n_episodes}} {{decode_topk}}

# Does the critic believe the stack is right -- and is it? Forks a live game at a
# chosen round, rigidly translates one SURPLUS squad off an over-stacked objective
# onto an empty one, and prices the move twice: `dV` is what the critic thinks it
# is worth, `dVP` is what it turns out to be worth when both branches are played
# out. The two signs together separate "the reward is wrong" from "the search is
# wrong" from "the stack was correct all along" -- on frozen weights, no GPU.
# Use: just measure-critic-probe <ckpt> configs/experiments/24v24_maps_spare_squads_refereed.yaml 10 3,6,10 3
measure-critic-probe ckpt env_config n_episodes='10' rounds='3,6,10' decode_topk='3' reverse='' *overrides:
	@uv run python -m scripts.measure_critic_probe {{ckpt}} {{env_config}} {{n_episodes}} {{rounds}} {{decode_topk}} {{reverse}} {{overrides}}

# THE MIXED-ROLES ARM. Three seeds in parallel, `ent_coef` 0.003 (the PPO
# default is 0.03 and is the worse arm here by +5.9 +/- 2.5 vp read paired),
# recording on so the behaviour can be eyeballed as it trains.
#
# ⚠ The arm configs are UNREFEREED, like every training config here. Score the
# resulting checkpoints on the `_refereed` twin, never on the config they
# trained on -- the referee taxes a policy by how often it breaks coherency, so
# scoring unrefereed flatters the scripts by ~16 vp.
#
# ⚠ `40v40_maps_mixed_roles_spares.yaml` changes `max_groups` 5 -> 8 and the
# model count 25 -> 40, so it is a TENSOR-SHAPE change: it orphans every
# existing checkpoint and removes the paired estimator against the
# `two_mode` lineage. `25v25_maps_mixed_roles*.yaml` keep both.
#
# Use: just train-mixed-roles configs/experiments/25v25_maps_mixed_roles.yaml 300 3
train-mixed-roles config max_epochs='300' n_seeds='3' first_seed='1' tag='-mixed':
	@trap 'kill 0' INT TERM && \
	for s in $(seq {{first_seed}} $(( {{first_seed}} + {{n_seeds}} - 1 ))); do \
		uv run train.py --record-during-training --record-threat-range --record-engagement-range --record-every-n-epochs 10 \
			--env-config-path {{config}} \
			--max-epochs {{max_epochs}} --n-eval-episodes 30 --seed "$s" \
			--ent-coef 0.003 --run-suffix "s$s{{tag}}" \
			--wandb-group mixed-roles & \
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
		uv run train.py --record-during-training --record-threat-range --record-engagement-range --env-config-path "$c" --run-suffix "$i" --wandb-group "$group" & \
		i=$((i+1)); \
	done && \
	wait

simulate-latest:
	uv run simulate.py

simulate checkpoint env_config_path overlays='':
	uv run simulate.py --checkpoint-path {{checkpoint}} --env-config-path {{env_config_path}} {{overlays}}

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
replay-render file out='' theme='tabletop' overlays='':
	uv run replay_events.py render {{file}} --theme {{theme}} {{overlays}} {{ if out != '' { '--out ' + out } else { '' } }}

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
#
# Every measure-* recipe takes trailing `key=value` scenario overrides --
# `rounds=5`, `weapon_range=24`, `turn_order=player` -- so one config can be
# scored at several settings of one number without copying it. See
# scripts/scenario_overrides.py for why they are not positional.
measure-baselines env_config n_episodes='100' record='' seed_base='' *overrides:
	@uv run python -m scripts.measure_baselines {{env_config}} {{n_episodes}} "{{record}}" "{{seed_base}}" {{overrides}}

# Score a checkpoint on held-out seeds through the same code path as the baselines,
# so the two are directly comparable. Pass `record` as the fourth argument for a trace.
#
# n=100, not 30. Per-episode vp_margin sd is ~45-50 on the 25v25 configs, so the
# standard error on the mean is ~8-9 at n=30 -- larger than most arm differences
# ever measured here (4-10 vp), which made the authoritative measurement unable
# to resolve what it was measuring. n=100 halves that to ~4.5 and costs minutes
# against the hours a training run costs. Scoring was the cheap half being
# under-sampled while the expensive half was over-sampled.
measure-checkpoint checkpoint env_config n_episodes='100' record='' decode_topk='1' *overrides:
	@uv run python -m scripts.measure_checkpoint {{checkpoint}} {{env_config}} {{n_episodes}} "{{record}}" {{decode_topk}} {{overrides}}

# Final evaluation: score a policy on the real table layouts, one row per map.
# Training uses `random_terrain`, so this is the only thing that asks how the
# policy does on the boards the game is actually played on. It runs the golden
# scenario unchanged and swaps only `terrain`, so evaluation cannot drift from
# what was trained. Takes a baseline name or a checkpoint, like
# measure-objective-split.
# Use it like: just measure-maps <ckpt> configs/golden/25v25_shooting_opponent.yaml
measure-maps policy env_config n_episodes='100' maps_dir='' decode_topk='1' decode_stay='' *overrides:
	@uv run python -m scripts.measure_maps {{policy}} {{env_config}} {{n_episodes}} "{{maps_dir}}" "{{decode_topk}}" "{{decode_stay}}" {{overrides}}

# Regenerate every evaluation table from the public layout API. The tables were
# originally traced by hand from this same source and the tracing lost detail;
# this reads the geometry instead. Overwrites configs/evaluation/maps/ and syncs
# the held-out copies, so re-measure baselines after running it.
# Use it like: just fetch-maps
fetch-maps owner='' maps_dir='':
	@uv run python -m scripts.fetch_map_layouts "{{owner}}" "{{maps_dir}}"

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
# Clone a scripted baseline into the policy network, producing a checkpoint
# `train.py --warm-start-ckpt-path` accepts. Crosses the coordination gap a
# gradient cannot: every unilateral step toward advancing is downhill, while the
# joint advancing policy scores HIGHER on the training reward (30.29 v 24.77).
# Use: just behaviour-clone squad_march_shoot configs/golden/25v25_maps_coherency.yaml
behaviour-clone policy env_config n_episodes='200' epochs='8' out='checkpoints/clone.ckpt' seed='0' decode_topk='1' reallocate='0':
	@uv run python -m scripts.behaviour_clone {{policy}} {{env_config}} {{n_episodes}} {{epochs}} {{out}} {{seed}} {{decode_topk}} {{reallocate}}

# Does a policy USE the charge phase, and does it use it competently?
# Read `stood/ep` (numerator only, hard floor at zero, monotone in competence),
# NOT the standing fraction -- its denominator is the policy's own declaration
# count, so it rises when a policy declares less. Quote the K: at topk 3 the
# joint decoder picks legal combinations FOR the network, so the counts measure
# the decoder. Training decodes at K=1.
# Use: just measure-charges squad_march_take_charge configs/evaluation/25v25_maps_melee_refereed.yaml
measure-charges policy env_config n_episodes='20' decode_topk='1' *overrides:
	@uv run python -m scripts.measure_charges {{policy}} {{env_config}} {{n_episodes}} {{decode_topk}} {{overrides}}

# The melee goal's FOUR cells together, plus the decode headroom that says
# whether a gain is the policy or the decode. The goal is conjunctive, so the
# cells are scored and printed together -- quoting the best one is selection.
# Use: just measure-melee-ladder checkpoints/barclone-s1.ckpt approach 45 3 1
measure-melee-ladder policy family='approach' n_episodes='45' decode_topk='3' charge_decode='1' *overrides:
	@uv run python -m scripts.measure_melee_ladder {{policy}} {{family}} {{n_episodes}} {{decode_topk}} {{charge_decode}} {{overrides}}

# Does a critic-directed reallocation decode buy vp at play (the panel's R5 kill screen)
measure-realloc checkpoint env_config n_episodes='20' decode_topk='3' min_stack='4' *overrides:
	@uv run python -m scripts.measure_reallocation_decode {{checkpoint}} {{env_config}} {{n_episodes}} {{decode_topk}} {{min_stack}} {{overrides}}

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
measure-paired policy_a policy_b env_config n_episodes='100' seed_base='700000' *overrides:
	@uv run python -m scripts.measure_paired_policies {{policy_a}} {{policy_b}} {{env_config}} {{n_episodes}} {{seed_base}} {{overrides}}

# Where the travel reward actually points. `closest_objective_v2` is the only
# calculator that pays a model to move BETWEEN objectives, and two gates inside
# it decide where: a candidate test that only pays for arrivals flipping control
# THIS STEP by ONE model, and a per-objective assignment that can leave a unit
# with nothing and fall it through to "walk to your nearest". This reports what
# both gates decided, so a reward change is aimed at a measured cause.
# Use it like: just measure-shaping-gates <ckpt> configs/golden/25v25_maps_two_mode.yaml 30 "" 3
measure-shaping-gates policy env_config n_episodes='30' maps_dir='' decode_topk='1' *overrides:
	@uv run python -m scripts.measure_shaping_gates {{policy}} {{env_config}} {{n_episodes}} "{{maps_dir}}" "{{decode_topk}}" {{overrides}}

# How often an ordered move produces NO movement, and whether freezing sticks.
# A model that asks to move and does not is invisible in every score here:
# vp_margin sees the consequence, `coherent` sees the formation, and nothing
# counts the order that evaporated. Read `absorbing` first -- P(f|f) minus
# P(f|moved). Above zero means freezing is self-sustaining, i.e. a subset of the
# army is permanently out of the game rather than occasionally delayed.
# Matters most for the LONGEST moves: an advance is the most likely to be
# stopped, so an advance arm can measure "no effect" when the moves never ran.
# Use: just measure-freezing squad_march_take configs/golden/25v25_maps_two_mode.yaml 20
measure-freezing policy env_config n_episodes='20' maps_dir='' decode_topk='1' *overrides:
	@uv run python -m scripts.measure_freezing {{policy}} {{env_config}} {{n_episodes}} "{{maps_dir}}" "{{decode_topk}}" {{overrides}}

# Why an objective was not held: abandoned, narrowly lost, or lost by a mile.
# `held` alone cannot separate those, and they call for different fixes. Also
# reports the redistribution ceiling -- what any pure re-allocation lever could
# buy at best -- so a reward-shaping idea can be ruled out before it is trained.
# Takes a baseline name or a checkpoint path.
measure-objective-split policy env_config n_episodes='100' decode_topk='1':
	@uv run python -m scripts.measure_objective_split {{policy}} {{env_config}} {{n_episodes}} "{{decode_topk}}"

# How often a policy is in unit coherency (rules 03-moving.md), which this env
# does not enforce. Measures both forces, at the rules' 2"/9" and at the config's
# own group_max_distance, so the cost of adopting the rule is known before any
# mechanism is built. Takes a baseline name or a checkpoint path.
measure-coherency policy env_config n_episodes='30':
	@uv run python -m scripts.measure_coherency {{policy}} {{env_config}} {{n_episodes}}

# One annotated frame showing where a squad broke coherency. Rings the models
# cut off from their unit and draws a line to the body they left, using
# `evaluate_coherency` itself so the annotation cannot disagree with the metric.
# Run it on two checkpoints at the SAME seed and step to get comparable boards.
render-coherency-figure env_config ckpt out seed='700000' step='20':
	@uv run python -m scripts.render_coherency_figures {{env_config}} {{ckpt}} {{seed}} {{step}} {{out}}

# How much of a config's outcome spread is dice rather than policy. Holds the
# layouts fixed and varies only the combat seed, so the within-layout spread is
# the noise floor any arm-to-arm difference has to clear.
measure-noise-floor env_config n_layouts='10' n_combat_seeds='10' policy='' *overrides:
	@uv run python -m scripts.measure_noise_floor {{env_config}} {{n_layouts}} {{n_combat_seeds}} "{{policy}}" {{overrides}}

# Which of our units wants to meet which of theirs, before a model has moved.
# The unit-level REDUCTION of the per-model expected-damage matrix that already
# ships as an observation input -- attacker axis sums, defender axis does not.
# Static: reads the config, runs no episodes, needs no GPU.
# Range is NEVER folded into the damage number; it appears as `reach` and `free`
# (rounds of unanswered fire while the shorter gun closes) and as an exchange
# ratio quoted at two distances. On a mirror config every cell is the same
# number and the report says so.
# Use: just measure-matchups configs/experiments/30v15_fast_horde_vs_elite.yaml
measure-matchups env_config *overrides:
	@uv run python -m scripts.measure_matchups {{env_config}} {{overrides}}

# Where a policy stands, priced against where the opponent can shoot NEXT turn.
# ⚠ THE OPPONENT MOVES BEFORE IT SHOOTS, so the current-turn threat map reads
# FALSE-SAFE and this prints by how much. The census then splits every model's
# exposure into on-objective and in-transit, which is the falsifier for "the
# agent hoards because it is avoiding danger".
# ⚠ Cover is not applied, and every objective on the real tables is a ruin, so
# the on-objective column is OVERSTATED -- read it beside measure-hold-hazard.
# Use: just measure-threat-field squad_march_take configs/golden/25v25_maps_two_mode.yaml 5 configs/evaluation/maps_heldout
measure-threat-field policy env_config n_episodes='5' maps_dir='' decode_topk='1' *overrides:
	@uv run python -m scripts.measure_threat_field {{policy}} {{env_config}} {{n_episodes}} "{{maps_dir}}" "{{decode_topk}}" {{overrides}}

# Terrain-layout statistics for a random_terrain config: coverage, how often a
# sightline is blocked, and how much of the board is genuinely out of sight.
# Tune a terrain profile here, not after a thousand epochs of training.
# Record one game per table and write a GIF of each, for the README.
# Frames go straight to the GIF -- never via an mp4, which drifts every flat
# colour. Use: just record-gifs <policy|ckpt> <config> [table_a,table_b]
record-gifs policy env_config tables='' maps_dir='' out='' seed='' decode_topk='' width='':
	@uv run python -m scripts.record_gifs {{policy}} {{env_config}} "{{tables}}" "{{maps_dir}}" "{{out}}" "{{seed}}" "{{decode_topk}}" "{{width}}"

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
# [R] and [E] toggle the threat and engagement overlays live; pass them in `overlays`
# to start with them on, e.g. just play <cfg> <policy> tabletop "--threat-range".
# Use it like: just play · just play configs/dev/tiny.yaml random · just play <cfg> squad_march tabletop
play env_config_path='configs/golden/25v25_shooting_opponent.yaml' policy='squad_march_shoot' theme='default' overlays='':
	uv run play.py {{env_config_path}} {{policy}} {{theme}} {{overlays}}

# Step a match by hand and rewind it. Takes a baseline name or a .ckpt path.
# [R] shooting threat, [E] engagement range; `overlays` starts them on and can tune
# the sweep, e.g. "--threat-range --threat-grid 2.0 --threat-smoothing 0".
# Opens paused: [.] steps forward, [,] steps back, [Space] plays, [Tab] lists the keys.
# A config with `skip_phases: []` steps one sub-phase at a time instead of one round.
# Use it like: just debug · just debug configs/dev/tiny.yaml random · just debug <cfg> <run>/last.ckpt
debug env_config_path='configs/golden/25v25_shooting_opponent.yaml' driver='squad_march_shoot' theme='default' overlays='':
	uv run debug.py {{env_config_path}} {{driver}} {{theme}} {{overlays}}

# Recreate the episode a recording came from and step it by hand. The recording
# carries its own config, seed, dice and driver, so nothing else is needed --
# pass a driver only to override the one it names. Replays the recording's own
# actions until you change something, then the driver takes over.
# Use: just debug-recording recordings/my_events.jsonl
debug-recording file driver='squad_march_shoot' theme='default' overlays='':
	uv run debug.py --from-recording {{file}} configs/golden/25v25_shooting_opponent.yaml {{driver}} {{theme}} {{overlays}}

# Is the player seat advantaged, beyond the zone and the first turn? One policy
# plays BOTH seats over the balanced four legs, so its rating difference is zero
# by construction and whatever margin survives is the seat itself.
#
# NO RATING ON A CONFIG MEANS ANYTHING UNTIL THIS READS ZERO. The reward,
# coherency and exposure trackers all sample the player's army only, so seat
# symmetry is a claim rather than a fact until it is measured.
# Use it like: just measure-seat-parity configs/golden/25v25_shooting_opponent.yaml squad_march_shoot 30
measure-seat-parity env_config policy='squad_march_shoot' n_layouts='30':
	@uv run python -m scripts.measure_seat_parity {{env_config}} "{{policy}}" "{{n_layouts}}"

# Rate policies against each other on one scale, and fit the two structural
# advantages the board has. Each pairing plays four legs per layout -- every
# combination of (A's zone) x (who moves first) -- so the schedule is balanced
# in both axes and `h_zone` and `h_turn` are separately identifiable. Legs are
# appended to `ratings/<scenario>.json`, which is COMMITTED: a rating nobody can
# reproduce is worthless.
#
# The config must set both deployment zones explicitly and use equal armies;
# anything else is refused rather than measured, because a zone swap on derived
# zones is a silent no-op and `h_zone` would then be fitted from noise.
#
# WARNING: on a config whose own `turn_order` is `random`, a rated leg is played
# on a DIFFERENT layout stream from that config's `measure-baselines` numbers --
# side assignment draws from the layout RNG before terrain and objectives are
# placed. The four legs agree with each other, which is what the fit needs.
#
# Use it like: just measure-elo configs/golden/25v25_shooting_opponent.yaml 100 random squad_march squad_march_shoot
measure-elo env_config n_layouts='100' +entrants='random squad_march squad_march_shoot':
	@uv run python -m scripts.measure_elo {{env_config}} "{{n_layouts}}" {{entrants}}

# Fit and print the rating table from legs already played. Nothing is replayed,
# so an entrant can be added or the margin scale recalibrated without re-running
# a match. Read it beside `held` and `vp_margin` -- Elo ranks, it does not
# explain -- and quote the interval, not the point.
# Use it like: just elo-table configs/golden/25v25_shooting_opponent.yaml
elo-table env_config:
	@uv run python -m scripts.elo_table {{env_config}}

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

# The declaration census -- s31's S1 farm screen + S2 one-hot ablation data source
measure-declarations checkpoint env_config n_episodes='20' decode_topk='3' ablate='0' *overrides:
	@uv run python -m scripts.measure_declarations {{checkpoint}} {{env_config}} {{n_episodes}} {{decode_topk}} {{ablate}} {{overrides}}
