# Wargame RL Project Analysis


## Coding Principles for Claude (Python)

When generating or modifying Python code, follow these principles strictly.

### 1. Prefer Simplicity Over Cleverness

- Write the simplest solution that works.
- Avoid unnecessary abstractions, metaclasses, decorators, or design patterns unless clearly justified.
- Do not optimize prematurely.
- Prefer explicit code over "magic".
- If a one-liner reduces readability, expand it.

Bad:
```python
return [f(x) for x in items if x and not isinstance(x, (int, float))]
````

Better:

```python
result = []

for item in items:
    if not item:
        continue

    if isinstance(item, (int, float)):
        continue

    result.append(f(item))

return result
```

Clarity > brevity.

---

### 2. Code Should Read Like Prose

* Use descriptive variable and function names.
* Avoid single-letter variables (except trivial loop indices).
* Avoid abbreviations unless universally understood.
* Functions should do one thing.

Good:

```python
def calculate_total_price(items: list[Item]) -> float:
    ...
```

Bad:

```python
def calc(tp):
    ...
```

---

### 3. Keep Functions Small and Focused

* Each function should have a single responsibility.
* Prefer multiple small functions over one large function.
* If a function exceeds ~30–40 lines, consider splitting it.

---

### 4. Avoid Over-Engineering

* Do not introduce classes if a function is enough.
* Do not introduce dependency injection unless required.
* Do not introduce frameworks or libraries unless explicitly requested.
* Prefer standard library.

---

### 5. Explicit Is Better Than Implicit

Follow the spirit of PEP 20:

* Make data transformations obvious.
* Avoid hidden side effects.
* Avoid mutating arguments unless clearly documented.
* Be explicit about return values.

---

### 6. Type Hints Are Required

* Use Python type hints for all public functions.
* Prefer built-in generics (`list[str]`, `dict[str, int]`) over `typing.List`.
* Avoid overly complex type constructs.

Example:

```python
def group_users_by_role(users: list[User]) -> dict[str, list[User]]:
    ...
```

---

### 7. Error Handling Should Be Clear

* Raise meaningful exceptions.
* Do not swallow exceptions silently.
* Avoid broad `except Exception` unless re-raising.

Bad:

```python
try:
    do_something()
except Exception:
    pass
```

Better:

```python
try:
    do_something()
except ValueError as error:
    raise ConfigurationError("Invalid configuration value") from error
```

---

## Testing Principles

Tests must be written for humans first.

### 1. Tests Should Explain Behavior

* Test names must describe behavior, not implementation.
* A reader should understand system behavior by reading tests.

Good:

```python
def test_calculate_total_price_applies_discount_for_premium_user():
    ...
```

Bad:

```python
def test_total_price_1():
    ...
```

---

### 2. Test Behavior, Not Implementation

* Do not test private methods.
* Do not test internal variables.
* Test observable outcomes only.

---

### 3. Each Test Should Have One Clear Purpose

* One logical assertion per behavior.
* Avoid large “kitchen sink” tests.
* If a test needs many asserts, split it.

---

### 4. Use Simple Arrange–Act–Assert Structure

Tests should follow:

```python
def test_something():
    # Arrange
    ...

    # Act
    result = ...

    # Assert
    assert result == ...
```

Make the three phases visually clear.

---

### 5. Avoid Over-Mocking

* Prefer real objects over mocks.
* Only mock external systems (API, DB, filesystem).
* If a test needs heavy mocking, the design may be too complex.

---

### 6. Tests Should Be Deterministic

* No randomness without a fixed seed.
* No dependency on system time (unless controlled).
* No dependency on network.

---

### 7. Edge Cases Must Be Explicit

For every non-trivial function, consider:

* Empty input
* Invalid input
* Boundary values
* Typical valid case

---

## Documentation and Comments

* Write docstrings for public functions explaining:

  * What it does
  * Expected inputs
  * Return value
  * Possible exceptions
* Avoid obvious comments.
* Explain *why*, not *what*.

Bad:

```python
# increment i
i += 1
```

Good:

```python
# We retry once because the external service is eventually consistent.
```

---

## Final Rule

The code should look like it was written by a thoughtful, pragmatic senior engineer — not a code generator trying to be clever.

Prioritize:

* Readability
* Maintainability
* Explicitness
* Behavioral clarity in tests

If forced to choose between elegance and clarity, choose clarity.






## Project Overview
This is a reinforcement learning environment for wargame simulations built with:
- PyTorch for deep learning models
- Gymnasium for environment interface
- Python with modern development tools (UV, Just, Ruff, etc.)

## Key Components

### Environment
- `WargameEnv` in `wargame_rl.wargame.envs.wargame`
- Supports configurable number of wargame models and objectives
- 2D board environment with movement actions
- Action space: Tuple(Discrete(5), Discrete(5)) where 5 actions = [up, down, left, right, stay]

### Models
- `DQN_MLP` - Multi-layer perceptron DQN (Legacy model, support will be dropped in the future)
- `DQN_Transformer` - Transformer-based DQN (This is the model we are currently using)
- Both models support different network architectures

### Configuration
- Environment configs in `examples/env_config/`
- Support for different board sizes and numbers of models/objectives
- Configurable parameters like objective radius, group cohesion, etc.
- Support for opponent deployment zones (recently added feature)

## How to Run

Look at the justfile for available commands.

### Environment Test
You can use pytest
```bash
uv run pytest --verbose tests
```

### Training (with example config)
```bash
# Without wandb (since it requires API key)
python train.py --env-config-path examples/env_config/example.yaml --max-epochs 1 --network-type transformer
```

### Simulation
For now, you cannot really see what happens. So simulating is mostly to see if the codes runs.

## Key Features
1. Configurable wargame environments with different board sizes
2. Support for multiple wargame models and objectives
3. Both MLP and Transformer DQN architectures
4. Proper reinforcement learning interface with Gymnasium
5. Modular design with clear separation of environment and model components
6. Support for opponent deployment zones (recently added feature)
7. Support for per-entity configuration in environment configs

## Recent Changes
- Added support for opponent deployment zones to enable adversarial play
- Added support for optional per-entity configuration in environment configs
- Enhanced wandb logging with video recording capabilities

## Current Status
The environment and models work correctly. The main limitation is wandb configuration, but the core functionality is intact. The DQN_Transformer model is the current focus for training and development.
