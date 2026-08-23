"""Rating: put scripted baselines and learned checkpoints on one scale.

This package sits **above** `envs/` and beside `model/`. `score.py` and `elo.py`
import numpy and nothing from this repo at all, so the rating mathematics is
testable on synthetic arrays with no environment, no torch and no I/O.
`tests/test_import_direction.py` pins that.
"""
