from wargame_rl.func import random_sum


def test_random_sum():
    assert random_sum(1) < 101
    assert random_sum(100) < 200
