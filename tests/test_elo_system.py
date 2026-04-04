from wargame_rl.wargame.model.common.elo import EloRatingSystem
from wargame_rl.wargame.model.ppo.elo_evaluator import episode_score_from_vp


def test_episode_score_from_vp() -> None:
    assert episode_score_from_vp(10, 5) == 1.0
    assert episode_score_from_vp(5, 10) == 0.0
    assert episode_score_from_vp(7, 7) == 0.5


def test_elo_update_win_loss_and_draw() -> None:
    elo = EloRatingSystem(initial_rating=1000.0, k_factor=32.0)
    elo.ensure_player("a")
    elo.ensure_player("b")

    a1, b1 = elo.update("a", "b", 1.0)
    assert a1 > 1000.0
    assert b1 < 1000.0

    a2, b2 = elo.update("a", "b", 0.0)
    assert a2 < a1
    assert b2 > b1

    a3, b3 = elo.update("a", "b", 0.5)
    assert abs((a3 - a2) + (b3 - b2)) < 1e-6
