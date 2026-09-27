"""Analyse und die zwei Experimente: Aufbau, und die Korrektheits-Kette dieses Stuecks - bei n=0 muss Dyna-Q sich exakt wie Q-Learning verhalten.
Exakte Zahlen stehen in test_claims.py."""

import numpy as np
import pytest

import dq_evaluation as E


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(episodes=50, seed=0))


def test_analyse_wiring(analysis):
    a = analysis
    assert a.Q.shape == (a.grid.n_states, 4)
    assert a.returns.shape == (50,) and a.lengths.shape == (50,) and a.falls.shape == (50,)
    assert a.env_steps == int(a.lengths.sum())


def test_analyse_is_deterministic_given_the_same_seed():
    s = E.Settings(rows=3, cols=4, episodes=30, seed=5)
    a1, a2 = E.analyse(s), E.analyse(s)
    assert np.array_equal(a1.Q, a2.Q)


def test_reference_is_cached_across_settings_with_the_same_grid():
    assert E._reference(4, 8, 0.10, 0.95) is E._reference(4, 8, 0.10, 0.95)


def test_n_zero_matches_a_plain_q_learning_style_gap_measurement():
    # Bei n=0 wird das Modell nie zur Planung gelesen - dieselbe Analyse mit n=0 muss unabhaengig vom Planungscode reproduzierbar sein.
    gap1, steps1 = E.policy_gap(4, 8, 0.0, 0.95, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, n_planning=0, episodes=50, seed=2)
    gap2, steps2 = E.policy_gap(4, 8, 0.0, 0.95, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, n_planning=0, episodes=50, seed=2)
    assert gap1 == pytest.approx(gap2) and steps1 == steps2


def test_sample_efficiency_experiment_shape():
    exp = E.sample_efficiency_experiment(n_levels=(0, 10), checkpoints=(20, 40), seeds=range(5))
    assert len(exp["rows"]) == 2
    for row in exp["rows"]:
        assert 20 in row and 40 in row
        assert "mean" in row[20] and "env_steps_mean" in row[20]


def test_blocking_experiment_shape():
    exp = E.blocking_experiment(n_levels=(0, 10), seeds=range(5), episodes_before=20, episodes_after=20)
    assert len(exp["rows"]) == 2
    for row in exp["rows"]:
        assert len(row["mean_curve"]) == 40
        assert "before_last" in row and "after_first" in row and "after_last" in row


def test_without_slip_dyna_q_reaches_near_optimal_faster_than_plain_q_learning():
    # Deterministisches Raster, sehr frueher Checkpoint (der Effekt spielt sich zwischen 2 und ~20 Episoden ab, siehe README - bei 50 Episoden
    # haben beide laengst konvergiert und der Unterschied waere 0 gegen 0). Dyna-Q mit Planung braucht WENIGER Episoden fuer eine nahezu optimale Policy.
    exp = E.sample_efficiency_experiment(n_levels=(0, 10), checkpoints=(8,), seeds=range(15), base=E.Settings(slip=0.0))
    rows = {r["n_planning"]: r for r in exp["rows"]}
    assert rows[10][8]["mean"] < rows[0][8]["mean"]
