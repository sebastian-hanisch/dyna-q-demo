"""Dyna-Q von Hand: die Q-Learning-Update-Formel (fuer echte UND Planungs-Schritte identisch), das Modell-Dictionary, gierige/erkundende
Aktionswahl, eine unabhaengige Schritt-fuer-Schritt-Nachrechnung, UND die zentrale Korrektheits-Kette: bei n_planning=0 wird das Modell nie benutzt -
Dyna-Q reduziert sich dann EXAKT auf reines Q-Learning (identische Trajektorie, byte-gleiche Q-Tabelle bei gleichem Seed)."""

import numpy as np
import pytest

import dq_agent as A
import dq_constants as C
from dq_grid import Grid, step


def test_choose_action_epsilon_zero_is_always_greedy():
    Q = np.zeros((1, 4))
    Q[0] = [1.0, 5.0, 2.0, 0.0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert A.choose_action(Q, 0, epsilon=0.0, rng=rng) == 1


def test_q_update_by_hand():
    Q = np.array([[1.0, 2.0], [3.0, 4.0]])
    A.q_update(Q, s=0, a=0, r=1.0, s_next=1, done=False, alpha=0.5, gamma=0.9)
    target = 1.0 + 0.9 * 4.0
    assert Q[0, 0] == pytest.approx(1.0 + 0.5 * (target - 1.0))


def test_q_update_terminal_ignores_the_bootstrap():
    Q = np.array([[1.0, 2.0], [100.0, 100.0]])
    A.q_update(Q, s=0, a=0, r=10.0, s_next=1, done=True, alpha=0.5, gamma=0.9)
    assert Q[0, 0] == pytest.approx(1.0 + 0.5 * (10.0 - 1.0))


def test_run_episode_records_the_last_observed_outcome_in_the_model():
    grid = Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    Q = np.zeros((grid.n_states, A.N_ACTIONS))
    model = {}
    rng = np.random.default_rng(0)
    A.run_episode(grid, Q, model, n_planning=0, epsilon=0.5, alpha=0.1, rng=rng, max_steps=20)
    assert len(model) > 0
    for (s, a), (r, s_next, done) in model.items():
        assert isinstance(r, float) and 0 <= s_next < grid.n_states


def test_run_episode_reproduces_a_hand_composed_step_by_step_replay_with_planning():
    # Deterministisches Raster, damit die einzige Zufaelligkeit die epsilon-gierige Wahl (real UND Planung) ist.
    grid = Grid(rows=4, cols=8, slip=0.0, gamma=0.9)
    rng_a = np.random.default_rng(11)
    Q_a = np.zeros((grid.n_states, A.N_ACTIONS))
    model_a = {}
    total_a, steps_a, falls_a = A.run_episode(grid, Q_a, model_a, n_planning=3, epsilon=0.3, alpha=0.2, rng=rng_a, max_steps=15)

    rng_b = np.random.default_rng(11)
    Q_b = np.zeros((grid.n_states, A.N_ACTIONS))
    model_b = {}
    s = grid.state_of(grid.start)
    total_b, steps_b, falls_b = 0.0, 0, 0
    for _ in range(15):
        a = A.choose_action(Q_b, s, 0.3, rng_b)
        s_next, r, done = step(grid, s, a, rng_b)
        if r == C.CLIFF_PENALTY:
            falls_b += 1
        A.q_update(Q_b, s, a, r, s_next, done, alpha=0.2, gamma=grid.gamma)
        model_b[(s, a)] = (r, s_next, done)
        keys = list(model_b.keys())
        for _ in range(3):
            ks, ka = keys[int(rng_b.integers(len(keys)))]
            kr, ks_next, kdone = model_b[(ks, ka)]
            A.q_update(Q_b, ks, ka, kr, ks_next, kdone, alpha=0.2, gamma=grid.gamma)
        total_b += r
        steps_b += 1
        if done:
            break
        s = s_next
    assert steps_a == steps_b and total_a == pytest.approx(total_b) and falls_a == falls_b
    assert np.array_equal(Q_a, Q_b)


def test_zero_planning_steps_reduce_dyna_q_exactly_to_plain_q_learning():
    # Der zentrale Korrektheits-Check dieses Stuecks: ohne Planung wird das Modell nie gelesen, nur geschrieben - identisch zu reinem Q-Learning.
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    rng = np.random.default_rng(5)
    Q_dyna = np.zeros((grid.n_states, A.N_ACTIONS))
    model = {}
    A.run_episode(grid, Q_dyna, model, n_planning=0, epsilon=0.2, alpha=0.1, rng=rng, max_steps=50)

    rng2 = np.random.default_rng(5)
    Q_plain = np.zeros((grid.n_states, A.N_ACTIONS))
    s = grid.state_of(grid.start)
    for _ in range(50):
        a = A.choose_action(Q_plain, s, 0.2, rng2)
        s_next, r, done = step(grid, s, a, rng2)
        A.q_update(Q_plain, s, a, r, s_next, done, alpha=0.1, gamma=grid.gamma)
        if done:
            break
        s = s_next
    assert np.array_equal(Q_dyna, Q_plain)


def test_train_snapshot_at_a_checkpoint_equals_a_separately_truncated_run():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    _, _, _, _, snapshots = A.train(grid, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, n_planning=10, episodes=100, seed=3, checkpoints=(30,))
    Q_short, _, _, _, _ = A.train(grid, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, n_planning=10, episodes=30, seed=3)
    assert np.array_equal(snapshots[30], Q_short)


def test_train_switching_before_phase_matches_a_plain_train_on_the_same_grid():
    # Vor dem Wechsel ist train_switching nichts anderes als train() auf grid_before - dieselbe RNG-Reihenfolge, dieselbe Q-Tabelle.
    grid_before = Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    grid_after = Grid(rows=4, cols=8, slip=0.0, gamma=0.95, blocked=frozenset({(2, 3)}))
    Q_switch, returns_switch, _, _ = A.train_switching(grid_before, grid_after, episodes_before=20, episodes_after=0, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.0, n_planning=5, seed=7)
    Q_plain, returns_plain, _, _, _ = A.train(grid_before, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.0, n_planning=5, episodes=20, seed=7)
    assert np.array_equal(Q_switch, Q_plain) and np.allclose(returns_switch, returns_plain)


def test_train_returns_arrays_of_the_right_shape():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    Q, returns, lengths, falls, snapshots = A.train(grid, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, n_planning=10, episodes=30, seed=0)
    assert Q.shape == (grid.n_states, A.N_ACTIONS)
    assert returns.shape == (30,) and lengths.shape == (30,) and falls.shape == (30,)
    assert np.all(lengths >= 1) and np.all(lengths <= C.MAX_STEPS_PER_EPISODE)
