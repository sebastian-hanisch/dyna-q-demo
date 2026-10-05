"""Unabhängige Orakel für Dyna-Q: Übergangsmodell inkl. `blocked` per Koordinaten-Neuimplementierung, V* per LP (scipy HiGHS), Policy-Wert per Lineargleichungssystem,
`train`/`train_switching` Schritt für Schritt gegen eine Schleifen-Neuimplementierung (eigene Modell-Liste, identischer Zufallsstrom), Kennzahlen des
Blocking-Maze-Experiments aus dem Nachbau, und exakte Konvergenz auf Q* im deterministischen Raster."""

import numpy as np
import pytest

import dq_agent as A
import dq_constants as C
import dq_evaluation as E
import dq_grid as G
import dq_reference as R

linprog = pytest.importorskip("scipy.optimize").linprog

_MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_NAME = {0: "N", 1: "S", 2: "E", 3: "W"}
_SIDES = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}


def _outcomes(rows, cols, slip, blocked, s, a):
    r, c = divmod(s, cols)
    goal = (rows - 1, cols - 1)
    if (r, c) == goal:
        return [(1.0, s, 0.0, True)]
    hazard = {(rows - 1, k) for k in range(1, cols - 1)} | set(blocked)
    m = _NAME[a]
    out = []
    for p, d in ((1 - slip, m), (slip / 2, _SIDES[m][0]), (slip / 2, _SIDES[m][1])):
        nr, nc = r + _MOVES[d][0], c + _MOVES[d][1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = r, c
        if (nr, nc) == goal:
            out.append((p, nr * cols + nc, 10.0, True))
        elif (nr, nc) in hazard:
            out.append((p, (rows - 1) * cols, -100.0, False))
        else:
            out.append((p, nr * cols + nc, -1.0, False))
    return out


def _vstar_lp(P, Rw, gamma):
    S, nA = Rw.shape
    rows = []
    for s in range(S):
        for a in range(nA):
            row = gamma * P[s, a].copy()
            row[s] -= 1.0
            rows.append(row)
    return linprog(np.ones(S), A_ub=np.array(rows), b_ub=-Rw.reshape(-1), bounds=[(None, None)] * S, method="highs").x


def _replay(phases, alpha, eps0, decay, n_plan, seed):
    """phases: Liste (rows, cols, slip, gamma, blocked, episodes); Q und Modell laufen über die Phasen weiter."""
    rng = np.random.default_rng(seed)
    rows, cols = phases[0][0], phases[0][1]
    Q = [[0.0] * 4 for _ in range(rows * cols)]
    seen, outcome = [], {}
    rets, lens, falls = [], [], []
    e = 0
    for rows, cols, slip, gamma, blocked, neps in phases:
        for _ in range(neps):
            eps = eps0 if decay <= 0 else max(0.01, eps0 / (1 + decay * e))
            e += 1
            s, tot, n, nf = (rows - 1) * cols, 0.0, 0, 0
            for _ in range(400):
                if rng.random() < eps:
                    a = int(rng.integers(4))
                else:
                    best = [k for k in range(4) if Q[s][k] == max(Q[s])]
                    a = best[0] if len(best) == 1 else int(rng.choice(np.array(best)))
                u = rng.random()
                k = 0 if (slip == 0 or u < 1 - slip) else (1 if u < 1 - slip / 2 else 2)
                _, s2, r, d = _outcomes(rows, cols, slip, blocked, s, a)[k]
                nf += r == -100.0
                Q[s][a] += alpha * ((r if d else r + gamma * max(Q[s2])) - Q[s][a])
                if (s, a) not in outcome:
                    seen.append((s, a))
                outcome[(s, a)] = (r, s2, d)
                for _ in range(n_plan):
                    ps, pa = seen[int(rng.integers(len(seen)))]
                    pr, ps2, pd = outcome[(ps, pa)]
                    Q[ps][pa] += alpha * ((pr if pd else pr + gamma * max(Q[ps2])) - Q[ps][pa])
                tot, n = tot + r, n + 1
                if d:
                    break
                s = s2
            rets.append(tot)
            lens.append(n)
            falls.append(nf)
    return np.array(Q), np.array(rets), np.array(lens), np.array(falls)


def test_train_and_train_switching_replay_step_by_step_on_the_same_random_stream():
    rng = np.random.default_rng(21)
    for _ in range(25):
        rows, cols = int(rng.integers(3, 6)), int(rng.integers(4, 8))
        slip, gamma, alpha = float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.8, 0.95])), float(rng.choice([0.05, 0.5]))
        eps0, decay, npl = float(rng.choice([0.2, 1.0])), float(rng.choice([0, 0.02])), int(rng.choice([0, 1, 5, 10]))
        ep, ep2, seed = int(rng.integers(1, 20)), int(rng.integers(1, 12)), int(rng.integers(1000))
        g = G.Grid(rows, cols, slip, gamma)
        Q, ret, ln, fl, _ = A.train(g, alpha, eps0, decay, npl, ep, seed)
        Qo, ro, lo, fo = _replay([(rows, cols, slip, gamma, (), ep)], alpha, eps0, decay, npl, seed)
        assert np.allclose(Q, Qo, atol=1e-12, rtol=0) and np.array_equal(ret, ro) and np.array_equal(ln, lo) and np.array_equal(fl, fo)
        bl = frozenset((rows - 2, c) for c in range(2, min(5, cols - 1)))
        Q2, r2, _, f2 = A.train_switching(g, G.Grid(rows, cols, slip, gamma, blocked=bl), ep, ep2, alpha, eps0, decay, npl, seed)
        Qo2, ro2, _, fo2 = _replay([(rows, cols, slip, gamma, (), ep), (rows, cols, slip, gamma, tuple(bl), ep2)], alpha, eps0, decay, npl, seed)
        assert np.allclose(Q2, Qo2, atol=1e-12, rtol=0) and np.array_equal(r2, ro2) and np.array_equal(f2, fo2)


def test_model_with_blocked_cells_reference_and_policy_value_match_independent_computations():
    rng = np.random.default_rng(3)
    for rows, cols, slip, gamma in [(3, 4, 0.1, 0.9), (4, 8, 0.0, 0.95), (4, 6, 0.3, 0.99)]:
        bl = frozenset((rows - 2, c) for c in range(2, min(5, cols - 1)))
        g = G.Grid(rows, cols, slip, gamma, blocked=bl)
        P, Rw = G.build_model(g)
        goal_s = g.state_of(g.goal)
        for s in range(g.n_states):
            for a in range(4):
                Pm, Rm = np.zeros(g.n_states), 0.0
                for p, s2, r, d in _outcomes(rows, cols, slip, bl, s, a):
                    Pm[goal_s if d else s2] += p
                    Rm += p * r
                assert np.allclose(P[s, a], Pm) and Rw[s, a] == pytest.approx(Rm)
        V, _, _ = R.value_iteration(P, Rw, gamma)
        assert np.max(np.abs(V - _vstar_lp(P, Rw, gamma))) < 1e-5
        idx = np.arange(g.n_states)
        pol = rng.integers(0, 4, g.n_states)
        exact = np.linalg.solve(np.eye(g.n_states) - gamma * P[idx, pol], Rw[idx, pol])
        assert np.max(np.abs(exact - R.policy_evaluation(P, Rw, pol, gamma))) < 1e-4 * max(1.0, np.max(np.abs(exact)))


def test_blocking_experiment_metrics_and_analysis_against_the_replay():
    base = E.Settings(slip=0.0, epsilon_start=0.1, epsilon_decay=0.0)
    exp = E.blocking_experiment(n_levels=(0, 5), seeds=range(3), base=base)
    blocked = tuple((base.rows - 2, c) for c in C.BLOCK_COLS)
    for row in exp["rows"]:
        arr = np.array([_replay([(base.rows, base.cols, 0.0, base.gamma, (), 30), (base.rows, base.cols, 0.0, base.gamma, blocked, 30)], base.alpha, 0.1, 0.0, row["n_planning"], sd)[1] for sd in range(3)])
        assert row["before_last"] == pytest.approx(arr[:, 20:30].mean())
        assert row["after_first"] == pytest.approx(arr[:, 30:40].mean())
        assert row["after_last"] == pytest.approx(arr[:, -10:].mean())
        assert np.allclose(row["mean_curve"], arr.mean(axis=0))
    for npl in (0, 10):
        a = E.analyse(E.Settings(n_planning=npl, episodes=12, seed=3))
        P, Rw = G.build_model(a.grid)
        idx, s0 = np.arange(a.grid.n_states), a.grid.state_of(a.grid.start)
        Vpi = np.linalg.solve(np.eye(a.grid.n_states) - a.grid.gamma * P[idx, a.policy], Rw[idx, a.policy])
        assert a.gap == pytest.approx(_vstar_lp(P, Rw, a.grid.gamma)[s0] - Vpi[s0], abs=1e-4)
        assert a.env_steps == int(_replay([(4, 8, 0.0, 0.95, (), 12)], 0.1, 1.0, 0.005, npl, 3)[2].sum())


@pytest.mark.parametrize("n_planning", [0, 10])
def test_deterministic_grid_converges_exactly_to_qstar_on_all_visited_pairs(n_planning):
    g = G.Grid(3, 4, 0.0, 0.9)
    P, Rw = G.build_model(g)
    Qstar = Rw + 0.9 * P @ _vstar_lp(P, Rw, 0.9)
    Q = A.train(g, 0.5, 1.0, 0.0, n_planning, 300, 0)[0]
    mask = np.ones_like(Q, bool)
    mask[g.state_of(g.goal)] = False
    for rc in g.cliff:
        mask[g.state_of(rc)] = False                                                                 # Klippenzellen werden nie betreten
    assert np.max(np.abs(Q[mask] - Qstar[mask])) < 1e-3


def test_app_verdict_follows_the_measured_gap_not_the_slip_setting():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    # Rutschen 0,10, 5 Episoden, n=5 (Seed 0): Planung ist hier messbar BESSER (Abstand 2,43 gegen 11,12) - die App darf nicht "hilft nicht" melden.
    a, base = E.analyse(E.Settings(slip=0.1, n_planning=5, episodes=5)), E.analyse(E.Settings(slip=0.1, n_planning=0, episodes=5))
    assert a.gap < base.gap - 5.0
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=600)
    for k, v in {"slip_slider": 0.1, "n_planning_slider": 5, "episodes_slider": 5}.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception
    assert not any("Hier hilft Planung nicht" in el.value for el in at.warning)
    assert any("kleineren oder gleich guten" in el.value for el in at.success)
