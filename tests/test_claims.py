"""Jede Zahl aus dem README, nachgerechnet ueber die echten Auswertungsfunktionen. Dyna-Q ist stochastisch: Mehr-Seed-Zahlen tragen grosszuegige
Baender (CI-robust), der Standardfall (fester Seed) ist exakt."""

import pytest

import dq_constants as C
import dq_evaluation as E


def test_v_star_start():
    grid, P, R, V_star, _, _ = E._reference(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA)
    assert V_star[grid.state_of(grid.start)] == pytest.approx(-0.10, abs=0.02)


def test_standard_case():
    a = E.analyse(E.Settings())
    baseline = E.analyse(E.Settings(n_planning=0))
    assert a.gap == pytest.approx(0.0, abs=0.01)
    assert baseline.gap == pytest.approx(19.90, abs=1.0)
    assert a.env_steps == pytest.approx(3617, abs=200)
    assert baseline.env_steps == pytest.approx(4308, abs=200)


def test_sample_efficiency_experiment():
    exp = E.sample_efficiency_experiment()
    rows = {r["n_planning"]: r for r in exp["rows"]}
    assert rows[0][8]["mean"] == pytest.approx(17.25, abs=5.0)
    assert rows[5][8]["mean"] < 2.0
    assert rows[10][8]["mean"] < 2.0
    assert rows[25][8]["mean"] < 2.0
    assert rows[0][20]["mean"] == pytest.approx(1.33, abs=2.0)


def test_slip_reverses_the_planning_advantage():
    # README (Modell/Grenzen): bei Rutschen 0,10 schadet Planung (n=10), bei 0,02 noch nicht - 15 Seeds, 50 Episoden
    def means(slip):
        exp = E.sample_efficiency_experiment(n_levels=(0, 10), checkpoints=(50,), base=E.Settings(slip=slip))
        return {r["n_planning"]: r[50]["mean"] for r in exp["rows"]}
    m10, m02 = means(0.10), means(0.02)
    assert m10[10] == pytest.approx(12.73, abs=3.0) and m10[0] == pytest.approx(1.18, abs=1.0)
    assert m10[10] > m10[0] + 5.0
    assert m02[10] == pytest.approx(2.90, abs=1.5) and m02[0] == pytest.approx(3.38, abs=2.5)
    assert m02[10] < m02[0] + 1.0                                                                    # bei 0,02 noch kein Nachteil


def test_blocking_experiment():
    exp = E.blocking_experiment(base=E.Settings(slip=0.0, epsilon_start=0.1, epsilon_decay=0.0))
    rows = {r["n_planning"]: r for r in exp["rows"]}
    for n, row in rows.items():
        assert row["after_first"] < row["before_last"]  # der Ertrag faellt bei JEDEM n direkt nach dem Wechsel
        assert row["after_last"] > row["after_first"]    # ... und erholt sich danach wieder (nicht notwendig vollstaendig)
    assert rows[0]["before_last"] == pytest.approx(-45.95, abs=15.0)
    assert rows[0]["after_first"] == pytest.approx(-94.38, abs=20.0)
