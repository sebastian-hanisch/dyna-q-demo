"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Trainingsstand-Slider, Permalink-Grenzen/-Raster, Extremwerte, zwei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dq_constants as C
import dq_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info) + list(at.error):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 4 and len(at.get("plotly_chart")) >= 4
    assert len(at.success) + len(at.warning) + len(at.info) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == pytest.approx(p[key]) if isinstance(p[key], float) else at.session_state[state_key] == p[key]


def test_frame_slider_survives_a_smaller_episode_count():
    at = _run(dq_frame_idx=5)
    _ok(at)
    at.slider(key="episodes_slider").set_value(C.EPISODES_MIN).run()
    _ok(at)


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rows"] = "999"
    at.query_params["cols"] = "-5"
    at.query_params["slip"] = "abc"
    at.query_params["n"] = "999"
    at.query_params["episodes"] = "999999"
    at.run()
    _ok(at)
    s = at.session_state
    assert s["rows_slider"] == C.ROWS_MAX and s["cols_slider"] == C.COLS_MIN
    assert s["slip_slider"] == C.DEFAULT_SLIP and s["n_planning_slider"] == C.N_PLANNING_MAX and s["episodes_slider"] == C.EPISODES_MAX


@pytest.mark.parametrize("kw", [
    dict(rows_slider=C.ROWS_MIN, cols_slider=C.COLS_MIN, n_planning_slider=C.N_PLANNING_MIN),
    dict(rows_slider=C.ROWS_MAX, cols_slider=C.COLS_MAX, n_planning_slider=C.N_PLANNING_MAX, episodes_slider=C.EPISODES_MIN),
    dict(alpha_slider=C.ALPHA_MIN, epsilon_decay_slider=C.EPSILON_DECAY_MIN),
    dict(alpha_slider=C.ALPHA_MAX, epsilon_decay_slider=C.EPSILON_DECAY_MAX, slip_slider=C.SLIP_MAX),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_sample_efficiency_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_N_LEVELS", (0, 10))
    monkeypatch.setattr(C, "EXP_EPISODE_CHECKPOINTS", (20, 40))
    monkeypatch.setattr(C, "EXP_SEEDS", 3)
    at = _run()
    _click(at, "sample_start")
    assert at.session_state["sample_on"] and any("Befund" in w.value for w in at.warning)


def test_blocking_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_N_LEVELS", (0, 10))
    monkeypatch.setattr(C, "EXP_SEEDS", 3)
    monkeypatch.setattr(C, "BLOCK_EPISODES_BEFORE", 15)
    monkeypatch.setattr(C, "BLOCK_EPISODES_AFTER", 15)
    at = _run()
    _click(at, "block_start")
    assert at.session_state["block_on"] and any("Befund" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
